"""
MoE Action Expert: Shared Attention + Expert FFN for WAM.

Inspired by BAGEL's Mixture-of-Transformer-Experts (MoT) pattern:
- Action tokens are concatenated to the video token sequence
- Shared self-attention: action and video tokens attend to each other
  using the VIDEO DiT's Q/K/V projections (shared representational space)
- Expert FFN: at designated layers, action tokens receive additional
  expert FFN correction for modality-specific capacity
- Deterministic routing: no learned gate, routing by token type

This achieves cross-modal grounding via shared attention while preserving
modality-specific representational capacity via expert FFN layers.

Architecture at every DiT layer (PT1-0b alignment — previously only a
subset of layers had expert FFNs):
    Before DiT block: action tokens appended to video sequence
    DiT block self-attention: all tokens attend to each other (shared)
    DiT block FFN: standard video FFN processes all tokens (base transform)
    After DiT block: expert FFN corrects action token representations
    After all blocks: action tokens extracted, projected to action_dim

References:
- BAGEL (ByteDance Seed): Shared attention + expert FFN for multimodal
  understanding and generation (arXiv:2505.14683)
- DreamZero: Shared backbone WAM with action+video in same DiT
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from openwam.model.action_backbone.backbone import ActionBackbone
from openwam.model.action_backbone.components import (
    ActionEncoder,
    ActionOutputMLP,
    LearnedPositionalEncoding,
    TimestepEmbedding,
    TimestepModulation,
    sinusoidal_embedding_1d,  # noqa: F401
)

if TYPE_CHECKING:
    from openwam.model.base import ActionState, ExecutionPlan
    from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone


@dataclass
class MoEExpertState:
    """State container for MoE expert execution in model_fn_wan_video.

    Threaded through the video DiT block loop. Action tokens are
    concatenated to the video sequence before the loop, and the state
    tracks expert FFN application at designated layers.

    Fields:
        moe_dit: Reference to MoEExpertDiT module.
        action_tokens: (B, T_action, video_dim) projected action tokens.
            Updated in-place as they flow through the block loop.
        t_mod: Timestep modulation for expert FFN. Shape depends on the
            diffusion timestep granularity chosen by the loss:
                (B, 3, video_dim)     — per-sample (default).
                (B, T, 3, video_dim)  — per-token (action_timestep_per_token).
            ExpertFFNBlock.forward accepts both.
        t_embed: (B, video_dim) timestep embedding for output head.
        n_action_tokens: Number of action tokens appended to sequence.
        timestep: Raw action diffusion timestep preserved at the granularity
            sampled by the loss. Shape:
                (1,)  scalar broadcast across the batch (1-sample inference).
                (B,)  per-sample (default).
                (B, T_action) per-token (action_timestep_per_token).
            Consumed by model_fn_wan_video._build_action_t_mod to build
            action-position t_mod rows for the video DiT's AdaLN.
        expert_block_counter: Tracks which expert block to apply next.
        skip_prefix_tokens: Number of reference-frame prefix tokens in
            the video sequence that action tokens should NOT attend to
            during RoPE (they still attend via content-based attention).
        action_noise_pred: Filled after finalize_output.
    """

    moe_dit: "MoEExpertDiT"
    action_tokens: torch.Tensor
    t_mod: torch.Tensor
    t_embed: torch.Tensor
    n_action_tokens: int = 0
    # Raw action diffusion timestep preserved at loss-time granularity:
    # (1,) broadcast, (B,) per-sample, or (B, T_action) per-token.
    # Consumed by model_fn_wan_video._build_action_t_mod (see wan_video.py).
    timestep: Optional[torch.Tensor] = None
    expert_block_counter: int = 0
    skip_prefix_tokens: int = 0
    action_noise_pred: Optional[torch.Tensor] = None
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False


class ExpertFFNBlock(nn.Module):
    """Single expert FFN layer with AdaLN modulation.

    Applied as a residual correction to action tokens at designated
    video DiT layers. The standard video FFN has already processed
    the action tokens (since they're in the shared sequence); this
    expert provides modality-specific refinement.

    Computation:
        h = LayerNorm(x) * (1 + scale) + shift
        x = x + gate * FFN(h)
    """

    def __init__(self, dim: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        # AdaLN base modulation (3 params: shift, scale, gate)
        self.modulation = nn.Parameter(torch.randn(1, 3, dim) / dim**0.5)

        # Zero-initialize output so expert correction starts at zero,
        # preserving pretrained video DiT behavior at initialization.
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN-modulated expert FFN as a residual correction.

        Args:
            x: (B, T_action, dim) action tokens after the video DiT block.
            t_mod: Timestep modulation. Two supported layouts:
                - (B, 3, dim): per-sample modulation, broadcast across tokens.
                - (B, T_action, 3, dim): per-token modulation, aligned with
                  ``x`` along the token dimension (used when the loss samples
                  one diffusion timestep per action token).

        Returns:
            (B, T_action, dim) corrected action tokens.
        """
        base = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        if t_mod.dim() == 4:
            # Per-token: base (1, 3, dim) broadcasts over (B, T_action, 3, dim).
            # chunk(3) along the params dim (=2) → each is (B, T_action, 1, dim).
            shift, scale, gate = (base.unsqueeze(1) + t_mod).chunk(3, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            gate = gate.squeeze(2)
        else:
            shift, scale, gate = (base + t_mod).chunk(3, dim=1)

        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


class MoEExpertDiT(ActionBackbone):
    """MoE Action Expert module for WAM.

    Provides the action-side components for the MoE architecture:
    - Input projection: action_dim -> video_dim
    - Positional encoding in video_dim space
    - Expert FFN blocks at designated layers
    - Output head: video_dim -> action_dim

    Unlike ActionDiT (dual-system), this module does NOT have its own
    self-attention or cross-attention. Action tokens participate in the
    video DiT's shared self-attention by being concatenated to the
    video token sequence.

    The expert FFN blocks provide modality-specific capacity: after the
    video DiT block applies its standard FFN to all tokens (including
    action tokens), the expert FFN applies an additional correction
    specifically to action tokens. This is conceptually similar to
    BAGEL's MoT where different token types route through different FFNs.

    Init note: the expert FFN output layer is zero-initialized so the
    expert correction is zero at step 0 (pretrained video DiT behavior
    preserved for action tokens at init). The final output MLP uses
    small-random init (std=0.02) rather than zero, so initial action
    predictions are small-random, not exactly zero.

    Args:
        action_dim: Raw action vector dimension (e.g., 20 for bimanual)
        video_dim: Video DiT hidden dimension (action tokens projected here)
        expert_ffn_dim: Expert FFN intermediate dimension
        num_experts: Number of expert FFN layers (= len(expert_layers))
        freq_dim: Timestep embedding frequency dimension
        max_action_len: Maximum action sequence length
        expert_layers: Which video DiT layers have expert FFN blocks.
            Must have exactly num_experts entries.
        eps: Layer norm epsilon
    """

    def __init__(
        self,
        action_dim: int,
        video_dim: int,
        expert_ffn_dim: int,
        num_experts: int,
        expert_layers: Tuple[int, ...],
        freq_dim: int = 256,
        max_action_len: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert len(expert_layers) == num_experts, (
            f"expert_layers ({len(expert_layers)}) must equal "
            f"num_experts ({num_experts}). Each expert FFN connects to "
            f"exactly one video DiT layer."
        )
        self.action_dim = action_dim
        self.video_dim = video_dim
        self.freq_dim = freq_dim
        self.num_experts = num_experts
        self.expert_layers = expert_layers
        self.expert_layers_set = set(expert_layers)

        # Action token projection fuses the diffusion timestep into the
        # action embedding inside the encoder (see ActionEncoder).
        self.action_input_proj = ActionEncoder(action_dim, video_dim)

        # Learned positional encoding in video_dim space
        self.pos_encoding = LearnedPositionalEncoding(max_action_len, video_dim)

        # Timestep embedding (independent from video timestep)
        self.time_embedding = TimestepEmbedding(freq_dim, video_dim)

        # Time projection -> 3 modulation params (shift, scale, gate)
        self.time_projection = TimestepModulation(video_dim, 3)

        # Expert FFN blocks (one per expert layer)
        self.expert_blocks = nn.ModuleList([ExpertFFNBlock(video_dim, expert_ffn_dim, eps) for _ in range(num_experts)])

        # Output head: 2-layer MLP video_dim -> 64 -> action_dim.
        self.action_output_head = ActionOutputMLP(video_dim, 64, action_dim)

        # Per-modality bias on the video DiT's AdaLN modulation signal for action
        # tokens appended to the video sequence. See SharedBackboneVanillaArchitecture's
        # modality_tmod_bias for full rationale (zero-init, no-weight-decay).
        self.modality_tmod_bias = nn.Parameter(torch.zeros(1, 1, 6, video_dim))

        # Action normalization stats (saved as persistent buffers)
        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)

    def _prepare_expert_state(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> MoEExpertState:
        """Project actions to video_dim and prepare state for block loop.

        Args:
            action_tokens: (B, T_action, action_dim) noisy actions.
            timestep: Action diffusion timestep. Supported shapes:
                (1,)            scalar broadcast.
                (B,)            per-sample (default).
                (B, T_action)   per-token (action_timestep_per_token=True).
                ActionEncoder accepts all three. ExpertFFN AdaLN consumes
                the same granularity: per-sample inputs yield a
                (B, 3, dim) t_mod; per-token inputs yield (B, T, 3, dim).

        Returns:
            MoEExpertState ready for model_fn_wan_video.
        """
        B, T, _ = action_tokens.shape
        assert T <= self.pos_encoding.embedding.shape[1], (
            f"Action sequence length {T} exceeds max_action_len {self.pos_encoding.embedding.shape[1]}"
        )

        # ActionEncoder fuses timestep into the action embedding
        # internally. Accepts (1,), (B,), or (B, T) timestep.
        x = self.action_input_proj(action_tokens, timestep)
        x = self.pos_encoding(x)

        # Build ExpertFFN AdaLN modulation at the same granularity as the
        # incoming timestep. Per-token modulation is required so that each
        # action token's AdaLN shift/scale/gate track its own noise level
        # when action_timestep_per_token=True; otherwise the expert FFN
        # becomes a silent performance ceiling on per-token ablations.
        per_token = timestep.dim() == 2 and timestep.shape == (B, T)
        if per_token:
            flat = timestep.reshape(B * T)
            t_flat = self.time_embedding(flat)  # (B*T, dim)
            t = t_flat.view(B, T, -1)
            t_mod_flat = self.time_projection(t_flat)  # (B*T, 3, dim)
            t_mod = t_mod_flat.view(B, T, self.time_projection.n_params, -1)
        else:
            timestep_flat = timestep.flatten()
            if timestep_flat.numel() == 1:
                timestep_flat = timestep_flat.expand(B)
            elif timestep_flat.shape[0] != B:
                # Defensive: shape mismatch (e.g. numel==B*T but not (B,T)).
                timestep_flat = timestep.view(B, -1)[:, 0]
            t = self.time_embedding(timestep_flat)
            t_mod = self.time_projection(t)

        return MoEExpertState(
            moe_dit=self,
            action_tokens=x,
            t_mod=t_mod,
            t_embed=t,
            n_action_tokens=T,
            # Preserve the raw timestep shape — model_fn_wan_video's
            # _build_action_t_mod dispatches on ndim to build per-sample vs
            # per-token t_mod for the video DiT's AdaLN (see wan_video.py).
            timestep=timestep,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    def apply_expert(self, state: MoEExpertState, x_action: torch.Tensor) -> torch.Tensor:
        """Apply expert FFN correction at the current expert layer.

        Args:
            state: Current MoE state (expert_block_counter is used/updated)
            x_action: (B, T_action, video_dim) action tokens after video
                DiT block processing

        Returns:
            (B, T_action, video_dim) corrected action tokens
        """
        i = state.expert_block_counter
        block = self.expert_blocks[i]

        if state.use_gradient_checkpointing and self.training:

            def _ckpt_fn(x, t):
                return block(x, t)

            if state.use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x_action = torch.utils.checkpoint.checkpoint(
                        _ckpt_fn,
                        x_action,
                        state.t_mod,
                        use_reentrant=False,
                    )
            else:
                x_action = torch.utils.checkpoint.checkpoint(
                    _ckpt_fn,
                    x_action,
                    state.t_mod,
                    use_reentrant=False,
                )
        else:
            x_action = block(x_action, state.t_mod)

        state.expert_block_counter = i + 1
        return x_action

    def finalize_output(self, state: MoEExpertState) -> torch.Tensor:
        """Project action tokens from video_dim to action_dim.

        Args:
            state: MoE state after all expert blocks executed

        Returns:
            (B, T_action, action_dim) predicted action noise
        """
        return self.action_output_head(state.action_tokens)

    # === ActionBackbone interface ===

    @property
    def execution_plan(self) -> "ExecutionPlan":
        from openwam.model.base import ExecutionPlan

        return ExecutionPlan.INTERLEAVED_SPLIT_FFN

    @property
    def bridge_layers(self) -> Tuple[int, ...]:
        return self.expert_layers

    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        proprio_state: Optional[torch.Tensor] = None,  # noqa: ARG002 — MoE doesn't consume proprio
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> "ActionState":
        from openwam.model.base import ActionState, ExecutionPlan, RuntimeState

        moe_state = self._prepare_expert_state(
            action_tokens=noisy_actions,
            timestep=timestep,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            num_action_tokens=moe_state.n_action_tokens,
            runtime_state=RuntimeState(
                framework="shared_backbone",
                variant="moe",
                execution_plan=ExecutionPlan.INTERLEAVED_SPLIT_FFN,
                payload=moe_state,
            ),
        )

    def before_loop(
        self,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        moe_state: MoEExpertState = astate.runtime_state.payload
        vstate = vb.inject_action_tokens(
            vstate,
            moe_state.action_tokens,
            moe_state.n_action_tokens,
            timestep=moe_state.timestep,
            t_mod_bias=self.modality_tmod_bias,
        )
        return vstate, astate

    def run_block(
        self,
        block_id: int,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        moe_state: MoEExpertState = astate.runtime_state.payload
        vstate = vb.run_block(block_id, vstate)
        if block_id in self.expert_layers_set:
            # Manual slice + cat (do NOT use vb.inject/extract here — those
            # rebuild freqs / t_mod which were already extended once in
            # before_loop; calling them again would double-extend).
            n_action = moe_state.n_action_tokens
            n_video = vstate.x.shape[1] - n_action
            x_action = vstate.x[:, n_video:, :]
            x_action = self.apply_expert(moe_state, x_action)
            vstate.x = torch.cat([vstate.x[:, :n_video, :], x_action], dim=1)
            moe_state.action_tokens = x_action
        return vstate, astate

    def after_loop(
        self,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        moe_state: MoEExpertState = astate.runtime_state.payload
        vstate, action_tail = vb.extract_action_tokens(vstate, moe_state.n_action_tokens)
        moe_state.action_tokens = action_tail
        astate.final_hidden = action_tail
        return vstate, astate

    def extract_prediction(self, astate: "ActionState") -> torch.Tensor:
        moe_state: MoEExpertState = astate.runtime_state.payload
        return self.finalize_output(moe_state)

    @staticmethod
    def state_dict_converter():
        return MoEExpertDiTStateDictConverter()


class MoEExpertDiTStateDictConverter:
    """Handles loading MoEExpertDiT state dicts."""

    def from_civitai(self, state_dict):
        return state_dict, {}
