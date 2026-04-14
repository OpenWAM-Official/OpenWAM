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

Architecture at each expert layer:
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
from typing import Optional, Tuple

import torch
import torch.nn as nn

from openwam.model.video_backbone.diffsynth.models.action_dit import sinusoidal_embedding_1d


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
        t_mod: (B, 3, video_dim) timestep modulation for expert FFN.
        t_embed: (B, video_dim) timestep embedding for output head.
        n_action_tokens: Number of action tokens appended to sequence.
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
        """
        Args:
            x: (B, T_action, dim) action tokens after video DiT block
            t_mod: (B, 3, dim) timestep modulation
        Returns:
            (B, T_action, dim) corrected action tokens
        """
        shift, scale, gate = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(3, dim=1)

        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


class MoEExpertDiT(nn.Module):
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

    Design note: Both the expert FFN output layer and the final output
    head are zero-initialized, so at initialization the model behaves
    identically to the vanilla video DiT (action predictions are zero).

    Args:
        action_dim: Raw action vector dimension (e.g., 14 for bimanual)
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
        action_dim: int = 14,
        video_dim: int = 1536,
        expert_ffn_dim: int = 4096,
        num_experts: int = 8,
        freq_dim: int = 256,
        max_action_len: int = 512,
        expert_layers: Tuple[int, ...] = (3, 7, 11, 15, 19, 23, 26, 29),
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

        # Action token projection: action_dim -> video_dim
        self.action_input_proj = nn.Sequential(
            nn.Linear(action_dim, video_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(video_dim, video_dim),
        )

        # Learned positional encoding in video_dim space
        self.pos_embedding = nn.Parameter(torch.randn(1, max_action_len, video_dim) * 0.02)

        # Timestep embedding (independent from video timestep)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, video_dim),
            nn.SiLU(),
            nn.Linear(video_dim, video_dim),
        )

        # Time projection -> 3 modulation params (shift, scale, gate)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(video_dim, video_dim * 3),
        )

        # Expert FFN blocks (one per expert layer)
        self.expert_blocks = nn.ModuleList([ExpertFFNBlock(video_dim, expert_ffn_dim, eps) for _ in range(num_experts)])

        # Output head: video_dim -> action_dim
        self.output_norm = nn.LayerNorm(video_dim, eps=eps, elementwise_affine=False)
        self.output_head = nn.Linear(video_dim, action_dim)
        self.output_modulation = nn.Parameter(torch.randn(1, 2, video_dim) / video_dim**0.5)

        # Zero-initialize output for stable training start
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # Action normalization stats (saved as persistent buffers)
        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)

    def prepare_state(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> MoEExpertState:
        """Project actions to video_dim and prepare state for block loop.

        Args:
            action_tokens: (B, T_action, action_dim) noisy actions
            timestep: (B,) or (1,) action diffusion timestep

        Returns:
            MoEExpertState ready for model_fn_wan_video
        """
        B, T, _ = action_tokens.shape
        assert T <= self.pos_embedding.shape[1], (
            f"Action sequence length {T} exceeds max_action_len {self.pos_embedding.shape[1]}"
        )

        # Project to video_dim and add positional encoding
        x = self.action_input_proj(action_tokens)
        x = x + self.pos_embedding[:, :T, :]

        # Timestep modulation for expert FFN
        timestep = timestep.flatten()
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (3, self.video_dim))

        return MoEExpertState(
            moe_dit=self,
            action_tokens=x,
            t_mod=t_mod,
            t_embed=t,
            n_action_tokens=T,
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
        x = state.action_tokens
        t = state.t_embed

        shift_out, scale_out = (
            self.output_modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1).expand(-1, 2, -1)
        ).chunk(2, dim=1)

        return self.output_head(self.output_norm(x) * (1 + scale_out) + shift_out)

    @staticmethod
    def state_dict_converter():
        return MoEExpertDiTStateDictConverter()


class MoEExpertDiTStateDictConverter:
    """Handles loading MoEExpertDiT state dicts."""

    def from_civitai(self, state_dict):
        return state_dict, {}
