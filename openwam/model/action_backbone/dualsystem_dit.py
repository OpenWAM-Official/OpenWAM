"""ActionDiT: Lightweight Diffusion Transformer for Action Generation.

Two variants share this module:

- ``variant='joint_cross_attn'``: a stack of :class:`CrossAttnActionDiTBlock`
  blocks (self-attn over action tokens + cross-attn to a per-layer video
  feature + FFN). :meth:`ActionDiT.forward(action_tokens, bridges,
  timestep, ...)` is the single entry point; ``bridges`` is a
  ``{block_id: feat}`` dict keyed by the architecture-side video DiT layer
  index, and the architecture invokes it once after the video backbone
  has run to completion.

- ``variant='joint_self_attn'``: a stack of :class:`SelfAttnActionDiTBlock`
  blocks with the same Q/K/V split layout as the video DiT.
  :class:`MoTJointDriver` drives the per-layer loop via
  :meth:`pre_attn_at_layer` / :meth:`post_attn_at_layer`, concatenating
  Q/K/V across modalities and running a single mixed attention.

Inspired by:
- CoVAR: Bridge attention between video and action streams (cross_attn).
- UWM: Independent diffusion timesteps for video and action.
- HunyuanVideo-Foley: Dual-stream MMDiT.
- SD3 MMDiT / FastWAM MoT: True joint self-attention with independent
  per-modality projections (self_attn).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from openwam.model.action_backbone.backbone import ActionBackbone
from openwam.model.action_backbone.components import (
    RMSNorm,
    get_attention_fn,
    precompute_freqs_cis_1d,
    rope_apply_1d,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward

if TYPE_CHECKING:
    from openwam.model.base import ActionState


@dataclass
class ActionDiTState:
    """Mutable container threaded through the joint self-attention loop.

    Populated by :meth:`ActionDiT.prepare_state` and consumed by
    :meth:`ActionDiT.pre_attn_at_layer` / :meth:`ActionDiT.post_attn_at_layer`
    (called per layer from :class:`MoTJointDriver`) and finally by
    :meth:`ActionDiT.extract_prediction`.
    """

    x_action: torch.Tensor  # (B, T_action [+ proprio prefix], dim)
    t_mod: torch.Tensor  # (B, t_mod_params, dim)
    t_embed: torch.Tensor  # (B, dim) — for output head
    action_freqs: torch.Tensor  # 1D RoPE frequencies for action positions
    action_prefix_tokens: int = 0  # number of proprio tokens prepended (sequence_concat)


class ActionSelfAttention(nn.Module):
    """Self-attention over action tokens with FastWAM-style heterogeneous projection.

    Q/K/V project from the residual ``hidden_dim`` to a separate attention
    space ``num_heads * attn_head_dim``; ``o`` projects back. This is the
    layout that lets two MoT experts share a single mixed attention even
    when their residual streams have different widths
    ([wan_video_dit.py:179-184](references/FastWAM/src/fastwam/models/wan22/wan_video_dit.py#L179)).
    For ``attn_head_dim == hidden_dim // num_heads`` this collapses back to
    the homogeneous ``Linear(dim, dim)`` form.

    Applies 1D rotary position embedding (RoPE) to Q/K when ``freqs`` is
    supplied. Call with ``freqs=None`` to fall back to position-free attention.
    """

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, eps: float = 1e-6):
        super().__init__()
        if attn_head_dim <= 0:
            raise ValueError(f"attn_head_dim must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim must be even for RoPE, got {attn_head_dim}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        if freqs is not None:
            q = rope_apply_1d(q, freqs)
            k = rope_apply_1d(k, freqs)
        x = get_attention_fn()(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return self.o(x)


class BridgeCrossAttention(nn.Module):
    """Cross-attention from action tokens to video DiT features.

    The bridge that transfers dynamics information from the video stream
    to the action stream in the cross_attn variant. Q is projected from
    the action ``hidden_dim``, K/V are projected from the video
    ``kv_hidden_dim`` (which may differ from the action stream's hidden
    dim — both are flattened into the shared
    ``num_heads * attn_head_dim`` attention space). Inspired by CoVAR's
    Bridge Attention and mimic-video's cross-attention to intermediate
    video features.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        eps: float = 1e-6,
        kv_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        kv_hidden_dim = kv_hidden_dim if kv_hidden_dim is not None else hidden_dim
        self.hidden_dim = hidden_dim
        self.kv_hidden_dim = kv_hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(kv_hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(kv_hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x_action: torch.Tensor, x_video: torch.Tensor):
        q = self.norm_q(self.q(x_action))
        k = self.norm_k(self.k(x_video))
        v = self.v(x_video)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = get_attention_fn()(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return self.o(x)


class CrossAttnActionDiTBlock(nn.Module):
    """Single ActionDiT block for the cross_attn variant.

    Layout: self-attn → cross-attn(to video) → FFN, all with AdaLN
    modulation from the action diffusion timestep (9 modulation params).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        ffn_dim: int,
        eps: float = 1e-6,
        kv_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        self.self_attn = ActionSelfAttention(hidden_dim, num_heads, attn_head_dim, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=kv_hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.modulation = nn.Parameter(torch.randn(1, 9, hidden_dim) / hidden_dim**0.5)

    def forward(self, x_action, x_video, t_mod, freqs: Optional[torch.Tensor] = None):
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff) = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(9, dim=1)

        h = self.norm1(x_action) * (1 + scale_sa) + shift_sa
        x_action = x_action + gate_sa * self.self_attn(h, freqs=freqs)

        h = self.norm2(x_action) * (1 + scale_ca) + shift_ca
        x_action = x_action + gate_ca * self.cross_attn(h, x_video)

        h = self.norm3(x_action) * (1 + scale_ff) + shift_ff
        x_action = x_action + gate_ff * self.ffn(h)

        return x_action


class SelfAttnActionDiTBlock(nn.Module):
    """Standard transformer block used by both modalities under the MoT
    driver — same sub-module layout as the video DiT block (norm1 +
    self-attn + cross-attn + FFN, with a 6-param AdaLN modulation), so the
    same pre/post split logic handles either modality.

    The block's ``forward()`` is intentionally functional: it is **not**
    used in the joint self-attention path (the driver calls the sub-modules
    directly). It exists as a fallback for unit tests and equivalence
    checks against the video DiT block.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        ffn_dim: int,
        kv_hidden_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        # Self-attention with RoPE-aware Q/K — Q/K/V live in the shared
        # num_heads * attn_head_dim attention space; o() projects back to
        # ``hidden_dim`` so the residual stream can have its own width.
        self.self_attn = ActionSelfAttention(hidden_dim, num_heads, attn_head_dim, eps)
        # Cross-attention to text context (mirrors video DiT). KV comes from
        # the video backbone's text-conditioned context (already projected to
        # the video residual width by ``dit.text_embedding``), so
        # ``kv_hidden_dim`` is the video backbone's hidden dim, which may
        # differ from ``hidden_dim`` under FastWAM-Joint heterogeneous layout.
        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=kv_hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        # norm3 wraps the cross-attention input — name kept aligned with the
        # video DiT block where ``norm3`` is the cross-attn LayerNorm and is
        # ``elementwise_affine=True``.
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def gate(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        t_mod: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reference forward used in equivalence tests; not invoked by the MoT driver."""
        chunks = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        attn_input = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = self.gate(x, gate_msa, self.self_attn(attn_input, freqs=freqs))
        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context)
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = self.gate(x, gate_mlp, self.ffn(mlp_input))
        return x


class ActionDiT(ActionBackbone):
    """Lightweight Diffusion Transformer for action generation.

    Two variants share the rest of the module (action embedding, timestep
    conditioning, output head, optional proprioception). They differ in the
    block class and entry point used:

    - ``joint_cross_attn`` → :class:`CrossAttnActionDiTBlock` + :meth:`forward`.
    - ``joint_self_attn``  → :class:`SelfAttnActionDiTBlock` + :meth:`pre_attn_at_layer`
      / :meth:`post_attn_at_layer`, driven by :class:`MoTJointDriver`.

    Args:
        action_dim: Dimension of raw action vectors (e.g. 20 for bimanual).
        dim: Hidden dimension of the action residual stream. **May differ
            from the video backbone's hidden dim** under ``joint_self_attn``
            — Q/K/V are projected into the shared attention space
            ``num_heads * attn_head_dim`` (FastWAM-Joint pattern) so the two
            modalities can run a single mixed attention even with
            heterogeneous residual widths.
        ffn_dim: FFN intermediate dimension.
        num_heads: Attention heads — under ``joint_self_attn`` must equal
            the video backbone's ``num_heads``.
        num_layers: Number of action-side blocks. For ``joint_self_attn`` this
            **must** equal the video backbone's ``num_layers`` (validated by
            :class:`MoTJointDriver`).
        video_dim: Cross-attn variant: video feature dim, projected to ``dim``
            on entry. Self-attn variant: informational only — kept so the
            cfg → constructor signature is uniform; the architecture validates
            ``num_heads`` / ``attn_head_dim`` parity, not ``video_dim``.
        attn_head_dim: Per-head attention dim. Defaults to ``dim // num_heads``
            when not specified. Under ``joint_self_attn`` must equal the video
            backbone's ``head_dim`` (validated by :class:`MoTJointDriver`).
        bridge_layers: Cross-attn variant: which video DiT layers feed each
            action block (1:1 mapping). Self-attn variant: typically the full
            ``range(num_layers)`` — the driver runs joint attention at every
            layer regardless and this is informational only.
        variant: ``"joint_cross_attn"`` or ``"joint_self_attn"``.
        use_proprioception / state_dim / proprio_fusion / num_state_tokens:
            Optional proprioceptive state injection.
    """

    def __init__(
        self,
        action_dim: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
        video_dim: int,
        bridge_layers: Tuple[int, ...],
        variant: str = "joint_cross_attn",
        attn_head_dim: Optional[int] = None,
        freq_dim: int = 256,
        max_action_len: int = 1024,
        eps: float = 1e-6,
        use_proprioception: bool = False,
        state_dim: int = 0,
        proprio_fusion: str = "channel_concat",
        num_state_tokens: int = 4,
    ):
        super().__init__()
        if variant not in ("joint_cross_attn", "joint_self_attn"):
            raise ValueError(f"Unknown variant '{variant}'. Choose from: joint_cross_attn, joint_self_attn")
        if len(bridge_layers) != num_layers:
            raise ValueError(
                f"bridge_layers ({len(bridge_layers)}) must equal num_layers ({num_layers}). "
                "Each action block connects to exactly one video DiT layer."
            )
        if attn_head_dim is None:
            attn_head_dim = dim // num_heads
            if attn_head_dim * num_heads != dim:
                raise ValueError(
                    f"Cannot infer attn_head_dim from dim ({dim}) / num_heads ({num_heads}); "
                    "specify attn_head_dim explicitly."
                )
        if attn_head_dim <= 0 or attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim must be a positive even int, got {attn_head_dim}")

        self.action_dim = action_dim
        self.dim = dim
        self._head_dim = attn_head_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self.freq_dim = freq_dim
        self.max_action_len = max_action_len
        self.bridge_layers = bridge_layers
        self.bridge_layers_set = set(bridge_layers)
        self.variant = variant

        from openwam.model.action_backbone.components import (
            ActionEmbedding,
            ActionOutputHead,
            TimestepEmbedding,
            TimestepModulation,
        )

        # Action token embedding: action_dim -> dim
        self.action_embedding = ActionEmbedding(action_dim, dim)

        # Both variants rely on RoPE inside attention — no learned absolute PE.
        # Plain attribute (not a buffer): model.to(bf16) would cast complex → real.
        # RoPE freqs are sized by attn_head_dim (the per-head attention dim that
        # both modalities share), not by hidden_dim/num_heads.
        self.freqs = precompute_freqs_cis_1d(attn_head_dim, max_action_len)

        # Cross-attn variant: per-layer projection from video_dim to action dim.
        # Self-attn doesn't need this — the MoT driver handles modality coupling
        # through the joint attention itself.
        if variant == "joint_cross_attn":
            if video_dim != dim:
                self.video_projs = nn.ModuleList([nn.Linear(video_dim, dim) for _ in range(num_layers)])
            else:
                self.video_projs = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Timestep embedding (independent from video timestep)
        self.time_embedding = TimestepEmbedding(freq_dim, dim)
        # Modulation params per block: 9 for cross_attn (3 self + 3 cross + 3 ffn);
        # 6 for self_attn (3 self/joint + 3 ffn — text cross-attn has no modulation
        # in the MoT block, mirroring FastWAM's Wan-aligned DiTBlock).
        t_mod_params = 9 if variant == "joint_cross_attn" else 6
        self.t_mod_params = t_mod_params
        self.time_projection = TimestepModulation(dim, t_mod_params)

        # Transformer blocks
        if variant == "joint_self_attn":
            # SelfAttn variant: cross-attn KV comes from the video backbone's
            # text-conditioned context (already in video.dim space), so
            # kv_hidden_dim = video_dim.
            self.blocks = nn.ModuleList(
                [
                    SelfAttnActionDiTBlock(
                        dim,
                        num_heads,
                        attn_head_dim,
                        ffn_dim,
                        kv_hidden_dim=video_dim,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [CrossAttnActionDiTBlock(dim, num_heads, attn_head_dim, ffn_dim, eps) for _ in range(num_layers)]
            )

        # Output head
        self.action_output_head = ActionOutputHead(dim, action_dim, eps)

        # Action normalization stats (saved as persistent buffers for checkpoint)
        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)

        # Optional proprioceptive state conditioning
        self.use_proprioception = use_proprioception
        self.proprio_fusion = proprio_fusion if use_proprioception else None
        self.proprio_encoder: Optional[nn.Module] = None
        if use_proprioception:
            from openwam.model.action_backbone.proprioceptive import ProprioceptiveEncoder

            valid_fusions = ("sequence_concat", "channel_concat", "add")
            if proprio_fusion not in valid_fusions:
                raise ValueError(f"proprio_fusion must be one of {list(valid_fusions)}, got '{proprio_fusion}'")
            effective_state_dim = state_dim if state_dim and state_dim > 0 else action_dim
            self.state_dim = effective_state_dim
            self.proprio_encoder = ProprioceptiveEncoder(
                state_dim=effective_state_dim,
                hidden_dim=dim,
                mode=proprio_fusion,
                num_state_tokens=num_state_tokens,
            )
        else:
            self.state_dim = 0

    # ------------------------------------------------------------------
    # ActionBackbone interface
    # ------------------------------------------------------------------

    @property
    def uses_proprioception(self) -> bool:
        return self.proprio_encoder is not None

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_proprio_tokens(self) -> int:
        if self.proprio_encoder is None:
            return 0
        return self.proprio_encoder.extra_tokens

    # ------------------------------------------------------------------
    # Helpers shared by both variants
    # ------------------------------------------------------------------

    def _inject_proprio(self, x: torch.Tensor, proprio_state: Optional[torch.Tensor]) -> torch.Tensor:
        """Run the proprio encoder if enabled; no-op otherwise."""
        if self.proprio_encoder is None:
            return x
        assert proprio_state is not None, (
            "ActionDiT was built with use_proprioception=True but received "
            "proprio_state=None. Pass the state explicitly to avoid silently "
            "skipping the proprio path."
        )
        return self.proprio_encoder(x, proprio_state)

    def _get_rope_freqs(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds precomputed RoPE cache "
                f"length {self.freqs.shape[0]}; increase ``max_action_len``."
            )
        return self.freqs[:seq_len]

    def _embed_actions(self, action_tokens: torch.Tensor, proprio_state: Optional[torch.Tensor]) -> torch.Tensor:
        T = action_tokens.shape[1]
        if T > self.max_action_len:
            raise ValueError(f"Action sequence length {T} exceeds max_action_len {self.max_action_len}.")
        x = self.action_embedding(action_tokens)
        return self._inject_proprio(x, proprio_state)

    # ------------------------------------------------------------------
    # joint_cross_attn variant: standalone forward with bridge features
    # ------------------------------------------------------------------

    def forward(
        self,
        action_tokens: torch.Tensor,
        bridges: Dict[int, torch.Tensor],
        timestep: torch.Tensor,
        *,
        proprio_state: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Standalone action prediction conditioned on per-layer video features.

        ``DualSystemCrossAttnArchitecture.forward`` collects ``vstate.x`` at
        the configured bridge layers after the video backbone runs to
        completion and passes them in a ``{block_id: feat}`` dict. This entry
        point does **not** touch the video backbone or any joint-attention
        machinery.

        Args:
            action_tokens: ``(B, T_action, action_dim)`` noisy action sequence.
            bridges: ``{block_id: (B, T_video, video_dim)}`` mapping from
                video DiT block index to its captured hidden state. Must
                contain every index in ``self.bridge_layers``.
            timestep: ``(B,)`` or ``(1,)`` action diffusion timestep.
            proprio_state: ``(B, state_dim)`` optional robot state.

        Returns:
            ``(B, T_action, action_dim)`` predicted action noise.
        """
        missing = [bid for bid in self.bridge_layers if bid not in bridges]
        if missing:
            raise ValueError(
                f"bridges dict missing video block ids {missing}; "
                f"expected one entry per bridge_layers={self.bridge_layers}."
            )
        x = self._embed_actions(action_tokens, proprio_state)
        timestep = timestep.flatten()
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)
        freqs = self._get_rope_freqs(x.shape[1])

        for i, block in enumerate(self.blocks):
            x_video_i = self.video_projs[i](bridges[self.bridge_layers[i]])
            x = gradient_checkpoint_forward(
                block,
                self.training and use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                x_video_i,
                t_mod,
                freqs,
            )

        if self.num_proprio_tokens:
            x = x[:, self.num_proprio_tokens :, :]
        return self.action_output_head(x, t)

    # ------------------------------------------------------------------
    # joint_self_attn variant: MoT-driven entry points
    # ------------------------------------------------------------------

    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        proprio_state: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,  # noqa: ARG002 — driver handles ckpt itself
        use_gradient_checkpointing_offload: bool = False,  # noqa: ARG002
    ) -> "ActionState":
        from openwam.model.base import ActionState

        x = self._embed_actions(noisy_actions, proprio_state)
        timestep = timestep.flatten()
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)
        action_freqs = self._get_rope_freqs(x.shape[1])

        payload = ActionDiTState(
            x_action=x,
            t_mod=t_mod,
            t_embed=t,
            action_freqs=action_freqs,
            action_prefix_tokens=self.num_proprio_tokens,
        )
        return ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            payload=payload,
        )

    def pre_attn_at_layer(
        self, layer_id: int, astate: "ActionState"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """First half of an action block: norm1 + AdaLN modulate + Q/K/V + RoPE.

        Returns Q/K/V shaped ``(B, T_action, num_heads * head_dim)`` (matching
        the layout produced by Wan ``self_attn.q/k/v`` after RMSNorm and RoPE
        — ready to concatenate with the video Q/K/V).
        """
        payload: ActionDiTState = astate.payload
        block: SelfAttnActionDiTBlock = self.blocks[layer_id]

        chunks = (block.modulation.to(dtype=payload.t_mod.dtype, device=payload.t_mod.device) + payload.t_mod).chunk(
            6, dim=1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = payload.x_action
        attn_input = block.norm1(residual_x) * (1 + scale_msa) + shift_msa

        sa = block.self_attn
        q = sa.norm_q(sa.q(attn_input))
        k = sa.norm_k(sa.k(attn_input))
        v = sa.v(attn_input)

        # Apply RoPE in head-split layout to match flash_attn / Wan rope_apply.
        q = rearrange(q, "b s (n d) -> b n s d", n=self._num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self._num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self._num_heads)
        q = rope_apply_1d(q, payload.action_freqs)
        k = rope_apply_1d(k, payload.action_freqs)
        # Driver consumes (B, S, H*D) — keep modality streams in matching layout.
        q_out = rearrange(q, "b n s d -> b s (n d)", n=self._num_heads)
        k_out = rearrange(k, "b n s d -> b s (n d)", n=self._num_heads)
        v_out = rearrange(v, "b n s d -> b s (n d)", n=self._num_heads)

        post_state = {
            "block": block,
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q_out, k_out, v_out, post_state

    def post_attn_at_layer(
        self,
        layer_id: int,
        astate: "ActionState",
        attn_out: torch.Tensor,
        post_state: dict,
    ) -> "ActionState":
        """Second half of an action block: gate(residual, self_attn.o(attn_out))
        → (optional) text cross-attn → FFN.

        ``attn_out`` is the unprojected attention output for the action slice
        of the joint mixed attention; ``self_attn.o`` is applied here. If the
        driver placed ``text_context`` in ``post_state`` (the video
        backbone's text-conditioned context), the action stream cross-attends
        to it for language conditioning. ``video↔action`` coupling itself
        already happened in the joint self-attention.
        """
        payload: ActionDiTState = astate.payload
        block: SelfAttnActionDiTBlock = post_state["block"]

        x = block.gate(post_state["residual_x"], post_state["gate_msa"], block.self_attn.o(attn_out))
        text_context = post_state.get("text_context")
        if text_context is not None:
            x = x + block.cross_attn(block.norm3(x), text_context)
        mlp_input = block.norm2(x) * (1 + post_state["scale_mlp"]) + post_state["shift_mlp"]
        x = block.gate(x, post_state["gate_mlp"], block.ffn(mlp_input))

        payload.x_action = x
        return astate

    def extract_prediction(self, astate: "ActionState") -> torch.Tensor:
        payload: ActionDiTState = astate.payload
        x = payload.x_action
        if payload.action_prefix_tokens:
            x = x[:, payload.action_prefix_tokens :, :]
        return self.action_output_head(x, payload.t_embed)
