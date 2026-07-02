"""Shared ActionDiT for OpenWAM joint video-action architectures.

Dual-system and tri-system architectures share this module. It owns the
action-side parameters and exposes both a standalone cross-attention path and
a split pre/post interface for mixed self-attention drivers.

Three variants share this module:

- ``variant='joint_cross_attn'``: a stack of :class:`CrossAttnActionDiTBlock`
  blocks (self-attn over action tokens + cross-attn to a per-layer video
  feature + cross-attn to text/proprio context + FFN).
  :meth:`ActionDiT.forward(action_tokens, bridges, timestep, ...)` is the
  single entry point; ``bridges`` is a ``{block_id: feat}`` dict keyed by
  the architecture-side video DiT layer index, and the architecture invokes
  it once after the video backbone has run to completion.

- ``variant='joint_self_attn'``: a stack of :class:`SelfAttnActionDiTBlock`
  blocks with the same Q/K/V split layout as the video DiT.
  :class:`DualSystemMoTDriver` drives the per-layer loop via
  :meth:`pre_attn_at_layer` / :meth:`post_attn_at_layer`, concatenating
  Q/K/V across modalities and running a single mixed attention.

- ``variant='idm'``: reuses the same MoT-driven block layout as
  ``joint_self_attn``. The architecture changes the training/inference
  orchestration into IDM's two-stage denoising, but the action expert's
  per-layer computation is identical.

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

from openwam.model.action_backbone.base import ActionDiTBackbone
from openwam.model.action_backbone.components import (
    RMSNorm,
    get_attention_fn,
    precompute_freqs_cis_1d,
    rope_apply_1d,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward

if TYPE_CHECKING:
    from openwam.model.architectures.base import ActionState


_MOT_VARIANTS = ("joint_self_attn", "idm")


@dataclass
class ActionDiTState:
    """Mutable container threaded through the joint self-attention loop.

    Populated by :meth:`ActionDiT.prepare_state` and consumed by
    :meth:`ActionDiT.pre_attn_at_layer` / :meth:`ActionDiT.post_attn_at_layer`
    (called per layer from :class:`DualSystemMoTDriver`) and finally by
    :meth:`ActionDiT.extract_prediction`.
    """

    x_action: torch.Tensor  # (B, T_action, dim)
    t_mod: torch.Tensor  # (B, t_mod_params, dim)
    t_embed: torch.Tensor  # (B, dim)
    action_freqs: torch.Tensor  # 1D RoPE frequencies for action positions
    context: Optional[torch.Tensor] = None  # (B, T_context, dim), action-projected raw text/proprio context
    context_mask: Optional[torch.Tensor] = None  # (B, T_context) or (B, T_action, T_context), True means attendable


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

    def forward(self, x_action: torch.Tensor, x_video: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x_action))
        k = self.norm_k(self.k(x_video))
        v = self.v(x_video)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        if ctx_mask is None:
            x = get_attention_fn()(q, k, v)
        else:
            if ctx_mask.dim() == 3:
                ctx_mask = ctx_mask.unsqueeze(1)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return self.o(x)


class CrossAttnActionDiTBlock(nn.Module):
    """Single ActionDiT block for the cross_attn variant.

    Layout: self-attn → cross-attn(to video bridge) → optional
    cross-attn(to text/proprio context) → FFN. Self-attn, bridge
    cross-attn, and FFN are AdaLN-modulated by the action diffusion
    timestep; context cross-attn mirrors Wan/FastWAM text conditioning and
    is not timestep-gated.
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
        self.self_attn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=kv_hidden_dim)
        self.bridge_attn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.context_attn = BridgeCrossAttention(
            hidden_dim,
            num_heads,
            attn_head_dim,
            eps,
            kv_hidden_dim=hidden_dim,
        )
        self.context_attn_norm = nn.LayerNorm(hidden_dim, eps=eps)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.modulation = nn.Parameter(torch.randn(1, 9, hidden_dim) / hidden_dim**0.5)

    def forward(
        self,
        x_action,
        x_video,
        t_mod,
        freqs: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ):
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff) = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(9, dim=1)

        h = self.self_attn_norm(x_action) * (1 + scale_sa) + shift_sa
        x_action = x_action + gate_sa * self.self_attn(h, freqs=freqs)

        h = self.bridge_attn_norm(x_action) * (1 + scale_ca) + shift_ca
        x_action = x_action + gate_ca * self.cross_attn(h, x_video)

        if context is not None:
            x_action = x_action + self.context_attn(self.context_attn_norm(x_action), context, ctx_mask=context_mask)

        h = self.ffn_norm(x_action) * (1 + scale_ff) + shift_ff
        x_action = x_action + gate_ff * self.ffn(h)

        return x_action


class SelfAttnActionDiTBlock(nn.Module):
    """Action expert block driven by the MoT joint self-attention loop.

    The block is split around self-attention so :class:`DualSystemMoTDriver` can
    concatenate video/action Q/K/V, run one mixed attention, and return the
    action slice. After that mixed attention, the action stream cross-attends
    to its own text/proprio context embedding and then runs the FFN. The
    mixed self-attention and FFN are AdaLN-modulated by the action diffusion
    timestep; context cross-attention is not timestep-gated.

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
        # Cross-attention to the action expert's own text/proprio context
        # embedding. KV therefore lives in the action residual width.
        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=kv_hidden_dim)

        self.self_attn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.context_attn_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

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

        attn_input = self.self_attn_norm(x) * (1 + scale_msa) + shift_msa
        x = self.gate(x, gate_msa, self.self_attn(attn_input, freqs=freqs))
        if context is not None:
            x = x + self.cross_attn(self.context_attn_norm(x), context)
        mlp_input = self.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        x = self.gate(x, gate_mlp, self.ffn(mlp_input))
        return x


class ActionDiT(ActionDiTBackbone):
    """Lightweight Diffusion Transformer for action generation.

    Two variants share the action encoder, action-owned text/proprio context
    embedding, action timestep conditioning, RoPE cache, and simple action
    decoder. They differ in how video features enter the action stream:

    - ``joint_cross_attn`` → :class:`CrossAttnActionDiTBlock` + :meth:`forward`.
    - ``joint_self_attn`` / ``idm`` → :class:`SelfAttnActionDiTBlock` +
      :meth:`pre_attn_at_layer` / :meth:`post_attn_at_layer`, driven by
      :class:`DualSystemMoTDriver`.

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
            :class:`DualSystemMoTDriver`).
        video_dim: Cross-attn variant: video feature dim, projected to ``dim``
            on entry. Self-attn variant: informational only — kept so the
            cfg → constructor signature is uniform; the architecture validates
            ``num_heads`` / ``attn_head_dim`` parity, not ``video_dim``.
        text_dim: Raw context dimension before the action expert's own
            FastWAM-style text/proprio projection. Defaults to Wan/FastWAM
            text encoder width (4096).
        attn_head_dim: Per-head attention dim. Defaults to ``dim // num_heads``
            when not specified. Under ``joint_self_attn`` must equal the video
            backbone's ``head_dim`` (validated by :class:`DualSystemMoTDriver`).
        bridge_layers: Cross-attn variant: which video DiT layers feed each
            action block (1:1 mapping). Self-attn variant: typically the full
            ``range(num_layers)`` — the driver runs joint attention at every
            layer regardless and this is informational only.
        variant: ``"joint_cross_attn"``, ``"joint_self_attn"``, or ``"idm"``.
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
        text_dim: int = 4096,
        freq_dim: int = 256,
        max_action_len: int = 1024,
        eps: float = 1e-6,
        shift_action: Optional[float] = None,
    ):
        super().__init__()
        if variant not in ("joint_cross_attn", *_MOT_VARIANTS):
            raise ValueError(f"Unknown variant '{variant}'. Choose from: joint_cross_attn, {', '.join(_MOT_VARIANTS)}")
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
        self._shift_action = None if shift_action is None else float(shift_action)
        self.freq_dim = freq_dim
        self.max_action_len = max_action_len
        self._bridge_layers = tuple(bridge_layers)
        self.variant = variant
        self.text_dim = int(text_dim)

        from openwam.model.action_backbone.components import (
            TimestepEmbedding,
            TimestepModulation,
        )

        # FastWAM-compatible action token embedding: raw action -> hidden.
        self.action_encoder = nn.Linear(action_dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        # Both variants rely on RoPE inside attention — no learned absolute PE.
        # Keep this as a plain attribute, not a buffer: Module.to(dtype=bf16)
        # casts complex buffers to real and drops the imaginary part. _apply()
        # below moves it across devices while preserving complex dtype.
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
        if variant in _MOT_VARIANTS:
            # SelfAttn variant: action owns an independent text/proprio
            # embedding, so cross-attn KV lives in the action residual width.
            self.blocks = nn.ModuleList(
                [
                    SelfAttnActionDiTBlock(
                        dim,
                        num_heads,
                        attn_head_dim,
                        ffn_dim,
                        kv_hidden_dim=dim,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [CrossAttnActionDiTBlock(dim, num_heads, attn_head_dim, ffn_dim, eps) for _ in range(num_layers)]
            )

        # FastWAM-compatible simple action decoder; no AdaLN and no zero init.
        self.action_decoder = nn.Linear(dim, action_dim)

    def _sync_rope_freqs_device(self) -> None:
        if not isinstance(getattr(self, "freqs", None), torch.Tensor):
            return
        ref = next(self.parameters(), None)
        if ref is not None and self.freqs.device != ref.device:
            self.freqs = self.freqs.to(device=ref.device)

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        self._sync_rope_freqs_device()
        return module

    # ------------------------------------------------------------------
    # ActionDiTBackbone interface
    # ------------------------------------------------------------------

    @property
    def uses_proprioception(self) -> bool:
        return False

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    # ------------------------------------------------------------------
    # Helpers shared by both variants
    # ------------------------------------------------------------------

    def _get_rope_freqs(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds precomputed RoPE cache "
                f"length {self.freqs.shape[0]}; increase ``max_action_len``."
            )
        return self.freqs[:seq_len]

    def _embed_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        T = action_tokens.shape[1]
        if T > self.max_action_len:
            raise ValueError(f"Action sequence length {T} exceeds max_action_len {self.max_action_len}.")
        return self.action_encoder(action_tokens)

    def _prepare_context(
        self,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if context is None:
            if context_mask is not None:
                raise ValueError("context_mask was provided but context is None.")
            return None, None
        if context.ndim != 3:
            raise ValueError(f"context must have shape [B, L, text_dim], got {tuple(context.shape)}")
        if context.shape[0] != batch_size:
            raise ValueError(f"context batch size ({context.shape[0]}) must match actions batch size ({batch_size})")
        if context.shape[-1] != self.text_dim:
            raise ValueError(f"context last dim must match text_dim={self.text_dim}, got {context.shape[-1]}")

        context = context.to(device=device, dtype=dtype)
        context_emb = self.text_embedding(context)
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        else:
            if context_mask.shape != context.shape[:2]:
                raise ValueError(
                    f"context_mask must have shape {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
                )
            context_mask = context_mask.to(device=device, dtype=torch.bool)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        return context_emb, context_attn_mask

    def _prepare_timestep(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(f"action timestep must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"action timestep length must be 1 or batch size ({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        return timestep

    # ------------------------------------------------------------------
    # joint_cross_attn variant: standalone forward with bridge features
    # ------------------------------------------------------------------

    def bridge_tuple_from_dict(self, bridges: Dict[int, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Return bridges ordered by ``self._bridge_layers`` for compile-friendly calls."""

        missing = [bid for bid in self._bridge_layers if bid not in bridges]
        if missing:
            raise ValueError(
                f"bridges dict missing video block ids {missing}; "
                f"expected one entry per bridge_layers={self._bridge_layers}."
            )
        return tuple(bridges[bid] for bid in self._bridge_layers)

    def _validate_bridge_tuple(self, x: torch.Tensor, bridge_tuple: tuple[torch.Tensor, ...]) -> None:
        if len(bridge_tuple) != len(self._bridge_layers):
            raise ValueError(
                f"bridge_tuple length ({len(bridge_tuple)}) must match bridge_layers={self._bridge_layers}."
            )

        for i, bridge in enumerate(bridge_tuple):
            if bridge.ndim != 3:
                raise ValueError(
                    f"bridge tensor for video block {self._bridge_layers[i]} must have shape [B, T_video, C], "
                    f"got {tuple(bridge.shape)}."
                )
            if bridge.shape[0] != x.shape[0]:
                raise ValueError(
                    f"bridge tensor batch size ({bridge.shape[0]}) must match actions batch size ({x.shape[0]})."
                )
            if bridge.device != x.device:
                raise RuntimeError(
                    f"bridge device mismatch at action block {i}: bridge={bridge.device}, actions={x.device}."
                )
            if bridge.dtype != x.dtype:
                raise RuntimeError(
                    f"bridge dtype mismatch at action block {i}: bridge={bridge.dtype}, actions={x.dtype}."
                )
            expected_dim = getattr(self.video_projs[i], "in_features", self.dim)
            if bridge.shape[-1] != expected_dim:
                raise ValueError(
                    f"bridge tensor last dim ({bridge.shape[-1]}) must match expected video_dim={expected_dim} "
                    f"for action block {i}."
                )

    def forward_with_bridge_tuple(
        self,
        action_tokens: torch.Tensor,
        bridge_tuple: tuple[torch.Tensor, ...],
        timestep: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action_freqs: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Tuple-based cross-attn forward used by eager and compile paths."""

        x = self._embed_actions(action_tokens)
        self._validate_bridge_tuple(x, bridge_tuple)
        timestep = self._prepare_timestep(timestep, action_tokens.shape[0])
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)
        freqs = action_freqs if action_freqs is not None else self._get_rope_freqs(x.shape[1]).to(device=x.device)
        context_emb, context_attn_mask = self._prepare_context(
            context,
            context_mask,
            batch_size=action_tokens.shape[0],
            seq_len=x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )

        for i, block in enumerate(self.blocks):
            x_video_i = self.video_projs[i](bridge_tuple[i])
            x = gradient_checkpoint_forward(
                block,
                self.training and use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                x_video_i,
                t_mod,
                freqs,
                context_emb,
                context_attn_mask,
            )

        return self.action_decoder(x)

    def forward(
        self,
        action_tokens: torch.Tensor,
        bridges: Dict[int, torch.Tensor],
        timestep: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Standalone action prediction conditioned on per-layer video features.

        ``DualSystemCrossAttnArchitecture.forward`` collects ``vstate.hidden_states`` at
        the configured bridge layers after the video backbone runs to
        completion and passes them in a ``{block_id: feat}`` dict. This entry
        point does **not** touch the video backbone or any joint-attention
        machinery.

        Args:
            action_tokens: ``(B, T_action, action_dim)`` noisy action sequence.
            bridges: ``{block_id: (B, T_video, video_dim)}`` mapping from
                video DiT block index to its captured hidden state. Must
                contain every index in ``self._bridge_layers``.
            timestep: ``(B,)`` or ``(1,)`` action diffusion timestep.
            context: Optional raw text/proprio context ``(B, L, text_dim)``.
            context_mask: Optional bool mask ``(B, L)`` where True means attend.
        Returns:
            ``(B, T_action, action_dim)`` predicted action noise.
        """

        return self.forward_with_bridge_tuple(
            action_tokens,
            self.bridge_tuple_from_dict(bridges),
            timestep,
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    # ------------------------------------------------------------------
    # joint_self_attn / idm variants: MoT-driven entry points
    # ------------------------------------------------------------------

    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,  # noqa: ARG002 — driver handles ckpt itself
        use_gradient_checkpointing_offload: bool = False,  # noqa: ARG002
    ) -> "ActionState":
        from openwam.model.architectures.base import ActionState

        x = self._embed_actions(noisy_actions)
        timestep = self._prepare_timestep(timestep, noisy_actions.shape[0])
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)
        action_freqs = self._get_rope_freqs(x.shape[1]).to(device=x.device)

        action_context = None
        action_context_mask = None
        if self.variant in _MOT_VARIANTS:
            if context is None:
                raise ValueError("ActionDiT.prepare_state requires raw context for variant='joint_self_attn' or 'idm'.")
            action_context, action_context_mask = self._prepare_context(
                context,
                context_mask,
                batch_size=noisy_actions.shape[0],
                seq_len=x.shape[1],
                dtype=x.dtype,
                device=x.device,
            )

        payload = ActionDiTState(
            x_action=x,
            t_mod=t_mod,
            t_embed=t,
            action_freqs=action_freqs,
            context=action_context,
            context_mask=action_context_mask,
        )
        return ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            payload=payload,
        )

    def pre_attn_at_layer(
        self, layer_id: int, astate: "ActionState"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """First half of an action block: self-attn norm + AdaLN + Q/K/V + RoPE.

        Returns Q/K/V shaped ``(B, T_action, num_heads * head_dim)`` (matching
        the layout produced by Wan ``self_attn.q/k/v`` after RMSNorm and RoPE
        — ready to concatenate with the video Q/K/V).
        """
        q_out, k_out, v_out, post_tuple = self.pre_attn_at_layer_for_compile(layer_id, astate)
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_tuple
        block: SelfAttnActionDiTBlock = self.blocks[layer_id]
        post_state = {
            "block": block,
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q_out, k_out, v_out, post_state

    def pre_attn_at_layer_for_compile(self, layer_id: int, astate: "ActionState"):
        """Compile-friendly pre-attention half using a tensor tuple post-state.

        Returns a 4-tuple ``(q, k, v, post_state)`` where ``post_state`` is the
        tensor tuple ``(residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)``.
        """
        payload: ActionDiTState = astate.payload
        block: SelfAttnActionDiTBlock = self.blocks[layer_id]

        chunks = (block.modulation.to(dtype=payload.t_mod.dtype, device=payload.t_mod.device) + payload.t_mod).chunk(
            6, dim=1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = payload.x_action
        attn_input = block.self_attn_norm(residual_x) * (1 + scale_msa) + shift_msa

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

        post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        return q_out, k_out, v_out, post_state

    def post_attn_at_layer(
        self,
        layer_id: int,
        astate: "ActionState",
        attn_out: torch.Tensor,
        post_state: dict,
    ) -> "ActionState":
        """Second half of an action block: gate(residual, self_attn.o(attn_out))
        → action-owned text/proprio cross-attn → FFN.

        ``attn_out`` is the unprojected attention output for the action slice
        of the joint mixed attention; ``self_attn.o`` is applied here.
        Language/proprio conditioning uses the action expert's independent
        ``text_embedding`` stored in :class:`ActionDiTState`.
        """
        if isinstance(post_state, dict):
            post_state = (
                post_state["residual_x"],
                post_state["gate_msa"],
                post_state["shift_mlp"],
                post_state["scale_mlp"],
                post_state["gate_mlp"],
            )
        return self.post_attn_at_layer_for_compile(layer_id, astate, attn_out, post_state)

    def post_attn_at_layer_for_compile(
        self,
        layer_id: int,
        astate: "ActionState",
        attn_out: torch.Tensor,
        post_state: tuple[torch.Tensor, ...],
    ) -> "ActionState":
        """Compile-friendly post-attention half consuming a tensor tuple."""
        payload: ActionDiTState = astate.payload
        block: SelfAttnActionDiTBlock = self.blocks[layer_id]
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

        x = block.gate(residual_x, gate_msa, block.self_attn.o(attn_out))
        if payload.context is not None:
            text_mask = payload.context_mask
            if text_mask is not None:
                if text_mask.dim() == 2:
                    text_mask = text_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
                elif text_mask.dim() not in (3, 4):
                    raise ValueError(
                        "ActionDiTState.context_mask must be [B, L], [B, T_action, L], "
                        f"or broadcastable [B, heads, T_action, L], got {tuple(text_mask.shape)}"
                    )
            x = x + block.cross_attn(block.context_attn_norm(x), payload.context, ctx_mask=text_mask)
        mlp_input = block.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))

        payload.x_action = x
        return astate

    def extract_prediction(self, astate: "ActionState") -> torch.Tensor:
        payload: ActionDiTState = astate.payload
        return self.action_decoder(payload.x_action)
