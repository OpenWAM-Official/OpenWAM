"""
ActionDiT: Lightweight Diffusion Transformer for Action Generation.

This module implements a parallel action generation stream that runs alongside
the video DiT, inspired by:
- CoVAR: Bidirectional bridge attention between video and action streams
- UWM: Independent diffusion timesteps for video and action
- HunyuanVideo-Foley: Dual-stream MMDiT + single-stream refinement
- mimic-video: Cross-attention from action decoder to video features
- SD3 MMDiT: Joint self-attention with independent projections

Supports canonical DualSystem controls via ``variant`` + ``detach_bridge``:
- ``variant='joint_cross_attn', detach_bridge=False``: Unidirectional
  (video -> action) cross-attention.
- ``variant='joint_cross_attn', detach_bridge=True``: Same architecture, but
  gradients are detached at the call site to prevent action loss from flowing
  back to the video DiT.
- ``variant='joint_self_attn'``: MMDiT-style bidirectional joint self-attention.
  Video features carry across blocks (dual-stream).

Architecture (cross_attn):
    Video DiT (frozen/LoRA) --> intermediate features --> Bridge Cross-Attention
                                                              |
    Action Tokens --> ActionDiT Blocks --> Action Noise Prediction
         ^                    |
         |--- Self-Attention -|
         |--- Cross-Attention to Video Features ---|

Architecture (joint_self_attn):
    Video DiT --> intermediate features --+-- Joint Self-Attention --+--> updated video stream
                                          |                          |
    Action Tokens -----> ActionDiT Blocks -+---- Joint Self-Attn ----+--> Action Noise Prediction

The ActionDiT is designed to be non-invasive: it reads intermediate video
features but does not modify the video generation path.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

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
from openwam.model.video_backbone.wan.dit import precompute_freqs_cis_3d

if TYPE_CHECKING:
    from openwam.model.base import ActionState, ExecutionPlan
    from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone


def _ckpt_call_joint_block(
    block,
    x_action: torch.Tensor,
    x_video: torch.Tensor,
    t_mod: torch.Tensor,
    video_t_mod: Optional[torch.Tensor],
    freqs_action: Optional[torch.Tensor],
    freqs_video: Optional[torch.Tensor],
    *,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward through a JointActionDiTBlock, dispatching to the gradient
    checkpoint variant. Mirrors the (gradient-ckpt × offload) matrix used
    by ``ActionDiT.forward`` for the joint_self_attn path so the interleaved
    training loop and the standalone forward share identical numerics.
    """

    def _fwd(*args):
        return block(*args)

    if use_gradient_checkpointing:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                return torch.utils.checkpoint.checkpoint(
                    _fwd,
                    x_action,
                    x_video,
                    t_mod,
                    video_t_mod,
                    freqs_action,
                    freqs_video,
                    use_reentrant=False,
                )
        return torch.utils.checkpoint.checkpoint(
            _fwd,
            x_action,
            x_video,
            t_mod,
            video_t_mod,
            freqs_action,
            freqs_video,
            use_reentrant=False,
        )
    return block(
        x_action,
        x_video,
        t_mod,
        video_t_mod=video_t_mod,
        freqs_action=freqs_action,
        freqs_video=freqs_video,
    )


@dataclass
class ActionDiTState:
    """Mutable container threaded through model_fn_wan_video for interleaved
    joint_self_attn execution.

    At each bridge layer inside the video DiT block loop, the pipeline:
      1. Projects video hidden state ``x`` down via ``video_projs[i]``
      2. Accumulates into ``x_video_proj``
      3. Runs ``JointActionDiTBlock[i]`` updating both ``x_action`` and ``x_video_proj``
      4. Back-projects ``x_video_new`` and adds it as a residual to ``x``

    After the block loop, ``finalize_action_output`` produces ``action_noise_pred``.
    """

    action_dit: "ActionDiT"
    x_action: torch.Tensor  # (B, T_action, dim)
    t_mod: torch.Tensor  # (B, t_mod_params, dim)
    t_embed: torch.Tensor  # (B, dim) — for output head
    use_joint_rope: bool = False
    x_video_proj: Optional[torch.Tensor] = None  # running projected video features
    action_noise_pred: Optional[torch.Tensor] = None  # filled after finalize
    bridge_block_counter: int = 0
    # Number of reference-frame prefix tokens in the *video* sequence. Read by
    # wan_video.py to slice video hidden states before bridge projection.
    skip_prefix_tokens: int = 0
    # Number of proprio tokens prepended to the *action* sequence
    # (sequence_concat mode). Independent from skip_prefix_tokens — do not mix.
    action_prefix_tokens: int = 0
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False


class ActionSelfAttention(nn.Module):
    """Self-attention over action tokens.

    Applies 1D rotary position embedding (RoPE) to Q/K when ``freqs`` is
    supplied, mirroring FastWAM's action DiT. Call with ``freqs=None`` to
    fall back to position-free attention (e.g. when the caller injects
    a learned positional encoding upstream).
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

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
    """
    Cross-attention from action tokens to video DiT features.

    This is the "bridge" that transfers dynamics information from the
    video stream to the action stream. Inspired by CoVAR's Bridge Attention
    and mimic-video's cross-attention to intermediate video features.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x_action: torch.Tensor, x_video: torch.Tensor):
        """
        Args:
            x_action: (B, T_action, dim) - action token queries
            x_video: (B, T_video, dim) - video features as keys/values
        """
        q = self.norm_q(self.q(x_action))
        k = self.norm_k(self.k(x_video))
        v = self.v(x_video)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = get_attention_fn()(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return self.o(x)


class JointSelfAttention(nn.Module):
    """MMDiT-style joint self-attention over video and action tokens.

    Each modality has independent Q/K/V projections. Keys and values are
    concatenated across modalities so both can attend to each other,
    enabling bidirectional information flow.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Action-side projections
        self.q_action = nn.Linear(dim, dim)
        self.k_action = nn.Linear(dim, dim)
        self.v_action = nn.Linear(dim, dim)
        self.o_action = nn.Linear(dim, dim)
        self.norm_q_action = RMSNorm(dim, eps=eps)
        self.norm_k_action = RMSNorm(dim, eps=eps)

        # Video-side projections
        self.q_video = nn.Linear(dim, dim)
        self.k_video = nn.Linear(dim, dim)
        self.v_video = nn.Linear(dim, dim)
        self.o_video = nn.Linear(dim, dim)
        self.norm_q_video = RMSNorm(dim, eps=eps)
        self.norm_k_video = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x_action: torch.Tensor,
        x_video: torch.Tensor,
        freqs_action: Optional[torch.Tensor] = None,
        freqs_video: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x_action: (B, T_action, dim)
            x_video: (B, T_video, dim)
            freqs_action: (T_action, head_dim // 2) complex freqs for action positions, or None
            freqs_video: (T_video, head_dim // 2) complex freqs for video positions, or None.
                For joint_self_attn, both modalities need explicit position signals since
                bridge features lack explicit positional encoding. FastWAM applies RoPE to
                both video and action branches in MoT.

        Returns:
            out_action: (B, T_action, dim)
            out_video: (B, T_video, dim)

        Padding handling: this layer does NOT take a sample-level key padding
        mask. Following FastWAM's MoT design, padded action / video tokens
        still participate in joint attention; padded positions are masked
        out at loss time via ``action_is_pad`` / ``video_is_pad``.
        """

        # Project action
        q_a = self.norm_q_action(self.q_action(x_action))
        k_a = self.norm_k_action(self.k_action(x_action))
        v_a = self.v_action(x_action)

        # Project video
        q_v = self.norm_q_video(self.q_video(x_video))
        k_v = self.norm_k_video(self.k_video(x_video))
        v_v = self.v_video(x_video)

        # Reshape to (B, num_heads, T, head_dim)
        q_a = rearrange(q_a, "b s (n d) -> b n s d", n=self.num_heads)
        k_a = rearrange(k_a, "b s (n d) -> b n s d", n=self.num_heads)
        v_a = rearrange(v_a, "b s (n d) -> b n s d", n=self.num_heads)
        q_v = rearrange(q_v, "b s (n d) -> b n s d", n=self.num_heads)
        k_v = rearrange(k_v, "b s (n d) -> b n s d", n=self.num_heads)
        v_v = rearrange(v_v, "b s (n d) -> b n s d", n=self.num_heads)

        # Apply 1D RoPE to both branches (FastWAM alignment)
        if freqs_action is not None:
            assert freqs_video is not None, "RoPE requires both freqs_action and freqs_video for joint attention"
            q_a = rope_apply_1d(q_a, freqs_action)
            k_a = rope_apply_1d(k_a, freqs_action)
            q_v = rope_apply_1d(q_v, freqs_video)
            k_v = rope_apply_1d(k_v, freqs_video)
        else:
            assert freqs_video is None, "freqs_video requires freqs_action for joint attention"

        # Concat K, V across modalities for joint attention
        k = torch.cat([k_v, k_a], dim=2)  # (B, n_heads, T_video+T_action, head_dim)
        v = torch.cat([v_v, v_a], dim=2)

        _attn = get_attention_fn()
        out_a = _attn(q_a, k, v)
        out_v = _attn(q_v, k, v)

        out_a = rearrange(out_a, "b n s d -> b s (n d)", n=self.num_heads)
        out_v = rearrange(out_v, "b n s d -> b s (n d)", n=self.num_heads)

        return self.o_action(out_a), self.o_video(out_v)


class ActionDiTBlock(nn.Module):
    """
    A single ActionDiT block with:
    1. Self-attention over action tokens
    2. Cross-attention to video DiT features (bridge)
    3. Feed-forward network

    All with AdaLN modulation from timestep embedding.
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Self-attention
        self.self_attn = ActionSelfAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Cross-attention to video features
        self.cross_attn = BridgeCrossAttention(dim, num_heads, eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # FFN
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # AdaLN modulation (6 params: shift/scale for self-attn, cross-attn, ffn + gates)
        self.modulation = nn.Parameter(torch.randn(1, 9, dim) / dim**0.5)

    def forward(self, x_action, x_video, t_mod, freqs: Optional[torch.Tensor] = None):
        """
        Args:
            x_action: (B, T_action, dim)
            x_video: (B, T_video, dim) - video features for cross-attention
            t_mod: (B, 1, 9*dim) - timestep modulation
            freqs: (T_action, head_dim // 2) complex RoPE freqs, or None
        """
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff) = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(9, dim=1)

        # Self-attention
        h = self.norm1(x_action) * (1 + scale_sa) + shift_sa
        x_action = x_action + gate_sa * self.self_attn(h, freqs=freqs)

        # Cross-attention to video features
        h = self.norm2(x_action) * (1 + scale_ca) + shift_ca
        x_action = x_action + gate_ca * self.cross_attn(h, x_video)

        # FFN
        h = self.norm3(x_action) * (1 + scale_ff) + shift_ff
        x_action = x_action + gate_ff * self.ffn(h)

        return x_action


class JointActionDiTBlock(nn.Module):
    """Dual-stream block with MMDiT-style joint self-attention.

    Sub-layers:
    1. Joint self-attention (video <-> action, bidirectional)
    2. Action FFN
    3. Video FFN

    Action side: AdaLN modulated by diffusion timestep (6 params: 3 attn + 3 ffn)
    Video side: Learned static modulation (no timestep dependency)
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Joint self-attention
        self.joint_attn = JointSelfAttention(dim, num_heads, eps)
        self.norm_attn_action = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm_attn_video = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Action FFN
        self.ffn_action = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.norm_ffn_action = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Video FFN
        self.ffn_video = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.norm_ffn_video = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Action: base modulation (added to timestep)
        self.action_modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # Video: static modulation (no timestep)
        self.video_modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x_action,
        x_video,
        t_mod,
        video_t_mod: Optional[torch.Tensor] = None,
        freqs_action: Optional[torch.Tensor] = None,
        freqs_video: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x_action: (B, T_action, dim)
            x_video: (B, T_video, dim)
            t_mod: (B, 6, dim) - timestep modulation for action side
            video_t_mod: timestep modulation for video side, or None to keep the static
                learned bias. Two layouts supported:
                  - (B, 6, dim) per-sample, broadcasts across video tokens
                  - (B, T_video, 6, dim) per-token (Wan ``seperated_timestep=True``)
            freqs_action: (T_action, head_dim // 2) RoPE freqs for action, or None (uses 1..T)
            freqs_video: (T_video, head_dim // 2) RoPE freqs for video, or None
        """
        (shift_a, scale_a, gate_a, shift_ffa, scale_ffa, gate_ffa) = (
            self.action_modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        # Video side: keep the static learned bias when video_t_mod is None so the
        # standalone forward path stays timestep-agnostic.
        video_mod_base = self.video_modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        if video_t_mod is None:
            shift_v, scale_v, gate_v, shift_ffv, scale_ffv, gate_ffv = video_mod_base.chunk(6, dim=1)
        elif video_t_mod.dim() == 4:
            shift_v, scale_v, gate_v, shift_ffv, scale_ffv, gate_ffv = (
                c.squeeze(2) for c in (video_mod_base.unsqueeze(1) + video_t_mod).chunk(6, dim=2)
            )
        else:
            shift_v, scale_v, gate_v, shift_ffv, scale_ffv, gate_ffv = (video_mod_base + video_t_mod).chunk(6, dim=1)

        # Joint attention with RoPE options
        h_a = self.norm_attn_action(x_action) * (1 + scale_a) + shift_a
        h_v = self.norm_attn_video(x_video) * (1 + scale_v) + shift_v
        out_a, out_v = self.joint_attn(
            h_a,
            h_v,
            freqs_action=freqs_action,
            freqs_video=freqs_video,
        )
        x_action = x_action + gate_a * out_a
        x_video = x_video + gate_v * out_v

        # Separate FFNs
        h_a = self.norm_ffn_action(x_action) * (1 + scale_ffa) + shift_ffa
        x_action = x_action + gate_ffa * self.ffn_action(h_a)
        h_v = self.norm_ffn_video(x_video) * (1 + scale_ffv) + shift_ffv
        x_video = x_video + gate_ffv * self.ffn_video(h_v)

        return x_action, x_video


class ActionDiT(ActionBackbone):
    """
    Lightweight Diffusion Transformer for action generation.

    Generates continuous action predictions synchronized with video frames.
    Takes intermediate features from the video DiT as conditioning via
    bridge attention (cross-attention or joint self-attention).

    Supports canonical DualSystem controls via `variant` + `detach_bridge`:
    - ``variant='joint_cross_attn', detach_bridge=False``: Unidirectional cross-attention.
    - ``variant='joint_cross_attn', detach_bridge=True``: Same architecture; gradient
      detachment is handled at the call site (training loss function).
    - ``variant='joint_self_attn'``: MMDiT-style bidirectional joint self-attention.
      Video features carry across blocks (dual-stream) and both modalities
      attend to each other.

    Design principles:
    - Non-invasive: does not modify the video DiT
    - Extensible: action_dim and architecture are configurable
    - Lightweight: much smaller than video DiT (~84M params vs 1.3B)
    - Per-block bridge: each ActionDiT block connects to a dedicated
      video DiT layer (1:1 mapping), preserving multi-scale information

    Args:
        action_dim: Dimension of raw action vectors (e.g., 20 for bimanual)
        dim: Hidden dimension of the transformer
        ffn_dim: Feed-forward network dimension
        num_heads: Number of attention heads
        num_layers: Number of ActionDiT blocks (must equal len(bridge_layers))
        freq_dim: Frequency dimension for timestep embedding
        max_action_len: Maximum action sequence length
        video_dim: Dimension of video DiT features (for projection if different)
        bridge_layers: Which video DiT layers to extract features from.
            Must have exactly num_layers entries (1:1 mapping to ActionDiT blocks).
        variant: DualSystem variant. One of 'joint_cross_attn' or 'joint_self_attn'.
        detach_bridge: Whether the cross-attn bridge should detach gradients.
            Ignored when ``variant == 'joint_self_attn'``.
        eps: Epsilon for layer norm
        use_proprioception: If True, build a ProprioceptiveEncoder that injects
            robot state into the action token stream after positional encoding.
        state_dim: Dimension of the proprioceptive state vector. Defaults to
            ``action_dim`` when 0 or None.
        proprio_fusion: ``"sequence_concat"`` (prepend state tokens) or
            ``"channel_concat"`` (concat along channel and project back).
        num_state_tokens: Number of prepended state tokens when
            ``proprio_fusion == "sequence_concat"``.
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
        detach_bridge: bool = False,
        freq_dim: int = 256,
        max_action_len: int = 1024,
        eps: float = 1e-6,
        use_proprioception: bool = False,
        state_dim: int = 0,
        proprio_fusion: str = "channel_concat",
        num_state_tokens: int = 4,
    ):
        super().__init__()
        assert len(bridge_layers) == num_layers, (
            f"bridge_layers ({len(bridge_layers)}) must equal num_layers ({num_layers}). "
            f"Each ActionDiT block connects to exactly one video DiT layer."
        )
        assert variant in ("joint_cross_attn", "joint_self_attn"), (
            f"Unknown variant '{variant}'. Choose from: joint_cross_attn, joint_self_attn"
        )
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.action_dim = action_dim
        self.dim = dim
        self.head_dim = head_dim
        self.freq_dim = freq_dim
        self.num_layers = num_layers
        self.max_action_len = max_action_len
        self.bridge_layers = bridge_layers
        self.bridge_layers_set = set(bridge_layers)
        self.variant = variant
        self.detach_bridge = bool(detach_bridge)

        from openwam.model.action_backbone.components import (
            ActionEmbedding,
            ActionOutputHead,
            TimestepEmbedding,
            TimestepModulation,
        )

        # Action token embedding: projects raw action to hidden dim
        self.action_embedding = ActionEmbedding(action_dim, dim)

        # No learned absolute PE: both variants rely on RoPE inside attention
        # (joint_cross_attn → ActionSelfAttention; joint_self_attn → JointSelfAttention).
        self.pos_encoding: Optional[nn.Module] = None
        # Plain attribute, not a buffer: model.to(bf16) would cast complex → real and drop the imaginary part.
        self.freqs = precompute_freqs_cis_1d(head_dim, max_action_len)

        # 3D RoPE cache for video tokens in joint self-attention. Per-axis cap of
        # 1024 covers any realistic latent (f, h, w). Same plain-attribute trick.
        if variant == "joint_self_attn":
            self._video_freqs_3d = precompute_freqs_cis_3d(head_dim, end=1024)

        # Per-block video feature projections (video_dim -> dim)
        # Each ActionDiT block has its own projection since features from
        # different video DiT layers have different distributions.
        if video_dim != dim:
            self.video_projs = nn.ModuleList([nn.Linear(video_dim, dim) for _ in range(num_layers)])
        else:
            self.video_projs = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Back-projection layers (dim -> video_dim) for joint_self_attn only.
        # Zero-initialized so action→video injection starts at zero,
        # preserving pretrained video behavior at initialization.
        if variant == "joint_self_attn":
            self.video_back_projs = nn.ModuleList([nn.Linear(dim, video_dim) for _ in range(num_layers)])
            for proj in self.video_back_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

            # Project video DiT t_mod (in video_dim) into action dim so each joint
            # block gets a timestep-aware video modulation. Zero-init makes initial
            # behavior equivalent to the previous static-modulation path.
            self.video_t_mod_proj = nn.Linear(video_dim, dim)
            nn.init.zeros_(self.video_t_mod_proj.weight)
            nn.init.zeros_(self.video_t_mod_proj.bias)

        # Timestep embedding (independent from video timestep)
        self.time_embedding = TimestepEmbedding(freq_dim, dim)

        # Number of modulation params per block depends on variant:
        # joint_cross_attn: 9 (3 self-attn + 3 cross-attn + 3 ffn)
        # joint_self_attn: 6 (3 joint-attn + 3 ffn; video has static modulation)
        t_mod_params = 6 if variant == "joint_self_attn" else 9
        self.t_mod_params = t_mod_params
        self.time_projection = TimestepModulation(dim, t_mod_params)

        # Transformer blocks
        if variant == "joint_self_attn":
            self.blocks = nn.ModuleList([JointActionDiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        else:
            self.blocks = nn.ModuleList([ActionDiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])

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

            fusion_to_mode = {
                "sequence_concat": "sequence_concat",
                "channel_concat": "channel_concat",
                "add": "add",
            }
            if proprio_fusion not in fusion_to_mode:
                raise ValueError(f"proprio_fusion must be one of {list(fusion_to_mode)}, got '{proprio_fusion}'")
            effective_state_dim = state_dim if state_dim and state_dim > 0 else action_dim
            self.state_dim = effective_state_dim
            self.proprio_encoder = ProprioceptiveEncoder(
                state_dim=effective_state_dim,
                hidden_dim=dim,
                mode=fusion_to_mode[proprio_fusion],
                num_state_tokens=num_state_tokens,
            )
        else:
            self.state_dim = 0

    @property
    def num_proprio_tokens(self) -> int:
        """Extra tokens prepended to the action sequence by proprio injection."""
        if self.proprio_encoder is None:
            return 0
        return self.proprio_encoder.extra_tokens

    def _inject_proprio(self, x: torch.Tensor, proprio_state: Optional[torch.Tensor]) -> torch.Tensor:
        """Run the proprio encoder if enabled; no-op otherwise.

        Training always supplies ``proprio_state`` when the encoder is built,
        so a missing tensor signals a wiring bug (e.g. a future deployment
        path forgetting to forward the kwarg). Asserting here avoids a silent
        fallback to baseline behavior.
        """
        if self.proprio_encoder is None:
            return x
        assert proprio_state is not None, (
            "ActionDiT was built with use_proprioception=True but forward "
            "received proprio_state=None. The proprio path would be silently "
            "skipped — pass the state explicitly."
        )
        return self.proprio_encoder(x, proprio_state)

    def prepare_action_state(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        proprio_state: Optional[torch.Tensor] = None,
    ) -> ActionDiTState:
        """Embed actions, compute timestep modulation, return initial state.

        Used for interleaved joint_self_attn execution where ActionDiT blocks
        run inside the video DiT block loop rather than sequentially after it.

        Args:
            action_tokens: (B, T_action, action_dim) — noisy action sequence
            timestep: (B,) or (1,) — action diffusion timestep
            proprio_state: (B, state_dim) — optional proprioceptive state;
                ignored unless the module was built with ``use_proprioception``.

        Returns:
            ActionDiTState ready to be threaded through model_fn_wan_video
        """
        T = action_tokens.shape[1]
        assert T <= self.max_action_len, f"Action sequence length {T} exceeds max_action_len {self.max_action_len}"

        # Embed actions, add positional encoding, inject proprio state
        x = self.action_embedding(action_tokens)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self._inject_proprio(x, proprio_state)

        # Timestep conditioning
        timestep = timestep.flatten()
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)

        return ActionDiTState(
            action_dit=self,
            x_action=x,
            t_mod=t_mod,
            t_embed=t,
            use_joint_rope=self.variant == "joint_self_attn",
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            action_prefix_tokens=self.num_proprio_tokens,
        )

    def _get_rope_freqs(self, seq_len: int) -> torch.Tensor:
        """1D RoPE freqs for the action sequence."""
        assert seq_len <= self.freqs.shape[0], (
            f"Action sequence length {seq_len} exceeds precomputed freqs length "
            f"{self.freqs.shape[0]}. Increase max_action_len."
        )
        return self.freqs[:seq_len]

    def _get_video_rope_freqs(self, f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Compose 3D RoPE freqs for (f, h, w) video latent tokens.

        Returns a (f*h*w, head_dim/2) complex tensor whose layout matches what
        ``rope_apply_1d`` consumes (it only inspects the flat sequence shape).

        Pads the last dim to ``head_dim/2`` with identity rotation (1+0i) to
        cover head_dim values where axis dims (f-axis = head_dim - 2*(head_dim//3),
        h/w-axis = head_dim//3) are odd and ``precompute_freqs_cis`` drops a
        position. Wan's production head_dim=128 divides cleanly so the pad is
        zero-width; this branch only kicks in for tests with smaller head_dim.
        """
        fr, hr, wr = self._video_freqs_3d
        assert f <= fr.shape[0] and h <= hr.shape[0] and w <= wr.shape[0], (
            f"Latent dim out of cache range: (f={f}, h={h}, w={w}) vs cache "
            f"({fr.shape[0]}, {hr.shape[0]}, {wr.shape[0]})"
        )
        freqs = torch.cat(
            [
                fr[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                hr[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                wr[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, -1)
        target = self.head_dim // 2
        if freqs.shape[-1] < target:
            pad = torch.ones(freqs.shape[0], target - freqs.shape[-1], dtype=freqs.dtype, device=freqs.device)
            freqs = torch.cat([freqs, pad], dim=-1)
        return freqs.to(device)

    def finalize_action_output(self, state: ActionDiTState) -> torch.Tensor:
        """Run output normalization and head on the action hidden state.

        Args:
            state: ActionDiTState after all blocks have executed

        Returns:
            (B, T_action, action_dim) — predicted action noise
        """
        x = state.x_action
        # Strip prepended proprio tokens (sequence_concat mode) before the head,
        # so the predicted noise is aligned with the original action sequence.
        if state.action_prefix_tokens:
            x = x[:, state.action_prefix_tokens :, :]
        return self.action_output_head(x, state.t_embed)

    def forward(
        self,
        action_tokens: torch.Tensor,
        video_features: List[torch.Tensor],
        timestep: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        proprio_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for action noise prediction.

        Args:
            action_tokens: (B, T_action, action_dim) - noisy action sequence
            video_features: List of num_layers tensors, each (B, T_video, video_dim),
                one per ActionDiT block from the corresponding bridge layer
            timestep: (B,) or (1,) - action diffusion timestep
            proprio_state: (B, state_dim) - optional proprioceptive state;
                ignored unless the module was built with ``use_proprioception``.

        Returns:
            (B, T_action, action_dim) - predicted action noise
        """
        T = action_tokens.shape[1]
        assert T <= self.max_action_len, f"Action sequence length {T} exceeds max_action_len {self.max_action_len}"

        # Embed actions, add positional encoding, inject proprio state
        x = self.action_embedding(action_tokens)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self._inject_proprio(x, proprio_state)

        # Ensure timestep is 1D
        timestep = timestep.flatten()

        # Timestep conditioning
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)

        # Transformer blocks — each block gets its own projected video feature
        def create_custom_forward(block):
            def custom_forward(*inputs):
                return block(*inputs)

            return custom_forward

        if self.variant == "joint_self_attn":
            # Dual-stream: video features carry across blocks with residual addition.
            # Keep the non-interleaved path behavior aligned with the production
            # interleaved path by applying joint RoPE to both action and video tokens.
            freqs_action = self._get_rope_freqs(x.shape[1])
            x_video = None
            for i, block in enumerate(self.blocks):
                x_video_i = self.video_projs[i](video_features[i])
                if x_video is None:
                    x_video = x_video_i
                else:
                    x_video = x_video + x_video_i

                # Standalone forward has no (f, h, w) layout — fall back to 1D RoPE for video.
                # Production path (run_block) uses _get_video_rope_freqs with proper 3D layout.
                freqs_video = self._get_rope_freqs(x_video.shape[1])

                # Standalone forward has no vstate.t_mod; pass None so the block falls
                # back to its static learned video modulation.
                video_t_mod = None
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x, x_video = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x,
                                x_video,
                                t_mod,
                                video_t_mod,
                                freqs_action,
                                freqs_video,
                                use_reentrant=False,
                            )
                    else:
                        x, x_video = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            x_video,
                            t_mod,
                            video_t_mod,
                            freqs_action,
                            freqs_video,
                            use_reentrant=False,
                        )
                else:
                    x, x_video = block(
                        x,
                        x_video,
                        t_mod,
                        video_t_mod=video_t_mod,
                        freqs_action=freqs_action,
                        freqs_video=freqs_video,
                    )
        else:
            # joint_cross_attn: each block gets independent video features
            freqs = self._get_rope_freqs(x.shape[1])
            for i, block in enumerate(self.blocks):
                x_video_i = self.video_projs[i](video_features[i])
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x,
                                x_video_i,
                                t_mod,
                                freqs,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            x_video_i,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x = block(x, x_video_i, t_mod, freqs=freqs)

        # Strip prepended proprio tokens before the output head so the
        # returned noise prediction is aligned with the original action seq.
        if self.num_proprio_tokens:
            x = x[:, self.num_proprio_tokens :, :]
        return self.action_output_head(x, t)

    # === ActionBackbone interface ===

    @property
    def execution_plan(self) -> "ExecutionPlan":
        from openwam.model.base import ExecutionPlan

        return (
            ExecutionPlan.INTERLEAVED_SPLIT_SELF_ATTENTION
            if self.variant == "joint_self_attn"
            else ExecutionPlan.BRIDGE_COLLECTION
        )

    @property
    def uses_proprioception(self) -> bool:
        return self.proprio_encoder is not None

    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        proprio_state: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> "ActionState":
        from openwam.model.base import ActionState, ExecutionPlan, RuntimeState

        if self.variant == "joint_self_attn":
            dit_state = self.prepare_action_state(
                action_tokens=noisy_actions,
                timestep=timestep,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                proprio_state=proprio_state,
            )
            return ActionState(
                action_latents=noisy_actions,
                timestep=timestep,
                proprio_state=proprio_state,
                runtime_state=RuntimeState(
                    framework="dual_system",
                    variant="joint_self_attn",
                    execution_plan=ExecutionPlan.INTERLEAVED_SPLIT_SELF_ATTENTION,
                    payload=dit_state,
                ),
            )
        # joint_cross_attn — defer all action computation to extract_prediction.
        # The astate itself is the payload so callers (and tests) can inspect
        # it; gradient-checkpointing flags ride on the astate as transient
        # attributes consumed by extract_prediction.
        astate = ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            proprio_state=proprio_state,
            runtime_state=RuntimeState(
                framework="dual_system",
                variant="joint_cross_attn",
                execution_plan=ExecutionPlan.BRIDGE_COLLECTION,
                payload=None,
                options={"detach_bridge": self.detach_bridge},
            ),
        )
        astate.runtime_state.payload = astate
        astate._use_gradient_checkpointing = use_gradient_checkpointing
        astate._use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        return astate

    def run_block(
        self,
        block_id: int,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        vstate = vb.run_block(block_id, vstate)
        if block_id not in self.bridge_layers_set:
            return vstate, astate

        if self.variant == "joint_cross_attn":
            astate.bridge_features.append(vstate.x.detach() if self.detach_bridge else vstate.x)
            return vstate, astate

        # joint_self_attn: project / run JointActionDiTBlock / back-project / residual
        dit_state: ActionDiTState = astate.runtime_state.payload
        dit_state.skip_prefix_tokens = vstate.reference_prefix_len

        i = dit_state.bridge_block_counter
        skip = dit_state.skip_prefix_tokens
        x_for_action = vstate.x[:, skip:, :] if skip > 0 else vstate.x

        x_video_proj_i = self.video_projs[i](x_for_action)
        if dit_state.x_video_proj is None:
            dit_state.x_video_proj = x_video_proj_i
        else:
            dit_state.x_video_proj = dit_state.x_video_proj + x_video_proj_i

        if dit_state.use_joint_rope:
            freqs_action = self._get_rope_freqs(dit_state.x_action.shape[1])
            # Use 3D RoPE for video tokens, indexed by (f, h, w) of the latent grid.
            # Reference (conditioning) frame is stripped via skip_prefix_tokens, so
            # the effective frame count for joint attention is f - 1 when skipped.
            f_eff = vstate.f - (1 if vstate.reference_prefix_len > 0 else 0)
            freqs_video = self._get_video_rope_freqs(f=f_eff, h=vstate.h, w=vstate.w, device=dit_state.x_action.device)
            assert freqs_video.shape[0] == dit_state.x_video_proj.shape[1], (
                f"Video freqs len {freqs_video.shape[0]} != x_video_proj seq "
                f"{dit_state.x_video_proj.shape[1]} (f_eff={f_eff}, h={vstate.h}, w={vstate.w})"
            )
        else:
            freqs_action = None
            freqs_video = None

        # Project video DiT t_mod into action dim for video-side AdaLN.
        video_t_mod = self.video_t_mod_proj(vstate.t_mod)

        # Padding handling: following FastWAM's MoT design, this layer does NOT
        # apply a sample-level key padding mask. Padded action / video tokens
        # still participate in joint attention; padded positions are masked out
        # at loss time via ``action_is_pad`` / ``video_is_pad``.
        x_action_new, x_video_new = _ckpt_call_joint_block(
            self.blocks[i],
            dit_state.x_action,
            dit_state.x_video_proj,
            dit_state.t_mod,
            video_t_mod,
            freqs_action,
            freqs_video,
            use_gradient_checkpointing=dit_state.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=dit_state.use_gradient_checkpointing_offload,
        )

        back_proj = self.video_back_projs[i](x_video_new)
        if skip > 0:
            vstate.x = torch.cat([vstate.x[:, :skip, :], vstate.x[:, skip:, :] + back_proj], dim=1)
        else:
            vstate.x = vstate.x + back_proj

        dit_state.x_action = x_action_new
        dit_state.x_video_proj = x_video_new
        dit_state.bridge_block_counter = i + 1

        return vstate, astate

    def extract_prediction(self, astate: "ActionState") -> torch.Tensor:
        if self.variant == "joint_self_attn":
            dit_state: ActionDiTState = astate.runtime_state.payload
            return self.finalize_action_output(dit_state)
        # joint_cross_attn: run the full ActionDiT forward with collected bridge features
        return self.forward(
            action_tokens=astate.action_latents,
            video_features=astate.bridge_features,
            timestep=astate.timestep,
            proprio_state=astate.proprio_state,
            use_gradient_checkpointing=getattr(astate, "_use_gradient_checkpointing", False),
            use_gradient_checkpointing_offload=getattr(astate, "_use_gradient_checkpointing_offload", False),
        )

    @staticmethod
    def state_dict_converter():
        return ActionDiTStateDictConverter()


class ActionDiTStateDictConverter:
    """Handles loading ActionDiT state dicts."""

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        # ActionDiT uses its own state dict format directly
        return state_dict, {}
