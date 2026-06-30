"""3D-sequence Cosmos block forward for shared-backbone support.

The standard Cosmos block (``block_split.py`` / ``dit_forward.py``) operates on a
5D grid ``(B, T, H, W, D)`` with per-frame modulation and reshapes ``flat ↔ 5D``
with fixed ``T,H,W``. SharedBackbone appends non-grid action/state tokens into
the video DiT's own sequence, which breaks that reshape.

This module runs the same Cosmos block submodules (``layer_norm_*``,
``self_attn.{compute_qkv, output_proj, output_dropout}``, ``cross_attn``,
``mlp``, ``adaln_modulation_*``) entirely on a flat 3D sequence
``(B, S, D)`` with **per-token** modulation, so ``[video | action | state]``
tokens ride one block together. The video grid is restored only at
``extract_shared_tokens`` (before ``finalize``).

Per-token inputs (built by ``CosmosPredict25VideoBackbone.inject_shared_tokens``):
- ``emb_B_S_D``        : per-token AdaLN embedding (video per-frame emb expanded
  over H·W, plus action/state emb from their timestep).
- ``adaln_lora_B_S_3D``: matching per-token AdaLN-LoRA.
- ``rope_emb``         : ``(S, 1, 1, head_dim)`` — grid rope for video, zero-angle
  (identity) rows for action/state (position-agnostic).
- ``attn_mask``        : ``(S, S)`` bool cross-modal mask (``True`` = attend).

Self-attention uses ``F.scaled_dot_product_attention`` (so the bool mask is
honored), matching the MoT driver's ``_mixed_attention``; the split-helper GPU
parity test already locks ``compute_qkv + sdpa + output_proj`` against the
monolithic block.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def _compute_modulation_3d(block, emb_B_S_D: Tensor, adaln_lora_B_S_3D: Optional[Tensor]) -> dict:
    """Per-token AdaLN modulation: nine ``(B, S, D)`` tensors (no 5D broadcast)."""
    if getattr(block, "use_adaln_lora", False):
        if adaln_lora_B_S_3D is None:
            raise ValueError("adaln_lora_B_S_3D is required when block.use_adaln_lora=True")
        msa = (block.adaln_modulation_self_attn(emb_B_S_D) + adaln_lora_B_S_3D).chunk(3, dim=-1)
        mca = (block.adaln_modulation_cross_attn(emb_B_S_D) + adaln_lora_B_S_3D).chunk(3, dim=-1)
        mmlp = (block.adaln_modulation_mlp(emb_B_S_D) + adaln_lora_B_S_3D).chunk(3, dim=-1)
    else:
        msa = block.adaln_modulation_self_attn(emb_B_S_D).chunk(3, dim=-1)
        mca = block.adaln_modulation_cross_attn(emb_B_S_D).chunk(3, dim=-1)
        mmlp = block.adaln_modulation_mlp(emb_B_S_D).chunk(3, dim=-1)
    return {
        "shift_self_attn": msa[0], "scale_self_attn": msa[1], "gate_self_attn": msa[2],
        "shift_cross_attn": mca[0], "scale_cross_attn": mca[1], "gate_cross_attn": mca[2],
        "shift_mlp": mmlp[0], "scale_mlp": mmlp[1], "gate_mlp": mmlp[2],
    }


def _adaln_modulate_3d(x_B_S_D: Tensor, norm_layer: Any, scale_B_S_D: Tensor, shift_B_S_D: Tensor) -> Tensor:
    """``norm(x) * (1 + scale) + shift`` (mirrors block_split._adaln_modulate, 3D)."""
    return norm_layer(x_B_S_D) * (1 + scale_B_S_D.type_as(x_B_S_D)) + shift_B_S_D.type_as(x_B_S_D)


def extend_rope_with_shared_tokens(rope_emb_L_1_1_D: Optional[Tensor], n_shared: int) -> Optional[Tensor]:
    """Append ``n_shared`` zero-angle (identity) rope rows for non-grid tokens.

    Cosmos rope holds rotary *angles* (``pos × freq``); a zero angle yields no
    rotation, so action/state tokens become position-agnostic in self-attention
    (their order is encoded by the action backbone's own ``input_proj``)."""
    if rope_emb_L_1_1_D is None or n_shared <= 0:
        return rope_emb_L_1_1_D
    extra = torch.zeros(
        n_shared, *rope_emb_L_1_1_D.shape[1:], dtype=rope_emb_L_1_1_D.dtype, device=rope_emb_L_1_1_D.device
    )
    return torch.cat([rope_emb_L_1_1_D, extra], dim=0)


def expand_video_emb_to_tokens(emb_B_T_X: Tensor, grid_frames: int, tokens_per_frame: int) -> Tensor:
    """Expand a per-frame (or per-sample) video embedding to per-token ``(B, T·H·W, X)``.

    Matches the ``(t h w)`` flatten order: each frame's value repeats over its
    ``H·W`` spatial tokens."""
    s_video = grid_frames * tokens_per_frame
    if emb_B_T_X.shape[1] == grid_frames:
        return emb_B_T_X.repeat_interleave(tokens_per_frame, dim=1)
    if emb_B_T_X.shape[1] == 1:
        return emb_B_T_X.expand(emb_B_T_X.shape[0], s_video, emb_B_T_X.shape[2])
    raise ValueError(
        f"video emb has frame dim {emb_B_T_X.shape[1]}; expected grid_frames={grid_frames} or 1 (per-sample)."
    )


def run_block_3d(
    block: Any,
    x_B_S_D: Tensor,
    emb_B_S_D: Tensor,
    adaln_lora_B_S_3D: Optional[Tensor],
    rope_emb: Optional[Tensor],
    context: Tensor,
    attn_mask: Optional[Tensor],
) -> Tensor:
    """One Cosmos block on a flat ``[video | action | state]`` 3D sequence.

    Returns the updated ``(B, S, D)`` hidden state. ``self_attn`` honors
    ``attn_mask`` via SDPA; cross-attn + MLP run per-token.
    """
    mod = _compute_modulation_3d(block, emb_B_S_D, adaln_lora_B_S_3D)

    # --- Self-attention (masked) ---
    normed = _adaln_modulate_3d(x_B_S_D, block.layer_norm_self_attn, mod["scale_self_attn"], mod["shift_self_attn"])
    q_4d, k_4d, v_4d = block.self_attn.compute_qkv(normed, None, rope_emb=rope_emb)
    n_heads = q_4d.shape[2]
    q = rearrange(q_4d, "b s n d -> b n s d")
    k = rearrange(k_4d, "b s n d -> b n s d")
    v = rearrange(v_4d, "b s n d -> b n s d")
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    attn = rearrange(attn, "b n s d -> b s (n d)", n=n_heads)
    sa_out = block.self_attn.output_dropout(block.self_attn.output_proj(attn))
    gate_sa = mod["gate_self_attn"].type_as(x_B_S_D)
    x_B_S_D = x_B_S_D + gate_sa * sa_out

    # --- Cross-attention (to text context) ---
    normed = _adaln_modulate_3d(x_B_S_D, block.layer_norm_cross_attn, mod["scale_cross_attn"], mod["shift_cross_attn"])
    cross_out = block.cross_attn(normed, context, rope_emb=rope_emb)
    gate_ca = mod["gate_cross_attn"].type_as(x_B_S_D)
    # Upstream cross-attn ordering is ``result * gate + x`` (block_split.post_self_attn).
    x_B_S_D = cross_out * gate_ca + x_B_S_D

    # --- MLP ---
    normed = _adaln_modulate_3d(x_B_S_D, block.layer_norm_mlp, mod["scale_mlp"], mod["shift_mlp"])
    mlp_out = block.mlp(normed)
    gate_mlp = mod["gate_mlp"].type_as(x_B_S_D)
    x_B_S_D = x_B_S_D + gate_mlp * mlp_out
    return x_B_S_D


__all__ = [
    "run_block_3d",
    "extend_rope_with_shared_tokens",
    "expand_video_emb_to_tokens",
]
