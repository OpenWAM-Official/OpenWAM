"""3D-sequence Cosmos block forward for single-system support.

The standard Cosmos block (``block_split.py`` / ``dit_forward.py``) operates on a
5D grid ``(B, T, H, W, D)`` with per-frame modulation and reshapes ``flat ↔ 5D``
with fixed ``T,H,W``. SingleSystem appends non-grid action/state tokens into
the video DiT's own sequence, which breaks that reshape.

This module runs the same Cosmos block submodules (``layer_norm_*``,
``self_attn.{compute_qkv, output_proj, output_dropout}``, ``cross_attn``,
``mlp``, ``adaln_modulation_*``) entirely on a flat 3D sequence
``(B, S, D)`` with **per-token** modulation, so ``[video | action | state]``
tokens ride one block together. The video grid is restored only at
``extract_shared_tokens`` (before ``finalize``).

Modulation is computed on the **compact** embedding — video per-frame
``(B, T, D)`` plus one row per action/state token — and only the resulting
nine ``(shift/scale/gate)`` tensors are expanded to the full ``(B, S, D)``
sequence before they are applied. Because the AdaLN projections are pointwise
over the sequence axis, ``Linear(repeat(x)) == repeat(Linear(x))`` exactly, so
this is numerically identical to computing on the expanded sequence while
running the projections at ``H·W×`` fewer FLOPs (and keeping only the compact
embedding resident across the block loop). This mirrors ``block_split``, which
also runs ``_compute_modulation`` on the per-frame ``(B, T, D)`` emb and
broadcasts over ``H·W``.

Per-token inputs (built by ``CosmosPredict25VideoBackbone.inject_shared_tokens``):
- ``emb_B_C_D``        : compact per-row AdaLN embedding — video per-frame emb
  ``(B, T, D)`` concatenated with action/state emb ``(B, n_shared, D)``.
- ``adaln_lora_B_C_3D``: matching compact per-row AdaLN-LoRA (``None`` when the
  DiT was built with ``use_adaln_lora=False``).
- ``rope_emb``         : ``(S, 1, 1, head_dim)`` — grid rope for video, zero-angle
  (identity) rows for action/state (position-agnostic).
- ``attn_mask``        : ``(S, S)`` bool cross-modal mask (``True`` = attend).

Self-attention uses ``torch_sdpa`` (so the bool mask is honored), matching the
MoT driver's ``_mixed_attention``; the split-helper parity test locks
``compute_qkv + sdpa + output_proj`` against the monolithic block.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

# _adaln_modulate is broadcast-shape-agnostic (norm(x)·(1+scale)+shift), so the
# 5D-grid helper serves the flat 3D sequence unchanged — reuse it verbatim.
from openwam.model.video_backbone.cosmos_predict25.block_split import _adaln_modulate


def _compute_modulation_3d(block, emb_B_C_D: Tensor, adaln_lora_B_C_3D: Optional[Tensor]) -> dict:
    """Per-row AdaLN modulation: nine ``(B, C, D)`` tensors on the compact emb."""
    if getattr(block, "use_adaln_lora", False):
        if adaln_lora_B_C_3D is None:
            raise ValueError("adaln_lora_B_C_3D is required when block.use_adaln_lora=True")
        msa = (block.adaln_modulation_self_attn(emb_B_C_D) + adaln_lora_B_C_3D).chunk(3, dim=-1)
        mca = (block.adaln_modulation_cross_attn(emb_B_C_D) + adaln_lora_B_C_3D).chunk(3, dim=-1)
        mmlp = (block.adaln_modulation_mlp(emb_B_C_D) + adaln_lora_B_C_3D).chunk(3, dim=-1)
    else:
        msa = block.adaln_modulation_self_attn(emb_B_C_D).chunk(3, dim=-1)
        mca = block.adaln_modulation_cross_attn(emb_B_C_D).chunk(3, dim=-1)
        mmlp = block.adaln_modulation_mlp(emb_B_C_D).chunk(3, dim=-1)
    return {
        "shift_self_attn": msa[0],
        "scale_self_attn": msa[1],
        "gate_self_attn": msa[2],
        "shift_cross_attn": mca[0],
        "scale_cross_attn": mca[1],
        "gate_cross_attn": mca[2],
        "shift_mlp": mmlp[0],
        "scale_mlp": mmlp[1],
        "gate_mlp": mmlp[2],
    }


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


def expand_compact_to_tokens(t_B_C_X: Tensor, grid_frames: int, tokens_per_frame: int) -> Tensor:
    """Expand a compact per-row tensor ``(B, T + n_shared, X)`` to per-token ``(B, S, X)``.

    The leading ``grid_frames`` rows are the video per-frame values (each repeated
    over its ``H·W`` spatial tokens); any trailing rows are the already-per-token
    action/state values and pass through unchanged."""
    video = expand_video_emb_to_tokens(t_B_C_X[:, :grid_frames], grid_frames, tokens_per_frame)
    shared = t_B_C_X[:, grid_frames:]
    if shared.shape[1] == 0:
        return video
    return torch.cat([video, shared], dim=1)


def run_block_3d(
    block: Any,
    x_B_S_D: Tensor,
    emb_B_C_D: Tensor,
    adaln_lora_B_C_3D: Optional[Tensor],
    rope_emb: Optional[Tensor],
    context: Tensor,
    attn_mask: Optional[Tensor],
    *,
    grid_frames: int,
    tokens_per_frame: int,
) -> Tensor:
    """One Cosmos block on a flat ``[video | action | state]`` 3D sequence.

    ``emb_B_C_D`` / ``adaln_lora_B_C_3D`` are the compact per-row inputs; the
    nine modulation tensors are computed on them and expanded to ``(B, S, D)``
    before use. Returns the updated ``(B, S, D)`` hidden state. ``self_attn``
    honors ``attn_mask`` via SDPA; cross-attn + MLP run per-token.
    """
    from openwam.model.video_backbone.wan.shared.core.attention.attention import torch_sdpa

    mod_compact = _compute_modulation_3d(block, emb_B_C_D, adaln_lora_B_C_3D)
    mod = {k: expand_compact_to_tokens(v, grid_frames, tokens_per_frame) for k, v in mod_compact.items()}

    # --- Self-attention (masked) ---
    normed = _adaln_modulate(x_B_S_D, block.layer_norm_self_attn, mod["scale_self_attn"], mod["shift_self_attn"])
    q_4d, k_4d, v_4d = block.self_attn.compute_qkv(normed, None, rope_emb=rope_emb)
    attn = torch_sdpa(
        q_4d,
        k_4d,
        v_4d,
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s (n d)",
        attn_mask=attn_mask,
    )
    sa_out = block.self_attn.output_dropout(block.self_attn.output_proj(attn))
    gate_sa = mod["gate_self_attn"].type_as(x_B_S_D)
    x_B_S_D = x_B_S_D + gate_sa * sa_out

    # --- Cross-attention (to text context) ---
    normed = _adaln_modulate(x_B_S_D, block.layer_norm_cross_attn, mod["scale_cross_attn"], mod["shift_cross_attn"])
    cross_out = block.cross_attn(normed, context, rope_emb=rope_emb)
    gate_ca = mod["gate_cross_attn"].type_as(x_B_S_D)
    # Upstream cross-attn ordering is ``result * gate + x`` (block_split.post_self_attn).
    x_B_S_D = cross_out * gate_ca + x_B_S_D

    # --- MLP ---
    normed = _adaln_modulate(x_B_S_D, block.layer_norm_mlp, mod["scale_mlp"], mod["shift_mlp"])
    mlp_out = block.mlp(normed)
    gate_mlp = mod["gate_mlp"].type_as(x_B_S_D)
    x_B_S_D = x_B_S_D + gate_mlp * mlp_out
    return x_B_S_D


__all__ = [
    "run_block_3d",
    "extend_rope_with_shared_tokens",
    "expand_video_emb_to_tokens",
    "expand_compact_to_tokens",
]
