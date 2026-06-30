"""IDM teacher-forcing branch merge/split for the Cosmos DiT.

Mirrors the split-out style of ``block_split.py`` / ``dit_forward.py``: the
backbone (:class:`CosmosPredict25VideoBackbone`) stays a thin adapter and
delegates the IDM branch concatenation to the free functions here.

IDM teacher-forcing runs the noisy + cond video branches through one MoT pass.
On Wan the two branches are flat ``(B, L, D)`` sequences, so the driver merges
them with plain ``torch.cat`` on the Wan ``BlockLoopState`` fields. Cosmos keeps
its state as a 5D grid ``(B, T, H, W, D)`` with per-frame modulation stashed in
``state.extras`` (``t_embedding_B_T_D`` / ``adaln_lora_B_T_3D`` / per-token
``rope_emb_L_1_1_D``), so the concatenation axes differ:

- ``hidden_states``         ``(B, T, H, W, D)``      → concat dim=1 (frames)
- ``t_embedding_B_T_D``     ``(B, T, D)``            → concat dim=1 (frames)
- ``adaln_lora_B_T_3D``     ``(B, T, 3D)``           → concat dim=1 (frames)
- ``rope_emb_L_1_1_D``      ``(T·H·W, 1, 1, D)``     → concat dim=0 (tokens)
- ``extra_per_block_pos_emb`` ``(B, T, H, W, D)``    → concat dim=1 (frames)

The token sequence lengths returned by :func:`merge_branches` are ``T·H·W`` per
branch (the granularity the teacher-forcing attention mask is built at), NOT
``hidden_states.shape[1]`` — that is just ``T`` for the 5D grid.

For the per-frame modulation to be meaningful on each branch, both branches must
be prepared with ``force_per_token_t_mod=True`` so ``t_embedding_B_T_D`` is
per-frame ``(B, T, D)`` rather than the broadcast ``(B, 1, D)`` (see
``dit_forward.prepare_block_loop``).
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch

from openwam.model.video_backbone.base import BlockLoopState

_FRAME_EXTRAS = ("t_embedding_B_T_D", "adaln_lora_B_T_3D")


def _tokens(state: BlockLoopState) -> int:
    return int(state.grid_frames) * int(state.grid_height) * int(state.grid_width)


def merge_branches(
    noisy: BlockLoopState, cond: BlockLoopState
) -> Tuple[BlockLoopState, int, int]:
    """Concatenate the noisy + cond Cosmos video branches along the frame axis.

    Returns ``(merged, s_noisy_tokens, s_cond_tokens)`` with the two seq lengths
    as token counts (``T·H·W``).
    """
    if (noisy.grid_height, noisy.grid_width) != (cond.grid_height, cond.grid_width):
        raise ValueError(
            "IDM teacher-forcing requires noisy and cond video branches to share "
            f"spatial token layout, got noisy h/w={(noisy.grid_height, noisy.grid_width)} "
            f"and cond h/w={(cond.grid_height, cond.grid_width)}."
        )
    for key in (*_FRAME_EXTRAS, "rope_emb_L_1_1_D"):
        if key not in noisy.extras or key not in cond.extras:
            raise ValueError(
                f"IDM merge requires Cosmos extras['{key}'] on both branches; "
                "ensure both were prepared by the CosmosPredict25 backbone."
            )

    s_noisy = _tokens(noisy)
    s_cond = _tokens(cond)

    merged = copy.copy(noisy)
    # 5D hidden state: concat along the frame axis (dim=1).
    merged.hidden_states = torch.cat([noisy.hidden_states, cond.hidden_states], dim=1)

    ex_n = noisy.extras
    ex_c = cond.extras
    merged_extras = dict(ex_n)
    for key in _FRAME_EXTRAS:
        merged_extras[key] = torch.cat([ex_n[key], ex_c[key]], dim=1)
    # Per-token RoPE: concat along the token axis (dim=0).
    merged_extras["rope_emb_L_1_1_D"] = torch.cat(
        [ex_n["rope_emb_L_1_1_D"], ex_c["rope_emb_L_1_1_D"]], dim=0
    )
    epe_n = ex_n.get("extra_per_block_pos_emb")
    epe_c = ex_c.get("extra_per_block_pos_emb")
    if (epe_n is None) != (epe_c is None):
        raise ValueError(
            "IDM teacher-forcing requires both branches to have or both lack "
            "extra_per_block_pos_emb."
        )
    merged_extras["extra_per_block_pos_emb"] = (
        None if epe_n is None else torch.cat([epe_n, epe_c], dim=1)
    )
    merged.extras = merged_extras

    # The merged frame axis spans both branches so ``block_split``'s
    # ``b (t h w) d -> b t h w d`` unflatten reconstructs the full 2T-frame grid.
    merged.grid_frames = int(noisy.grid_frames) + int(cond.grid_frames)
    return merged, s_noisy, s_cond


def split_branches(
    merged: BlockLoopState, noisy: BlockLoopState, cond: BlockLoopState
) -> Tuple[BlockLoopState, BlockLoopState]:
    """Write the post-loop merged hidden state back onto the noisy/cond branches."""
    f_n = int(noisy.grid_frames)
    noisy.hidden_states = merged.hidden_states[:, :f_n]
    cond.hidden_states = merged.hidden_states[:, f_n:]
    # t_embedding is consumed by finalize() per branch; restore the per-branch slices.
    for key in _FRAME_EXTRAS:
        if key in merged.extras:
            noisy.extras[key] = merged.extras[key][:, :f_n]
            cond.extras[key] = merged.extras[key][:, f_n:]
    return noisy, cond


__all__ = ["merge_branches", "split_branches"]
