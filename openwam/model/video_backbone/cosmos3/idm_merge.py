"""IDM teacher-forcing branch merge/split for the Cosmos3-Edge backbone.

IDM runs the noisy + cond video branches through one MoT pass. Cosmos3's state
is a flat ``(B, S, D)`` gen sequence with ``S = T·H·W`` (text does NOT live in
``hidden_states`` — it rides the cached per-layer und K/V), so the merge is a
plain sequence concat, simpler than the predict2.5 5D-grid case:

- ``hidden_states``  ``(B, S, D)``          → concat dim=1 (tokens)
- ``cos_gen``/``sin_gen`` ``(B, S, head_dim)`` → concat dim=1 (tokens)
- ``grid_frames``                            → sum (mask granularity)

The und K/V cache, ``und_mask`` and the prefix-K/V declaration are *shared*: both
branches condition on the same prompt, so the merged state keeps the noisy
branch's entries unchanged. Timestep conditioning is already baked into each
branch's tokens by ``prepare_block_loop`` (additive scatter, per branch), so
nothing timestep-shaped needs merging — the reason this file has no analogue of
predict2.5's per-frame ``t_embedding``/``adaln_lora`` handling.

Returned sequence lengths are token counts (``T·H·W``) — the granularity the
teacher-forcing mask is built at.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch

from openwam.model.video_backbone.base import BlockLoopState

_TOKEN_EXTRAS = ("cos_gen", "sin_gen")


def _tokens(state: BlockLoopState) -> int:
    return int(state.grid_frames) * int(state.grid_height) * int(state.grid_width)


def merge_branches(noisy: BlockLoopState, cond: BlockLoopState) -> Tuple[BlockLoopState, int, int]:
    """Concatenate the noisy + cond Cosmos3 gen sequences along the token axis.

    Returns ``(merged, s_noisy_tokens, s_cond_tokens)``.
    """
    if (noisy.grid_height, noisy.grid_width) != (cond.grid_height, cond.grid_width):
        raise ValueError(
            "IDM teacher-forcing requires noisy and cond branches to share the spatial token "
            f"layout; got noisy h/w={(noisy.grid_height, noisy.grid_width)} and "
            f"cond h/w={(cond.grid_height, cond.grid_width)}."
        )
    for key in _TOKEN_EXTRAS:
        if key not in noisy.extras or key not in cond.extras:
            raise ValueError(
                f"cosmos3 IDM merge requires extras['{key}'] on both branches; "
                "ensure both were prepared by the Cosmos3-Edge backbone."
            )
    if int(getattr(noisy, "prefix_kv_len", 0) or 0) != int(getattr(cond, "prefix_kv_len", 0) or 0):
        raise ValueError(
            "cosmos3 IDM merge requires both branches to share the und (text) prefix; "
            "prepare them from the same prompt encoding."
        )
    for name, branch in (("noisy", noisy), ("cond", cond)):
        got = int(branch.hidden_states.shape[1])
        want = _tokens(branch)
        if got != want:
            raise ValueError(
                f"cosmos3 IDM merge: {name} branch has {got} tokens but grid T·H·W={want}; "
                "the gen sequence must hold exactly the video tokens at merge time "
                "(single-system tokens must not be injected before IDM merge)."
            )

    s_noisy = _tokens(noisy)
    s_cond = _tokens(cond)

    merged = copy.copy(noisy)
    merged.hidden_states = torch.cat([noisy.hidden_states, cond.hidden_states], dim=1)

    merged_extras = dict(noisy.extras)
    for key in _TOKEN_EXTRAS:
        merged_extras[key] = torch.cat([noisy.extras[key], cond.extras[key]], dim=1)
    merged.extras = merged_extras
    # Frame count spans both branches so the token/mask arithmetic downstream
    # (grid_frames × tokens_per_frame) covers the merged sequence.
    merged.grid_frames = int(noisy.grid_frames) + int(cond.grid_frames)
    return merged, s_noisy, s_cond


def split_branches(
    merged: BlockLoopState, noisy: BlockLoopState, cond: BlockLoopState
) -> Tuple[BlockLoopState, BlockLoopState]:
    """Write the post-loop merged gen sequence back onto the noisy/cond branches.

    Only ``hidden_states`` is written back — the sole field the block loop mutates.
    Each branch keeps its own ``cos_gen``/``sin_gen`` and grid for ``finalize()``.
    """
    s_noisy = _tokens(noisy)
    noisy.hidden_states = merged.hidden_states[:, :s_noisy]
    cond.hidden_states = merged.hidden_states[:, s_noisy:]
    return noisy, cond


__all__ = ["merge_branches", "split_branches"]
