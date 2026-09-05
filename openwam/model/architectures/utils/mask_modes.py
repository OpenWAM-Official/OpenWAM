"""Cross-modal attention mask modes: visibility between video and action.

All four modes share two invariants: the v<->v sub-block is controlled
separately by ``video_attention_mask_mode``, and a<->a is always mutually
visible. The modes only decide whether (and how far) video and action can
see each other:

- ``mutual``             video sees action; action sees all of video
- ``action_sees_video``  video does not see action; action sees all of video
- ``video_sees_action``  video sees action; action sees only the first video frame
- ``isolated``           video does not see action; action sees only the first video frame (FastWAM-style)

In the two modes where video sees action, the first video frame (the clean
conditioning frame) does not attend to action.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

MUTUAL = "mutual"
ACTION_SEES_VIDEO = "action_sees_video"
VIDEO_SEES_ACTION = "video_sees_action"
ISOLATED = "isolated"
VALID_ATTENTION_MASK_MODES = (MUTUAL, ACTION_SEES_VIDEO, VIDEO_SEES_ACTION, ISOLATED)


def validate_attention_mask_mode(mode: str) -> str:
    if mode not in VALID_ATTENTION_MASK_MODES:
        raise ValueError(f"unknown attention_mask_mode '{mode}'. Choose from: {VALID_ATTENTION_MASK_MODES}.")
    return mode


def set_video_attention_mask_mode(video_backbone, mode: Optional[str]) -> None:
    """Best-effort override of the video v<->v mask sub-mode."""
    if mode is None:
        return
    try:
        video_backbone.video_attention_mask_mode = mode
    except AttributeError:
        # Custom test doubles may expose the minimal surface only;
        # production WanVideoBackbone supports this property.
        fallback = getattr(video_backbone, "video_attention_mask_mode", "<unavailable>")
        logger.warning(
            "video_attention_mask_mode='%s' supplied but %s does not expose a settable property; falling back to %s.",
            mode,
            type(video_backbone).__name__,
            fallback,
        )


def fill_cross_modal_va_blocks(
    mask: torch.Tensor,
    *,
    v_start: int,
    v_end: int,
    a_start: int,
    a_end: int,
    mode: str,
    video_tokens_per_frame: int,
) -> None:
    """Fill the v->a and a->v blocks in place.

    Both directions are assigned explicitly (nothing relies on the initial
    fill), so masks starting from zeros and from ones are equally correct.
    """
    s_video = v_end - v_start
    ff = min(video_tokens_per_frame, s_video)  # tokens in the first frame

    # a->v: does action see video?
    if mode in (ACTION_SEES_VIDEO, MUTUAL):
        mask[a_start:a_end, v_start:v_end] = True
    else:  # VIDEO_SEES_ACTION, ISOLATED: first frame only
        mask[a_start:a_end, v_start:v_end] = False
        mask[a_start:a_end, v_start : v_start + ff] = True

    # v->a: does video see action?
    if mode in (MUTUAL, VIDEO_SEES_ACTION):
        mask[v_start:v_end, a_start:a_end] = True
        mask[v_start : v_start + ff, a_start:a_end] = False  # except the first-frame rows
    else:  # ACTION_SEES_VIDEO, ISOLATED
        mask[v_start:v_end, a_start:a_end] = False


def build_cross_modal_attention_mask(
    video_backbone,
    *,
    s_video: int,
    s_action: int,
    video_tokens_per_frame: int,
    mode: str,
    device: torch.device,
    n_readonly_tail: int = 0,
) -> torch.Tensor:
    """Build the ``[video, action, tail]`` bool attention mask (True = visible).

    - v<->v: ``video_backbone.build_video_to_video_mask`` (driven by
      ``video_attention_mask_mode``)
    - a<->a: True
    - v<->a: per ``mode`` (see :func:`fill_cross_modal_va_blocks`)
    - read-only ``tail`` (tri-system understanding tokens / single-system state
      tokens, structurally the same): video and action see the tail; the tail
      sees only itself and never video/action.
    """
    validate_attention_mask_mode(mode)
    total = s_video + s_action + int(n_readonly_tail)
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)

    mask[:s_video, :s_video] = video_backbone.build_video_to_video_mask(
        video_seq_len=s_video,
        video_tokens_per_frame=video_tokens_per_frame,
        device=device,
    )
    a_start, a_end = s_video, s_video + s_action
    mask[a_start:a_end, a_start:a_end] = True
    fill_cross_modal_va_blocks(
        mask,
        v_start=0,
        v_end=s_video,
        a_start=a_start,
        a_end=a_end,
        mode=mode,
        video_tokens_per_frame=video_tokens_per_frame,
    )

    if n_readonly_tail:
        tail_start = a_end
        mask[:tail_start, tail_start:] = True  # video + action see the tail
        mask[tail_start:, tail_start:] = True  # the tail sees only itself
    return mask


def widen_mask_for_prefix_kv(mask, state):
    """Prepend prefix-K/V key columns to a query×key mask.

    Some backbones return per-layer keys/values that carry a leading block with
    no matching query rows — Cosmos3's cached und (text) stream, declared via
    ``BlockLoopState.prefix_kv_len`` / ``prefix_kv_mask``. Every query row may
    attend that prefix (the video stream reads its own text natively; action
    rows reading text mirrors the cross-attention context the other variants
    provide), gated per sample by ``prefix_kv_mask`` when padding is present.

    Returns ``mask`` unchanged when the state declares no prefix, so callers can
    apply this unconditionally and backbones without a prefix stay byte-identical.

    Rank is preserved: a 2-D ``(S_q, S_k)`` mask widens to 2-D and a 4-D
    ``(B, 1, S_q, S_k)`` mask to 4-D. The batch axis is only introduced when a
    per-sample gate is actually present — ``prefix_kv_mask is None`` means every
    prefix key is real (every B=1 case, and any uniform-length batch), which
    keeps the cheap batch-shared 2-D mask instead of materializing B copies.
    Backbones must signal "no padding" with ``None`` rather than an all-True
    tensor: testing the tensor here would force a device sync and, inside the
    compiled MoT region, a dynamo graph break.
    """
    prefix = int(getattr(state, "prefix_kv_len", 0) or 0)
    if prefix <= 0:
        return mask
    rows = mask.shape[-2]
    prefix_mask = getattr(state, "prefix_kv_mask", None)
    if prefix_mask is None:
        pad_shape = (*mask.shape[:-1], prefix)
        pad = torch.ones(pad_shape, dtype=torch.bool, device=mask.device)
        return torch.cat([pad, mask], dim=-1)
    bsz = prefix_mask.shape[0]
    pad = prefix_mask.view(bsz, 1, 1, prefix).expand(bsz, 1, rows, prefix)
    base = mask if mask.dim() == 4 else mask.view(1, 1, rows, mask.shape[-1]).expand(bsz, 1, rows, mask.shape[-1])
    return torch.cat([pad, base], dim=-1)


__all__ = [
    "MUTUAL",
    "ACTION_SEES_VIDEO",
    "VIDEO_SEES_ACTION",
    "ISOLATED",
    "VALID_ATTENTION_MASK_MODES",
    "validate_attention_mask_mode",
    "set_video_attention_mask_mode",
    "fill_cross_modal_va_blocks",
    "build_cross_modal_attention_mask",
    "widen_mask_for_prefix_kv",
]
