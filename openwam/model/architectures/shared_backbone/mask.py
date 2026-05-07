"""Attention mask helpers for SharedBackbone architectures."""

from __future__ import annotations

import logging
from typing import Optional

import torch

VALID_SHARED_ATTENTION_MASK_MODES = ("bidirectional", "joint")
logger = logging.getLogger(__name__)


def validate_shared_attention_mask_mode(mode: str) -> str:
    if mode not in VALID_SHARED_ATTENTION_MASK_MODES:
        raise ValueError(
            f"SharedBackbone: unknown attention_mask_mode '{mode}'. Choose from: {VALID_SHARED_ATTENTION_MASK_MODES}."
        )
    return mode


def set_video_attention_mask_mode(video_backbone, mode: Optional[str]) -> None:
    """Best-effort override of the video v<->v mask sub-mode."""
    if mode is None:
        return
    try:
        video_backbone.video_attention_mask_mode = mode
    except AttributeError:
        # Custom test doubles may expose the minimal SharedBackbone surface only.
        # Production WanVideoBackbone supports this property.
        fallback = getattr(video_backbone, "video_attention_mask_mode", "<unavailable>")
        logger.warning(
            "video_attention_mask_mode='%s' supplied to SharedBackbone but %s does not expose "
            "a settable property; falling back to %s.",
            mode,
            type(video_backbone).__name__,
            fallback,
        )
        return


def build_shared_backbone_attention_mask(
    video_backbone,
    state,
    n_action: int,
    attention_mask_mode: str,
) -> Optional[torch.Tensor]:
    """Build the SharedBackbone self-attention mask.

    Layout for ``joint`` mirrors the dual-system MoT mask after action tokens
    have been appended to the video sequence:

    - video -> video: delegated to ``video_backbone.build_video_to_video_mask``
    - video -> action: blocked
    - action -> video: allowed
    - action -> action: allowed

    ``bidirectional`` returns ``None`` so the standard fused Wan block path can
    be used unchanged.
    """
    mode = validate_shared_attention_mask_mode(attention_mask_mode)
    if mode == "bidirectional":
        return None

    if n_action <= 0:
        raise ValueError(f"n_action must be positive, got {n_action}")

    total = int(state.x.shape[1])
    s_video = total - int(n_action)
    if s_video <= 0:
        raise ValueError(
            f"SharedBackbone attention mask expected video tokens before the action tail, "
            f"got total={total}, n_action={n_action}."
        )

    h = int(getattr(state, "h", 0))
    w = int(getattr(state, "w", 0))
    video_tokens_per_frame = h * w
    if video_tokens_per_frame <= 0:
        raise ValueError(f"SharedBackbone cannot derive video_tokens_per_frame from BlockLoopState (h={h}, w={w}).")

    device = state.x.device
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:s_video, :s_video] = video_backbone.build_video_to_video_mask(
        video_seq_len=s_video,
        video_tokens_per_frame=video_tokens_per_frame,
        device=device,
    )
    mask[s_video:, :s_video] = True
    mask[s_video:, s_video:] = True
    return mask


def attach_shared_attention_mask(
    video_backbone,
    state,
    n_action: int,
    *,
    attention_mask_mode: str,
) -> None:
    """Attach the computed mask to ``state.extras`` for WanVideoBackbone.run_block."""
    mode = validate_shared_attention_mask_mode(attention_mask_mode)
    extras = getattr(state, "extras", None)
    if extras is None:
        if mode == "bidirectional":
            return
        raise RuntimeError(
            "SharedBackbone attention_mask_mode='joint' requires BlockLoopState.extras "
            "so video_backbone.run_block() can consume shared_attention_mask."
        )
    if mode == "joint" and extras.get("use_usp", False):
        raise NotImplementedError(
            "SharedBackbone attention_mask_mode='joint' does not support unified sequence parallel yet."
        )
    mask = build_shared_backbone_attention_mask(video_backbone, state, n_action, mode)
    if mask is None:
        extras.pop("shared_attention_mask", None)
    else:
        extras["shared_attention_mask"] = mask
