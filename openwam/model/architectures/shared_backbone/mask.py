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
    n_state: int = 0,
) -> Optional[torch.Tensor]:
    """Build the SharedBackbone self-attention mask.

    Layout for ``joint`` after action/state tokens have been appended:

    - video -> video: delegated to ``video_backbone.build_video_to_video_mask``
    - video -> action: blocked
    - video -> state: allowed
    - action -> video: allowed, when action tokens are present
    - action -> action: allowed, when action tokens are present
    - action -> state: allowed, when action tokens are present
    - state -> state: allowed
    - state -> video/action: blocked

    ``bidirectional`` returns ``None`` so the standard fused Wan block path can
    be used unchanged.
    """
    mode = validate_shared_attention_mask_mode(attention_mask_mode)
    if mode == "bidirectional":
        return None

    if n_action < 0:
        raise ValueError(f"n_action must be non-negative, got {n_action}")

    n_state = int(n_state or 0)
    if n_state < 0:
        raise ValueError(f"n_state must be non-negative, got {n_state}")
    if n_action + n_state <= 0:
        raise ValueError("SharedBackbone attention mask requires at least one action or state token.")

    total = int(state.x.shape[1])
    s_video = total - int(n_action) - n_state
    if s_video <= 0:
        raise ValueError(
            f"SharedBackbone attention mask expected video tokens before action/state tails, "
            f"got total={total}, n_action={n_action}, n_state={n_state}."
        )

    # Reuse the MoT helper so the v↔v mask uses the same tokens_per_frame
    # value across SharedBackbone and DualSystem paths.
    from openwam.model.architectures._mot_utils import compute_video_tokens_per_frame

    video_tokens_per_frame = compute_video_tokens_per_frame(state, "SharedBackbone")

    device = state.x.device
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:s_video, :s_video] = video_backbone.build_video_to_video_mask(
        video_seq_len=s_video,
        video_tokens_per_frame=video_tokens_per_frame,
        device=device,
    )
    action_start = s_video
    action_end = action_start + int(n_action)
    state_start = action_end

    # video queries: video sub-mask + state conditioning, but no action.
    if n_state:
        mask[:s_video, state_start:] = True

    # action queries: video, action, and state are all visible.
    if n_action:
        mask[action_start:action_end, :] = True

    # state queries: state tokens are conditioning anchors; they only see state.
    if n_state:
        mask[state_start:, state_start:] = True
    return mask


def attach_shared_attention_mask(
    video_backbone,
    state,
    n_action: int,
    *,
    n_state: int = 0,
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
    mask = build_shared_backbone_attention_mask(video_backbone, state, n_action, mode, n_state=n_state)
    if mask is None:
        extras.pop("shared_attention_mask", None)
    else:
        extras["shared_attention_mask"] = mask
