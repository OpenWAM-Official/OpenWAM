"""State-token helpers for SingleSystem architectures."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from openwam.model.architectures.utils.common import compute_video_tokens_per_frame
from openwam.model.architectures.utils.mask_modes import (
    build_cross_modal_attention_mask,
    validate_attention_mask_mode,
)


def align_state_tokens_to_action_batch(state_tokens: Optional[Tensor], action_batch_size: int) -> Optional[Tensor]:
    """Match shared state-token batch semantics to dual-system proprio context.

    Dual-system accepts a single deploy-time ``[D]`` proprio vector, converts it
    to batch size one, then expands it when the model batch is larger. Shared
    state tokens use the same rule after encoding.
    """
    if state_tokens is None:
        return None
    if state_tokens.shape[0] == action_batch_size:
        return state_tokens
    if state_tokens.shape[0] == 1 and action_batch_size > 1:
        return state_tokens.expand(action_batch_size, -1, -1)
    raise ValueError(
        f"Batch mismatch between action tokens and proprio: {action_batch_size} vs {state_tokens.shape[0]}"
    )


def attach_shared_attention_mask(
    video_backbone,
    state,
    n_action: int,
    *,
    n_state: int = 0,
    attention_mask_mode: str,
) -> None:
    """Build the shared-sequence mask and stash it on ``state.extras``.

    The single system concatenates action (and an optional state token) into
    the video DiT sequence, so the mask travels through ``state.extras`` for
    ``WanVideoBackbone.run_block`` to consume. Layout ``[video, action, state]``
    is the unified cross-modal mask with the state token as a read-only tail.
    """
    mode = validate_attention_mask_mode(attention_mask_mode)
    n_action = int(n_action)
    n_state = int(n_state or 0)
    if n_action < 0 or n_state < 0:
        raise ValueError(f"n_action and n_state must be non-negative, got {n_action}, {n_state}")
    if n_action + n_state <= 0:
        raise ValueError("SingleSystem attention mask requires at least one action or state token.")

    extras = getattr(state, "extras", None)
    if extras is None:
        raise RuntimeError(
            "SingleSystem attention_mask_mode requires BlockLoopState.extras "
            "so video_backbone.run_block() can consume shared_attention_mask."
        )

    total = int(state.hidden_states.shape[1])
    s_video = total - n_action - n_state
    if s_video <= 0:
        raise ValueError(
            f"SingleSystem attention mask expected video tokens before action/state tails, "
            f"got total={total}, n_action={n_action}, n_state={n_state}."
        )

    extras["shared_attention_mask"] = build_cross_modal_attention_mask(
        video_backbone,
        s_video=s_video,
        s_action=n_action,
        video_tokens_per_frame=compute_video_tokens_per_frame(state, "SingleSystem"),
        mode=mode,
        device=state.hidden_states.device,
        n_readonly_tail=n_state,
    )
