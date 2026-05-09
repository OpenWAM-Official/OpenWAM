"""State-token helpers for SharedBackbone architectures."""

from __future__ import annotations

from typing import Optional

from torch import Tensor


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
        f"Batch mismatch between action tokens and proprio_state: {action_batch_size} vs {state_tokens.shape[0]}"
    )
