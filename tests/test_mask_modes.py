"""Unit tests for the unified cross-modal attention-mask builder.

Validates the four modes' v<->a layout, the first-frame-row exclusion, and the
read-only tail (tri's understanding / shared's state) block.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    ISOLATED,
    MUTUAL,
    VIDEO_SEES_ACTION,
    build_cross_modal_attention_mask,
    validate_attention_mask_mode,
)


class _VBFirstFrameCausal:
    """Minimal video backbone exposing first_frame_causal v<->v sub-mask."""

    video_attention_mask_mode = "first_frame_causal"

    def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        ff = min(video_tokens_per_frame, video_seq_len)
        mask[:ff, ff:] = False  # first frame only sees itself
        return mask


SV, SA, TPF = 6, 3, 2  # 3 video frames x 2 tokens, 3 action tokens
FF = min(TPF, SV)  # first-frame token count = 2


def _build(mode, n_tail=0):
    return build_cross_modal_attention_mask(
        _VBFirstFrameCausal(),
        s_video=SV,
        s_action=SA,
        video_tokens_per_frame=TPF,
        mode=mode,
        device=torch.device("cpu"),
        n_readonly_tail=n_tail,
    )


def test_validate_rejects_unknown_mode():
    with pytest.raises(ValueError, match="attention_mask_mode"):
        validate_attention_mask_mode("joint")  # old name no longer valid


@pytest.mark.parametrize("mode", [MUTUAL, ACTION_SEES_VIDEO, VIDEO_SEES_ACTION, ISOLATED])
def test_shared_invariants(mode):
    """v<->v sub-mask and a<->a hold for all modes."""
    mask = _build(mode)
    assert mask.shape == (SV + SA, SV + SA)
    assert mask.dtype == torch.bool
    # a<->a fully connected
    assert mask[SV:, SV:].all()
    # v<->v first_frame_causal: first frame rows don't see later video
    assert not mask[:FF, FF:SV].any()
    assert mask[:FF, :FF].all()
    assert mask[FF:SV, :SV].all()


def test_action_sees_video():
    mask = _build(ACTION_SEES_VIDEO)
    assert mask[SV:, :SV].all()  # a -> all video
    assert not mask[:SV, SV:].any()  # v -> a blocked


def test_isolated():
    mask = _build(ISOLATED)
    assert mask[SV:, :FF].all()  # a -> first frame only
    assert not mask[SV:, FF:SV].any()  # a -> later video blocked
    assert not mask[:SV, SV:].any()  # v -> a blocked


def test_mutual():
    mask = _build(MUTUAL)
    assert mask[SV:, :SV].all()  # a -> all video
    # v -> a: all video rows EXCEPT first-frame rows
    assert not mask[:FF, SV:].any()  # first-frame rows do NOT see action
    assert mask[FF:SV, SV:].all()  # later video rows see action


def test_video_sees_action():
    mask = _build(VIDEO_SEES_ACTION)
    assert mask[SV:, :FF].all()  # a -> first frame only
    assert not mask[SV:, FF:SV].any()  # a -> later video blocked
    assert not mask[:FF, SV:].any()  # v -> a: first-frame rows excluded
    assert mask[FF:SV, SV:].all()  # later video rows see action


def test_readonly_tail():
    """tail (understanding/state): everyone sees it, it sees only itself."""
    n_tail = 4
    mask = _build(ACTION_SEES_VIDEO, n_tail=n_tail)
    total = SV + SA + n_tail
    assert mask.shape == (total, total)
    tail = SV + SA
    assert mask[:tail, tail:].all()  # video + action -> tail
    assert mask[tail:, tail:].all()  # tail -> tail
    assert not mask[tail:, :tail].any()  # tail -> video/action blocked
