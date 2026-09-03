"""``BasePipeline.check_resize_height_width`` num_frames round-up logic.

When ``time_division_factor > 1``, the method rounds ``num_frames`` up to
the next value satisfying ``N % time_division_factor == time_division_remainder``
(Wan VAE: factor=4 / V-JEPA 2.1: factor=2, both with remainder=1).

When ``time_division_factor == 1`` the constraint becomes meaningless —
``N % 1`` is always 0 and ``remainder=1`` is mathematically unsatisfiable —
so the round-up gate must be disabled entirely. Pre-fix the bare ``!=``
check would fire on every generate (a per-frame encoder such as DINOv3 sets
factor=1 / remainder=1 in ``WanVideoBackbone.from_pretrained``), silently
shifting num_frames by 1 each call. The fix adds a ``time_division_factor > 1``
guard so the no-compression path is a true no-op.
"""

from __future__ import annotations

import pytest

from openwam.model.video_backbone.wan.preprocess import check_resize_height_width


class _StubPipe:
    """Minimal stand-in carrying only the four attributes the method reads.

    ``height_division_factor`` / ``width_division_factor`` are set to 16 only
    so the H/W round-up branch doesn't fire (test inputs 384x320 are already
    16-aligned); the test bodies only exercise the time-division code path.
    """

    def __init__(self, time_division_factor: int, time_division_remainder: int):
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.height_division_factor = 16
        self.width_division_factor = 16


def _call(pipe, num_frames: int) -> int:
    height, width, out_frames = check_resize_height_width(
        384,
        320,
        num_frames,
        height_division_factor=pipe.height_division_factor,
        width_division_factor=pipe.width_division_factor,
        time_division_factor=pipe.time_division_factor,
        time_division_remainder=pipe.time_division_remainder,
    )
    assert (height, width) == (384, 320)
    return out_frames


def test_factor_one_no_round_up(capsys):
    """Per-frame encoder (factor=1, remainder=1). The remainder constraint is
    mathematically unsatisfiable when factor=1 (``N % 1`` is always 0),
    so the fix disables the round-up entirely and the method must return
    num_frames unchanged with no print. Pre-fix this printed
    ``num_frames % 1 != 1. We round it up to 10.`` on every generate."""
    pipe = _StubPipe(time_division_factor=1, time_division_remainder=1)
    assert _call(pipe, 9) == 9
    out = capsys.readouterr().out
    assert "num_frames" not in out, f"unexpected round-up message: {out!r}"


def test_wan22_vae_factor_four_already_compliant_no_round_up(capsys):
    """Wan VAE: factor=4, remainder=1, num_frames=9 satisfies 9 % 4 == 1.
    Method returns 9 unchanged and prints nothing."""
    pipe = _StubPipe(time_division_factor=4, time_division_remainder=1)
    assert _call(pipe, 9) == 9
    out = capsys.readouterr().out
    assert "num_frames" not in out


def test_wan22_vae_factor_four_round_up_still_works(capsys):
    """Regression guard: the fix must NOT break legitimate round-up for
    factor > 1. num_frames=8, factor=4, remainder=1 → round up to 9."""
    pipe = _StubPipe(time_division_factor=4, time_division_remainder=1)
    assert _call(pipe, 8) == 9
    out = capsys.readouterr().out
    assert "num_frames" in out and "9" in out


def test_vjepa_factor_two_already_compliant_no_round_up(capsys):
    """V-JEPA 2.1: factor=2, remainder=1, num_frames=9 satisfies 9 % 2 == 1.
    Method returns 9 unchanged and prints nothing."""
    pipe = _StubPipe(time_division_factor=2, time_division_remainder=1)
    assert _call(pipe, 9) == 9
    out = capsys.readouterr().out
    assert "num_frames" not in out


def test_vjepa_factor_two_round_up_still_works(capsys):
    """Regression guard: V-JEPA round-up still works. num_frames=10, factor=2,
    remainder=1 → round up to 11."""
    pipe = _StubPipe(time_division_factor=2, time_division_remainder=1)
    assert _call(pipe, 10) == 11
    out = capsys.readouterr().out
    assert "num_frames" in out and "11" in out


@pytest.mark.parametrize("num_frames", [1, 2, 5, 9, 17, 33, 65])
def test_factor_one_is_always_pass_through(num_frames):
    """Belt-and-braces: with factor=1, the method must return num_frames
    verbatim for any positive N."""
    pipe = _StubPipe(time_division_factor=1, time_division_remainder=1)
    assert _call(pipe, num_frames) == num_frames
