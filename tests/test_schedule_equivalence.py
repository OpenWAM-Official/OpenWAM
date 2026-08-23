"""Verify synchronous and asynchronous denoising schedules."""

import pytest


def _two_schedulers():
    """Build a (video, action) scheduler pair for tests.

    Both implement the same shifted-sigmoid Wan-equivalent formula.
    """
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    return FlowMatchScheduler("Wan"), ActionScheduler()


class _StubScheduler:
    """Minimal duck-typed scheduler for the ``variance_shift`` schedule tests.

    ``schedule_variance_shift`` only reads ``num_train_timesteps`` (it derives
    timesteps from the curve rather than indexing a grid), so the tests need no
    heavy backbone scheduler. Keeps them fast and import-light.
    """

    num_train_timesteps = 1000


def _two_stub_schedulers():
    return _StubScheduler(), _StubScheduler()


def test_schedule_sync():
    from openwam.deploy.denoise_schedule import schedule_sync

    v, a = _two_schedulers()
    result = schedule_sync(v, a, num_steps=20, shift=5.0)
    assert len(result) > 0
    assert all(isinstance(item, tuple) for item in result)


def test_make_schedule_sync():
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    result = make_schedule("sync", v, a, num_steps=20, shift=5.0)
    assert len(result) > 0
    assert result[-1] == (0.0, 0.0)


def test_make_schedule_rejects_invalid_mode():
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    with pytest.raises(ValueError, match="Unsupported denoise mode"):
        make_schedule("unsupported", v, a)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "variance_shift_alpha must be >= 1"),
        ({"alpha": 0.5}, "variance_shift_alpha must be >= 1"),
        ({"alpha": float("nan")}, "variance_shift_alpha must be a finite number"),
        ({"offset": -0.1}, "linear_offset must satisfy"),
        ({"offset": 1.0}, "linear_offset must satisfy"),
        ({"offset": float("inf")}, "linear_offset must be a finite number"),
    ],
)
def test_make_schedule_rejects_invalid_async_parameters(kwargs, match):
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    with pytest.raises(ValueError, match=match):
        make_schedule("async", v, a, **kwargs)


def test_make_schedule_sync_ignores_inactive_async_parameters():
    """Per-request path builds what it is handed; the shape check lives at startup."""
    from openwam.deploy.denoise_schedule import make_schedule, schedule_sync

    v, a = _two_schedulers()
    expected = schedule_sync(v, a, num_steps=20, shift=5.0)

    v, a = _two_schedulers()
    assert make_schedule("sync", v, a, num_steps=20, shift=5.0, lead="action", alpha=9.0, offset=0.3) == expected


def test_normalize_denoise_config_rejects_inactive_controls_by_default():
    from openwam.deploy.denoise_schedule import normalize_denoise_config

    with pytest.raises(ValueError, match="denoise_mode='async'"):
        normalize_denoise_config({"denoise_mode": "sync", "variance_shift_alpha": 9.0})


def test_normalize_denoise_config_reset_inactive_drops_async_controls():
    from openwam.deploy.denoise_schedule import DenoiseConfig, normalize_denoise_config

    resolved = normalize_denoise_config(
        {"denoise_mode": "sync", "lead_modality": "action", "variance_shift_alpha": 9.0, "linear_offset": 0.3},
        reset_inactive=True,
    )
    assert resolved == DenoiseConfig()


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"denoise_mode": "async"}, True),
        ({"denoise_mode": "async", "variance_shift_alpha": 1.0, "linear_offset": 0.0}, True),
        ({"denoise_mode": "async", "variance_shift_alpha": 3.0}, False),
        ({"denoise_mode": "async", "linear_offset": 0.1}, False),
        ({"denoise_mode": "sync"}, False),
    ],
)
def test_denoise_async_is_noop(cfg, expected):
    from openwam.deploy.denoise_schedule import denoise_async_is_noop, normalize_denoise_config

    assert denoise_async_is_noop(normalize_denoise_config(cfg)) is expected


def test_make_schedule_async_structure():
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    result = make_schedule("async", v, a, num_steps=20, lead="action", alpha=9.0)
    assert len(result) == 21  # num_steps pairs + (0.0, 0.0) sentinel
    assert result[-1] == (0.0, 0.0)


def test_schedule_variance_shift_lead_is_cleaner_and_monotonic():
    from openwam.deploy.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    result = schedule_variance_shift(v, a, num_steps=20, lead="action", alpha=9.0)
    v_ts = [tv for tv, _ in result[:-1]]
    a_ts = [ta for _, ta in result[:-1]]
    assert v_ts == sorted(v_ts, reverse=True)
    assert a_ts == sorted(a_ts, reverse=True)
    # action leads -> action stays at lower-or-equal timestep (cleaner) every step
    assert all(ta <= tv + 1e-9 for tv, ta in zip(v_ts, a_ts))
    assert any(ta < tv - 1e-6 for tv, ta in zip(v_ts, a_ts))  # strictly leads somewhere


def test_schedule_variance_shift_alpha1_is_diagonal():
    from openwam.deploy.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    result = schedule_variance_shift(v, a, num_steps=16, lead="action", alpha=1.0)
    for tv, ta in result[:-1]:
        assert abs(tv - ta) < 1e-9  # alpha=1 -> both streams identical (sync diagonal)


def test_variance_shift_alpha1_matches_sync_bitwise():
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    sync = make_schedule("sync", v, a, num_steps=50, shift=5.0, shift_video=3.0)
    for lead in ("video", "action"):
        vs = make_schedule("async", v, a, num_steps=50, shift=5.0, shift_video=3.0, lead=lead, alpha=1.0)
        assert vs == sync  # exact float equality, not approx


def test_schedule_variance_shift_lead_direction_flips():
    from openwam.deploy.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    res_a = schedule_variance_shift(v, a, num_steps=12, lead="action", alpha=9.0)
    res_v = schedule_variance_shift(v, a, num_steps=12, lead="video", alpha=9.0)
    assert [tv for tv, _ in res_a] == [ta for _, ta in res_v]
    assert [ta for _, ta in res_a] == [tv for tv, _ in res_v]


@pytest.mark.parametrize("lead", ["action", "video"])
def test_schedule_variance_shift_offset_delays_lag(lead):
    from openwam.deploy.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    # Dyadic num_steps/offset: s[k] and s/(1-off) are exact in float32, so the
    # pinned head has a crisp boundary at k = off*num_steps (inclusive: s=1-off
    # maps to sigma exactly 1.0).
    num_steps, off = 16, 0.5
    base = schedule_variance_shift(v, a, num_steps=num_steps, lead=lead, alpha=9.0)
    res = schedule_variance_shift(v, a, num_steps=num_steps, lead=lead, alpha=9.0, offset=off)
    lag_idx = 0 if lead == "action" else 1  # offset delays the non-lead stream
    lead_idx = 1 - lag_idx
    # The lead stream's code path is untouched by offset.
    assert [p[lead_idx] for p in res] == [p[lead_idx] for p in base]
    lag = [p[lag_idx] for p in res[:-1]]
    pin = int(off * num_steps)
    assert all(t == 1000.0 for t in lag[: pin + 1])
    assert all(t < 1000.0 for t in lag[pin + 1 :])
    # Both streams stay monotonically non-increasing.
    v_ts = [tv for tv, _ in res[:-1]]
    a_ts = [ta for _, ta in res[:-1]]
    assert v_ts == sorted(v_ts, reverse=True)
    assert a_ts == sorted(a_ts, reverse=True)
    assert len(res) == num_steps + 1
    assert res[-1] == (0.0, 0.0)


def test_schedule_variance_shift_offset_zero_is_noop_bitwise():
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    default = make_schedule("async", v, a, num_steps=50, shift=5.0, lead="action", alpha=9.0)
    explicit = make_schedule("async", v, a, num_steps=50, shift=5.0, lead="action", alpha=9.0, offset=0.0)
    assert explicit == default  # exact float equality, not approx


def test_schedule_variance_shift_offset_lead_flip_swaps_streams():
    from openwam.deploy.denoise_schedule import schedule_variance_shift

    v, a = _two_stub_schedulers()
    res_a = schedule_variance_shift(v, a, num_steps=12, lead="action", alpha=9.0, offset=0.3)
    res_v = schedule_variance_shift(v, a, num_steps=12, lead="video", alpha=9.0, offset=0.3)
    assert [tv for tv, _ in res_a] == [ta for _, ta in res_v]
    assert [ta for _, ta in res_a] == [tv for tv, _ in res_v]


def test_action_scheduler_is_action_scheduler_instance():
    """Architecture's action_scheduler must be an ActionScheduler (not FlowMatchScheduler)."""
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    assert isinstance(arch.action_scheduler, ActionScheduler)
