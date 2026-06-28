"""Verify that the sync and independent schedules produce valid output and
that removed strategies are rejected loudly."""

import pytest


def _two_schedulers():
    """Build a (video, action) scheduler pair for tests.

    Both implement the same shifted-sigmoid Wan-equivalent formula, so
    schedule outputs are identical to the legacy single-template path.
    """
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    return FlowMatchScheduler("Wan"), ActionScheduler()


class _StubScheduler:
    """Minimal duck-typed scheduler for the ``independent`` schedule tests.

    ``schedule_independent`` only reads ``num_train_timesteps`` (it samples
    timesteps directly rather than indexing a grid), so the tests need no
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


@pytest.mark.parametrize(
    "removed",
    ["video_leading", "cascade", "action_only", "bogus"],
)
def test_make_schedule_rejects_removed_strategies(removed):
    from openwam.deploy.denoise_schedule import make_schedule

    # make_schedule raises on an unknown strategy before touching the
    # schedulers, so a lightweight stub pair suffices.
    v, a = _two_stub_schedulers()
    with pytest.raises(NotImplementedError, match="not supported"):
        make_schedule(removed, v, a, num_steps=20, shift=5.0)


def test_make_schedule_independent_structure():
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_stub_schedulers()
    result = make_schedule("independent", v, a, num_steps=20, shift=5.0, seed=0)
    assert len(result) == 21  # num_steps pairs + (0.0, 0.0) sentinel
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    assert result[-1] == (0.0, 0.0)


def test_schedule_independent_monotonic_descending():
    from openwam.deploy.denoise_schedule import schedule_independent

    v, a = _two_stub_schedulers()
    result = schedule_independent(v, a, num_steps=16, shift=5.0, shift_video=3.0, seed=123)
    v_ts = [t_v for t_v, _ in result[:-1]]
    a_ts = [t_a for _, t_a in result[:-1]]
    assert v_ts == sorted(v_ts, reverse=True)
    assert a_ts == sorted(a_ts, reverse=True)
    assert all(0.0 <= t <= v.num_train_timesteps for t in v_ts)
    assert all(0.0 <= t <= a.num_train_timesteps for t in a_ts)


def test_schedule_independent_seed_reproducible():
    from openwam.deploy.denoise_schedule import schedule_independent

    v, a = _two_stub_schedulers()
    r1 = schedule_independent(v, a, num_steps=12, shift=5.0, seed=42)
    r2 = schedule_independent(v, a, num_steps=12, shift=5.0, seed=42)
    r3 = schedule_independent(v, a, num_steps=12, shift=5.0, seed=7)
    assert r1 == r2  # same seed -> identical schedule
    assert r1 != r3  # different seed -> different schedule


def test_schedule_independent_streams_decoupled():
    """Video and action draw independent trajectories (not lockstep)."""
    from openwam.deploy.denoise_schedule import schedule_independent

    v, a = _two_stub_schedulers()
    result = schedule_independent(v, a, num_steps=24, shift=5.0, seed=1)
    v_ts = [t_v for t_v, _ in result[:-1]]
    a_ts = [t_a for _, t_a in result[:-1]]
    assert v_ts != a_ts


def test_build_timestep_sampler_default_is_none():
    from openwam.model.architectures.utils.timestep_sampling import build_timestep_sampler

    assert build_timestep_sampler(None) is None
    assert build_timestep_sampler("default") is None
    assert build_timestep_sampler("randint") is None
    assert build_timestep_sampler("independent_randint") is None


def test_build_timestep_sampler_rejects_unknown():
    from openwam.model.architectures.utils.timestep_sampling import build_timestep_sampler

    with pytest.raises(ValueError, match="Unknown training.timestep_sampling"):
        build_timestep_sampler("bogus")


def test_independent_timestep_sampler_shapes_and_seed():
    import torch

    from openwam.model.architectures.utils.timestep_sampling import (
        IndependentTimestepSampler,
        build_timestep_sampler,
    )

    sampler = build_timestep_sampler("independent_uniform_shift", num_train_timesteps=1000, seed=0)
    assert isinstance(sampler, IndependentTimestepSampler)

    v1, a1 = sampler.sample_timesteps(8, device="cpu")
    assert v1.shape == (8,) and a1.shape == (8,)
    assert float(v1.min()) >= 0.0 and float(v1.max()) <= 1000.0
    assert not torch.allclose(v1, a1)  # independent per-stream draws

    v2, a2 = build_timestep_sampler("independent_uniform_shift", seed=0).sample_timesteps(8, device="cpu")
    assert torch.allclose(v1, v2) and torch.allclose(a1, a2)  # seed reproducible


def test_action_scheduler_is_action_scheduler_instance():
    """Architecture's action_scheduler must be an ActionScheduler (not FlowMatchScheduler)."""
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    assert isinstance(arch.action_scheduler, ActionScheduler)
