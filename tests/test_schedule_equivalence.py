"""Verify that the sync schedule produces valid output and that removed
strategies are rejected loudly."""

import pytest


def _two_schedulers():
    """Build a (video, action) scheduler pair for tests.

    Both implement the same shifted-sigmoid Wan-equivalent formula, so
    schedule outputs are identical to the legacy single-template path.
    """
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    return FlowMatchScheduler("Wan"), ActionScheduler()


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
    ["video_leading", "cascade", "action_only", "decoupled_flash", "decoupled_asymmetric", "bogus"],
)
def test_make_schedule_rejects_removed_strategies(removed):
    from openwam.deploy.denoise_schedule import make_schedule

    v, a = _two_schedulers()
    with pytest.raises(NotImplementedError, match="only 'sync' is supported"):
        make_schedule(removed, v, a, num_steps=20, shift=5.0)


def test_action_scheduler_is_action_scheduler_instance():
    """Architecture's action_scheduler must be an ActionScheduler (not FlowMatchScheduler)."""
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    assert isinstance(arch.action_scheduler, ActionScheduler)
