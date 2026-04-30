"""Verify that schedule functions produce valid output."""


def _two_schedulers():
    """Build a (video, action) scheduler pair for tests.

    Both implement the same shifted-sigmoid Wan-equivalent formula, so
    schedule outputs are identical to the legacy single-template path.
    """
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    return FlowMatchScheduler("Wan"), ActionScheduler()


def test_schedule_sync():
    from openwam.deploy.schedule import schedule_sync

    v, a = _two_schedulers()
    result = schedule_sync(v, a, num_steps=20, shift=5.0)
    assert len(result) > 0
    assert all(isinstance(item, tuple) for item in result)


def test_schedule_video_leading():
    from openwam.deploy.schedule import schedule_video_leading

    v, a = _two_schedulers()
    result = schedule_video_leading(v, a, num_steps=20, shift=5.0, lead_steps=5)
    assert len(result) > 0


def test_schedule_cascade():
    from openwam.deploy.schedule import schedule_cascade

    v, a = _two_schedulers()
    result = schedule_cascade(v, a, video_steps=20, action_steps=10, shift=5.0)
    assert len(result) > 0


def test_schedule_action_only():
    from openwam.deploy.schedule import schedule_action_only

    v, a = _two_schedulers()
    result = schedule_action_only(v, a, num_steps=20, shift=5.0)
    assert len(result) > 0


def test_make_schedule_all_strategies():
    from openwam.deploy.schedule import make_schedule

    for strategy in ["sync", "video_leading", "cascade", "action_only"]:
        kwargs = {}
        if strategy == "video_leading":
            kwargs["lead_steps"] = 5
        elif strategy == "cascade":
            kwargs["video_steps"] = 15
            kwargs["action_steps"] = 15

        v, a = _two_schedulers()
        result = make_schedule(strategy, v, a, num_steps=20, shift=5.0, **kwargs)
        assert len(result) > 0, f"make_schedule({strategy}) returned empty"


def test_action_scheduler_is_action_scheduler_instance():
    """Architecture's action_scheduler must be an ActionScheduler (not FlowMatchScheduler)."""
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    assert isinstance(arch.action_scheduler, ActionScheduler)
