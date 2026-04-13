"""Verify that schedule functions produce valid output."""


def test_schedule_sync():
    from openwam.deployment.schedule import schedule_sync

    result = schedule_sync(num_steps=20, shift=5.0)
    assert len(result) > 0
    assert all(isinstance(item, tuple) for item in result)


def test_schedule_video_leading():
    from openwam.deployment.schedule import schedule_video_leading

    result = schedule_video_leading(num_steps=20, shift=5.0, lead_steps=5)
    assert len(result) > 0


def test_schedule_cascade():
    from openwam.deployment.schedule import schedule_cascade

    result = schedule_cascade(video_steps=20, action_steps=10, shift=5.0)
    assert len(result) > 0


def test_schedule_action_only():
    from openwam.deployment.schedule import schedule_action_only

    result = schedule_action_only(num_steps=20, shift=5.0)
    assert len(result) > 0


def test_make_schedule_all_strategies():
    from openwam.deployment.schedule import make_schedule

    for strategy in ["sync", "video_leading", "cascade", "action_only"]:
        kwargs = {}
        if strategy == "video_leading":
            kwargs["lead_steps"] = 5
        elif strategy == "cascade":
            kwargs["video_steps"] = 15
            kwargs["action_steps"] = 15

        result = make_schedule(strategy, num_steps=20, shift=5.0, **kwargs)
        assert len(result) > 0, f"make_schedule({strategy}) returned empty"
