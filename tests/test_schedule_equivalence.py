"""Verify that new and legacy schedule functions produce identical output."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"
if str(WAM_DIR) not in sys.path:
    sys.path.insert(0, str(WAM_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_schedule_sync():
    from open_wam.inference.schedule import schedule_sync as new_sync
    from joint_inference import schedule_sync as old_sync

    new = new_sync(num_steps=20, shift=5.0)
    old = old_sync(num_steps=20, shift=5.0)
    assert new == old, f"sync schedules differ:\nnew={new[:3]}...\nold={old[:3]}..."


def test_schedule_video_leading():
    from open_wam.inference.schedule import schedule_video_leading as new_vl
    from joint_inference import schedule_video_leading as old_vl

    new = new_vl(num_steps=20, shift=5.0, lead_steps=5)
    old = old_vl(num_steps=20, shift=5.0, lead_steps=5)
    assert new == old


def test_schedule_cascade():
    from open_wam.inference.schedule import schedule_cascade as new_c
    from joint_inference import schedule_cascade as old_c

    new = new_c(video_steps=20, action_steps=10, shift=5.0)
    old = old_c(video_steps=20, action_steps=10, shift=5.0)
    assert new == old


def test_schedule_action_only():
    from open_wam.inference.schedule import schedule_action_only as new_ao
    from joint_inference import schedule_action_only as old_ao

    new = new_ao(num_steps=20, shift=5.0)
    old = old_ao(num_steps=20, shift=5.0)
    assert new == old


def test_make_schedule_all_strategies():
    from open_wam.inference.schedule import make_schedule as new_ms
    from joint_inference import make_schedule as old_ms

    for strategy in ["sync", "video_leading", "cascade", "action_only"]:
        kwargs = {}
        if strategy == "video_leading":
            kwargs["lead_steps"] = 5
        elif strategy == "cascade":
            kwargs["video_steps"] = 15
            kwargs["action_steps"] = 15

        new = new_ms(strategy, num_steps=20, shift=5.0, **kwargs)
        old = old_ms(strategy, num_steps=20, shift=5.0, **kwargs)
        assert new == old, f"make_schedule({strategy}) differs"
