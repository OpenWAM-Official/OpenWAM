"""Tests for RoboCOIN camera resolution (`_resolve_robocoin_cameras`).

``_resolve_robocoin_cameras`` picks head / left-wrist / right-wrist camera keys
from a bucket's ``info.json`` features by priority order. Only the *keys* matter,
so we feed plain ``{camera_key: {}}`` dicts — no real dataset on disk, no reader
instantiation.
"""

from __future__ import annotations

from openwam.dataloader.robocoin import _resolve_robocoin_cameras


def _features(*keys: str) -> dict:
    """Build a features dict from camera keys (only the keys are consulted)."""
    return {k: {} for k in keys}


class TestResolveRoboCOINCameras:
    def test_ai2_alphabot2_resolves_dedicated_head(self):
        """Regression: ai2_alphabot2 must resolve its own head cam, not chest.

        Every AI2_Alphabot_2_* bucket exposes exactly
        {cam_front_chest_rgb, cam_front_head_rgb, cam_left_wrist_rgb,
        cam_right_wrist_rgb}. Before the fix, cam_front_head_rgb was absent from
        HEAD_CAMERA_PRIORITY so the resolver fell through to the chest camera.
        """
        head, left_wrist, right_wrist = _resolve_robocoin_cameras(
            _features(
                "observation.images.cam_front_chest_rgb",
                "observation.images.cam_front_head_rgb",
                "observation.images.cam_left_wrist_rgb",
                "observation.images.cam_right_wrist_rgb",
            )
        )
        assert head == "observation.images.cam_front_head_rgb"
        assert left_wrist == "observation.images.cam_left_wrist_rgb"
        assert right_wrist == "observation.images.cam_right_wrist_rgb"

    def test_dedicated_head_not_preempted(self):
        """No-preemption: a dedicated cam_head_rgb still outranks the new key."""
        head, _, _ = _resolve_robocoin_cameras(
            _features(
                "observation.images.cam_head_rgb",
                "observation.images.cam_front_head_rgb",
                "observation.images.cam_front_chest_rgb",
            )
        )
        assert head == "observation.images.cam_head_rgb"

    def test_chest_fallback_preserved(self):
        """Chest fallback: with only chest (+ wrists) present, head is chest."""
        head, left_wrist, right_wrist = _resolve_robocoin_cameras(
            _features(
                "observation.images.cam_front_chest_rgb",
                "observation.images.cam_left_wrist_rgb",
                "observation.images.cam_right_wrist_rgb",
            )
        )
        assert head == "observation.images.cam_front_chest_rgb"
        assert left_wrist == "observation.images.cam_left_wrist_rgb"
        assert right_wrist == "observation.images.cam_right_wrist_rgb"
