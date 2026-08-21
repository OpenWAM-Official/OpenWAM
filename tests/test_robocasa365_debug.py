"""Tests for the RoboCasa365 inspection helpers (build_obs_payload + dump_obs_debug).

These run offline (no sim, no server): synthetic obs in, the per-step debug bundle
out. The bundle mirrors robotwin's debug layout (ep{N}/step_{N}/ with per-camera
JPGs + meta.json) and adds a labeled cameras.png montage + pass/fail checks.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "robocasa365"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import openwam2robocasa365_interface as adapter  # noqa: E402


def _make_obs():
    head = np.zeros((64, 64, 3), dtype=np.uint8)
    head[..., 0] = 200
    left = np.zeros((64, 64, 3), dtype=np.uint8)
    left[..., 1] = 200
    right = np.zeros((64, 64, 3), dtype=np.uint8)
    right[..., 2] = 200
    return {
        "state.base_position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "state.base_rotation": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "state.end_effector_position_relative": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "state.end_effector_rotation_relative": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        "state.gripper_qpos": np.array([0.01, 0.02], dtype=np.float32),
        "video.robot0_agentview_left": head,
        "video.robot0_eye_in_hand": left,
        "video.robot0_agentview_right": right,
        "annotation.human.task_description": "open the drawer",
    }


def _payload(obs, *, right_wrist_camera_key="video.robot0_agentview_right"):
    # The production mapping fills all three slots. Passing None explicitly still
    # exercises the supported legacy/debug black-slot path below.
    return adapter.build_obs_payload(
        obs,
        head_camera_key="video.robot0_agentview_left",
        left_wrist_camera_key="video.robot0_eye_in_hand",
        right_wrist_camera_key=right_wrist_camera_key,
        image_transform="none",
        state_keys=adapter.DEFAULT_STATE_KEYS,
        prompt=obs.get("annotation.human.task_description", ""),
    )


def test_build_obs_payload_missing_head_raises():
    obs = _make_obs()
    del obs["video.robot0_agentview_left"]
    with pytest.raises(KeyError):
        adapter.build_obs_payload(
            obs,
            head_camera_key="video.robot0_agentview_left",
            left_wrist_camera_key=None,
            right_wrist_camera_key=None,
            image_transform="none",
            state_keys=adapter.DEFAULT_STATE_KEYS,
            prompt="x",
        )


def test_dump_obs_debug_robotwin_layout_and_meta(tmp_path):
    obs = _make_obs()
    out = adapter.dump_obs_debug(
        obs,
        _payload(obs),
        tmp_path / "ep0000" / "step_0000",
        action=np.arange(12, dtype=np.float32),
        episode=0,
        step=0,
        server_step=5,
        latency_ms=12.3,
    )
    # Lossless per-camera PNGs + the montage.
    for stem in ("head", "left", "right"):
        assert (out / f"{stem}.png").is_file()
    assert (out / "cameras.png").is_file()

    meta = json.loads((out / "meta.json").read_text())
    # robotwin fields
    assert meta["episode"] == 0
    assert meta["step"] == 0
    assert meta["server_step"] == 5
    assert meta["latency_ms"] == 12.3
    assert meta["prompt"] == "open the drawer"
    assert len(meta["state"]) == 19  # compact EEF10 + base pose9
    assert meta["action"] == [float(i) for i in range(12)]
    # enhancements
    assert meta["action_sliced"]["action.base_motion"] == [7.0, 8.0, 9.0, 10.0]
    assert meta["action_sliced"]["action.control_mode"] == [11.0]
    assert meta["state_breakdown"]["state.gripper_qpos"] == pytest.approx([0.01, 0.02])
    assert meta["image_slots"] == {
        "head_camera": [320, 256],
        "left_wrist_camera": [160, 128],
        "right_wrist_camera": [160, 128],
    }
    assert meta["checks"] == {
        "state_dim_ok": True,
        "head_and_wrist_present": True,
        "action_dim_is_12": True,
    }


def test_dump_obs_debug_missing_wrist_writes_stub(tmp_path):
    obs = _make_obs()
    out = adapter.dump_obs_debug(obs, _payload(obs, right_wrist_camera_key=None), tmp_path / "s", action=None)
    assert (out / "head.png").is_file()
    assert (out / "left.png").is_file()
    assert not (out / "right.jpg").exists()
    assert (out / "right_missing.txt").is_file()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["image_slots"]["right_wrist_camera"] is None
    assert meta["checks"]["head_and_wrist_present"] is True  # head + left wrist remain present
    assert "action_dim_is_12" not in meta["checks"]  # action omitted
