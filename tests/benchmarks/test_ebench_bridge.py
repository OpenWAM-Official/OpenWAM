"""Offline tests for the EBench bridge — no GenManip, no Isaac Sim, no server.

Three layers:
1. Trainer↔bridge mirrors pinned byte-for-byte (base rendering, quat→rot6d,
   raw-23 proprio construction, prompt template) — the eval and training ends
   of one contract must never drift.
2. Action round-trip: raw-23 → EBench ``ee_pose`` action dict → recovered
   rotation/position/gripper/base match the model output.
3. Wire contract: the dict survives the exact consumption pattern of GenManip's
   ``parse_embodiment_action`` (list concatenation into cuRobo IK) and the
   driver's episode bookkeeping (reset, prev-base differencing) works against
   a fake south client.
"""

import math

import numpy as np
import pytest

from benchmarks.ebench.openwam2ebench_interface import EBenchOpenWAMDriver
from benchmarks.ebench.prompt_template import format_prompt_for_inference as bridge_prompt
from benchmarks.utils.action_conversion import (
    EBENCH_GRIPPER_OPEN,
    ebench_obs_to_raw23,
    ebench_quat_wxyz_to_rot6d,
    ebench_render_state_base,
    ebench_wrap_angle_rad,
    raw23_to_ebench_action,
)

# Trainer-side (openwam) counterparts — repo tests run in the training env.
from openwam.dataloader.ebench import (
    _ee_pose_gripper_base_to_raw23,
    _quat_wxyz_to_rot6d,
    render_ebench_state_base,
    wrap_angle_rad,
)
from openwam.dataloader.transforms.multiview import (
    format_prompt_for_inference as trainer_prompt,
)


def _rand_unit_quat_wxyz(rng, n=1):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


# ------------------------------------------------------------------ mirrors


def test_prompt_template_pinned_to_trainer():
    for text in ("pick the apple", "把苹果放进果盘", ""):
        assert bridge_prompt(text) == trainer_prompt(text)


def test_wrap_angle_mirror():
    angles = np.linspace(-10, 10, 999)
    np.testing.assert_array_equal(ebench_wrap_angle_rad(angles), wrap_angle_rad(angles))


def test_render_state_base_mirror():
    rng = np.random.default_rng(0)
    for _ in range(200):
        cur = rng.normal(scale=3.0, size=3)
        prev = rng.normal(scale=3.0, size=3)
        for mode in ("delta", "cumulative"):
            np.testing.assert_array_equal(
                ebench_render_state_base(cur, prev, mode),
                render_ebench_state_base(cur, prev, mode),
            )
        np.testing.assert_array_equal(
            ebench_render_state_base(cur, None, "delta"),
            render_ebench_state_base(cur, None, "delta"),
        )
    with pytest.raises(ValueError):
        ebench_render_state_base(np.zeros(3), None, "velocity")


def test_quat_to_rot6d_mirror():
    rng = np.random.default_rng(1)
    q = _rand_unit_quat_wxyz(rng, 128)
    np.testing.assert_array_equal(ebench_quat_wxyz_to_rot6d(q), _quat_wxyz_to_rot6d(q))


def test_obs_to_raw23_mirror():
    rng = np.random.default_rng(2)
    for _ in range(50):
        lq, rq = _rand_unit_quat_wxyz(rng)[0], _rand_unit_quat_wxyz(rng)[0]
        lpos, rpos = rng.normal(size=3).astype(np.float32), rng.normal(size=3).astype(np.float32)
        grip = rng.uniform(0, 0.044, size=4).astype(np.float32)
        base = rng.normal(size=3).astype(np.float32)
        # bridge input: the obs nested-pair shape
        nested = [[lpos.tolist(), lq.tolist()], [rpos.tolist(), rq.tolist()]]
        got = ebench_obs_to_raw23(nested, grip, base)
        # trainer input: flat rows
        flat14 = np.concatenate([lpos, lq, rpos, rq])[None, :]
        want = _ee_pose_gripper_base_to_raw23(flat14, grip[None, :], base[None, :])[0]
        np.testing.assert_array_equal(got, want)


# ------------------------------------------------------------------ round-trip


def _rot6d_normalize(r6d):
    a1, a2 = np.asarray(r6d[:3], np.float64), np.asarray(r6d[3:6], np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    return np.concatenate([b1, b2])


def test_action_round_trip_delta():
    rng = np.random.default_rng(3)
    raw = rng.normal(scale=0.3, size=23)
    raw[9], raw[19] = 0.03, 0.01  # in-range grippers
    raw[3:9] = ebench_quat_wxyz_to_rot6d(_rand_unit_quat_wxyz(rng)[0])
    raw[13:19] = ebench_quat_wxyz_to_rot6d(_rand_unit_quat_wxyz(rng)[0])
    out = raw23_to_ebench_action(raw, "delta")

    assert out["control_type"] == "ee_pose" and out["is_rel"] is False
    assert out["base_is_rel"] is True
    np.testing.assert_allclose(out["base_motion"], raw[20:23], atol=1e-9)

    for arm_idx, (sl_pos, sl_rot, g_idx) in enumerate(
        [(slice(0, 3), slice(3, 9), 9), (slice(10, 13), slice(13, 19), 19)]
    ):
        pos, quat_wxyz, grip = out["action"][arm_idx]
        np.testing.assert_allclose(pos, raw[sl_pos], atol=1e-7)
        assert grip == pytest.approx([raw[g_idx], raw[g_idx]])
        # recovered quaternion must encode the same rotation as the rot6d
        back = ebench_quat_wxyz_to_rot6d(np.asarray(quat_wxyz, np.float32))
        np.testing.assert_allclose(back, _rot6d_normalize(raw[sl_rot]), atol=1e-5)


def test_action_round_trip_cumulative_and_gripper_clip():
    raw = np.zeros(23)
    raw[3:9] = [1, 0, 0, 0, 1, 0]
    raw[13:19] = [1, 0, 0, 0, 1, 0]
    raw[9], raw[19] = 0.09, -0.02  # out of physical range -> clipped
    raw[20:23] = [1.5, -0.5, 90.0]
    out = raw23_to_ebench_action(raw, "cumulative")
    assert out["base_is_rel"] is False
    np.testing.assert_allclose(out["base_motion"], [1.5, -0.5, 90.0])
    assert out["action"][0][2] == pytest.approx([EBENCH_GRIPPER_OPEN] * 2)
    assert out["action"][1][2] == pytest.approx([0.0, 0.0])


def test_action_rejects_wrong_width_and_mode():
    with pytest.raises(ValueError, match="23-D"):
        raw23_to_ebench_action(np.zeros(20), "delta")
    with pytest.raises(ValueError, match="base_action_source"):
        raw23_to_ebench_action(np.zeros(23), "velocity")


# ------------------------------------------------------------------ wire contract


def test_wire_types_survive_genmanip_consumption():
    """Emulate parse_embodiment_action's exact consumption of the dict."""
    raw = np.zeros(23)
    raw[3:9] = [1, 0, 0, 0, 1, 0]
    raw[13:19] = [1, 0, 0, 0, 1, 0]
    out = raw23_to_ebench_action(raw, "delta")
    for position, orientation, gripper_width in out["action"]:
        combined = position + orientation  # list concat — ndarray would broadcast
        assert isinstance(combined, list) and len(combined) == 7
        assert all(isinstance(v, float) for v in combined)
        assert isinstance(gripper_width, list) and len(gripper_width) == 2
        assert abs(np.linalg.norm(orientation) - 1.0) < 1e-5
    assert all(isinstance(v, float) for v in out["base_motion"])
    import pickle

    pickle.dumps(out)  # must be plain-pickleable for the EvalClient wire


# ------------------------------------------------------------------ driver bookkeeping


class _FakeSouth:
    def __init__(self):
        self.resets = 0
        self.payloads = []

    def reset(self):
        self.resets += 1
        return {"type": "reset_ack"}

    def predict(self, payload):
        self.payloads.append(payload)
        return {"action": [0.0] * 23, "step": len(self.payloads), "latency_ms": 1.0}


def _obs(t, reset=False, base=None, instruction="pick the apple"):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "reset": reset,
        "timestep": t,
        "instruction": instruction,
        "video.overlook_camera_view": rgb,
        "video.left_camera_view": rgb,
        "video.right_camera_view": rgb,
        "state.ee_pose": [[[0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]], [[-0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]]],
        "state.gripper": np.array([0.02, 0.02, 0.04, 0.04], dtype=np.float32),
        "state.base": np.asarray(base if base is not None else [0.0, 0.0, 0.0], dtype=np.float32),
    }


def test_driver_episode_flow_and_prev_base_differencing():
    south = _FakeSouth()
    driver = EBenchOpenWAMDriver(south, base_mode="delta")

    a0 = driver.act(_obs(0, reset=True, base=[0.0, 0.0, 0.0]))
    assert south.resets == 1
    # first step: no previous base -> zeros rendered into proprio slots [20:23)
    assert south.payloads[0]["state"][20:23] == pytest.approx([0.0, 0.0, 0.0])
    assert south.payloads[0]["prompt"] == bridge_prompt("pick the apple")
    assert a0["control_type"] == "ee_pose"

    driver.act(_obs(1, base=[0.01, -0.005, 0.02]))
    # measured diff vs previous obs, yaw rad->deg
    assert south.payloads[1]["state"][20:23] == pytest.approx([0.01, -0.005, math.degrees(0.02)], abs=1e-6)

    # new episode: reset flag -> south reset + prev_base cleared + prompt re-read
    driver.act(_obs(0, reset=True, base=[5.0, 5.0, 1.0], instruction="stack the cups"))
    assert south.resets == 2
    assert south.payloads[2]["state"][20:23] == pytest.approx([0.0, 0.0, 0.0])
    assert south.payloads[2]["prompt"] == bridge_prompt("stack the cups")

    # proprio ee/gripper rendering matches the shared conversion
    expected = ebench_obs_to_raw23(_obs(0)["state.ee_pose"], _obs(0)["state.gripper"], np.zeros(3, dtype=np.float32))
    assert south.payloads[0]["state"] == pytest.approx(list(expected), abs=1e-6)


def test_driver_missing_instruction_fails_fast():
    driver = EBenchOpenWAMDriver(_FakeSouth(), base_mode="delta")
    with pytest.raises(ValueError, match="instruction"):
        driver.act(_obs(0, reset=True, instruction=""))


def test_driver_rejects_wrong_action_width():
    class _BadSouth(_FakeSouth):
        def predict(self, payload):
            return {"action": [0.0] * 27}

    driver = EBenchOpenWAMDriver(_BadSouth(), base_mode="delta")
    with pytest.raises(ValueError, match="27-D"):
        driver.act(_obs(0, reset=True))


def test_driver_no_send_state():
    south = _FakeSouth()
    driver = EBenchOpenWAMDriver(south, base_mode="delta", send_state=False)
    driver.act(_obs(0, reset=True))
    assert "state" not in south.payloads[0]
