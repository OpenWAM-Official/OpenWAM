from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.robocasa365 import openwam2robocasa365_interface as adapter
from benchmarks.utils import transport


def _rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _obs(*, eef_position=(0.1, -0.2, 0.4), eef_quat=(0, 0, 0, 1)) -> dict:
    return {
        "state.base_position": np.array([1.0, 2.0, 0.0], np.float32),
        "state.base_rotation": np.array([0.0, 0.0, 0.0, 1.0], np.float32),
        "state.end_effector_position_relative": np.array(eef_position, np.float32),
        "state.end_effector_rotation_relative": np.array(eef_quat, np.float32),
        "state.gripper_qpos": np.array([0.04, -0.04], np.float32),
        "video.robot0_agentview_left": np.zeros((16, 16, 3), np.uint8),
        "video.robot0_eye_in_hand": np.zeros((16, 16, 3), np.uint8),
        "video.robot0_agentview_right": np.zeros((16, 16, 3), np.uint8),
    }


class _Client:
    def __init__(self, action, *, representation="robocasa365"):
        self.action = action
        self.representation = representation

    def ping(self):
        return {"type": transport.PONG, "representation": self.representation}

    def reset(self):
        return {"type": transport.RESET_ACK}

    def predict(self, payload):
        return {"action": self.action, "step": 0}

    def close(self):
        pass


def _action15(delta_xyz=(0.5, -0.25, 0.1), delta_rot=(0.1, -0.2, 0.8)) -> np.ndarray:
    return np.concatenate(
        [
            delta_xyz,
            _rot6d(Rotation.from_rotvec(delta_rot).as_matrix()),
            [1.0],
            [0.2, -0.3, 0.4, 0.7, 1.0],
        ]
    ).astype(np.float32)


def test_delta_action15_restores_native_action_without_current_state_or_half_scale():
    action = _action15()
    policy = adapter.OpenWAMRoboCasa365Policy(_client=_Client(action))

    first = policy.act(_obs(eef_position=(0.1, 0.2, 0.3)), "prompt")
    second = policy.act(_obs(eef_position=(8.0, 9.0, 10.0)), "prompt")

    for result in (first, second):
        np.testing.assert_allclose(result["action.end_effector_position"], [0.5, -0.25, 0.1], atol=2e-6)
        np.testing.assert_allclose(result["action.end_effector_rotation"], [0.1, -0.2, 0.8], atol=2e-6)
        np.testing.assert_allclose(result["action.gripper_close"], [-1.0])
        np.testing.assert_allclose(result["action.base_motion"], [0.2, -0.3, 0.4, 0.7])
        np.testing.assert_allclose(result["action.control_mode"], [1.0])


def test_state19_keeps_xyzw_base_and_eef_quaternion_conversion():
    state = np.asarray(adapter.assemble_state19_proprio(_obs()))
    assert state.shape == (19,)
    np.testing.assert_allclose(state[3:9], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(state[13:19], [1, 0, 0, 0, 1, 0])


def test_representation_handshake_rejects_wrong_contract():
    with pytest.raises(RuntimeError, match="representation mismatch"):
        adapter.OpenWAMRoboCasa365Policy(_client=_Client(_action15(), representation="wrong_contract"))
