from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image
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
    }


class _Client:
    def __init__(self, action, *, representation="robocasa365"):
        self.action = action
        self.payloads = []
        self.representation = representation

    def ping(self):
        return {
            "type": transport.PONG,
            "representation": self.representation,
        }

    def reset(self):
        return {"type": transport.RESET_ACK}

    def predict(self, payload):
        self.payloads.append(payload)
        return {"action": self.action, "step": 0}

    def close(self):
        pass


def _action15(obs: dict, *, delta_xyz=(0.5, -0.25, 0.1), delta_rot=(0.0, 0.0, 0.2)) -> np.ndarray:
    current_position = np.asarray(obs["state.end_effector_position_relative"], np.float64)
    current_rotation = Rotation.from_quat(obs["state.end_effector_rotation_relative"]).as_matrix()
    target_position = current_position + 0.05 * np.asarray(delta_xyz)
    target_rotation = Rotation.from_rotvec(0.5 * np.asarray(delta_rot)).as_matrix() @ current_rotation
    return np.concatenate(
        [target_position, _rot6d(target_rotation), [1.0], [0.2, -0.3, 0.4, 0.7, 1.0]]
    ).astype(np.float32)


def test_state19_matches_training_layout():
    state = np.asarray(adapter.assemble_state19_proprio(_obs()))
    assert state.shape == (19,)
    np.testing.assert_allclose(state[:3], [0.1, -0.2, 0.4])
    np.testing.assert_allclose(state[3:9], [1, 0, 0, 0, 1, 0])
    assert state[9] == pytest.approx(0.6)  # width .08 -> 2*.08/.1-1
    np.testing.assert_allclose(state[10:13], [1, 2, 0])
    np.testing.assert_allclose(state[13:19], [1, 0, 0, 0, 1, 0])


def test_action15_roundtrip_uses_current_state_and_preserves_base_torso_mode():
    obs = _obs()
    action = _action15(obs)
    client = _Client(action)
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=client, osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    result = policy.act(obs, "prompt")
    np.testing.assert_allclose(result["action.end_effector_position"], [0.5, -0.25, 0.1], atol=2e-6)
    np.testing.assert_allclose(result["action.end_effector_rotation"], [0, 0, 0.2], atol=2e-6)
    np.testing.assert_allclose(result["action.gripper_close"], [-1.0])
    np.testing.assert_allclose(result["action.base_motion"], [0.2, -0.3, 0.4, 0.7])
    np.testing.assert_allclose(result["action.control_mode"], [1.0])
    assert len(client.payloads[0]["state"]) == 19
    sent_images = client.payloads[0]["images"]
    head = Image.open(io.BytesIO(base64.b64decode(sent_images["head_camera"])))
    wrist = Image.open(io.BytesIO(base64.b64decode(sent_images["left_wrist_camera"])))
    assert head.format == wrist.format == "PNG"
    assert head.size == (320, 256)
    assert wrist.size == (160, 128)
    assert sent_images["right_wrist_camera"] is None


def test_base_mode_still_anchors_to_current_observation_not_previous_target():
    first_obs = _obs(eef_position=(0.1, 0.0, 0.4))
    first_action = _action15(first_obs, delta_xyz=(0.5, 0, 0), delta_rot=(0, 0, 0))
    client = _Client(first_action)
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=client, osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    first = policy.act(first_obs, "prompt")
    np.testing.assert_allclose(first["action.end_effector_position"], [0.5, 0, 0], atol=2e-6)

    second_obs = _obs(eef_position=tuple(first_action[:3]))
    client.action = first_action  # target now equals the current achieved position
    second = policy.act(second_obs, "prompt")
    np.testing.assert_allclose(second["action.end_effector_position"], [0, 0, 0], atol=2e-6)


def test_missing_scales_raise():
    obs = _obs()
    policy = adapter.OpenWAMRoboCasa365Policy(_client=_Client(_action15(obs)))
    with pytest.raises(ValueError, match="osc_pos_scale"):
        policy.act(obs, "prompt")


def test_representation_handshake_rejects_legacy_checkpoint():
    obs = _obs()
    with pytest.raises(RuntimeError, match="representation mismatch"):
        adapter.OpenWAMRoboCasa365Policy(
            _client=_Client(_action15(obs), representation="eef_base"),
            osc_pos_scale=0.05,
            osc_rot_scale=0.5,
        )


def test_raw_action12_passthrough():
    raw = np.arange(12, dtype=np.float32)
    raw[6] = 0.49
    raw[11] = 0.5
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=_Client(raw), osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    result = policy.act(_obs(), "prompt")
    flattened = np.concatenate(list(result.values()))
    expected = raw.copy()
    expected[6] = -1.0
    expected[11] = 1.0
    np.testing.assert_allclose(flattened, expected)


@pytest.mark.parametrize(
    ("gripper_open_scale", "mode", "expected_gripper_close", "expected_mode"),
    [
        (-0.51, 0.49, 1.0, -1.0),
        (-0.50, 0.50, 1.0, 1.0),
        (-0.49, 0.51, -1.0, 1.0),
        (0.00, 0.00, -1.0, -1.0),
    ],
)
def test_eval_binarizes_gripper_and_mode_at_official_boundaries(
    gripper_open_scale, mode, expected_gripper_close, expected_mode
):
    obs = _obs()
    action = _action15(obs)
    action[9] = gripper_open_scale
    action[14] = mode
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=_Client(action), osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    result = policy.act(obs, "prompt")
    np.testing.assert_allclose(result["action.gripper_close"], [expected_gripper_close])
    np.testing.assert_allclose(result["action.control_mode"], [expected_mode])
