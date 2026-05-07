"""RoboTwin deploy proprio extraction must match training action spaces."""

import numpy as np

from benchmarks.robotwin import openwam2robotwin_interface as iface
from benchmarks.utils import action_conversion
from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d


class _Model:
    def __init__(self, action_type: str):
        self._action_type = action_type


def test_robotwin_endpose_conversion_matches_dataset_eef_layout():
    left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    eef = action_conversion.robotwin_endpose_to_eef20d(left, right, 0.25, 0.75)
    expected = np.concatenate(
        [
            left[:3],
            quat_xyzw_to_rotation_6d(left[3:]),
            np.array([0.25], dtype=np.float32),
            right[:3],
            quat_xyzw_to_rotation_6d(right[3:]),
            np.array([0.75], dtype=np.float32),
        ]
    ).astype(np.float32)

    assert eef.shape == (20,)
    np.testing.assert_allclose(eef, expected, atol=1e-6)


def test_ee_action_type_sends_20d_eef_proprio():
    observation = {
        "endpose": {
            "left_endpose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "right_endpose": np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "left_gripper": np.array(0.25, dtype=np.float32),
            "right_gripper": np.array(0.75, dtype=np.float32),
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }

    proprio = iface._extract_proprio(_Model("ee"), observation)

    assert proprio.shape == (20,)
    np.testing.assert_allclose(proprio[[0, 1, 2, 9, 10, 11, 12, 19]], [0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.75])


def test_qpos_action_type_sends_joint_vector():
    joint = np.arange(14, dtype=np.float32)
    observation = {"joint_action": {"vector": joint}}

    proprio = iface._extract_proprio(_Model("qpos"), observation)

    assert proprio.shape == (14,)
    np.testing.assert_allclose(proprio, joint)


def test_ee_action_type_rejects_joint_only_observation():
    observation = {"joint_action": {"vector": np.zeros(14, dtype=np.float32)}}

    try:
        iface._extract_proprio(_Model("ee"), observation)
    except KeyError as exc:
        assert "action_mode='eef'" in str(exc)
    else:
        raise AssertionError("expected ee action_type to require EEF proprio")
