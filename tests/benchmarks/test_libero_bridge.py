"""Pin the LIBERO train/eval EEF10 contract together byte-for-byte.

The dataloader (``openwam.dataloader.libero``) and the pure-numpy client bridge
(``benchmarks.utils.action_conversion``) implement the two ends of one
representation; these tests cross-check them on the same physical state and
verify the EEF10 full-pose -> 7-D OSC delta inverse.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import eef10_to_libero7d, libero_gripper_qpos_to_cmd, libero_obs_to_eef10
from openwam.dataloader.libero import (
    LIBERO_GRIPPER_WIDTH_OPEN,
    axis_angle_to_matrix,
    gripper_qpos_to_cmd,
    matrix_to_rot6d,
    state8_to_eef10,
)

_IDENT_R6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32)


def _axis_angle_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], np.float64)
    axis = aa / angle
    return np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])


def test_client_proprio_matches_dataloader_repr():
    """libero_obs_to_eef10 (live quat) == state8_to_eef10 (stored axis-angle)."""
    rng = np.random.RandomState(7)
    for _ in range(8):
        aa = rng.uniform(-1.5, 1.5, size=3)
        pos = rng.uniform(-0.5, 0.5, size=3)
        qpos = np.array([rng.uniform(0, 0.04), rng.uniform(-0.04, 0)])
        state8 = np.concatenate([pos, aa, qpos]).astype(np.float32)[None]
        from_state = state8_to_eef10(state8)[0]
        from_obs = libero_obs_to_eef10(pos, _axis_angle_to_quat_xyzw(aa), qpos)
        np.testing.assert_allclose(from_obs, from_state, atol=1e-5)


def test_gripper_render_is_shared_between_train_and_eval():
    for width in (0.0, 0.02, LIBERO_GRIPPER_WIDTH_OPEN, 0.2, -0.1):
        assert libero_gripper_qpos_to_cmd(width) == pytest.approx(
            float(gripper_qpos_to_cmd(np.array(width))), abs=1e-7
        )
    assert libero_gripper_qpos_to_cmd(LIBERO_GRIPPER_WIDTH_OPEN) == -1.0  # open
    assert libero_gripper_qpos_to_cmd(0.0) == 1.0  # closed


def test_bridge_zero_delta_is_zero_command():
    eef10 = np.concatenate([[0.1, 0.2, 0.3], _IDENT_R6D, [-1.0]]).astype(np.float32)
    out = eef10_to_libero7d(eef10, ref_pos=[0.1, 0.2, 0.3], ref_rot6d=_IDENT_R6D)
    assert out.shape == (7,)
    assert out[:6] == pytest.approx(np.zeros(6), abs=1e-6)
    assert out[6] == pytest.approx(-1.0)


def test_bridge_pos_delta_scaling_and_clip():
    eef10 = np.concatenate([[0.025, 0.0, 0.0], _IDENT_R6D, [0.3]]).astype(np.float32)
    out = eef10_to_libero7d(eef10, ref_pos=[0.0, 0.0, 0.0], ref_rot6d=_IDENT_R6D, pos_scale=0.05, rot_scale=0.5)
    assert out[0:3] == pytest.approx([0.5, 0.0, 0.0], abs=1e-5)
    # Gripper is a CONTINUOUS command in [-1, +1]: passed through, not thresholded.
    assert out[6] == pytest.approx(0.3, abs=1e-6)

    huge = np.concatenate([[1.0, 0.0, 0.0], _IDENT_R6D, [2.0]]).astype(np.float32)
    out = eef10_to_libero7d(huge, ref_pos=[0.0, 0.0, 0.0], ref_rot6d=_IDENT_R6D, pos_scale=0.05, rot_scale=0.5)
    assert out[0] == pytest.approx(1.0)
    assert out[6] == pytest.approx(1.0)  # command clipped to +1


def test_bridge_rotation_axis_angle():
    """90 deg about z, identity reference, rot_scale=pi/2 -> rot_cmd ~ [0, 0, 1]."""
    rot90z = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], np.float32)
    eef10 = np.concatenate([[0.0, 0.0, 0.0], rot90z, [0.0]]).astype(np.float32)
    out = eef10_to_libero7d(
        eef10, ref_pos=[0.0, 0.0, 0.0], ref_rot6d=_IDENT_R6D, pos_scale=0.05, rot_scale=np.pi / 2
    )
    assert out[3:6] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)


def test_bridge_roundtrip_recovers_full_pose():
    """Forward-integrating the OSC delta from the reference reproduces the EEF10 target."""
    rng = np.random.RandomState(21)
    pos_scale, rot_scale = 0.05, 0.5
    for _ in range(8):
        ref_aa = rng.uniform(-1.0, 1.0, size=3)
        R_ref = axis_angle_to_matrix(ref_aa[None])[0]
        ref_pos = rng.uniform(-0.3, 0.3, size=3)
        # Small reachable target (inside the per-step clip bounds).
        d_aa = rng.uniform(-0.3, 0.3, size=3)
        R_tgt = axis_angle_to_matrix(d_aa[None])[0] @ R_ref
        tgt_pos = ref_pos + rng.uniform(-0.04, 0.04, size=3)
        eef10 = np.concatenate([tgt_pos, matrix_to_rot6d(R_tgt), [0.5]]).astype(np.float32)

        out = eef10_to_libero7d(
            eef10,
            ref_pos,
            matrix_to_rot6d(R_ref),
            pos_scale=pos_scale,
            rot_scale=rot_scale,
        )
        # Integrate: pos = ref + cmd * scale; R = R_delta @ R_ref (world-frame OSC delta).
        rec_pos = ref_pos + out[0:3].astype(np.float64) * pos_scale
        R_rec = axis_angle_to_matrix((out[3:6].astype(np.float64) * rot_scale)[None])[0] @ R_ref
        np.testing.assert_allclose(rec_pos, tgt_pos, atol=1e-5)
        np.testing.assert_allclose(R_rec, R_tgt, atol=1e-5)


def test_bridge_rejects_bad_inputs():
    with pytest.raises(ValueError, match="10-D"):
        eef10_to_libero7d(np.zeros(7), ref_pos=np.zeros(3), ref_rot6d=_IDENT_R6D)
    with pytest.raises(ValueError, match="> 0"):
        eef10_to_libero7d(
            np.concatenate([np.zeros(3), _IDENT_R6D, [0.0]]),
            ref_pos=np.zeros(3),
            ref_rot6d=_IDENT_R6D,
            pos_scale=0.0,
        )
    with pytest.raises(ValueError, match="eef_pos must be 3"):
        libero_obs_to_eef10(np.zeros(2), np.array([0, 0, 0, 1.0]), np.zeros(2))
