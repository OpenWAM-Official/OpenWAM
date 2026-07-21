from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openwam.dataloader.utils.gr1_kinematics import (
    EEF33_DIM,
    EEF33_UNIFY_DST,
    JOINT44_DIM,
    JOINT44_SLICES,
    ROT6D_DIMS_EEF33,
    GR1Kinematics,
    IKSolution,
    matrix_to_rot6d,
    matrix_to_rotvec,
    rot6d_to_matrix,
)
from openwam.dataloader.utils.unify_action import map_to_unify, unmap_from_unify


def test_joint44_slices_match_nvidia_modality_contract():
    vector = np.arange(JOINT44_DIM)
    assert vector[JOINT44_SLICES["left_arm"]].tolist() == list(range(0, 7))
    assert vector[JOINT44_SLICES["left_hand"]].tolist() == list(range(7, 13))
    assert vector[JOINT44_SLICES["right_arm"]].tolist() == list(range(22, 29))
    assert vector[JOINT44_SLICES["right_hand"]].tolist() == list(range(29, 35))
    assert vector[JOINT44_SLICES["waist"]].tolist() == list(range(41, 44))


def test_eef33_unify_mapping_roundtrips_and_leaves_gripper_slots_masked():
    raw = np.arange(EEF33_DIM, dtype=np.float32)[None]
    unified, mask = map_to_unify(raw, EEF33_UNIFY_DST, unify_dim=80)
    np.testing.assert_array_equal(unmap_from_unify(unified, EEF33_UNIFY_DST), raw)
    assert unified.shape == (1, 80)
    assert mask.sum() == EEF33_DIM
    assert unified[0, 9] == 0
    assert unified[0, 43] == 0
    assert EEF33_UNIFY_DST.tolist() == [
        *range(0, 9),
        *range(10, 16),
        *range(34, 43),
        *range(44, 50),
        *range(68, 71),
    ]


def test_rot6d_matrix_roundtrip_is_orthonormal():
    angle = 0.73
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    rot6d = matrix_to_rot6d(matrix)
    rebuilt = rot6d_to_matrix(rot6d)
    np.testing.assert_allclose(rebuilt, matrix, atol=1e-7)
    np.testing.assert_allclose(rebuilt.T @ rebuilt, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rebuilt), 1.0, atol=1e-7)
    np.testing.assert_allclose(matrix_to_rotvec(matrix), [0, 0, angle], atol=1e-7)


def test_eef33_rot6d_indices_cover_both_arms_only():
    assert ROT6D_DIMS_EEF33 == (*range(3, 9), *range(18, 24))
    assert not set(ROT6D_DIMS_EEF33).intersection(range(9, 15))
    assert not set(ROT6D_DIMS_EEF33).intersection(range(24, 33))


def test_degenerate_rot6d_is_rejected_by_nonfinite_contract():
    with pytest.raises(ValueError, match="degenerate"):
        rot6d_to_matrix(np.zeros(6, dtype=np.float32))


def test_ik_failure_holds_current_arms_and_preserves_hand_waist(monkeypatch):
    kin = object.__new__(GR1Kinematics)
    kin.data = SimpleNamespace(qpos=np.arange(17, dtype=np.float64))
    kin.qpos_index = {
        "left_arm": np.arange(0, 7),
        "right_arm": np.arange(7, 14),
        "waist": np.arange(14, 17),
    }
    failed = IKSolution(
        left_arm=np.full(7, 99, dtype=np.float32),
        right_arm=np.full(7, 99, dtype=np.float32),
        position_error=1.0,
        rotation_error=1.0,
        converged=False,
    )
    monkeypatch.setattr(kin, "solve_eef33", lambda target, **kwargs: failed)
    target = np.zeros(EEF33_DIM, dtype=np.float32)
    target[9:15] = np.arange(6)
    target[24:30] = np.arange(6) + 10
    target[30:33] = [0.1, 0.2, 0.3]
    action, result = kin.eef33_to_action_dict(target)
    assert result is failed
    np.testing.assert_array_equal(action["action.left_arm"], np.arange(7))
    np.testing.assert_array_equal(action["action.right_arm"], np.arange(7, 14))
    np.testing.assert_array_equal(action["action.left_hand"], target[9:15])
    np.testing.assert_array_equal(action["action.right_hand"], target[24:30])
    np.testing.assert_array_equal(action["action.waist"], target[30:33])
