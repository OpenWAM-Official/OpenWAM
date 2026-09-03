from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.libero.openwam2libero_interface import (
    LIBERO_ACTION_MODE,
    OpenWAMLiberoPolicy,
    native_eef10_to_libero7d,
)


# Rotation fixtures inlined from the retired LIBERO EEF10 converter.
def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to rotation matrices with Rodrigues' formula."""
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape[-1:] != (3,):
        raise ValueError(f"axis-angle values must end in dimension 3, got {value.shape}")
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    small = angle[..., 0] < 1e-8
    axis = np.where(angle > 1e-8, value / np.maximum(angle, 1e-8), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    cosine = np.cos(angle[..., 0])
    sine = np.sin(angle[..., 0])
    one_minus_cosine = 1.0 - cosine
    matrix = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cosine + x * x * one_minus_cosine
    matrix[..., 0, 1] = x * y * one_minus_cosine - z * sine
    matrix[..., 0, 2] = x * z * one_minus_cosine + y * sine
    matrix[..., 1, 0] = y * x * one_minus_cosine + z * sine
    matrix[..., 1, 1] = cosine + y * y * one_minus_cosine
    matrix[..., 1, 2] = y * z * one_minus_cosine - x * sine
    matrix[..., 2, 0] = z * x * one_minus_cosine - y * sine
    matrix[..., 2, 1] = z * y * one_minus_cosine + x * sine
    matrix[..., 2, 2] = cosine + z * z * one_minus_cosine
    matrix[small] = np.eye(3)
    return matrix


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Store the first two columns of a rotation matrix, column by column."""
    value = np.asarray(matrix)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in shape (3, 3), got {value.shape}")
    return np.concatenate([value[..., :, 0], value[..., :, 1]], axis=-1).astype(np.float32)


class _Client:
    def __init__(self, representation: str):
        self.representation = representation

    def ping(self):
        return {"type": "pong", "representation": self.representation}

    def close(self):
        pass


def test_canonical_benchmark_uses_native_action_contract() -> None:
    root = Path("benchmarks")
    cfg = yaml.safe_load((root / "libero/policy_config.yml").read_text())
    assert cfg["action_mode"] == LIBERO_ACTION_MODE
    assert cfg["state_dim"] == 10
    assert (root / "libero/single_eval.py").is_file()
    assert (root / "libero/openwam2libero_interface.py").is_file()


def test_native_eef10_bridge_preserves_native_delta_and_flips_gripper_only() -> None:
    rotvec = np.array([0.8, -0.1, 0.05], dtype=np.float32)
    action10 = np.concatenate(
        [
            np.array([0.3, -0.4, 0.5], dtype=np.float32),
            matrix_to_rot6d(axis_angle_to_matrix(rotvec[None])[0]),
            np.array([0.8], dtype=np.float32),
        ]
    )

    action7 = native_eef10_to_libero7d(action10)

    np.testing.assert_array_equal(action7[:3], action10[:3])
    np.testing.assert_allclose(action7[3:6], rotvec, atol=2e-6)
    assert action7[6] == pytest.approx(-0.8)


def test_native_eef10_bridge_clips_runtime_command_directly() -> None:
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action10 = np.array([1.5, -1.5, 0.0, *identity, -1.2], dtype=np.float32)

    action7 = native_eef10_to_libero7d(action10)

    np.testing.assert_array_equal(action7[:3], [1.0, -1.0, 0.0])
    np.testing.assert_array_equal(action7[3:6], np.zeros(3))
    assert action7[6] == 1.0


def test_policy_rejects_wrong_representation() -> None:
    with pytest.raises(RuntimeError, match="representation mismatch"):
        OpenWAMLiberoPolicy(_client=_Client("wrong_contract"))
