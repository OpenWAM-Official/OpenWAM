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
from scripts.convert_lerobot_libero_to_eef10_v3 import (
    axis_angle_to_matrix,
    matrix_to_rot6d,
)


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
    assert action7[6] == -0.8


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
