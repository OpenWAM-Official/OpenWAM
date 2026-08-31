from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.registry import DATASET_REGISTRY
from scripts.convert_lerobot_libero_to_eef10_v3 import (
    DEFAULT_OUTPUT_ROOT,
    NATIVE_ACTION_NAMES,
    axis_angle_to_matrix,
    convert_state_action,
    rot6d_to_matrix,
)


def test_canonical_config_and_registry_use_native_action_reader() -> None:
    config = yaml.safe_load(Path("configs/dataloader/libero.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "libero"
    assert config["action_mode"] == "libero"
    assert DATASET_REGISTRY["libero"] is LiberoDataset


def test_native_action_converter_uses_canonical_default_path() -> None:
    assert DEFAULT_OUTPUT_ROOT.name == "libero_native_action_v3"


def test_native_action_keeps_delta_xyz_and_rotvec_without_controller_scaling() -> None:
    state = np.array(
        [
            [0.10, -0.20, 0.70, 3.10, 0.10, -0.20, 0.04, -0.04],
            [0.20, 0.30, 0.80, 2.90, -0.30, 0.20, 0.02, -0.02],
        ],
        dtype=np.float32,
    )
    action = np.array(
        [
            [0.50, -0.25, 0.10, 0.80, -0.10, 0.05, -1.0],
            [-0.20, 0.40, -0.30, -0.10, 0.15, -0.80, 0.35],
        ],
        dtype=np.float32,
    )

    state10, action10, errors = convert_state_action(state, action)

    assert state10.shape == action10.shape == (2, 10)
    np.testing.assert_array_equal(action10[:, :3], action[:, :3])
    np.testing.assert_allclose(
        rot6d_to_matrix(action10[:, 3:9]),
        axis_angle_to_matrix(action[:, 3:6]),
        atol=2e-6,
    )
    np.testing.assert_array_equal(action10[:, 9], -action[:, 6])
    # The first rotation pins the native unit scale: Exp(r), not Exp(0.5*r).
    assert not np.allclose(
        rot6d_to_matrix(action10[:1, 3:9]),
        axis_angle_to_matrix(action[:1, 3:6] * 0.5),
        atol=1e-3,
    )
    assert errors["position"] == 0.0
    assert errors["rotation"] < 3e-6
    assert errors["gripper"] == 0.0
    assert len(NATIVE_ACTION_NAMES) == 10
    assert NATIVE_ACTION_NAMES[0] == "eef_native_delta_x"
    assert NATIVE_ACTION_NAMES[-1] == "gripper_open_command"


def test_state_is_achieved_pose_and_gripper_is_open_scale() -> None:
    state = np.array(
        [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02, -0.02],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    action = np.zeros((3, 7), dtype=np.float32)

    state10, action10, _ = convert_state_action(state, action)

    np.testing.assert_array_equal(state10[:, :3], state[:, :3])
    np.testing.assert_allclose(state10[:, 3:9], np.array([[1, 0, 0, 0, 1, 0]] * 3), atol=1e-6)
    np.testing.assert_allclose(state10[:, 9], [1.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(action10[:, 3:9], np.array([[1, 0, 0, 0, 1, 0]] * 3), atol=1e-6)
    np.testing.assert_array_equal(action10[:, 9], np.zeros(3))


def test_reader_hard_preserves_rot6d_under_custom_stats() -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    # Deliberately non-identity rotation stats: the reader must still leave
    # rot6d untouched rather than relying only on the generated stats artifact.
    reader._normalization_stats = {
        "min": np.zeros(10, dtype=np.float32),
        "max": np.ones(10, dtype=np.float32),
    }
    raw = np.array(
        [[0.25, 0.5, 0.75, 0.998, 0.02, -0.04, 0.01, 0.999, 0.02, -0.5]],
        dtype=np.float32,
    )
    normalized = reader._normalize_array(raw)
    np.testing.assert_array_equal(normalized[:, 3:9], raw[:, 3:9])
    expected_non_rot = np.clip(raw[:, [0, 1, 2, 9]] * 2.0 - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(normalized[:, [0, 1, 2, 9]], expected_non_rot)


def test_reader_rejects_incomplete_compatibility_stats(tmp_path) -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    reader._source_stats_path = str(tmp_path / "normalization_stats.npy")
    reader._dataset_dir = tmp_path
    reader._raw_action_dim = 10
    np.save(reader._source_stats_path, {"eef": {}}, allow_pickle=True)
    with pytest.raises(KeyError, match="libero"):
        reader._load_stats({})
