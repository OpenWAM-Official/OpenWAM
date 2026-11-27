"""Native RoboTwin wxyz must describe the same physical rotation in every EEF path."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.utils import action_conversion
from openwam.dataloader.robotwin import RoboTwinDataset
from openwam.dataloader.utils.stats_computation import robotwin_stats_computation as stats


@pytest.mark.parametrize(
    "quat",
    [
        [1, 0, 0, 0],  # Identity, not Rx(180).
        [2**-0.5, 2**-0.5, 0, 0],
        [2**-0.5, 0, 2**-0.5, 0],
        [2**-0.5, 0, 0, 2**-0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0.5, -0.5, 0.5, 0.5],
    ],
    ids=["identity", "x90", "y90", "z90", "x180", "y180", "z180", "mixed"],
)
@pytest.mark.parametrize("sign", [1, -1])
def test_native_wxyz_matches_physical_rotation_across_eef_paths(tmp_path, quat, sign):
    left_q = sign * np.asarray(quat, dtype=np.float64)
    right_q = left_q[[0, 3, 1, 2]]
    left = np.r_[[0.1, 0.2, 0.3], left_q]
    right = np.r_[[0.4, 0.5, 0.6], right_q]
    rotations = Rotation.from_quat(np.stack([left_q, right_q])[:, [1, 2, 3, 0]]).as_matrix()
    rot6d = np.concatenate([rotations[:, :, 0], rotations[:, :, 1]], axis=-1)
    expected = np.r_[left[:3], rot6d[0], 0.25, right[:3], rot6d[1], 0.75]

    # Live observation encoding: compare physical matrices, not an inverse pair.
    proprio = action_conversion.robotwin_endpose_to_eef20d(left, right, 0.25, 0.75)
    np.testing.assert_allclose(proprio, expected, atol=1e-6)

    # Both the training loader and normalization-stat reader consume native HDF5.
    with h5py.File(tmp_path / "episode.hdf5", "w") as f:
        f.create_dataset("endpose/left_endpose", data=[left, left])
        f.create_dataset("endpose/right_endpose", data=[right, right])
        f.create_dataset("endpose/left_gripper", data=[0.25, 0.25])
        f.create_dataset("endpose/right_gripper", data=[0.75, 0.75])
        dataset = RoboTwinDataset.__new__(RoboTwinDataset)
        np.testing.assert_allclose(dataset._read_eef_actions(f, 1, 2), expected[None], atol=1e-6)
        np.testing.assert_allclose(stats._read_eef_actions(f), np.stack([expected, expected]), atol=1e-6)

    # Decode independently constructed physical rot6d, not the encoder's output.
    action = action_conversion.eef20d_to_ee16d(expected.astype(np.float32))
    np.testing.assert_allclose(
        action[[0, 1, 2, 7, 8, 9, 10, 15]],
        expected[[0, 1, 2, 9, 10, 11, 12, 19]],
    )
    native_quats = np.stack([action[3:7], action[11:15]])
    actual_rotations = Rotation.from_quat(native_quats[:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(actual_rotations, rotations, atol=1e-6)
    assert proprio.dtype == action.dtype == np.float32


def test_stats_recompute_instead_of_reusing_legacy_quaternion_shards(tmp_path):
    root = tmp_path / "task_a" / "test-robot_clean_50" / "data"
    root.mkdir(parents=True)
    native_identity = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float64)
    with h5py.File(root / "episode0.hdf5", "w") as f:
        for arm in ("left", "right"):
            f.create_dataset(f"endpose/{arm}_endpose", data=native_identity)
            f.create_dataset(f"endpose/{arm}_gripper", data=[0.5])

    checkpoint = str(tmp_path / "stats.npy")
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    legacy_shard = Path(f"{checkpoint}.partial/shards_v1/{digest}.npz")
    legacy_shard.parent.mkdir(parents=True)
    # An old shard must not be read, even if its task-root cache key matches.
    legacy_eef = np.array([[0, 0, 0, 1, 0, 0, 0, -1, 0, 0.5] * 2], dtype=np.float64)
    np.savez(legacy_shard, joint=np.empty((0, 14)), eef=legacy_eef)
    legacy_bytes = legacy_shard.read_bytes()
    stats.compute_multitask_robotwin_stats(
        dataset_dir=str(tmp_path),
        embodiment="test-robot",
        variant="clean_50",
        tasks=["task_a"],
        checkpoint_path=checkpoint,
    )
    with np.load(stats._shard_path_for_root(checkpoint, str(root)), allow_pickle=False) as payload:
        np.testing.assert_allclose(payload["eef"][0, 3:9], [1, 0, 0, 0, 1, 0])
    assert legacy_shard.read_bytes() == legacy_bytes
