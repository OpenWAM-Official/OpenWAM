"""Tests for RoboTwin dataloader: joint mode, EEF mode, multi-variant, normalization."""

import io
import os
import tempfile

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_jpeg(height=16, width=16, seed=0):
    """Create a tiny JPEG-encoded byte string (like RoboTwin HDF5 stores)."""
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _create_mock_episode(path, T=20, action_dim=14, seed=0):
    """Create a mock RoboTwin HDF5 episode with both joint and endpose keys."""
    rng = np.random.default_rng(seed)
    quats = Rotation.random(T, random_state=seed).as_quat()  # (T, 4) xyzw

    # Pre-encode JPEG frames
    jpeg_frames = [_encode_jpeg(16, 16, seed=seed + i) for i in range(T)]
    with h5py.File(path, "w") as f:
        # Joint actions — values in [0, 1] range
        joint_actions = rng.random((T, action_dim)).astype(np.float32)
        # Set gripper dims to clearly distinguishable open/closed values
        joint_actions[: T // 2, 6] = 0.8  # open  (> 0.5)
        joint_actions[T // 2 :, 6] = 0.2  # closed (< 0.5)
        joint_actions[: T // 2, 13] = 0.9  # open
        joint_actions[T // 2 :, 13] = 0.1  # closed
        f.create_dataset("joint_action/vector", data=joint_actions)

        # EEF endpose
        f.create_dataset(
            "endpose/left_endpose",
            data=np.c_[rng.random((T, 3)).astype(np.float64), quats],
        )
        f.create_dataset(
            "endpose/right_endpose",
            data=np.c_[rng.random((T, 3)).astype(np.float64), quats],
        )
        # Gripper: first half open (1.0), second half closed (0.0)
        left_grip = np.ones(T, dtype=np.float64)
        left_grip[T // 2 :] = 0.0
        right_grip = np.ones(T, dtype=np.float64)
        right_grip[T // 2 :] = 0.0
        f.create_dataset("endpose/left_gripper", data=left_grip)
        f.create_dataset("endpose/right_gripper", data=right_grip)

        # JPEG-encoded camera frames (variable-length byte arrays)
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        cam_ds = f.create_dataset("observation/head_camera/rgb", shape=(T,), dtype=dt)
        for i, jpeg in enumerate(jpeg_frames):
            cam_ds[i] = jpeg


def _create_action_stats(path, action_dim=14):
    """Create a mock action_stats.npy with min/max keys."""
    stats = {
        "mean": np.zeros(action_dim, dtype=np.float64),
        "std": np.ones(action_dim, dtype=np.float64),
        "min": np.full(action_dim, -1.0, dtype=np.float64),
        "max": np.full(action_dim, 1.0, dtype=np.float64),
        "q01": np.full(action_dim, -0.99, dtype=np.float64),
        "q99": np.full(action_dim, 0.99, dtype=np.float64),
    }
    np.save(path, stats)


# ---------------------------------------------------------------------------
# Joint mode tests
# ---------------------------------------------------------------------------


def test_joint_mode_basic():
    """Joint mode loads correctly and returns action_dim=14."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            val_ratio=0.0,
        )
        assert ds.action_dim == 14
        assert ds.action_mode == "joint"

        sample = ds[0]
        assert sample["action_trajectory"].shape == (5, 14)
        assert sample["action_mask"].shape == (5,)


def test_joint_mode_minmax_normalization():
    """Joint mode applies min-max normalization to joints and binary to grippers."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=10, seed=42)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path, action_dim=14)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            action_stats_path=stats_path,
            val_ratio=0.0,
        )

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()

        # Joint dims should be in [-1, 1] (min-max normalized)
        joint_mask = np.ones(14, dtype=bool)
        joint_mask[[6, 13]] = False
        assert actions[:, joint_mask].min() >= -1.0 - 1e-6
        assert actions[:, joint_mask].max() <= 1.0 + 1e-6

        # Gripper dims should be binary {0, 1}
        gripper_vals = set(actions[:, 6].tolist() + actions[:, 13].tolist())
        assert gripper_vals.issubset({0.0, 1.0}), f"Gripper values not binary: {gripper_vals}"


def test_joint_mode_gripper_convention():
    """Joint mode gripper: 1=closed, 0=open (inverted from raw)."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create episode where gripper raw > 0.5 (open) for all frames
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=10, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path, action_dim=14)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=10,
            height=32,
            width=32,
            action_mode="joint",
            action_stats_path=stats_path,
            val_ratio=0.0,
        )

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()

        # First half: raw gripper > 0.5 (open) → normalized = 1 - 1 = 0 (open)
        assert all(actions[:5, 6] == 0.0), "First half should be 0 (open)"
        # Second half: raw gripper < 0.5 (closed) → normalized = 1 - 0 = 1 (closed)
        assert all(actions[5:, 6] == 1.0), "Second half should be 1 (closed)"


def test_joint_mode_denormalize_roundtrip():
    """Joint mode denormalize_action inverts normalization for joint dims."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=10, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")

        # Custom stats: min=0, max=2 for all dims
        stats = {
            "mean": np.ones(14, dtype=np.float64),
            "std": np.ones(14, dtype=np.float64),
            "min": np.zeros(14, dtype=np.float64),
            "max": np.full(14, 2.0, dtype=np.float64),
            "q01": np.zeros(14, dtype=np.float64),
            "q99": np.full(14, 2.0, dtype=np.float64),
        }
        np.save(stats_path, stats)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            action_stats_path=stats_path,
            val_ratio=0.0,
        )

        # Test a known normalized value: normalized = -1 → original = min = 0
        test_normalized = np.full((1, 14), -1.0)
        denormed = ds.denormalize_action(test_normalized)
        # Joint dims (non-gripper) should map -1 → 0
        for d in range(14):
            if d in (6, 13):
                continue  # gripper handled separately
            np.testing.assert_allclose(denormed[0, d], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# EEF mode tests
# ---------------------------------------------------------------------------


def test_eef_mode_basic():
    """EEF mode loads correctly and returns action_dim=20."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            val_ratio=0.0,
        )
        assert ds.action_dim == 20
        assert ds.action_mode == "eef"
        assert ds.action_stats is None

        sample = ds[0]
        assert sample["action_trajectory"].shape == (5, 20)


def test_eef_mode_no_normalization():
    """EEF mode returns raw values, even if stats file exists."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=10, seed=0)
        # Create stats file that would be loaded in joint mode
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path, action_dim=14)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            action_stats_path=stats_path,
            val_ratio=0.0,
        )
        assert ds.action_stats is None  # EEF ignores stats


def test_eef_gripper_inversion():
    """EEF mode inverts gripper: raw 1=open → output 0, raw 0=closed → output 1."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=10,
            height=32,
            width=32,
            action_mode="eef",
            val_ratio=0.0,
        )

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()

        # First half: raw gripper = 1.0 (open) → inverted = 0.0 (open in our convention)
        np.testing.assert_allclose(actions[:5, 9], 0.0, atol=1e-6)
        np.testing.assert_allclose(actions[:5, 19], 0.0, atol=1e-6)

        # Second half: raw gripper = 0.0 (closed) → inverted = 1.0 (closed)
        np.testing.assert_allclose(actions[5:, 9], 1.0, atol=1e-6)
        np.testing.assert_allclose(actions[5:, 19], 1.0, atol=1e-6)


def test_eef_denormalize_inverts_gripper():
    """EEF mode denormalize_action inverts gripper dims back to raw convention."""
    from openwam.dataloader.robotwin_dataset import EEF_GRIPPER_INDICES, RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            val_ratio=0.0,
        )

        # Non-gripper dims should pass through unchanged
        test_data = np.random.randn(5, 20).astype(np.float32)
        result = ds.denormalize_action(test_data)
        non_grip_mask = np.ones(20, dtype=bool)
        non_grip_mask[EEF_GRIPPER_INDICES] = False
        np.testing.assert_array_equal(result[:, non_grip_mask], test_data[:, non_grip_mask])

        # Gripper dims: binary inversion (1=closed → 0=open raw, 0=open → 1=open raw)
        for gi in EEF_GRIPPER_INDICES:
            expected = 1.0 - (test_data[:, gi] > 0.5).astype(np.float32)
            np.testing.assert_array_equal(result[:, gi], expected)


# ---------------------------------------------------------------------------
# Multi-variant tests
# ---------------------------------------------------------------------------


def test_multi_variant_discovery():
    """MultiTaskRoboTwinDataset with variant='both' discovers clean and randomized."""
    from openwam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure for 2 tasks × 2 variants
        for task in ["task_a", "task_b"]:
            for variant in ["clean_50", "randomized_500"]:
                data_dir = os.path.join(tmpdir, task, f"test-robot_{variant}", "data")
                os.makedirs(data_dir)
                _create_mock_episode(os.path.join(data_dir, "episode0.hdf5"), T=10, seed=hash(task + variant) % 1000)

        ds = MultiTaskRoboTwinDataset(
            dataset_dir=tmpdir,
            robot="test-robot",
            variant="both",
            tasks=["task_a", "task_b"],
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            val_ratio=0.0,
        )

        # Should have 4 sub-datasets (2 tasks × 2 variants)
        assert len(ds._sub_datasets) == 4
        assert len(ds) > 0


def test_multi_variant_single_variant_compat():
    """MultiTaskRoboTwinDataset with variant='clean_50' loads a single variant."""
    from openwam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["task_a"]:
            data_dir = os.path.join(tmpdir, task, "test-robot_clean_50", "data")
            os.makedirs(data_dir)
            _create_mock_episode(os.path.join(data_dir, "episode0.hdf5"), T=10, seed=0)

        ds = MultiTaskRoboTwinDataset(
            dataset_dir=tmpdir,
            robot="test-robot",
            variant="clean_50",
            tasks=["task_a"],
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            val_ratio=0.0,
        )

        assert len(ds._sub_datasets) == 1


# ---------------------------------------------------------------------------
# Rotation conversion tests
# ---------------------------------------------------------------------------


def test_rotation_conversion_roundtrip():
    """quat_xyzw → rot6d → quat_xyzw should approximately roundtrip."""
    from openwam.dataloader.transforms.rotation import (
        quat_xyzw_to_rotation_6d,
        rotation_6d_to_quat_xyzw,
    )

    quats = Rotation.random(50, random_state=42).as_quat()  # (50, 4) xyzw

    rot6d = quat_xyzw_to_rotation_6d(quats)
    assert rot6d.shape == (50, 6)

    recovered = rotation_6d_to_quat_xyzw(rot6d)
    assert recovered.shape == (50, 4)

    # Quaternions can differ by sign (q and -q represent the same rotation)
    for i in range(len(quats)):
        q_orig = quats[i]
        q_rec = recovered[i]
        # Check that either q or -q matches
        err = min(
            np.linalg.norm(q_orig - q_rec),
            np.linalg.norm(q_orig + q_rec),
        )
        assert err < 1e-5, f"Quaternion roundtrip failed at index {i}: err={err}"


# ---------------------------------------------------------------------------
# Action stats computation tests
# ---------------------------------------------------------------------------


def test_action_stats_eef_mode():
    """Action stats computation works in EEF mode."""
    from openwam.dataloader.robotwin_stats_computation import compute_action_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        stats = compute_action_stats(tmpdir, action_mode="eef")
        assert stats["mean"].shape == (20,)
        assert stats["min"].shape == (20,)
        assert stats["max"].shape == (20,)


def test_action_stats_joint_mode():
    """Action stats computation works in joint mode."""
    from openwam.dataloader.robotwin_stats_computation import compute_action_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        stats = compute_action_stats(tmpdir, action_mode="joint")
        assert stats["mean"].shape == (14,)
        assert stats["min"].shape == (14,)


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------


def test_registry_robotwin_is_multitask():
    """Registry 'robotwin' type maps to MultiTaskRoboTwinDataset."""
    from openwam.dataloader.registry import DATASET_REGISTRY
    from openwam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    assert "robotwin" in DATASET_REGISTRY
    assert DATASET_REGISTRY["robotwin"] is MultiTaskRoboTwinDataset


def test_registry_robotwin_multitask_removed():
    """The 'robotwin_multitask' alias should no longer be registered."""
    from openwam.dataloader.registry import DATASET_REGISTRY

    assert "robotwin_multitask" not in DATASET_REGISTRY


def test_from_config_task_resolution_single_task():
    """from_config with task_name set resolves to [task_name]."""
    from openwam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    config = {
        "type": "robotwin",
        "dataset_dir": "/dummy",
        "task_name": "adjust_bottle",
        "robot": "aloha-agilex",
        "variant": "clean_50",
    }

    # We can't construct a real dataset without data, but we can test
    # that from_config calls __init__ with the right tasks list by
    # monkeypatching __init__.
    captured = {}
    original_init = MultiTaskRoboTwinDataset.__init__

    def mock_init(self, **kwargs):
        captured.update(kwargs)
        raise _SkipInit()

    class _SkipInit(Exception):
        pass

    MultiTaskRoboTwinDataset.__init__ = mock_init
    try:
        MultiTaskRoboTwinDataset.from_config(config, split="train")
    except _SkipInit:
        pass
    finally:
        MultiTaskRoboTwinDataset.__init__ = original_init

    assert captured["tasks"] == ["adjust_bottle"]
    assert captured["split"] == "train"


def test_from_config_task_resolution_holdout():
    """from_config with holdout_tasks excludes them from training tasks."""
    from openwam.dataloader.robotwin_dataset import (
        ROBOTWIN_ALL_TASKS,
        MultiTaskRoboTwinDataset,
    )

    holdout = ["open_laptop", "turn_switch"]
    config = {
        "type": "robotwin",
        "dataset_dir": "/dummy",
        "task_name": None,
        "train_tasks": None,
        "holdout_tasks": holdout,
        "robot": "aloha-agilex",
        "variant": "clean_50",
    }

    captured = {}
    original_init = MultiTaskRoboTwinDataset.__init__

    def mock_init(self, **kwargs):
        captured.update(kwargs)
        raise _SkipInit()

    class _SkipInit(Exception):
        pass

    MultiTaskRoboTwinDataset.__init__ = mock_init
    try:
        MultiTaskRoboTwinDataset.from_config(config, split="train")
    except _SkipInit:
        pass
    finally:
        MultiTaskRoboTwinDataset.__init__ = original_init

    expected = sorted(t for t in ROBOTWIN_ALL_TASKS if t not in holdout)
    assert captured["tasks"] == expected


def test_from_config_via_registry():
    """build_dataset dispatches to MultiTaskRoboTwinDataset.from_config."""
    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    config = {
        "type": "robotwin",
        "dataset_dir": "/dummy",
        "task_name": "adjust_bottle",
        "robot": "aloha-agilex",
        "variant": "clean_50",
    }

    captured = {}
    original_init = MultiTaskRoboTwinDataset.__init__

    def mock_init(self, **kwargs):
        captured.update(kwargs)
        raise _SkipInit()

    class _SkipInit(Exception):
        pass

    MultiTaskRoboTwinDataset.__init__ = mock_init
    try:
        build_dataset(config, split="train")
    except _SkipInit:
        pass
    finally:
        MultiTaskRoboTwinDataset.__init__ = original_init

    assert captured["dataset_dir"] == "/dummy"
    assert captured["tasks"] == ["adjust_bottle"]
    assert captured["action_mode"] == "eef"
