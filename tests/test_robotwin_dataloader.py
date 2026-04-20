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


def _flat_stats(action_dim: int, mean: float = 0.0, std: float = 1.0, low: float = -1.0, high: float = 1.0) -> dict:
    return {
        "mean": np.full(action_dim, mean, dtype=np.float64),
        "std": np.full(action_dim, std, dtype=np.float64),
        "min": np.full(action_dim, low, dtype=np.float64),
        "max": np.full(action_dim, high, dtype=np.float64),
        "q01": np.full(action_dim, low + 0.01 * (high - low), dtype=np.float64),
        "q99": np.full(action_dim, high - 0.01 * (high - low), dtype=np.float64),
    }


def _create_action_stats(path, joint_dim: int = 14, eef_dim: int = 20, joint_flat: bool = False) -> None:
    """Create a mock action_stats.npy with both joint and eef sub-dicts.

    When ``joint_flat=True`` writes the legacy flat schema instead (for
    backward-compat tests); the flat dict uses ``joint_dim``.
    """
    if joint_flat:
        np.save(path, _flat_stats(joint_dim))
        return
    nested = {
        "joint": _flat_stats(joint_dim),
        "eef": _flat_stats(eef_dim),
        "num_timesteps": 1000,
    }
    np.save(path, nested)


# ---------------------------------------------------------------------------
# Joint mode tests
# ---------------------------------------------------------------------------


def test_joint_mode_basic():
    """Joint mode loads correctly and returns action_dim=14.

    With the new semantics num_frames = *sampled* frames; one window covers
    (num_frames-1)*video_stride + 1 raw frames. Action trajectory has
    num_frames-1 steps; proprio is the first sampled action.
    """
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=20, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            val_ratio=0.0,
            video_stride=1,
            filter_static_segments=False,
        )
        assert ds.action_dim == 14
        assert ds.action_mode == "joint"

        sample = ds[0]
        # action horizon = num_frames - 1
        assert sample["action_trajectory"].shape == (4, 14)
        assert sample["action_mask"].shape == (4,)
        # proprio is a single frame with time dim kept (shape (1, D))
        assert sample["proprio"].shape == (1, 14)
        assert sample["proprio_mask"].shape == (1,)
        # video_mask length matches sampled frames
        assert sample["video_mask"].shape == (5,)


def test_short_episode_pads_and_masks():
    """Episode shorter than the requested window must pad frames + action_mask.

    Layout for T=10, num_frames=17:
      - raw_window_len   = 17
      - actual_raw_len   = 10  (ep_len < window)
      - pad_len          = 7   (last frame repeated in video + actions)
      - action_mask[t]   = (t + 1) < 10  → first 9 True, remaining 7 False
    """
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=10, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,  # (17-1)%4==0 ✓ video_frames=5 (5-1)%4==0 ✓
            height=32,
            width=32,
            action_mode="joint",
            val_ratio=0.0,
            filter_static_segments=False,
            normalize_mode=None,
        )

        # One window starting at 0 (max_start = max(0, 10 - 17) = 0)
        assert len(ds) == 1
        sample = ds[0]

        # Shapes are still the full horizon regardless of episode length
        assert sample["action_trajectory"].shape == (16, 14)
        assert len(sample["video"]) == 5

        # Only steps whose source raw frame exists are unmasked.
        # action_mask[t] is (t + 1) < actual_raw_len = 10 → True for t in 0..8
        mask = sample["action_mask"].bool().tolist()
        assert mask[:9] == [True] * 9
        assert mask[9:] == [False] * 7

        # The padded tail actions should exactly repeat the last real action.
        last_real = sample["action_trajectory"][8]
        for t in range(9, 16):
            assert (sample["action_trajectory"][t] == last_real).all(), (
                f"padded step {t} does not equal last real action"
            )


def test_video_stride_does_not_affect_action_length():
    """video_stride must subsample VIDEO only; state/action stay at raw HDF5 rate.

    Layout under num_frames=17, video_stride=4:
      - raw window      = 17 HDF5 frames
      - video frames    = (17-1)//4 + 1 = 5  (subsampled, also satisfies VAE (5-1)%4==0)
      - action horizon  = num_frames - 1 = 16  (full raw rate)
      - proprio         = raw_actions[0:1], shape (1, D)
    """
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=30, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,  # (17-1) % 4 == 0 ✓ and (5-1) % 4 == 0 for VAE ✓
            height=32,
            width=32,
            action_mode="joint",
            val_ratio=0.0,
            filter_static_segments=False,
            normalize_mode=None,
        )

        sample = ds[0]
        assert sample["action_trajectory"].shape == (16, 14)
        assert sample["action_mask"].shape == (16,)
        assert len(sample["video"]) == 5
        assert sample["video_mask"].shape == (5,)
        assert sample["proprio"].shape == (1, 14)
        assert sample["proprio_mask"].shape == (1,)


def test_invalid_video_stride_rejected():
    """(num_frames - 1) must be divisible by video_stride; else ValueError."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        # num_frames=10, video_stride=4 → (10-1) % 4 == 1, invalid
        try:
            RoboTwinDataset(
                data_root=tmpdir,
                num_frames=10,
                video_stride=4,
                height=32,
                width=32,
                action_mode="joint",
                val_ratio=0.0,
            )
        except ValueError as e:
            assert "divisible" in str(e)
        else:
            raise AssertionError("Expected ValueError for (num_frames-1) not divisible by video_stride")


def test_joint_mode_minmax_normalization():
    """Joint mode applies min-max normalization to joints and binary to grippers."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=42)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            action_stats_path=stats_path,
            val_ratio=0.0,
            video_stride=1,
            filter_static_segments=False,
        )

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()
        proprio = sample["proprio"].numpy()

        # All dims (including gripper) should be in [-1, 1] (min-max normalized)
        assert actions.min() >= -1.0 - 1e-6
        assert actions.max() <= 1.0 + 1e-6
        # proprio shares the normalized space
        assert proprio.min() >= -1.0 - 1e-6
        assert proprio.max() <= 1.0 + 1e-6


def test_joint_mode_gripper_continuous():
    """Joint mode gripper uses raw continuous values, min-max normalized like all dims."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            action_stats_path=stats_path,
            val_ratio=0.0,
            video_stride=1,  # num_video_frames=5 → (5-1)%4=0 ✓
            filter_static_segments=False,
        )

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()

        # Gripper dims should be continuous (min-max normalized), not binary
        gripper_vals = actions[:, 6]
        assert gripper_vals.min() >= -1.0 - 1e-6
        assert gripper_vals.max() <= 1.0 + 1e-6


def test_joint_mode_denormalize_roundtrip():
    """denormalize_action inverts normalization back to raw units."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")

        # Joint stats: min=0, max=2 for all dims; eef filler
        stats = {
            "joint": _flat_stats(14, mean=1.0, std=1.0, low=0.0, high=2.0),
            "eef": _flat_stats(20),
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
            video_stride=1,
            filter_static_segments=False,
        )

        # normalized = -1 → raw = min = 0 under min-max
        test_normalized = np.full((1, 14), -1.0, dtype=np.float32)
        denormed = ds.denormalize_action(test_normalized)
        np.testing.assert_allclose(denormed[0], 0.0, atol=1e-5)


def test_legacy_flat_stats_file_backward_compat():
    """Old flat-schema stats files still load with a DeprecationWarning."""
    import warnings

    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path, joint_flat=True)  # legacy flat schema

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = RoboTwinDataset(
                data_root=tmpdir,
                num_frames=5,
                height=32,
                width=32,
                action_mode="joint",
                action_stats_path=stats_path,
                val_ratio=0.0,
                video_stride=1,
                filter_static_segments=False,
            )
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        sample = ds[0]
        actions = sample["action_trajectory"].numpy()
        assert actions.min() >= -1.0 - 1e-6
        assert actions.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# EEF mode tests
# ---------------------------------------------------------------------------


def test_eef_mode_basic():
    """EEF mode loads correctly and returns action_dim=20."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=20, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            val_ratio=0.0,
            video_stride=1,
            filter_static_segments=False,
            normalize_mode=None,  # raw values for this test
        )
        assert ds.action_dim == 20
        assert ds.action_mode == "eef"
        assert ds.action_stats is None  # no stats loaded when normalize_mode=None

        sample = ds[0]
        # action horizon = num_frames - 1
        assert sample["action_trajectory"].shape == (4, 20)
        assert sample["proprio"].shape == (1, 20)


def test_eef_mode_minmax_normalization():
    """EEF mode with normalize_mode='min-max' rescales all 20 dims into [-1, 1]."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        _create_action_stats(stats_path)  # nested {joint, eef}

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            action_stats_path=stats_path,
            normalize_mode="min-max",
            val_ratio=0.0,
            video_stride=1,
            filter_static_segments=False,
        )
        assert ds.action_stats is not None and "min" in ds.action_stats

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()
        proprio = sample["proprio"].numpy()
        # With stats {min:-1, max:+1} and raw values roughly in that range,
        # normalized outputs should stay in [-1, 1] after min-max.
        assert actions.min() >= -1.0 - 1e-6
        assert actions.max() <= 1.0 + 1e-6
        assert proprio.min() >= -1.0 - 1e-6
        assert proprio.max() <= 1.0 + 1e-6


def test_eef_mode_zscore_normalization():
    """EEF mode with normalize_mode='z-score' centers around the stats mean."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=40, seed=0)
        stats_path = os.path.join(tmpdir, "action_stats.npy")
        # Deliberately small std so normalized magnitudes are large — makes it
        # easy to tell the mapping actually applied.
        nested = {
            "joint": _flat_stats(14, mean=0.0, std=1.0, low=-1.0, high=1.0),
            "eef": _flat_stats(20, mean=0.5, std=0.25, low=0.0, high=1.0),
        }
        np.save(stats_path, nested)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            action_stats_path=stats_path,
            normalize_mode="z-score",
            val_ratio=0.0,
            video_stride=1,
            filter_static_segments=False,
        )
        sample = ds[0]
        actions = sample["action_trajectory"].numpy()
        # z-score: (x - 0.5) / 0.25 → scale-up by 4.
        # Raw xyz/gripper are in [0, 1] so they map into roughly [-2, 2].
        # rot6d can extend beyond [0, 1], so allow a wider envelope.
        assert actions.min() >= -8.0
        assert actions.max() <= 8.0
        # Confirm the mapping actually applied (not a no-op): mean should shift
        # substantially away from 0.5 because we subtracted 0.5 before dividing.
        assert abs(actions.mean()) > 0.1


def test_eef_roundtrip_denormalize():
    """EEF normalize → denormalize should recover the raw value for both modes."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    for mode in ("min-max", "z-score"):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
            stats_path = os.path.join(tmpdir, "action_stats.npy")
            _create_action_stats(stats_path)

            ds = RoboTwinDataset(
                data_root=tmpdir,
                num_frames=5,
                height=32,
                width=32,
                action_mode="eef",
                action_stats_path=stats_path,
                normalize_mode=mode,
                val_ratio=0.0,
                video_stride=1,
                filter_static_segments=False,
            )
            raw = np.random.RandomState(0).uniform(-1, 1, size=(7, 20)).astype(np.float32)
            normed = ds._action_normalizer.normalize(raw)
            recovered = ds.denormalize_action(normed)
            np.testing.assert_allclose(recovered, raw, atol=1e-4, err_msg=f"mode={mode}")


def test_eef_gripper_raw_values():
    """EEF mode uses raw continuous gripper values from HDF5 without inversion."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # T=10, num_frames=9, video_stride=2 → window [0..8], num_video_frames=5.
        # Mock data: gripper 1.0 (open) for frames 0-4, 0.0 (closed) for frames 5-9.
        # action_trajectory = raw[1..8], so first 4 actions open, last 4 closed.
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=9,
            height=32,
            width=32,
            action_mode="eef",
            val_ratio=0.0,
            video_stride=2,  # (9-1)%2==0 and (5-1)%4==0 for VAE ✓
            filter_static_segments=False,
            normalize_mode=None,  # keep raw gripper values for this assertion
        )

        sample = ds[0]
        actions = sample["action_trajectory"].numpy()  # (8, 20)
        proprio = sample["proprio"].numpy()  # (1, 20)

        # proprio is from frame 0 → gripper open (1.0)
        np.testing.assert_allclose(proprio[0, 9], 1.0, atol=1e-6)
        np.testing.assert_allclose(proprio[0, 19], 1.0, atol=1e-6)

        # First 4 action steps: raw gripper = 1.0 (open) → remain 1.0
        np.testing.assert_allclose(actions[:4, 9], 1.0, atol=1e-6)
        np.testing.assert_allclose(actions[:4, 19], 1.0, atol=1e-6)

        # Remaining 4 action steps: raw gripper = 0.0 (closed) → remain 0.0
        np.testing.assert_allclose(actions[4:, 9], 0.0, atol=1e-6)
        np.testing.assert_allclose(actions[4:, 19], 0.0, atol=1e-6)


def test_eef_denormalize_passthrough():
    """With normalize_mode=None, denormalize_action is an identity."""
    from openwam.dataloader.robotwin_dataset import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=20, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            val_ratio=0.0,
            video_stride=1,
            filter_static_segments=False,
            normalize_mode=None,
        )

        # All dims should pass through unchanged (no inversion, no normalization)
        test_data = np.random.randn(5, 20).astype(np.float32)
        result = ds.denormalize_action(test_data)
        np.testing.assert_array_equal(result, test_data)


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
                _create_mock_episode(os.path.join(data_dir, "episode0.hdf5"), T=20, seed=hash(task + variant) % 1000)

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
            video_stride=1,
            filter_static_segments=False,
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
            _create_mock_episode(os.path.join(data_dir, "episode0.hdf5"), T=20, seed=0)

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
            video_stride=1,
            filter_static_segments=False,
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


def test_action_stats_nested_schema_contains_both_modes():
    """compute_action_stats returns a nested dict with both 'joint' and 'eef'."""
    from openwam.dataloader.robotwin_stats_computation import compute_action_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        stats = compute_action_stats(tmpdir)
        assert "joint" in stats and "eef" in stats
        assert stats["joint"]["mean"].shape == (14,)
        assert stats["joint"]["std"].shape == (14,)
        assert stats["eef"]["mean"].shape == (20,)
        assert stats["eef"]["std"].shape == (20,)
        # Per-dim stats cover a non-zero range (mock actions are uniform in [0, 1])
        assert stats["joint"]["max"].max() > stats["joint"]["min"].min()
        assert stats["eef"]["max"].max() > stats["eef"]["min"].min()
        assert stats["num_timesteps"] > 0


def test_multitask_action_stats_nested_schema():
    """compute_multitask_robotwin_stats also returns both modes."""
    from openwam.dataloader.robotwin_stats_computation import compute_multitask_robotwin_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["task_a", "task_b"]:
            for variant in ["clean_50", "randomized_500"]:
                data_dir = os.path.join(tmpdir, task, f"test-robot_{variant}", "data")
                os.makedirs(data_dir)
                _create_mock_episode(
                    os.path.join(data_dir, "episode0.hdf5"),
                    T=10,
                    seed=hash(task + variant) % 1000,
                )

        stats = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            robot="test-robot",
            variant="both",
            tasks=["task_a", "task_b"],
        )
        assert "joint" in stats and "eef" in stats
        assert stats["joint"]["mean"].shape == (14,)
        assert stats["eef"]["mean"].shape == (20,)


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
