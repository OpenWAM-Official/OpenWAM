"""Tests for RoboTwin dataloader: joint mode, EEF mode, multi-variant, normalization."""

import io
import os
import tempfile
import threading
import time

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


def _create_normalization_stats(path, joint_dim: int = 14, eef_dim: int = 20) -> None:
    """Create a mock nested-schema normalization_stats.npy with both joint and eef sub-dicts."""
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
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=20, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            video_stride=1,
        )
        assert ds.action_dim == 14
        assert ds.action_mode == "joint"

        sample = ds[0]
        # action horizon = num_frames - 1
        assert sample["action"].shape == (4, 14)
        # 2-D mask: (T_action, action_dim)
        assert sample["action_mask"].shape == (4, 14)
        # proprio is a single frame with time dim kept (shape (1, D))
        assert sample["proprio"].shape == (1, 14)
        # 2-D mask: (1, action_dim)
        assert sample["proprio_mask"].shape == (1, 14)
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
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=10, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,  # (17-1)%4==0 ✓ video_frames=5 (5-1)%4==0 ✓
            height=32,
            width=32,
            action_mode="joint",
            normalize_mode=None,
        )

        # Every start must include at least one real future action label.
        assert len(ds) == 9
        sample = ds[0]

        # Shapes are still the full horizon regardless of episode length
        assert sample["action"].shape == (16, 14)
        assert len(sample["video"]) == 5

        # Only steps whose source raw frame exists are unmasked.
        # action_mask[t] is (t + 1) < actual_raw_len = 10 → True for t in 0..8.
        # Post 2-D mask migration: mask is (T, D). Collapse to per-step validity
        # via .any(dim=-1) — RoboTwin bimanual sets all D dims uniformly per step.
        mask_per_step = sample["action_mask"].any(dim=-1).bool().tolist()
        assert mask_per_step[:9] == [True] * 9
        assert mask_per_step[9:] == [False] * 7

        # The padded tail actions should exactly repeat the last real action.
        last_real = sample["action"][8]
        for t in range(9, 16):
            assert (sample["action"][t] == last_real).all(), f"padded step {t} does not equal last real action"


def test_long_episode_tail_windows_are_included_and_padded():
    """Long episodes must include tail starts, not only full windows.

    For T=20, num_frames=17, every start in 0..18 is valid. A start near the
    end should be padded while still containing at least one real action label.
    """
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,
            height=32,
            width=32,
            action_mode="joint",
            normalize_mode=None,
        )

        assert len(ds) == 19
        assert ds._window_index[:3] == [(0, 0), (0, 1), (0, 2)]
        assert ds._window_index[-1] == (0, 18)

        # start=15 leaves raw frames 15..19 available. Actions are frames
        # 16..19 (4 valid steps), then the last action repeats.
        sample = ds[15]
        # 2-D mask (T, D); collapse to per-step via any(dim=-1).
        mask = sample["action_mask"].any(dim=-1).bool().tolist()
        assert mask[:4] == [True] * 4
        assert mask[4:] == [False] * 12

        video_mask = sample["video_mask"].bool().tolist()
        assert video_mask == [True, True, False, False, False]

        last_real = sample["action"][3]
        for t in range(4, 16):
            assert (sample["action"][t] == last_real).all(), f"padded step {t} does not equal last real action"


def test_tail_windows_always_have_at_least_one_valid_action():
    """The loader must not enumerate samples with fully padded action labels."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,
            height=32,
            width=32,
            action_mode="joint",
            normalize_mode=None,
        )

        assert ds._window_index[-1] == (0, 18)
        sample = ds[len(ds) - 1]
        assert sample["start_frame"] == 18
        # 2-D mask (T, D); collapse to per-step via any(dim=-1).
        assert sample["action_mask"].any(dim=-1).bool().tolist() == [True] + [False] * 15


def test_single_frame_episode_has_no_valid_action_window():
    """A sample must contain at least one future action label."""
    import pytest

    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=1, seed=0)

        with pytest.raises(ValueError, match="at least one action label"):
            RoboTwinDataset(
                data_root=tmpdir,
                num_frames=17,
                video_stride=4,
                height=32,
                width=32,
                action_mode="joint",
                normalize_mode=None,
            )


def test_build_sample_rejects_window_without_valid_action_label():
    """Internal sample construction also enforces the action-label invariant."""
    import pytest

    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,
            height=32,
            width=32,
            action_mode="joint",
            normalize_mode=None,
        )

        with pytest.raises(IndexError, match="no valid action label"):
            ds._build_sample(0, 19)


def test_tail_masks_flow_through_prepare_inputs_and_loss():
    """Dataset tail masks must become loss masks and suppress padded errors."""
    import torch

    from openwam.dataloader.robotwin import RoboTwinDataset
    from tests.test_openwam_trainer import _make_tiny_arch

    class _UnitScheduler:
        def training_weight(self, timestep_ids):
            return torch.ones_like(timestep_ids, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,
            height=32,
            width=32,
            action_mode="joint",
            normalize_mode=None,
        )
        arch = _make_tiny_arch()

        # start=15 has four real action labels (frames 16..19) followed by pad.
        partial = ds[15]
        inputs = arch.prepare_inputs(partial)
        # Post 2-D mask migration: action_is_pad is (B, T, action_dim). Collapse
        # to per-step via any(dim=-1) to recover the legacy time-mask comparison.
        expected_action_is_pad = torch.tensor([[False] * 4 + [True] * 12])
        assert inputs["action_is_pad"].shape == (1, 16, ds.action_dim)
        assert torch.equal(inputs["action_is_pad"].cpu().any(dim=-1), expected_action_is_pad)
        # Video frames are [15,19,pad,pad,pad]. The single tail latent group
        # contains frame 19, so it is not considered padded.
        assert torch.equal(inputs["video_is_pad"].cpu(), torch.tensor([[False]]))

        target_action = torch.zeros(1, 16, 2)
        pred_action = torch.zeros_like(target_action)
        pred_action[:, :4] = 0.5
        pred_action[:, 4:] = 1000.0
        action_loss = arch._compute_action_loss(
            pred_action,
            target_action,
            torch.tensor([0]),
            _UnitScheduler(),
            inputs,
            device="cpu",
        )
        assert abs(action_loss.item() - 0.25) < 1e-5

        # Last valid start has one real action label and then padding. Padded
        # action errors are ignored, while the one valid step still contributes.
        last = ds[18]
        last_inputs = arch.prepare_inputs(last)
        # 2-D action_is_pad collapsed to per-step.
        assert last_inputs["action_is_pad"].shape == (1, 16, ds.action_dim)
        assert torch.equal(
            last_inputs["action_is_pad"].cpu().any(dim=-1),
            torch.tensor([[False] + [True] * 15]),
        )
        assert torch.equal(last_inputs["video_is_pad"].cpu(), torch.tensor([[True]]))

        pred_action = torch.full((1, 16, 2), 1000.0)
        pred_action[:, :1] = 0.5
        target_action = torch.zeros_like(pred_action)
        action_loss = arch._compute_action_loss(
            pred_action,
            target_action,
            torch.tensor([0]),
            _UnitScheduler(),
            last_inputs,
            device="cpu",
        )
        assert abs(action_loss.item() - 0.25) < 1e-5

        video_inputs = dict(last_inputs)
        video_inputs["first_frame_latents"] = torch.zeros(1, 2, 1, 1, 1)
        pred_video = torch.full((1, 2, 2, 1, 1), 1000.0)
        target_video = torch.zeros_like(pred_video)
        video_loss = arch._compute_video_loss(
            pred_video,
            target_video,
            torch.tensor([0]),
            video_inputs,
            device="cpu",
        )
        assert video_loss.item() == 0.0


def test_video_stride_does_not_affect_action_length():
    """video_stride must subsample VIDEO only; state/action stay at raw HDF5 rate.

    Layout under num_frames=17, video_stride=4:
      - raw window      = 17 HDF5 frames
      - video frames    = (17-1)//4 + 1 = 5  (subsampled, also satisfies VAE (5-1)%4==0)
      - action horizon  = num_frames - 1 = 16  (full raw rate)
      - proprio         = raw_actions[0:1], shape (1, D)
    """
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=30, seed=0)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=17,
            video_stride=4,  # (17-1) % 4 == 0 ✓ and (5-1) % 4 == 0 for VAE ✓
            height=32,
            width=32,
            action_mode="joint",
            normalize_mode=None,
        )

        sample = ds[0]
        assert sample["action"].shape == (16, 14)
        # 2-D mask: (T_action, action_dim)
        assert sample["action_mask"].shape == (16, 14)
        assert len(sample["video"]) == 5
        assert sample["video_mask"].shape == (5,)
        assert sample["proprio"].shape == (1, 14)
        # 2-D mask: (1, action_dim)
        assert sample["proprio_mask"].shape == (1, 14)


def test_joint_mode_minmax_normalization():
    """Joint mode applies min-max normalization to joints and binary to grippers."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=42)
        stats_path = os.path.join(tmpdir, "normalization_stats.npy")
        _create_normalization_stats(stats_path)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            normalization_stats_path=stats_path,
            video_stride=1,
        )

        sample = ds[0]
        actions = sample["action"].numpy()
        proprio = sample["proprio"].numpy()

        # All dims (including gripper) should be in [-1, 1] (min-max normalized)
        assert actions.min() >= -1.0 - 1e-6
        assert actions.max() <= 1.0 + 1e-6
        # proprio shares the normalized space
        assert proprio.min() >= -1.0 - 1e-6
        assert proprio.max() <= 1.0 + 1e-6


def test_joint_mode_gripper_continuous():
    """Joint mode gripper uses raw continuous values, min-max normalized like all dims."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "normalization_stats.npy")
        _create_normalization_stats(stats_path)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            normalization_stats_path=stats_path,
            video_stride=1,  # num_video_frames=5 → (5-1)%4=0 ✓
        )

        sample = ds[0]
        actions = sample["action"].numpy()

        # Gripper dims should be continuous (min-max normalized), not binary
        gripper_vals = actions[:, 6]
        assert gripper_vals.min() >= -1.0 - 1e-6
        assert gripper_vals.max() <= 1.0 + 1e-6


def test_joint_mode_denormalize_roundtrip():
    """denormalize_action inverts normalization back to raw units."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "normalization_stats.npy")

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
            normalization_stats_path=stats_path,
            video_stride=1,
        )

        # normalized = -1 → raw = min = 0 under min-max
        test_normalized = np.full((1, 14), -1.0, dtype=np.float32)
        denormed = ds.denormalize_action(test_normalized)
        np.testing.assert_allclose(denormed[0], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# EEF mode tests
# ---------------------------------------------------------------------------


def test_eef_mode_basic():
    """EEF mode loads correctly and returns action_dim=20."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=20, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            video_stride=1,
            normalize_mode=None,  # raw values for this test
        )
        assert ds.action_dim == 20
        assert ds.action_mode == "eef"
        assert ds.normalization_stats is None  # no stats loaded when normalize_mode=None

        sample = ds[0]
        # action horizon = num_frames - 1
        assert sample["action"].shape == (4, 20)
        assert sample["proprio"].shape == (1, 20)


def test_eef_mode_minmax_normalization():
    """EEF mode with normalize_mode='min-max' rescales all 20 dims into [-1, 1]."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
        stats_path = os.path.join(tmpdir, "normalization_stats.npy")
        _create_normalization_stats(stats_path)  # nested {joint, eef}

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            normalization_stats_path=stats_path,
            normalize_mode="min-max",
            video_stride=1,
        )
        assert ds.normalization_stats is not None and "min" in ds.normalization_stats

        sample = ds[0]
        actions = sample["action"].numpy()
        proprio = sample["proprio"].numpy()
        # With stats {min:-1, max:+1} and raw values roughly in that range,
        # normalized outputs should stay in [-1, 1] after min-max.
        assert actions.min() >= -1.0 - 1e-6
        assert actions.max() <= 1.0 + 1e-6
        assert proprio.min() >= -1.0 - 1e-6
        assert proprio.max() <= 1.0 + 1e-6


def test_eef_mode_zscore_normalization():
    """EEF mode with normalize_mode='z-score' centers around the stats mean."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=40, seed=0)
        stats_path = os.path.join(tmpdir, "normalization_stats.npy")
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
            normalization_stats_path=stats_path,
            normalize_mode="z-score",
            video_stride=1,
        )
        sample = ds[0]
        actions = sample["action"].numpy()
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
    from openwam.dataloader.robotwin import RoboTwinDataset

    for mode in ("min-max", "z-score"):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_mock_episode(os.path.join(tmpdir, "episode0.hdf5"), T=20, seed=0)
            stats_path = os.path.join(tmpdir, "normalization_stats.npy")
            _create_normalization_stats(stats_path)

            ds = RoboTwinDataset(
                data_root=tmpdir,
                num_frames=5,
                height=32,
                width=32,
                action_mode="eef",
                normalization_stats_path=stats_path,
                normalize_mode=mode,
                video_stride=1,
            )
            raw = np.random.RandomState(0).uniform(-1, 1, size=(7, 20)).astype(np.float32)
            normed = ds._normalizer.normalize(raw)
            recovered = ds.denormalize_action(normed)
            np.testing.assert_allclose(recovered, raw, atol=1e-4, err_msg=f"mode={mode}")


def test_eef_gripper_raw_values():
    """EEF mode uses raw continuous gripper values from HDF5 without inversion."""
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # T=10, num_frames=9, video_stride=2 → window [0..8], num_video_frames=5.
        # Mock data: gripper 1.0 (open) for frames 0-4, 0.0 (closed) for frames 5-9.
        # action = raw[1..8], so first 4 actions open, last 4 closed.
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=9,
            height=32,
            width=32,
            action_mode="eef",
            video_stride=2,  # (9-1)%2==0 and (5-1)%4==0 for VAE ✓
            normalize_mode=None,  # keep raw gripper values for this assertion
        )

        sample = ds[0]
        actions = sample["action"].numpy()  # (8, 20)
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
    from openwam.dataloader.robotwin import RoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=20, seed=i)

        ds = RoboTwinDataset(
            data_root=tmpdir,
            num_frames=5,
            height=32,
            width=32,
            action_mode="eef",
            video_stride=1,
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
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure for 2 tasks × 2 variants
        for task in ["task_a", "task_b"]:
            for variant in ["clean_50", "randomized_500"]:
                data_dir = os.path.join(tmpdir, task, f"test-robot_{variant}", "data")
                os.makedirs(data_dir)
                _create_mock_episode(os.path.join(data_dir, "episode0.hdf5"), T=20, seed=hash(task + variant) % 1000)

        ds = MultiTaskRoboTwinDataset(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="both",
            tasks=["task_a", "task_b"],
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            video_stride=1,
        )

        # Should have 4 sub-datasets (2 tasks × 2 variants)
        assert len(ds._sub_datasets) == 4
        assert len(ds) > 0


def test_multi_variant_single_variant_compat():
    """MultiTaskRoboTwinDataset with variant='clean_50' loads a single variant."""
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["task_a"]:
            data_dir = os.path.join(tmpdir, task, "test-robot_clean_50", "data")
            os.makedirs(data_dir)
            _create_mock_episode(os.path.join(data_dir, "episode0.hdf5"), T=20, seed=0)

        ds = MultiTaskRoboTwinDataset(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a"],
            num_frames=5,
            height=32,
            width=32,
            action_mode="joint",
            video_stride=1,
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


def test_normalization_stats_nested_schema_contains_both_modes():
    """compute_normalization_stats returns a nested dict with both 'joint' and 'eef'."""
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_normalization_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _create_mock_episode(os.path.join(tmpdir, f"episode{i}.hdf5"), T=10, seed=i)

        stats = compute_normalization_stats(tmpdir)
        assert "joint" in stats and "eef" in stats
        assert stats["joint"]["mean"].shape == (14,)
        assert stats["joint"]["std"].shape == (14,)
        assert stats["eef"]["mean"].shape == (20,)
        assert stats["eef"]["std"].shape == (20,)
        # Per-dim stats cover a non-zero range (mock actions are uniform in [0, 1])
        assert stats["joint"]["max"].max() > stats["joint"]["min"].min()
        assert stats["eef"]["max"].max() > stats["eef"]["min"].min()
        assert stats["num_timesteps"] > 0


def test_multitask_normalization_stats_nested_schema():
    """compute_multitask_robotwin_stats also returns both modes."""
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_multitask_robotwin_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["task_a", "task_b"]:
            for variant in ["clean_50", "randomized_500"]:
                data_dir = os.path.join(tmpdir, task, f"test-robot_{variant}", "data")
                os.makedirs(data_dir)
                _create_mock_episode(
                    os.path.join(data_dir, "episode0.hdf5"),
                    T=10,
                )

        stats = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="both",
            tasks=["task_a", "task_b"],
        )
        assert "joint" in stats and "eef" in stats
        assert stats["joint"]["mean"].shape == (14,)
        assert stats["eef"]["mean"].shape == (20,)


# ---------------------------------------------------------------------------
# Resumable multi-task stats checkpointing
# ---------------------------------------------------------------------------


def _make_multitask_layout(root, tasks, embodiment="test-robot", variant="clean_50", T=10):
    """Lay down a minimal RoboTwin multi-task tree under ``root``."""
    for task in tasks:
        data_dir = os.path.join(root, task, f"{embodiment}_{variant}", "data")
        os.makedirs(data_dir, exist_ok=True)
        _create_mock_episode(
            os.path.join(data_dir, "episode0.hdf5"),
            T=T,
        )


def test_multitask_action_stats_can_resume_from_partial_checkpoint():
    """Second run reuses already-persisted task shards instead of starting over."""
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_multitask_robotwin_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multitask_layout(tmpdir, ["task_a", "task_b"])
        checkpoint_path = os.path.join(tmpdir, "test-robot_clean_50_stats.npy")

        partial_stats = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a"],
            checkpoint_path=checkpoint_path,
        )
        assert partial_stats["num_timesteps"] > 0
        assert os.path.isdir(f"{checkpoint_path}.partial/shards_v1")

        resumed_stats = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a", "task_b"],
            checkpoint_path=checkpoint_path,
        )
        full_stats = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a", "task_b"],
        )

        assert resumed_stats["num_timesteps"] == full_stats["num_timesteps"]
        for mode in ["joint", "eef"]:
            for key in ["mean", "std", "min", "max", "q01", "q99"]:
                assert np.allclose(resumed_stats[mode][key], full_stats[mode][key])


def test_resume_does_not_recompute_already_checkpointed_shards():
    """task_a's shard mtime must be unchanged after a resume that adds task_b.

    Without this, the resume path could silently rerun every task on every
    invocation and the prior test would still pass. Pins the actual contract
    PR #77 review flagged: already-persisted task-roots are *skipped*.
    """
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_multitask_robotwin_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multitask_layout(tmpdir, ["task_a", "task_b"])
        checkpoint_path = os.path.join(tmpdir, "test-robot_clean_50_stats.npy")

        compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a"],
            checkpoint_path=checkpoint_path,
        )

        shards_dir = f"{checkpoint_path}.partial/shards_v1"
        shards_before = {name: os.stat(os.path.join(shards_dir, name)).st_mtime_ns for name in os.listdir(shards_dir)}
        assert len(shards_before) == 1

        # Force a discernible mtime delta even on filesystems with low resolution.
        time.sleep(0.05)

        compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a", "task_b"],
            checkpoint_path=checkpoint_path,
        )

        shards_after = {name: os.stat(os.path.join(shards_dir, name)).st_mtime_ns for name in os.listdir(shards_dir)}
        assert len(shards_after) == 2
        # task_a's shard must not have been rewritten.
        for name, mtime in shards_before.items():
            assert shards_after[name] == mtime, f"{name} was rewritten on resume"


def test_resume_ignores_shards_dropped_from_tasks_list():
    """Shrinking tasks only rebuilds from the current task_roots' shards."""
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_multitask_robotwin_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multitask_layout(tmpdir, ["task_a", "task_b"])
        checkpoint_path = os.path.join(tmpdir, "test-robot_clean_50_stats.npy")

        # Initial run creates both task shards.
        compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a", "task_b"],
            checkpoint_path=checkpoint_path,
        )
        # Re-run with a shrunk tasks list. Extra shards remain on disk but are
        # ignored because rebuild only looks up shards for current task_roots.
        narrowed = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a"],
            checkpoint_path=checkpoint_path,
        )
        ground_truth = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a"],
        )

        # Counts and stats must match the from-scratch single-task run.
        assert narrowed["num_timesteps"] == ground_truth["num_timesteps"]
        for mode in ["joint", "eef"]:
            for key in ["mean", "std", "min", "max", "q01", "q99"]:
                assert np.allclose(narrowed[mode][key], ground_truth[mode][key])


def test_resume_shards_are_keyed_by_data_root():
    """Reusing checkpoint_path across robots does not mix incompatible shards."""
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import compute_multitask_robotwin_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multitask_layout(tmpdir, ["task_a"], embodiment="robot-x", variant="clean_50")
        _make_multitask_layout(tmpdir, ["task_a"], embodiment="robot-y", variant="clean_50")
        checkpoint_path = os.path.join(tmpdir, "shared_stats.npy")

        compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="robot-x",
            variant="clean_50",
            tasks=["task_a"],
            checkpoint_path=checkpoint_path,
        )
        # The robot-y data_root maps to a different deterministic shard, so the
        # robot-x shard is not reused even with the same checkpoint_path.
        stats_y = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="robot-y",
            variant="clean_50",
            tasks=["task_a"],
            checkpoint_path=checkpoint_path,
        )
        ground_truth_y = compute_multitask_robotwin_stats(
            dataset_dir=tmpdir,
            embodiment="robot-y",
            variant="clean_50",
            tasks=["task_a"],
        )
        for mode in ["joint", "eef"]:
            for key in ["mean", "std", "min", "max"]:
                assert np.allclose(stats_y[mode][key], ground_truth_y[mode][key])


def test_atomic_save_stats_npy_roundtrip(tmp_path):
    """atomic_save_stats_npy should produce a fully-formed .npy at the target path."""
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import atomic_save_stats_npy

    target = str(tmp_path / "stats.npy")
    payload = {"joint": {"mean": np.zeros(14, dtype=np.float32)}, "num_timesteps": 7}
    atomic_save_stats_npy(target, payload)
    loaded = np.load(target, allow_pickle=True).item()
    assert loaded["num_timesteps"] == 7
    assert loaded["joint"]["mean"].shape == (14,)
    # The tmp file must not linger.
    assert not os.path.exists(f"{target}.tmp")
    assert not os.path.exists(f"{target}.tmp.npy")


def test_multitask_peer_rank_waits_for_shared_stats(monkeypatch, tmp_path):
    """Peer ranks should wait for rank 0's final .npy instead of computing stats."""
    import openwam.dataloader.robotwin as ds_mod
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import atomic_save_stats_npy

    dataset_dir = str(tmp_path)
    os.makedirs(os.path.join(dataset_dir, "meta"), exist_ok=True)
    stats_path = os.path.join(dataset_dir, "meta", "robotwin_clean_50_normalization_stats.npy")

    class _FakeSubDataset:
        def __init__(self, **kwargs):
            self.action_dim = 14
            self.normalization_stats = {"mean": np.zeros(14, dtype=np.float32)}

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return idx

        def denormalize_action(self, action):
            return action

    def _publish_stats_later():
        time.sleep(0.02)
        atomic_save_stats_npy(
            stats_path,
            {"joint": _flat_stats(14), "eef": _flat_stats(20), "num_timesteps": 1},
        )

    worker = threading.Thread(target=_publish_stats_later)
    worker.start()

    monkeypatch.setenv("OPENWAM_STATS_POLL_INTERVAL_S", "0.005")
    monkeypatch.setenv("OPENWAM_STATS_WAIT_TIMEOUT_S", "1")
    monkeypatch.setattr(ds_mod, "discover_robotwin_roots", lambda *args, **kwargs: [("task_a", dataset_dir)])
    monkeypatch.setattr(ds_mod, "RoboTwinDataset", _FakeSubDataset)
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)

    try:
        ds = ds_mod.MultiTaskRoboTwinDataset(
            dataset_dir=dataset_dir,
            embodiment="test-robot",
            variant="clean_50",
            tasks=["task_a"],
            action_mode="joint",
            num_frames=5,
            height=32,
            width=32,
            video_stride=1,
        )
    finally:
        worker.join(timeout=1)

    assert len(ds) == 1
    assert os.path.exists(stats_path)


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------


def test_registry_robotwin_is_multitask():
    """Registry 'robotwin' type maps to MultiTaskRoboTwinDataset."""
    from openwam.dataloader.registry import DATASET_REGISTRY
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    assert "robotwin" in DATASET_REGISTRY
    assert DATASET_REGISTRY["robotwin"] is MultiTaskRoboTwinDataset


def test_registry_robotwin_multitask_removed():
    """The 'robotwin_multitask' alias should no longer be registered."""
    from openwam.dataloader.registry import DATASET_REGISTRY

    assert "robotwin_multitask" not in DATASET_REGISTRY


def test_from_config_uses_directory_discovery_without_task_selectors():
    """from_config leaves task discovery to the multi-task dataset."""
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    config = {
        "type": "robotwin",
        "dataset_dir": "/dummy",
        "embodiment": "aloha-agilex",
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

    assert "tasks" not in captured
    assert "task_name" not in captured
    assert captured["split"] == "train"


def test_discover_robotwin_roots_loads_every_task_directory(tmp_path):
    from openwam.dataloader.robotwin import discover_robotwin_roots

    for task in ("task_b", "task_a", "custom_task"):
        (tmp_path / task / "test-robot_clean_50" / "data").mkdir(parents=True)
    (tmp_path / "wrong_variant" / "test-robot_randomized_500" / "data").mkdir(parents=True)
    (tmp_path / "not_a_task").mkdir()

    roots = discover_robotwin_roots(
        str(tmp_path),
        "test-robot",
        "clean_50",
    )
    assert [task for task, _ in roots] == ["custom_task", "task_a", "task_b"]


def test_from_config_via_registry():
    """build_dataset dispatches to MultiTaskRoboTwinDataset.from_config."""
    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    config = {
        "type": "robotwin",
        "dataset_dir": "/dummy",
        "embodiment": "aloha-agilex",
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
    assert "tasks" not in captured
    assert "task_name" not in captured
    assert captured["action_mode"] == "eef"
