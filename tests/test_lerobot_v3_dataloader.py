"""Tests for LeRobot v3 dataloader.

Covers (no GPU / real dataset required):
  - ActionComposer: field extraction, multi-field concatenation, error handling
  - _flatten_layout: flat and 2-D inputs
  - assemble_multiview_layout: L-shape geometry, missing camera, bad camera count
  - _pad_actions / _pad_video: T < target / T == target / T > target + mask correctness
  - _normalize / denormalize_action: none / min-max / z-score + roundtrip
  - LeRobot3Dataset.__getitem__: return dict keys/shapes, window split, proprio_source
  - LeRobot3Dataset startup guards: normalize_mode + missing stats raises at init
  - MixtureDataset: virtual-len, dispatch, stats=None degradation, action-dim padding
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Fake-data helpers
# ---------------------------------------------------------------------------


def _make_info(action_fields: list[tuple[str, list[int]]], with_state: bool = False, state_dim: int = 7) -> dict:
    """Build a minimal info.json features dict."""
    features: dict = {}
    for field, shape in action_fields:
        features[field] = {"shape": shape, "dtype": "float32"}
    features["index"] = {"shape": [], "dtype": "int64"}
    features["episode_index"] = {"shape": [], "dtype": "int64"}
    if with_state:
        features["observation.state"] = {"shape": [state_dim], "dtype": "float32"}
    return {"features": features}


def _make_stats(action_fields: list[tuple[str, list[int]]]) -> dict:
    """Build a minimal stats.json for min-max and z-score normalization."""
    stats: dict = {}
    for field, shape in action_fields:
        d = shape[0] if shape else 1
        stats[field] = {
            "mean": [0.0] * d,
            "std": [1.0] * d,
            "min": [-1.0] * d,
            "max": [1.0] * d,
        }
    return stats


def _create_fake_task(
    base_dir: str,
    *,
    n_episodes: int = 2,
    ep_len: int = 20,
    action_fields: list[tuple[str, list[int]]] | None = None,
    cameras: list[str] | None = None,
    with_state: bool = False,
    state_dim: int = 7,
) -> str:
    """Create a minimal LeRobot v3 task directory (no real video files).

    Returns the task directory path.
    """
    if action_fields is None:
        action_fields = [("action", [14])]
    if cameras is None:
        cameras = ["head"]

    task_dir = base_dir
    os.makedirs(os.path.join(task_dir, "meta", "episodes"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "data", "chunk-000"), exist_ok=True)

    # meta/info.json
    with open(os.path.join(task_dir, "meta", "info.json"), "w") as f:
        json.dump(_make_info(action_fields, with_state=with_state, state_dim=state_dim), f)

    # meta/stats.json
    with open(os.path.join(task_dir, "meta", "stats.json"), "w") as f:
        json.dump(_make_stats(action_fields), f)

    # meta/episodes/0.parquet
    ep_meta = pd.DataFrame(
        {
            "episode_index": list(range(n_episodes)),
            "length": [ep_len] * n_episodes,
            "data/chunk_index": [0] * n_episodes,
            "data/file_index": [0] * n_episodes,
            "tasks": [["do something"]] * n_episodes,
        }
    )
    ep_meta.to_parquet(os.path.join(task_dir, "meta", "episodes", "0.parquet"), index=False)

    # data/chunk-000/file-000.parquet
    rng = np.random.default_rng(0)
    rows = []
    global_idx = 0
    for ep_idx in range(n_episodes):
        for _ in range(ep_len):
            row: dict = {"episode_index": ep_idx, "index": global_idx}
            for field, shape in action_fields:
                d = shape[0] if shape else 1
                row[field] = rng.standard_normal(d).astype(np.float32)
            if with_state:
                row["observation.state"] = rng.standard_normal(state_dim).astype(np.float32)
            rows.append(row)
            global_idx += 1
    pd.DataFrame(rows).to_parquet(os.path.join(task_dir, "data", "chunk-000", "file-000.parquet"), index=False)

    # videos/{camera}/ directory stubs (no real mp4 — patched in integration tests)
    for cam in cameras:
        os.makedirs(os.path.join(task_dir, "videos", cam, "chunk-000"), exist_ok=True)

    return task_dir


@contextmanager
def _mock_video(n_total_frames: int = 10000, height: int = 8, width: int = 8):
    """Patch video frame loading to return black PIL images without real mp4 files."""

    def _fake_frame_map(video_dir: str, camera: str):
        return [("fake.mp4", 0, n_total_frames)]

    def _fake_decode(path: str, frame_indices, h: int, w: int):
        return [Image.new("RGB", (w, h)) for _ in frame_indices]

    with (
        patch("openwam.dataloader.lerobot_v3_base._get_video_frame_map", _fake_frame_map),
        patch("openwam.dataloader.lerobot_v3_base._decode_video_frames", _fake_decode),
    ):
        yield


# ---------------------------------------------------------------------------
# _flatten_layout
# ---------------------------------------------------------------------------


def test_flatten_layout_flat_list():
    from openwam.dataloader.lerobot_v3_base import _flatten_layout

    cams = ["cam_a", "cam_b", "cam_c"]
    assert _flatten_layout(cams) == cams


def test_flatten_layout_2d_list():
    from openwam.dataloader.lerobot_v3_base import _flatten_layout

    layout = [["cam_a", "cam_b"], ["cam_c", "cam_d"]]
    assert _flatten_layout(layout) == ["cam_a", "cam_b", "cam_c", "cam_d"]


def test_flatten_layout_empty():
    from openwam.dataloader.lerobot_v3_base import _flatten_layout

    assert _flatten_layout([]) == []


# ---------------------------------------------------------------------------
# ActionComposer
# ---------------------------------------------------------------------------


def test_action_composer_single_field_shape():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info = {"action": {"shape": [14], "dtype": "float32"}}
    comp = ActionComposer(["action"], info)
    assert comp.action_dim == 14

    T = 5
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"action": [rng.standard_normal(14).astype(np.float32) for _ in range(T)]})
    out = comp.extract(df)
    assert out.shape == (T, 14)
    assert out.dtype == np.float32


def test_action_composer_multi_field_concat():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info = {
        "action.left_arm": {"shape": [6], "dtype": "float32"},
        "action.right_arm": {"shape": [6], "dtype": "float32"},
        "action.left_gripper": {"shape": [1], "dtype": "float32"},
    }
    comp = ActionComposer(["action.left_arm", "action.right_arm", "action.left_gripper"], info)
    assert comp.action_dim == 13

    T = 4
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "action.left_arm": [rng.standard_normal(6).astype(np.float32) for _ in range(T)],
            "action.right_arm": [rng.standard_normal(6).astype(np.float32) for _ in range(T)],
            "action.left_gripper": rng.standard_normal(T).astype(np.float32),
        }
    )
    out = comp.extract(df)
    assert out.shape == (T, 13)


def test_action_composer_missing_field_raises():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    with pytest.raises(KeyError, match="action.missing"):
        ActionComposer(["action.missing"], {"action": {"shape": [14]}})


def test_action_composer_scalar_reshaped_to_column():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info = {"grip": {"shape": [1], "dtype": "float32"}}
    comp = ActionComposer(["grip"], info)
    assert comp.action_dim == 1

    T = 3
    df = pd.DataFrame({"grip": np.array([0.1, 0.5, 0.9], dtype=np.float32)})
    out = comp.extract(df)
    assert out.shape == (T, 1)


# ---------------------------------------------------------------------------
# assemble_multiview_layout
# ---------------------------------------------------------------------------


def test_assemble_multiview_lshape_canvas_size():
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    cameras = ["top", "bot_l", "bot_r"]
    frames = {c: Image.new("RGB", (100, 100), color=(255, 0, 0)) for c in cameras}
    out = assemble_multiview_layout(frames, cameras, out_h=120, out_w=160)
    assert out.size == (160, 120)


def test_assemble_multiview_missing_camera_black_fill():
    """assemble_multiview_layout leaves a black region for any missing key.

    In normal __getitem__ flow the caller pre-pads short cameras with the last
    frame, so missing keys only occur on genuine data-absence.  The function
    itself still produces black for any absent camera_layout key.
    """
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    cameras = ["top", "bot_l", "bot_r"]
    # Only provide top camera; bot_l and bot_r are missing → black region
    frames = {"top": Image.new("RGB", (100, 100), color=(200, 200, 200))}
    out = assemble_multiview_layout(frames, cameras, out_h=60, out_w=60)
    arr = np.array(out)
    top_h = int(round(60 * 2 / 3))
    bottom_region = arr[top_h:, :, :]
    assert bottom_region.max() == 0, "Missing cameras should produce black regions"


def test_assemble_multiview_wrong_camera_count_raises():
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    with pytest.raises(ValueError, match="3 cameras"):
        assemble_multiview_layout({}, ["cam_a", "cam_b"], out_h=60, out_w=60)


# ---------------------------------------------------------------------------
# _pad_actions / _pad_video  (tested via dataset instance, no __getitem__ call)
# ---------------------------------------------------------------------------


def _make_ds(tmpdir, normalize_mode="none"):
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    _create_fake_task(tmpdir, n_episodes=2, ep_len=20)
    return LeRobot3Dataset(
        data_root=tmpdir,
        action_fields=["action"],
        target_camera="head",
        cameras=["head"],
        camera_layout=None,
        num_frames=9,
        height=16,
        width=16,
        split="train",
        val_ratio=0.0,
        multiview=False,
        normalize_mode=normalize_mode,
    )


def test_pad_actions_short():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds(tmpdir)
        actions = np.ones((5, 14), dtype=np.float32)
        padded, mask = ds._pad_actions(actions, target=10)
        assert padded.shape == (10, 14)
        assert mask[:5].all()
        assert not mask[5:].any()
        # Padded tail repeats last real action
        assert (padded[5:] == actions[-1]).all()


def test_pad_actions_exact():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds(tmpdir)
        actions = np.ones((8, 14), dtype=np.float32)
        padded, mask = ds._pad_actions(actions, target=8)
        assert padded.shape == (8, 14)
        assert mask.all()


def test_pad_actions_long():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds(tmpdir)
        actions = np.ones((15, 14), dtype=np.float32)
        padded, mask = ds._pad_actions(actions, target=8)
        assert padded.shape == (8, 14)
        assert mask.all()


def test_pad_video_short():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds(tmpdir)
        frames = [Image.new("RGB", (16, 16)) for _ in range(3)]
        padded, mask = ds._pad_video(frames, target=7)
        assert len(padded) == 7
        assert mask[:3].all()
        assert not mask[3:].any()


def test_pad_video_exact():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds(tmpdir)
        frames = [Image.new("RGB", (16, 16)) for _ in range(5)]
        padded, mask = ds._pad_video(frames, target=5)
        assert len(padded) == 5
        assert mask.all()


# ---------------------------------------------------------------------------
# _normalize / denormalize_action
# ---------------------------------------------------------------------------


def _make_ds_with_stats(tmpdir, normalize_mode: str, mean=0.0, std=1.0, low=-1.0, high=1.0):
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    _create_fake_task(tmpdir, n_episodes=2, ep_len=20)
    return LeRobot3Dataset(
        data_root=tmpdir,
        action_fields=["action"],
        target_camera="head",
        cameras=["head"],
        num_frames=9,
        height=16,
        width=16,
        split="train",
        val_ratio=0.0,
        multiview=False,
        normalize_mode=normalize_mode,
    )


def test_normalize_none_is_noop():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds(tmpdir, normalize_mode="none")
        actions = np.array([[0.5] * 14], dtype=np.float32)
        out = ds._normalize(actions)
        np.testing.assert_array_equal(out, actions)


def test_normalize_minmax_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds_with_stats(tmpdir, normalize_mode="min-max")
        # With min=-1, max=1 stats and input in [-1, 1], output must be in [-1, 1]
        rng = np.random.default_rng(0)
        actions = rng.uniform(-1.0, 1.0, (10, 14)).astype(np.float32)
        out = ds._normalize(actions)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6


def test_normalize_zscore_shifts_mean():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds_with_stats(tmpdir, normalize_mode="z-score")
        # With mean=0, std=1 stats, the output should equal the input
        actions = np.array([[1.0, 2.0, -1.0] + [0.0] * 11], dtype=np.float32)
        out = ds._normalize(actions)
        np.testing.assert_allclose(out, actions, atol=1e-5)


def test_denormalize_minmax_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds_with_stats(tmpdir, normalize_mode="min-max")
        rng = np.random.default_rng(2)
        original = rng.uniform(-1.0, 1.0, (5, 14)).astype(np.float32)
        normed = ds._normalize(original)
        recovered = ds.denormalize_action(normed)
        np.testing.assert_allclose(recovered, original, atol=1e-5)


def test_denormalize_zscore_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_ds_with_stats(tmpdir, normalize_mode="z-score")
        rng = np.random.default_rng(3)
        original = rng.standard_normal((5, 14)).astype(np.float32)
        normed = ds._normalize(original)
        recovered = ds.denormalize_action(normed)
        np.testing.assert_allclose(recovered, original, atol=1e-5)


# ---------------------------------------------------------------------------
# LeRobot3Dataset integration tests
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "video",
    "vace_video",
    "first_frame_image",
    "action",
    "action_mask",
    "video_mask",
    "proprio",
    "proprio_mask",
    "prompt",
    "episode_index",
    "episode_path",
    "start_frame",
    "end_frame",
    "episode_length",
    "task_name",
    "proprio_source",
}


def test_getitem_return_dict_keys():
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=20)
        ds = LeRobot3Dataset(
            data_root=tmpdir,
            action_fields=["action"],
            target_camera="head",
            cameras=["head"],
            num_frames=9,
            height=16,
            width=16,
            split="train",
            val_ratio=0.0,
            multiview=False,
            normalize_mode="none",
        )
        with _mock_video():
            sample = ds[0]
    assert _REQUIRED_KEYS.issubset(sample.keys()), f"Missing keys: {_REQUIRED_KEYS - sample.keys()}"


def test_getitem_action_and_proprio_shapes():
    """Window split: proprio=(1,D), action=(num_frames-1,D)."""
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    num_frames = 9
    action_dim = 14

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=30)
        ds = LeRobot3Dataset(
            data_root=tmpdir,
            action_fields=["action"],
            target_camera="head",
            cameras=["head"],
            num_frames=num_frames,
            height=16,
            width=16,
            split="train",
            val_ratio=0.0,
            multiview=False,
            normalize_mode="none",
        )
        with _mock_video():
            sample = ds[0]

    assert sample["proprio"].shape == (1, action_dim)
    assert sample["proprio_mask"].shape == (1,)
    assert sample["action"].shape == (num_frames - 1, action_dim)
    assert sample["action_mask"].shape == (num_frames - 1,)


def test_getitem_proprio_source_action_fallback():
    """Without observation.state in info.json, proprio_source should be 'action_fallback'."""
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=20, with_state=False)
        ds = LeRobot3Dataset(
            data_root=tmpdir,
            action_fields=["action"],
            target_camera="head",
            cameras=["head"],
            num_frames=9,
            height=16,
            width=16,
            split="train",
            val_ratio=0.0,
            multiview=False,
            normalize_mode="none",
        )
        assert ds._state_composer is None
        with _mock_video():
            sample = ds[0]

    assert sample["proprio_source"] == "action_fallback"


def test_getitem_proprio_source_native_with_observation_state():
    """With observation.state in info.json, proprio_source should be 'native'."""
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    # state_dim must equal action_dim (14) for the dimension check to pass
    state_dim = 14
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=20, with_state=True, state_dim=state_dim)
        ds = LeRobot3Dataset(
            data_root=tmpdir,
            action_fields=["action"],
            target_camera="head",
            cameras=["head"],
            num_frames=9,
            height=16,
            width=16,
            split="train",
            val_ratio=0.0,
            multiview=False,
            normalize_mode="none",
        )
        assert ds._state_composer is not None
        with _mock_video():
            sample = ds[0]

    assert sample["proprio_source"] == "native"
    assert sample["proprio"].shape == (1, state_dim)


def test_normalize_mode_raises_when_eef_stats_missing():
    """normalize_mode + action_transform with no eef_stats.json must raise at init.
    Auto-compute was removed to prevent DDP multi-rank file write races."""
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=20)
        with pytest.raises(RuntimeError, match="action_stats"):
            LeRobot3Dataset(
                data_root=tmpdir,
                action_fields=["action"],
                target_camera="head",
                cameras=["head"],
                num_frames=9,
                height=16,
                width=16,
                split="train",
                val_ratio=0.0,
                multiview=False,
                normalize_mode="min-max",
                action_transform=lambda x: x,
                action_out_dim=14,
                action_stats_path=None,
            )


# ---------------------------------------------------------------------------
# MixtureDataset
# ---------------------------------------------------------------------------


class _FakeDS:
    """Minimal fake dataset for MixtureDataset tests (no disk I/O)."""

    def __init__(self, n: int, action_dim: int, stats: dict | None = None):
        self._n = n
        self._action_dim = action_dim
        self._stats = stats

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "action": torch.zeros(self._action_dim),
            "action_mask": torch.ones(self._action_dim, dtype=torch.bool),
        }

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_stats(self) -> dict | None:
        return self._stats

    def denormalize_action(self, x):
        return x


def _fake_stats(dim: int, mean: float = 0.0, std: float = 1.0) -> dict:
    return {
        "mean": np.full(dim, mean, dtype=np.float32),
        "std": np.full(dim, std, dtype=np.float32),
        "min": np.full(dim, -1.0, dtype=np.float32),
        "max": np.full(dim, 1.0, dtype=np.float32),
    }


def test_mixture_virtual_len_covers_total_real():
    """Total virtual samples should equal round(total_real * w) summed."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(100, 14)
    ds_b = _FakeDS(200, 14)
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    total_real = 300
    # Each dataset gets ~round(total_real * 0.5) = 150 virtual samples
    # Total virtual = 300
    assert len(mix) == total_real


def test_mixture_dispatch_reaches_all_subdatasets():
    """Every sub-dataset index should appear in the virtual index map."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(50, 14)
    ds_b = _FakeDS(50, 14)
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    seen_ds = {di for di, _ in mix._index_map}
    assert seen_ds == {0, 1}


def test_mixture_stats_none_when_any_subdataset_has_no_stats():
    """If any sub-dataset has action_stats=None, mixture stats should be None."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(50, 14, stats=_fake_stats(14))
    ds_b = _FakeDS(50, 14, stats=None)  # no stats
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    assert mix.action_stats is None


def test_mixture_stats_computed_when_all_have_stats():
    """If all sub-datasets have stats, mixture stats should be non-None."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(50, 14, stats=_fake_stats(14))
    ds_b = _FakeDS(50, 14, stats=_fake_stats(14))
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    assert mix.action_stats is not None
    assert "mean" in mix.action_stats and "std" in mix.action_stats


def test_mixture_action_dim_padding():
    """action from a smaller-dim sub-dataset must be zero-padded to action_dim_override."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_small = _FakeDS(10, 14)  # 14-D
    ds_large = _FakeDS(10, 20)  # 20-D
    mix = MixtureDataset([ds_small, ds_large], weights=[1.0, 1.0], action_dim_override=20)

    # Find an index that dispatches to ds_small (di=0)
    target_idx = next(i for i, (di, _) in enumerate(mix._index_map) if di == 0)
    sample = mix[target_idx]
    assert sample["action"].shape[-1] == 20
    # Padded dims should be zero
    assert (sample["action"][14:] == 0.0).all()


# ---------------------------------------------------------------------------
# quat → rot6d correctness (critical path for EEF transform)
# ---------------------------------------------------------------------------


def test_eef_transform_identity_pose_wxyz():
    """Identity quaternion [1,0,0,0] (wxyz) should produce rot6d = [1,0,0, 0,1,0]."""
    from openwam.dataloader.agibot import make_eef_transform

    transform = make_eef_transform(quat_convention="wxyz")

    # 1-frame 40-D action, only left/right quaternions set to identity.
    # Layout: [gripper_l(1), gripper_r(1), xyz_l(3), xyz_r(3), q_l(4), q_r(4), ...]
    action = np.zeros((1, 40), dtype=np.float32)
    action[0, 8:12] = [1, 0, 0, 0]  # q_l wxyz identity
    action[0, 12:16] = [1, 0, 0, 0]  # q_r wxyz identity

    out = transform(action)  # (1, 20)
    assert out.shape == (1, 20)

    # left rot6d at cols 3:9, right rot6d at cols 13:19
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(out[0, 3:9], identity_rot6d, atol=1e-5)
    np.testing.assert_allclose(out[0, 13:19], identity_rot6d, atol=1e-5)


def test_eef_transform_identity_pose_xyzw():
    """Identity quaternion [0,0,0,1] (xyzw) should produce rot6d = [1,0,0, 0,1,0]."""
    from openwam.dataloader.agibot import make_eef_transform

    transform = make_eef_transform(quat_convention="xyzw")

    action = np.zeros((1, 40), dtype=np.float32)
    action[0, 8:12] = [0, 0, 0, 1]  # q_l xyzw identity
    action[0, 12:16] = [0, 0, 0, 1]  # q_r xyzw identity

    out = transform(action)
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(out[0, 3:9], identity_rot6d, atol=1e-5)
    np.testing.assert_allclose(out[0, 13:19], identity_rot6d, atol=1e-5)


# ---------------------------------------------------------------------------
# Boundary: single-frame episode / last-frame pad
# ---------------------------------------------------------------------------


def _make_ds_ep(tmpdir, ep_len: int, num_frames: int = 9):
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    return LeRobot3Dataset(
        data_root=tmpdir,
        action_fields=["action"],
        target_camera="head",
        cameras=["head"],
        num_frames=num_frames,
        height=16,
        width=16,
        split="train",
        val_ratio=0.0,
        multiview=False,
        normalize_mode="none",
    )


def test_single_frame_episode_pads_correctly():
    """Single-frame episode: action should be all-pad with mask all False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=1)
        ds = _make_ds_ep(tmpdir, ep_len=1)
        with _mock_video():
            sample = ds[0]
        # action shape: (num_frames-1, D)
        assert sample["action"].shape[0] == 8
        # No real actions in a single-frame episode — mask should be all False
        assert not sample["action_mask"].any()


def test_last_frame_pad_repeats_last_action():
    """Episode shorter than num_frames: padded tail should repeat the last real action."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_task(tmpdir, n_episodes=2, ep_len=5)
        ds = _make_ds_ep(tmpdir, ep_len=5)
        with _mock_video():
            sample = ds[0]
        action = sample["action"]  # (8, D)
        mask = sample["action_mask"]  # (8,)
        # ep_len=5: proprio=raw[0], action=raw[1:5] → 4 valid frames, 4 padded
        assert mask[:4].all()
        assert not mask[4:].any()
        # Padded frames must equal the last valid action frame
        last = action[3].numpy()
        np.testing.assert_array_equal(action[4:].numpy(), np.tile(last, (4, 1)))


# ---------------------------------------------------------------------------
# MultiTaskLeRobot3Dataset bisect boundary
# ---------------------------------------------------------------------------


def test_multitask_bisect_first_and_last_index():
    """idx=0 should dispatch to the first sub-dataset; idx=total-1 to the last."""
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _create_fake_task(d1, n_episodes=2, ep_len=20)
        _create_fake_task(d2, n_episodes=2, ep_len=20)

        ds1 = _make_ds_ep(d1, ep_len=20)
        ds2 = _make_ds_ep(d2, ep_len=20)
        multi = MultiTaskLeRobot3Dataset([ds1, ds2])

        total = len(multi)
        assert total == len(ds1) + len(ds2)

        with _mock_video():
            # idx=0 → first sample of ds1
            s0 = multi[0]
            assert s0["task_name"] == ds1.task_name

            # idx=total-1 → last sample of ds2
            s_last = multi[total - 1]
            assert s_last["task_name"] == ds2.task_name
