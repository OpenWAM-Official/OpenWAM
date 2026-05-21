"""Shared fake-data helpers for LeRobot v3 dataloader test files.

This module is intentionally not a test module (no ``test_`` prefix) so pytest
ignores it during collection. It is imported by:

  - tests/test_lerobot_v3_core.py
  - tests/test_lerobot_v3_discovery.py
  - tests/test_lerobot_v3_stats.py
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
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


def _make_lerobot3(data_root, **overrides):
    """LeRobot3Dataset with test defaults; pass overrides for any kwarg."""
    from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset

    defaults = dict(
        action_fields=["action"], target_camera="head", cameras=["head"],
        num_frames=9, height=16, width=16, split="train", val_ratio=0.0,
        multiview=False, normalize_mode="none",
    )
    defaults.update(overrides)
    return LeRobot3Dataset(data_root=str(data_root), **defaults)


def _fake_task_and_ds(tmp_path, *, task_kwargs=None, **ds_overrides):
    """Create a fake task at tmp_path then build LeRobot3Dataset. task_kwargs
    are forwarded to _create_fake_task (defaults: n_episodes=2, ep_len=20)."""
    kw = dict(n_episodes=2, ep_len=20)
    if task_kwargs:
        kw.update(task_kwargs)
    _create_fake_task(str(tmp_path), **kw)
    return _make_lerobot3(tmp_path, **ds_overrides)


def _mkdir_subs(tmp_path, *names):
    """Create named subdirs under tmp_path and return the Path objects."""
    out = []
    for n in names:
        sub = tmp_path / n
        sub.mkdir()
        out.append(sub)
    return out


# ---------------------------------------------------------------------------
# OXE helpers
# ---------------------------------------------------------------------------


_OXE_RAW_FIELDS = [
    ("action.cartesian_position", [6]),
    ("action.gripper_position", [1]),
]
_OXE_CAM = "observation.images.exterior_1_left"


def _oxe_subset(name, dataset_dir, cam, *, enabled=True, **overrides):
    """Build an OXE subset entry dict with the common camera/dir scaffolding."""
    base = dict(name=name, enabled=enabled, dataset_dir=str(dataset_dir),
                target_camera=cam, camera_layout=[cam])
    base.update(overrides)
    return base


def _make_oxe_task(path, *, cam=_OXE_CAM):
    """Create a fake task wired for the OXE raw-fields/camera layout."""
    return _create_fake_task(
        str(path), n_episodes=2, ep_len=20, cameras=[cam], action_fields=_OXE_RAW_FIELDS,
    )


def _oxe_config(subsets, defaults=None):
    """Build a minimal OXE yaml-like config (flat top-level + subsets list).

    Top-level fields (the test "defaults") are merged with sensible test-only
    values so individual tests only override what they care about. The schema
    matches the real configs/dataloader/oxe.yaml: shared fields live at the
    top level (not in a nested ``defaults:`` block — Hydra reserves that key).
    """
    base_defaults = dict(
        type="oxe", action_format="eef",
        action_fields=["action.cartesian_position", "action.gripper_position"],
        multiview=False, num_frames=9, height=16, width=16,
        val_ratio=0.0, seed=42, video_stride=1, window_stride=1, repeat=1,
        fps=30, normalize_mode=None, action_stats_path=None,
    )
    if defaults:
        base_defaults.update(defaults)
    base_defaults["subsets"] = subsets
    return SimpleNamespace(**base_defaults)


# ---------------------------------------------------------------------------
# Union stats helpers (used by multiple test files)
# ---------------------------------------------------------------------------


def _write_union_stats_npy(path: str, dim: int = 20, mode_key: str = "eef") -> str:
    """Write a deploy-compatible nested-schema .npy with all-zero stats."""
    zeros = np.zeros(dim, dtype=np.float32).tolist()
    ones = np.ones(dim, dtype=np.float32).tolist()
    stats = {
        mode_key: {"mean": zeros, "std": ones, "min": [-v for v in ones], "max": ones},
        "num_timesteps": 1000,
    }
    np.save(path, stats, allow_pickle=True)
    # np.save appends ".npy" if missing — normalize so caller's path is correct
    if not os.path.exists(path) and os.path.exists(path + ".npy"):
        os.rename(path + ".npy", path)
    return path
