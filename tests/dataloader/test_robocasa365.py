"""Tests for the compact RoboCasa365 state19/action15 reader."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import openwam.dataloader.robocasa365 as rc
from openwam.dataloader.robocasa365 import ACTION_DIM, STATE_DIM, RoboCasa365Dataset

EP_LENGTH = 40
PROMPT = "Open the right drawer."
HEAD = rc.HEAD_CAMERA
WRIST = rc.WRIST_CAMERA
RIGHT = rc.RIGHT_CAMERA


def _stats(dim: int) -> dict:
    return {
        "mean": np.zeros(dim, np.float32),
        "std": np.ones(dim, np.float32),
        "min": -np.ones(dim, np.float32),
        "max": np.ones(dim, np.float32),
        "q01": -np.ones(dim, np.float32),
        "q99": np.ones(dim, np.float32),
    }


def _make_repo(root: Path, tasks=("OpenDrawer",), episodes_per_task: int = 2) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    total_episodes = len(tasks) * episodes_per_task
    info = {
        "codebase_version": "v3.0",
        "robot_type": "PandaOmron",
        "fps": 20,
        "total_episodes": total_episodes,
        "total_frames": total_episodes * EP_LENGTH,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            HEAD: {"dtype": "video", "shape": [256, 256, 3]},
            WRIST: {"dtype": "video", "shape": [256, 256, 3]},
            RIGHT: {"dtype": "video", "shape": [256, 256, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    with (root / "meta" / "robocasa365_normalization_stats.npy").open("wb") as handle:
        np.save(handle, {rc.ACTION_STATS_KEY: _stats(ACTION_DIM), rc.STATE_STATS_KEY: _stats(STATE_DIM)})

    state_rows, action_rows, episode_rows = [], [], []
    offset = episode_index = 0
    for task_index, task in enumerate(tasks):
        for _ in range(episodes_per_task):
            state = np.zeros((EP_LENGTH, STATE_DIM), np.float32)
            state[:, 0] = np.arange(EP_LENGTH) + 100 * episode_index
            state[:, 3] = state[:, 7] = 1.0
            state[:, 9] = 1.0
            state[:, 10:13] = [1.0, 2.0, 3.0]
            state[:, 13] = state[:, 17] = 1.0
            action = np.zeros((EP_LENGTH, ACTION_DIM), np.float32)
            action[:, 0] = state[:, 0] + 0.025
            action[:, 3] = action[:, 7] = 1.0
            action[:, 9] = -1.0
            action[:, 10:13] = [0.1, -0.2, 0.3]
            action[:, 13] = 0.0
            action[:, 14] = np.where(np.arange(EP_LENGTH) % 2 == 0, -1.0, 1.0)
            state_rows.extend(state)
            action_rows.extend(action)
            row = {
                "episode_index": episode_index,
                "dataset_from_index": offset,
                "dataset_to_index": offset + EP_LENGTH,
                "length": EP_LENGTH,
                "tasks": [PROMPT],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "source_prefix": f"pretrain/atomic/{task}/20250820",
                "source_episode_index": episode_index,
            }
            for camera in (HEAD, WRIST, RIGHT):
                row[f"videos/{camera}/chunk_index"] = 0
                row[f"videos/{camera}/file_index"] = 0
                row[f"videos/{camera}/from_timestamp"] = offset / 20
                row[f"videos/{camera}/to_timestamp"] = (offset + EP_LENGTH) / 20
            episode_rows.append(row)
            offset += EP_LENGTH
            episode_index += 1
    pd.DataFrame({"observation.state": state_rows, "action": action_rows}).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet"
    )
    pd.DataFrame(episode_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for camera in (HEAD, WRIST, RIGHT):
        path = root / "videos" / camera / "chunk-000"
        path.mkdir(parents=True)
        (path / "file-000.mp4").write_bytes(b"")
    return root


@contextmanager
def _mock_video():
    def decode(path, frame_indices, height, width):
        if RIGHT in path:
            color = (0, 0, 255)
        elif WRIST in path:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        return [Image.new("RGB", (width, height), color) for _ in frame_indices]

    with patch.object(rc, "decode_video_frames", side_effect=decode):
        yield


def _dataset(tmp_path: Path, **kwargs) -> RoboCasa365Dataset:
    root = _make_repo(tmp_path / "repo")
    kwargs.setdefault("task_name", "OpenDrawer")
    kwargs.setdefault("normalize_mode", None)
    kwargs.setdefault("multiview", False)
    kwargs.setdefault("height", 64)
    kwargs.setdefault("width", 96)
    return RoboCasa365Dataset(str(root), **kwargs)


def test_compact_nonunified_shapes_and_same_row_action(tmp_path):
    with _mock_video():
        dataset = _dataset(tmp_path)
        sample = dataset._build_sample(0, 1)
    assert sample["proprio"].shape == (1, STATE_DIM)
    assert sample["action"].shape == (32, ACTION_DIM)
    # State row 1 is x=1 and action row 1 is target x=1.025. A next-state
    # implementation would have produced/read x around 2 here.
    np.testing.assert_allclose(sample["proprio"][0, 0], 1.0)
    np.testing.assert_allclose(sample["action"][0, 0], 1.025)


def test_unified_maps_and_masks_are_asymmetric(tmp_path):
    with _mock_video():
        dataset = _dataset(
            tmp_path,
            unify_action=True,
            unify_action_map=["0-9", "68-72"],
            unify_state_map=["0-9", "68-76"],
        )
        sample = dataset[0]
    assert sample["action"].shape == (32, 80)
    assert sample["proprio"].shape == (1, 80)
    action_mask = sample["action_mask"][0].numpy()
    state_mask = sample["proprio_mask"][0].numpy()
    assert action_mask.sum() == 15 and action_mask[:10].all() and action_mask[68:73].all()
    assert state_mask.sum() == 19 and state_mask[:10].all() and state_mask[68:77].all()
    assert action_mask[71]  # torso is retained
    assert not action_mask[73:77].any()
    assert state_mask[73:77].all()


def test_action_and_state_values_land_in_requested_slots(tmp_path):
    with _mock_video():
        dataset = _dataset(tmp_path, unify_action=True)
        sample = dataset[0]
    action = sample["action"][0].numpy()
    state = sample["proprio"][0].numpy()
    np.testing.assert_allclose(action[68:73], [0.1, -0.2, 0.3, 0.0, -1.0])
    np.testing.assert_allclose(state[68:71], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state[71:77], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_separate_normalization_and_denormalization(tmp_path):
    root = _make_repo(tmp_path / "repo")
    with _mock_video():
        dataset = RoboCasa365Dataset(
            str(root),
            task_name="OpenDrawer",
            normalize_mode="min-max",
            multiview=False,
            height=64,
            width=96,
            unify_action=True,
        )
        sample = dataset[0]
    raw = dataset.denormalize_action(sample["action"].numpy())
    assert raw.shape == (32, ACTION_DIM)
    np.testing.assert_allclose(raw[0, 10:15], [0.1, -0.2, 0.3, 0.0, -1.0], atol=1e-6)


def test_prompt_and_video_contract(tmp_path):
    with _mock_video():
        dataset = _dataset(tmp_path, multiview=True, height=384, width=320)
        sample = dataset[0]
    assert sample["prompt"] == PROMPT
    assert len(sample["video"]) == 9
    assert sample["video"][0].size == (320, 384)
    pixels = np.asarray(sample["video"][0])
    np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(pixels[300, 0], [0, 255, 0])
    np.testing.assert_array_equal(pixels[300, 319], [0, 0, 255])


def test_rejects_native_unconverted_schema(tmp_path):
    root = _make_repo(tmp_path / "repo")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["shape"] = [16]
    info["features"]["action"]["shape"] = [12]
    info_path.write_text(json.dumps(info))
    try:
        RoboCasa365Dataset(str(root), normalize_mode=None)
    except ValueError as error:
        assert "state19/action15" in str(error)
    else:
        raise AssertionError("native schema should be rejected")


def test_multiview_rejects_dataset_without_right_agentview(tmp_path):
    root = _make_repo(tmp_path / "repo")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"][RIGHT]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="robot0_agentview_right"):
        RoboCasa365Dataset(str(root), normalize_mode=None, multiview=True)
