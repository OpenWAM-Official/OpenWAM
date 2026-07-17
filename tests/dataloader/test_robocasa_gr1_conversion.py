from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.robocasa_gr1 import RoboCasaGR1Dataset
from scripts.convert_robocasa_gr1_v20_to_v30 import convert_bucket, discover_buckets

VIDEO_KEY = "observation.images.ego_view"
EP_LENGTH = 6


def _write_v20_bucket(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "GR1ArmsAndWaistFourierHands",
        "total_episodes": 1,
        "total_frames": EP_LENGTH,
        "total_tasks": 2,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20.0,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {"dtype": "video", "shape": [256, 256, 3]},
            "observation.state": {"dtype": "object", "shape": [44]},
            "action": {"dtype": "object", "shape": [44]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.coarse_action": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": EP_LENGTH, "tasks": ["pick the squash"]}) + "\n"
    )
    (root / "meta" / "tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": ""}),
                json.dumps({"task_index": 1, "task": "pick the squash"}),
            ]
        )
        + "\n"
    )
    (root / "meta" / "modality.json").write_text(json.dumps({"state": {}, "action": {}}))
    state = np.arange(EP_LENGTH * 44, dtype=np.float32).reshape(EP_LENGTH, 44) / 100
    action = state + 0.1
    frame = pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "task_index": np.ones(EP_LENGTH, dtype=np.int64),
            "episode_index": np.zeros(EP_LENGTH, dtype=np.int64),
            "annotation.human.coarse_action": np.full(EP_LENGTH, 6, dtype=np.int64),
        }
    )
    pq.write_table(pa.Table.from_pandas(frame), root / "data" / "chunk-000" / "episode_000000.parquet")
    (root / "videos" / "chunk-000" / VIDEO_KEY / "episode_000000.mp4").write_bytes(b"video")


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height), (100, 150, 200)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def test_converter_reindexes_v20_without_copying_payloads(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_v20_bucket(source)
    report = convert_bucket(source, output)

    assert report["native_action_dim"] == 44
    assert report["representation"] == "native_joint"
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["splits"] == {"train": "0:1"}
    assert info["data_path"].endswith("file-{file_index:03d}.parquet")
    assert os.stat(source / "data/chunk-000/episode_000000.parquet").st_ino == os.stat(
        output / "data/chunk-000/file-000.parquet"
    ).st_ino

    episodes = pd.read_parquet(output / "meta" / "episodes" / "chunk-000.parquet")
    assert episodes.loc[0, "dataset_from_index"] == 0
    assert episodes.loc[0, f"videos/{VIDEO_KEY}/file_index"] == 0
    tasks = pd.read_parquet(output / "meta" / "tasks.parquet")
    assert tasks.index.tolist() == ["pick the squash"]


def test_converted_real_schema_loads_joint44_and_task_prompt(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_v20_bucket(source)
    convert_bucket(source, output)
    config = OmegaConf.create(
        {
            "dataset_dir": str(output),
            "action_mode": "joint",
            "action_dim": 44,
            # Deliberately include the native integer annotation: the reader
            # must reject it as text and fall back through task_index.
            "prompt_columns": ["annotation.human.coarse_action"],
            "num_frames": 5,
            "video_stride": 1,
            "height": 384,
            "width": 320,
            "multiview": True,
            "normalize_mode": None,
        }
    )
    with _mock_decoder():
        dataset = RoboCasaGR1Dataset.from_config(config)
        sample = dataset[0]
    assert sample["action"].shape == (4, 44)
    assert sample["proprio"].shape == (1, 44)
    assert sample["prompt"] == "pick the squash"
    image = np.asarray(sample["video"][0])
    assert np.any(image[:256] != 0)
    assert not np.any(image[256:] != 0)


def test_discover_multibucket_root(tmp_path: Path):
    _write_v20_bucket(tmp_path / "a")
    _write_v20_bucket(tmp_path / "b")
    assert [path.name for path in discover_buckets(tmp_path)] == ["a", "b"]


def test_shipped_configs_separate_native_joint_and_eef_unify():
    native = OmegaConf.load("configs/dataloader/robocasa_gr1.yaml")
    assert native.action_mode == "joint"
    assert native.action_dim == 44
    assert native.unify_action is False
    assert native.unify_action_map is None
    assert list(native.prompt_columns) == []

    unified = OmegaConf.load("configs/dataloader/robocasa_gr1_unify.yaml")
    assert unified.action_mode == "unify"
    assert unified.action_dim == 20
    assert unified.unify_action is True
    assert list(unified.unify_action_map) == ["0-9", "34-43"]


def test_eef_unify_profile_fails_fast_on_native_joint44(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_v20_bucket(source)
    convert_bucket(source, output)
    config = OmegaConf.load("configs/dataloader/robocasa_gr1_unify.yaml")
    config.dataset_dir = str(output)
    with pytest.raises(KeyError, match="native joint44 only"):
        RoboCasaGR1Dataset.from_config(config)
