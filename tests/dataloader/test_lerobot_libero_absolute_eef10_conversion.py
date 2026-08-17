from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert_lerobot_libero_to_absolute_eef10_v3 import (
    DEFAULT_OUTPUT_ROOT,
    EEF10_NAMES,
    POSITION_SCALE,
    ROTATION_SCALE,
    convert_dataset,
    convert_state_action,
)


def test_official_lerobot_converter_owns_canonical_default_path() -> None:
    assert DEFAULT_OUTPUT_ROOT == Path("/path/to/libero")


def test_native_continuous_libero_action_converts_and_roundtrips() -> None:
    state = np.array(
        [
            [0.1, -0.2, 0.7, 3.1, 0.1, -0.2, 0.04, -0.04],
            [0.2, 0.3, 0.8, 2.9, -0.3, 0.2, 0.02, -0.02],
            [-0.1, 0.1, 0.6, 3.0, 0.2, 0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    action = np.array(
        [
            [0.5, -0.25, 0.1, 0.2, -0.1, 0.05, -1.0],
            [-0.2, 0.4, -0.3, -0.1, 0.15, -0.2, 0.35],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    state10, action10, errors = convert_state_action(state, action)

    assert state10.shape == action10.shape == (3, 10)
    np.testing.assert_allclose(action10[:, :3], state[:, :3] + action[:, :3] * POSITION_SCALE, atol=2e-7)
    np.testing.assert_array_equal(action10[:, 9], -action[:, 6])
    np.testing.assert_allclose(state10[:, 9], [1.0, 0.0, -1.0], atol=1e-7)
    assert errors["position"] < 2e-6
    assert errors["rotation"] < 2e-6
    assert errors["gripper"] == 0.0
    assert POSITION_SCALE == 0.05
    assert ROTATION_SCALE == 0.5


def _write_mock_source(root: Path) -> tuple[np.ndarray, np.ndarray]:
    state = np.array(
        [
            [0.0, 0.0, 0.5, 3.14, 0.0, 0.0, 0.04, -0.04],
            [0.01, 0.0, 0.5, 3.14, 0.01, 0.0, 0.03, -0.03],
            [0.0, 0.1, 0.6, 3.10, 0.0, 0.1, 0.02, -0.02],
            [0.0, 0.11, 0.6, 3.10, 0.0, 0.11, 0.01, -0.01],
        ],
        dtype=np.float32,
    )
    action = np.array(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, -0.25],
            [-0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 1.0],
        ],
        dtype=np.float32,
    )
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array([0.0, 0.1, 0.0, 0.1], type=pa.float32()),
            "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
            "index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(table, data_path)

    episodes = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "length": pa.array([2, 2], type=pa.int64()),
            "dataset_from_index": pa.array([0, 2], type=pa.int64()),
            "dataset_to_index": pa.array([2, 4], type=pa.int64()),
            "data/chunk_index": pa.array([0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(episodes, episodes_path)
    pq.write_table(pa.table({"task_index": [0], "task": ["move the object"]}), root / "meta/tasks.parquet")

    video_features = {}
    for key in ("observation.images.image", "observation.images.image2"):
        video_features[key] = {
            "dtype": "video",
            "shape": [2, 2, 3],
            "names": ["height", "width", "channel"],
            "fps": 10,
        }
        video = root / "videos" / key / "chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes((key + " mock video").encode())

    scalar_feature = {"dtype": "int64", "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            **video_features,
            "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"], "fps": 10},
            "action": {"dtype": "float32", "shape": [7], "names": ["actions"], "fps": 10},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": scalar_feature,
            "episode_index": scalar_feature,
            "index": scalar_feature,
            "task_index": scalar_feature,
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    base_stats = {
        key: {field: [0.0] for field in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")}
        for key in info["features"]
    }
    (root / "meta/stats.json").write_text(json.dumps(base_stats))
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n")
    return state, action


def test_single_dataset_conversion_preserves_packing_and_regenerates_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    state, action = _write_mock_source(source)
    source_episode_bytes = (source / "meta/episodes/chunk-000/file-000.parquet").read_bytes()

    result = convert_dataset(source, output, workers=2)

    assert result["frames"] == 4
    assert result["parquet_files"] == 1
    info = json.loads((output / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 10
    assert info["features"]["observation.state"]["shape"] == [10]
    assert info["features"]["action"]["shape"] == [10]
    assert info["features"]["action"]["names"]["motors"] == EEF10_NAMES

    converted = pq.read_table(output / "data/chunk-000/file-000.parquet")
    state10 = np.asarray(converted["observation.state"].to_pylist(), np.float32)
    action10 = np.asarray(converted["action"].to_pylist(), np.float32)
    expected_state, expected_action, _ = convert_state_action(state, action)
    np.testing.assert_array_equal(state10, expected_state)
    np.testing.assert_array_equal(action10, expected_action)
    np.testing.assert_array_equal(converted["episode_index"].to_numpy(), [0, 0, 1, 1])
    assert (output / "meta/episodes/chunk-000/file-000.parquet").read_bytes() == source_episode_bytes

    source_video = source / "videos/observation.images.image/chunk-000/file-000.mp4"
    output_video = output / "videos/observation.images.image/chunk-000/file-000.mp4"
    assert output_video.read_bytes() == source_video.read_bytes()
    assert output_video.stat().st_ino != source_video.stat().st_ino

    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["observation.state"]["count"] == [4]
    assert stats["action"]["count"] == [4]
    assert not (output / "meta/libero_normalization_stats.npy").exists()
    pooled = np.load(output / "meta/normalization_stats.npy", allow_pickle=True).item()["eef"]
    assert pooled["action_rows"] == pooled["state_rows"] == 4
    assert pooled["num_timesteps"] == 8
    assert pooled["gripper_convention"] == "minus1_closed_plus1_open"
    assert pooled["representation"] == "absolute_eef10"
    assert set(("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")) <= set(pooled)
    np.testing.assert_array_equal(np.asarray(pooled["mean"])[3:9], np.zeros(6))
    np.testing.assert_array_equal(np.asarray(pooled["std"])[3:9], np.ones(6))

    conversion = json.loads((output / "meta/conversion.json").read_text())
    assert conversion["gripper_formula"] == "open_scale = -source_native_close_scale"
    assert conversion["preserved"]["episode_metadata_and_offsets"] is True
