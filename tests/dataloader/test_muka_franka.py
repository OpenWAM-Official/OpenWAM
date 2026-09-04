from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.muka_franka import (
    EEF10_DIM,
    GRIPPER_CONVENTION,
    GRIPPER_TRANSFORM,
    RAW_GRIPPER_CONVENTION,
    ROT6D_DIMS_EEF10,
    ROTATION_CONVENTION,
    STATS_POPULATION,
    MukaFrankaDataset,
    closedness_to_open_scale,
    euler7_to_eef10,
)
from openwam.dataloader.registry import build_dataset, list_registered_datasets
from openwam.dataloader.utils.stats_computation.muka_franka_stats_computation import (
    _compute_global_stats,
)
from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec, unmap_from_unify

EP_LENGTH = 6
HEAD = "observation.images.left"
WRIST = "observation.images.left_wrist"
UNIFY_MAP = ["0-9"]


def _source_state() -> np.ndarray:
    state = np.zeros((EP_LENGTH, 7), dtype=np.float32)
    state[:, :3] = np.arange(EP_LENGTH, dtype=np.float32)[:, None] * np.array([0.01, -0.02, 0.03], dtype=np.float32)
    state[:, 3:6] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2],
            [0.1, -0.2, 0.3],
            [-0.3, 0.2, -0.1],
            [0.4, 0.1, -0.2],
            [0.4, 0.1, -0.2],
        ],
        dtype=np.float32,
    )
    state[:, 6] = np.linspace(0.0, 1.0, EP_LENGTH, dtype=np.float32)
    return state


def _write_bucket(bucket: Path) -> tuple[np.ndarray, np.ndarray]:
    (bucket / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    for camera in (HEAD, WRIST):
        video_dir = bucket / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"")

    names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_closedness"]
    info = {
        "fps": 15,
        "robot_type": "muka_franka_single_arm",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7], "names": names},
            "action": {"dtype": "float32", "shape": [7], "names": names},
            HEAD: {"dtype": "video", "shape": [480, 640, 3]},
            WRIST: {"dtype": "video", "shape": [480, 640, 3]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
        "splits": {"train": "0:1"},
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    source_schema = {
        "position_frame": "robot_base",
        "rotation_storage": "Euler XYZ",
        "rotation_convention": "extrinsic; R = Rz @ Ry @ Rx",
        "rotation_unit": "radians",
        "action_type": "absolute achieved EEF state",
        "action_alignment": "action[t] = observation.state[t+1]; terminal row repeated and masked by reader",
        "gripper": "continuous closedness; 0=open, 1=closed",
    }
    (bucket / "meta" / "muka_schema.json").write_text(json.dumps(source_schema), encoding="utf-8")

    episode = {
        "episode_index": 0,
        "length": EP_LENGTH,
        "dataset_from_index": 0,
        "data/chunk_index": 0,
        "data/file_index": 0,
        f"videos/{HEAD}/chunk_index": 0,
        f"videos/{HEAD}/file_index": 0,
        f"videos/{WRIST}/chunk_index": 0,
        f"videos/{WRIST}/file_index": 0,
    }
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame([episode])),
        bucket / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["place the object"], name="task")).to_parquet(
        bucket / "meta" / "tasks.parquet"
    )

    state = _source_state()
    action = np.concatenate((state[1:], state[-1:]), axis=0)
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "observation.state": list(state),
                    "action": list(action),
                    "task_index": np.zeros(EP_LENGTH, dtype=np.int64),
                }
            )
        ),
        bucket / "data" / "chunk-000" / "file-000.parquet",
    )
    return state, action


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, height, width):
        color = (255, 0, 0) if HEAD in str(path) else (0, 255, 0)
        return [Image.new("RGB", (width, height), color) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _dataset(bucket: Path, **overrides) -> MukaFrankaDataset:
    config = {
        "dataset_dir": str(bucket),
        "num_frames": 5,
        "video_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "normalize_mode": None,
    }
    config.update(overrides)
    return MukaFrankaDataset.from_config(OmegaConf.create(config), split="train")


def test_closedness_and_euler_conversion_match_pretrain_contract():
    np.testing.assert_array_equal(
        closedness_to_open_scale(np.array([[0.0], [0.5], [1.0]], dtype=np.float32)),
        np.array([[1.0], [0.0], [-1.0]], dtype=np.float32),
    )
    converted = euler7_to_eef10(_source_state())
    assert converted.shape == (EP_LENGTH, EEF10_DIM)
    np.testing.assert_allclose(converted[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(converted[1, 3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(converted[:, 3:6], axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(converted[:, 6:9], axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(converted[:, 3:6] * converted[:, 6:9], axis=-1), 0.0, atol=1e-6)


def test_reader_converts_both_streams_and_masks_repeated_terminal_action(tmp_path: Path):
    raw_state, raw_action = _write_bucket(tmp_path)
    dataset = _dataset(tmp_path)
    with _mock_decoder():
        first = dataset[0]
        terminal = dataset[len(dataset) - 1]

    assert dataset.action_dim == EEF10_DIM
    np.testing.assert_allclose(first["action"].numpy(), euler7_to_eef10(raw_action[:4]))
    np.testing.assert_allclose(first["proprio"].numpy(), euler7_to_eef10(raw_state[:1]))
    assert first["action_mask"].all()
    assert first["proprio_mask"].all()
    assert not terminal["action_mask"].any()
    assert first["prompt"] == "place the object"
    assert first["video"][0].size == (320, 384)


def test_unify_scatter_uses_only_left_arm_slots_zero_through_nine(tmp_path: Path):
    _write_bucket(tmp_path)
    raw_dataset = _dataset(tmp_path)
    unified_dataset = _dataset(tmp_path, unify_action=True, unify_action_map=UNIFY_MAP)
    with _mock_decoder():
        raw = raw_dataset[0]
        unified = unified_dataset[0]

    dst = parse_unify_spec(UNIFY_MAP, UNIFY_DIM)
    assert unified_dataset.action_dim == UNIFY_DIM
    np.testing.assert_array_equal(dst, np.arange(10))
    np.testing.assert_allclose(unmap_from_unify(unified["action"].numpy(), dst), raw["action"].numpy())
    np.testing.assert_allclose(unmap_from_unify(unified["proprio"].numpy(), dst), raw["proprio"].numpy())
    assert unified["action_mask"][:, :10].all()
    assert not unified["action_mask"][:, 10:].any()


def test_stats_pool_real_actions_and_states_and_pin_rot6d_identity(tmp_path: Path):
    raw_state, raw_action = _write_bucket(tmp_path)
    dataset = _dataset(tmp_path)
    mode, dim, stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap=100)
    expected = np.concatenate((euler7_to_eef10(raw_action[:-1]), euler7_to_eef10(raw_state)), axis=0)
    ordinary = np.setdiff1d(np.arange(EEF10_DIM), ROT6D_DIMS_EEF10)

    assert mode == "eef"
    assert dim == EEF10_DIM
    assert action_rows == EP_LENGTH - 1
    assert state_rows == EP_LENGTH
    np.testing.assert_allclose(np.asarray(stats["mean"])[ordinary], expected.mean(0)[ordinary], atol=1e-7)
    for key, value in {
        "mean": 0.0,
        "std": 1.0,
        "min": -1.0,
        "max": 1.0,
        "q01": -1.0,
        "q99": 1.0,
    }.items():
        np.testing.assert_array_equal(np.asarray(stats[key])[list(ROT6D_DIMS_EEF10)], value)
    assert stats["gripper_convention"] == GRIPPER_CONVENTION
    assert stats["raw_gripper_convention"] == RAW_GRIPPER_CONVENTION
    assert stats["gripper_transform"] == GRIPPER_TRANSFORM
    assert stats["rotation_convention"] == ROTATION_CONVENTION
    assert stats["stats_population"] == STATS_POPULATION
    assert stats["split"] == "train"
    assert stats["num_episodes"] == 1


def test_registry_and_training_yaml_select_muka_franka(tmp_path: Path):
    _write_bucket(tmp_path)
    dataset = build_dataset(
        OmegaConf.create(
            {
                "type": "muka_franka",
                "dataset_dir": str(tmp_path),
                "num_frames": 5,
                "height": 384,
                "width": 320,
                "multiview": True,
                "normalize_mode": None,
            }
        )
    )
    assert isinstance(dataset, MukaFrankaDataset)
    assert "muka_franka" in list_registered_datasets()

    config = OmegaConf.load("configs/dataloader/pretrain_data/muka_franka.yaml")
    assert config.type == "muka_franka"
    assert config.dataset_dir == "/path/to/muka_franka_lerobot_v3"
    assert config.gripper_convention == GRIPPER_CONVENTION
    assert list(config.unify_action_map) == UNIFY_MAP
    assert config.unify_action is True
