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

from openwam.dataloader.registry import list_registered_datasets
from openwam.dataloader.robocasa_gr1 import RoboCasaGR1Dataset
from openwam.dataloader.robocasa_gr1_stats import compute_array_stats, neutralize_rot6d_stats
from openwam.dataloader.transforms.builder import build_transforms
from openwam.dataloader.transforms.video import VideoColorJitter

EP_LENGTH = 8
HEAD_CAM = "observation.images.ego_view"


def _write_bucket(bucket: Path, *, include_wrist: bool = False) -> None:
    (bucket / "meta" / "episodes").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    (bucket / "videos" / HEAD_CAM / "chunk-000").mkdir(parents=True)
    (bucket / "videos" / HEAD_CAM / "chunk-000" / "file-000.mp4").write_bytes(b"")

    features = {
        HEAD_CAM: {"dtype": "video"},
        "eef_sim_pose_action": {"shape": [12]},
        "gripper_open_scale_action": {"shape": [2]},
        "eef_sim_pose_state": {"shape": [12]},
        "gripper_open_scale_state": {"shape": [2]},
        "annotation.human.coarse_action": {"dtype": "string"},
    }
    episode_row = {
        "episode_index": 0,
        "length": EP_LENGTH,
        "dataset_from_index": 0,
        "data/chunk_index": 0,
        "data/file_index": 0,
        f"videos/{HEAD_CAM}/chunk_index": 0,
        f"videos/{HEAD_CAM}/file_index": 0,
    }
    if include_wrist:
        left = "observation.images.left_wrist"
        right = "observation.images.right_wrist"
        for cam in (left, right):
            features[cam] = {"dtype": "video"}
            (bucket / "videos" / cam / "chunk-000").mkdir(parents=True)
            (bucket / "videos" / cam / "chunk-000" / "file-000.mp4").write_bytes(b"")
            episode_row[f"videos/{cam}/chunk_index"] = 0
            episode_row[f"videos/{cam}/file_index"] = 0

    info = {"fps": 20, "features": features}
    (bucket / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([episode_row])), bucket / "meta" / "episodes" / "chunk-000.parquet")

    eef = np.zeros((EP_LENGTH, 12), dtype=np.float32)
    eef[:, 0] = np.linspace(0.0, 0.7, EP_LENGTH)
    eef[:, 6] = np.linspace(1.0, 1.7, EP_LENGTH)
    grip = np.stack(
        [
            np.linspace(0.0, 1.0, EP_LENGTH),
            np.linspace(1.0, 0.0, EP_LENGTH),
        ],
        axis=1,
    ).astype(np.float32)
    df = pd.DataFrame(
        {
            "annotation.human.coarse_action": ["pick cup"] * EP_LENGTH,
            "eef_sim_pose_action": list(eef),
            "gripper_open_scale_action": list(grip),
            "eef_sim_pose_state": list(eef + 0.1),
            "gripper_open_scale_state": list(grip),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), bucket / "data" / "chunk-000" / "file-000.parquet")


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), (255, 0, 0)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _dataset(bucket: Path, **overrides) -> RoboCasaGR1Dataset:
    cfg = {
        "dataset_dir": str(bucket),
        "num_frames": 5,
        "video_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "prompt_columns": ["annotation.human.coarse_action"],
        "normalize_mode": None,
    }
    cfg.update(overrides)
    return RoboCasaGR1Dataset.from_config(OmegaConf.create(cfg), split="train")


def test_registry_includes_robocasa_gr1():
    assert "robocasa_gr1" in list_registered_datasets()


def test_eef_sample_and_missing_wrist_black_slots(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        ds = _dataset(tmp_path)
        sample = ds[0]

    assert ds.action_dim == 20
    assert ds.action_mode == "eef"
    assert sample["action"].shape == (4, 20)
    assert sample["action_mask"].shape == (4, 20)
    assert sample["action_mask"].all()
    assert sample["proprio"].shape == (1, 20)
    assert sample["proprio_mask"].shape == (1, 20)
    assert sample["prompt"] == "pick cup"

    img = sample["video"][0]
    assert img.size == (320, 384)
    assert img.getpixel((10, 10)) == (255, 0, 0)
    assert img.getpixel((10, 300)) == (0, 0, 0)
    assert img.getpixel((250, 300)) == (0, 0, 0)


def test_unify_mode_maps_eef20_to_80_and_masks_unmapped_dims(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        ds = _dataset(tmp_path, action_mode="unify", unify_action=True)
        sample = ds[0]

    assert ds.action_dim == 80
    assert sample["action"].shape == (4, 80)
    assert sample["action_mask"].shape == (4, 80)
    assert sample["action_mask"][0].sum().item() == 20
    assert not sample["action_mask"][0, 10:34].any()
    assert not sample["action_mask"][0, 44:68].any()
    assert sample["action"][0, 0].item() == 0.0
    assert sample["action"][0, 34].item() == 1.0


def test_unify_normalizes_raw_eef_before_mapping(tmp_path: Path):
    _write_bucket(tmp_path)
    stats = {
        "unify": {
            "min": np.zeros(20, dtype=np.float32),
            "max": np.ones(20, dtype=np.float32),
            "mean": np.zeros(20, dtype=np.float32),
            "std": np.ones(20, dtype=np.float32),
        }
    }
    stats_path = tmp_path / "normalization_stats.npy"
    np.save(stats_path, stats)

    with _mock_decoder():
        ds = _dataset(
            tmp_path,
            action_mode="unify",
            unify_action=True,
            normalize_mode="min-max",
            normalization_stats_path=str(stats_path),
        )
        sample = ds[0]

    # raw EEF left x=0 -> min-max -1, then maps to unified slot 0.
    assert sample["action"][0, 0].item() == -1.0
    # raw EEF right x=1 -> min-max +1, then maps to unified slot 34.
    assert sample["action"][0, 34].item() == 1.0


def test_rot6d_stats_are_neutralized():
    raw = {
        "min": np.full(20, -5.0, dtype=np.float32),
        "max": np.full(20, 5.0, dtype=np.float32),
        "mean": np.full(20, 2.0, dtype=np.float32),
        "std": np.full(20, 3.0, dtype=np.float32),
    }
    stats = neutralize_rot6d_stats(raw, [(3, 9), (13, 19)])
    np.testing.assert_allclose(stats["min"][3:9], -1.0)
    np.testing.assert_allclose(stats["max"][13:19], 1.0)
    np.testing.assert_allclose(stats["mean"][3:9], 0.0)
    np.testing.assert_allclose(stats["std"][13:19], 1.0)


def test_compute_stats_neutralizes_rot6d():
    arr = np.arange(40, dtype=np.float32).reshape(2, 20)
    stats = compute_array_stats([arr], rot6d_slices=[(3, 9), (13, 19)])
    np.testing.assert_allclose(stats["min"][3:9], -1.0)
    np.testing.assert_allclose(stats["max"][3:9], 1.0)
    np.testing.assert_allclose(stats["mean"][13:19], 0.0)
    np.testing.assert_allclose(stats["std"][13:19], 1.0)


def test_color_jitter_defaults_are_02():
    jitter = VideoColorJitter()
    assert jitter.brightness == 0.2
    assert jitter.contrast == 0.2
    assert jitter.saturation == 0.2
    assert jitter.hue == 0.0

    pipeline = build_transforms(OmegaConf.create({"augmentation": {"color_jitter": {}}}))
    built = pipeline.transforms[0]
    assert isinstance(built, VideoColorJitter)
    assert built.brightness == 0.2
    assert built.contrast == 0.2
    assert built.saturation == 0.2
