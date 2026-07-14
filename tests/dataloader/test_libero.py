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

from openwam.dataloader.libero import LiberoDataset, MultiLiberoDataset
from openwam.dataloader.registry import build_dataset, list_registered_datasets
from openwam.deploy.model_loader import _build_normalizer
from openwam.train.utils.checkpointing import save_normalization_stats
from scripts.libero_compute_stats import _iter_action_arrays

EP_LENGTH = 8
HEAD = "observation.images.image"
WRIST = "observation.images.wrist_image"


def _write_bucket(bucket: Path) -> None:
    (bucket / "meta" / "episodes").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    for camera in (HEAD, WRIST):
        video_dir = bucket / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"")

    info = {
        "fps": 10.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            HEAD: {"dtype": "video", "shape": [256, 256, 3]},
            WRIST: {"dtype": "video", "shape": [256, 256, 3]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

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
        bucket / "meta" / "episodes" / "chunk-000.parquet",
    )
    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["pick up the red mug"], name="task"),
    ).to_parquet(bucket / "meta" / "tasks.parquet")

    rng = np.random.RandomState(11)
    frame = pd.DataFrame(
        {
            "action": list(rng.uniform(-1, 1, size=(EP_LENGTH, 7)).astype(np.float32)),
            "observation.state": list(rng.uniform(-1, 1, size=(EP_LENGTH, 8)).astype(np.float32)),
            "task_index": np.zeros(EP_LENGTH, dtype=np.int64),
        }
    )
    pq.write_table(pa.Table.from_pandas(frame), bucket / "data" / "chunk-000" / "file-000.parquet")


def _add_prompt_column(bucket: Path, column: str, values: list[str]) -> None:
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][column] = {"dtype": "string", "shape": [1]}
    info_path.write_text(json.dumps(info), encoding="utf-8")

    data_path = bucket / "data" / "chunk-000" / "file-000.parquet"
    frame = pd.read_parquet(data_path)
    frame[column] = values
    pq.write_table(pa.Table.from_pandas(frame), data_path)


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, h, w):
        color = (255, 0, 0) if HEAD in str(path) else (0, 255, 0)
        return [Image.new("RGB", (w, h), color) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _dataset(bucket: Path, **overrides) -> LiberoDataset:
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
    return LiberoDataset.from_config(OmegaConf.create(config), split="train")


def test_registry_and_yaml_include_libero():
    assert "libero" in list_registered_datasets()
    config = OmegaConf.load("configs/dataloader/libero.yaml")
    assert config.type == "libero"
    assert config.unify_action is False


def test_libero_sample_uses_native_action_and_masks_proprio(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path)
        sample = dataset[0]

    assert dataset.action_dim == 7
    assert sample["action"].shape == (4, 7)
    assert sample["action_mask"].all()
    assert sample["proprio"].shape == (1, 7)
    assert not sample["proprio_mask"].any()
    assert sample["prompt"] == "pick up the red mug"
    assert sample["video"][0].size == (320, 384)
    assert sample["video"][0].getpixel((10, 10)) == (255, 0, 0)
    assert sample["video"][0].getpixel((10, 300)) == (0, 255, 0)
    assert sample["video"][0].getpixel((250, 300)) == (0, 0, 0)


def test_libero_prompt_column_falls_back_to_tasks_parquet(tmp_path: Path):
    fallback_dir = tmp_path / "fallback"
    _write_bucket(fallback_dir)
    _add_prompt_column(fallback_dir, "language_instruction", [""] * EP_LENGTH)
    with _mock_decoder():
        fallback = _dataset(fallback_dir)[0]
    assert fallback["prompt"] == "pick up the red mug"

    direct_dir = tmp_path / "direct"
    _write_bucket(direct_dir)
    _add_prompt_column(direct_dir, "language_instruction", ["lift the crimson cup"] * EP_LENGTH)
    with _mock_decoder():
        direct = _dataset(direct_dir)[0]
    assert direct["prompt"] == "lift the crimson cup"


def test_libero_rejects_unify_action(tmp_path: Path):
    _write_bucket(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "does not support unify_action"):
        _dataset(tmp_path, unify_action=True, unify_action_map=list(range(7)))


def test_libero_stats_generate_deploy_artifact(tmp_path: Path):
    _write_bucket(tmp_path)
    stats_path = tmp_path / "source_stats.npy"
    np.save(
        stats_path,
        {
            "libero": {
                "mean": np.zeros(7, dtype=np.float32),
                "std": np.ones(7, dtype=np.float32),
                "min": -np.ones(7, dtype=np.float32),
                "max": np.ones(7, dtype=np.float32),
                "q01": -np.ones(7, dtype=np.float32),
                "q99": np.ones(7, dtype=np.float32),
            }
        },
    )
    dataset = _dataset(
        tmp_path,
        normalize_mode="quantile",
        normalization_stats_path=str(stats_path),
    )

    deploy_path = Path(dataset.normalization_stats_path)
    assert deploy_path == tmp_path / "meta" / "normalization_stats.npy"
    checkpoint = tmp_path / "checkpoint"
    save_normalization_stats(str(checkpoint), dataset)
    copied = np.load(checkpoint / "normalization_stats.npy", allow_pickle=True).item()
    assert copied["libero"]["q99"].shape == (7,)
    with _mock_decoder():
        sample = dataset[0]
    normalizer = _build_normalizer(
        OmegaConf.create(
            {
                "dataloader": {
                    "normalize_mode": "quantile",
                    "action_mode": "libero",
                    "unify_action": False,
                }
            }
        ),
        str(checkpoint),
    )
    raw = np.stack(dataset._load_data_table(0, 0).to_pandas()["action"].values)[:4]
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), raw, atol=1e-6)


def test_libero_stats_stream_reads_each_shard_once(tmp_path: Path):
    _write_bucket(tmp_path)
    arrays = list(_iter_action_arrays(_dataset(tmp_path)))
    assert len(arrays) == 1
    assert arrays[0].shape == (EP_LENGTH, 7)


def test_registry_builds_libero_dataset(tmp_path: Path):
    _write_bucket(tmp_path)
    dataset = build_dataset(
        OmegaConf.create(
            {
                "type": "libero",
                "dataset_dir": str(tmp_path),
                "num_frames": 5,
                "normalize_mode": None,
            }
        )
    )
    assert isinstance(dataset, LiberoDataset)


def test_multibucket_forwards_generated_deploy_stats(tmp_path: Path):
    root = tmp_path / "root"
    _write_bucket(root / "suite_a")
    _write_bucket(root / "suite_b")
    stats_path = tmp_path / "source_stats.npy"
    unit_stats = {
        "mean": np.zeros(7, dtype=np.float32),
        "std": np.ones(7, dtype=np.float32),
        "min": -np.ones(7, dtype=np.float32),
        "max": np.ones(7, dtype=np.float32),
        "q01": -np.ones(7, dtype=np.float32),
        "q99": np.ones(7, dtype=np.float32),
    }
    np.save(stats_path, {"libero": unit_stats})

    dataset = LiberoDataset.from_config(
        OmegaConf.create(
            {
                "dataset_dir": str(root),
                "num_frames": 5,
                "normalize_mode": "quantile",
                "normalization_stats_path": str(stats_path),
            }
        )
    )
    assert isinstance(dataset, MultiLiberoDataset)
    assert dataset.normalization_stats_path == dataset.buckets[0].normalization_stats_path
    checkpoint = tmp_path / "multibucket_checkpoint"
    save_normalization_stats(str(checkpoint), dataset)
    assert (checkpoint / "normalization_stats.npy").is_file()
