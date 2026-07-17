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

from benchmarks.robocasa_gr1.openwam2robocasa_gr1_interface import action_vector_to_dict, build_state
from openwam.dataloader.bases.lerobot_v3_reader import _read_data_table_cached
from openwam.dataloader.registry import list_registered_datasets
from openwam.dataloader.robocasa_gr1 import MultiRoboCasaGR1Dataset, RoboCasaGR1Dataset
from openwam.dataloader.transforms.builder import build_transforms
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.normalization import load_stats_file
from openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation import _iter_bucket_arrays
from openwam.deploy.model_loader import _build_normalizer, _UnifyAwareNormalizer
from openwam.train.utils.checkpointing import save_normalization_stats

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
        "task_index": {"dtype": "int64"},
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
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame([episode_row])), bucket / "meta" / "episodes" / "chunk-000.parquet"
    )

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
            "task_index": [0] * EP_LENGTH,
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
        ds = _dataset(
            tmp_path,
            action_mode="unify",
            unify_action=True,
            unify_action_map=["0-9", "34-43"],
        )
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
            unify_action_map=["0-9", "34-43"],
            normalize_mode="min-max",
            normalization_stats_path=str(stats_path),
        )
        sample = ds[0]

    # raw EEF left x=0 -> min-max -1, then maps to unified slot 0.
    assert sample["action"][0, 0].item() == -1.0
    # raw EEF right x=1 -> min-max +1, then maps to unified slot 34.
    assert sample["action"][0, 34].item() == 1.0
    assert ds.normalization_stats_path == str(tmp_path / "meta" / "normalization_stats.npy")
    deploy_stats = np.load(ds.normalization_stats_path, allow_pickle=True).item()
    assert set(deploy_stats["unify"]) == {"mean", "std", "min", "max", "q01", "q99"}
    assert deploy_stats["unify"]["mean"].shape == (20,)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "unify",
                "unify_action": True,
                "unify_action_map": ["0-9", "34-43"],
            }
        }
    )
    normalizer = _build_normalizer(cfg, str(tmp_path / "meta"))
    assert isinstance(normalizer, _UnifyAwareNormalizer)
    raw = ds._raw_action(ds._load_data_table(0, 0).to_pandas())[:4]
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), np.clip(raw, 0, 1), atol=1e-5)


def test_unify_mode_rejects_mismatched_base_flag(tmp_path: Path):
    _write_bucket(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "requires unify_action=true"):
        _dataset(tmp_path, action_mode="unify", unify_action=False)


def test_unify_mode_requires_explicit_map(tmp_path: Path):
    _write_bucket(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "requires an explicit unify_action_map"):
        _dataset(tmp_path, action_mode="unify", unify_action=True, unify_action_map=None)


def test_missing_optional_prompt_column_is_not_projected(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        ds = _dataset(
            tmp_path,
            prompt_columns=["annotation.missing", "annotation.human.coarse_action"],
        )
        sample = ds[0]
    assert "annotation.missing" not in ds.NEEDED_COLS
    assert sample["prompt"] == "pick cup"


def test_prompt_falls_back_to_tasks_parquet(tmp_path: Path):
    _write_bucket(tmp_path)
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].pop("annotation.human.coarse_action")
    info_path.write_text(json.dumps(info))
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    frame = pq.read_table(data_path).to_pandas().drop(columns=["annotation.human.coarse_action"])
    pq.write_table(pa.Table.from_pandas(frame), data_path)
    pd.DataFrame({"task_index": [0]}, index=["fallback task"]).to_parquet(tmp_path / "meta" / "tasks.parquet")

    with _mock_decoder():
        sample = _dataset(tmp_path, prompt_columns=["annotation.human.coarse_action"])[0]
    assert sample["prompt"] == "fallback task"


def test_dot_path_projection_fallback_stays_cached():
    _read_data_table_cached.cache_clear()
    table = pa.table({"annotation.human.coarse_action": ["pick cup"]})
    with patch(
        "openwam.dataloader.bases.lerobot_v3_reader.pq.read_table",
        side_effect=[pa.ArrowInvalid("Dot path does not exist"), table],
    ) as read:
        first = _read_data_table_cached("/tmp/dotted.parquet", ("annotation.human.coarse_action",))
        second = _read_data_table_cached("/tmp/dotted.parquet", ("annotation.human.coarse_action",))
    assert first is second
    assert read.call_count == 2


def test_quantile_stats_are_materialized(tmp_path: Path):
    raw = {
        "q01": np.full(20, -1.0, dtype=np.float32),
        "q99": np.full(20, 1.0, dtype=np.float32),
    }
    path = tmp_path / "stats.npy"
    np.save(path, {"eef": raw})
    stats = load_stats_file(path, action_mode="eef", normalize_mode="quantile", dim=20)
    assert set(stats) == {"min", "max", "mean", "std", "q01", "q99"}
    np.testing.assert_allclose(stats["q01"], -1.0)


def test_stats_stream_pools_action_and_state(tmp_path: Path):
    _write_bucket(tmp_path)
    arrays = list(_iter_bucket_arrays(_dataset(tmp_path)))
    assert len(arrays) == 2
    assert arrays[0].shape == arrays[1].shape == (EP_LENGTH, 20)
    assert arrays[0][0, 0] == 0.0
    np.testing.assert_allclose(arrays[1][0, 0], 0.1)


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


def test_robocasa_config_wires_reader_color_jitter():
    cfg = OmegaConf.load("configs/dataloader/robocasa_gr1.yaml")
    assert cfg.color_jitter.brightness == 0.2
    assert "transforms" not in cfg


def test_multibucket_forwards_generated_deploy_stats(tmp_path: Path):
    root = tmp_path / "root"
    for name in ("a", "b"):
        _write_bucket(root / name)
    source = tmp_path / "stats.npy"
    base = np.arange(20, dtype=np.float32)
    np.save(
        source,
        {
            "eef": {
                "min": base - 1,
                "max": base + 1,
                "mean": base,
                "std": np.ones(20, dtype=np.float32),
            }
        },
    )
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(root),
            "num_frames": 5,
            "prompt_columns": ["annotation.human.coarse_action"],
            "normalize_mode": "z-score",
            "normalization_stats_path": str(source),
        }
    )
    ds = RoboCasaGR1Dataset.from_config(cfg)
    assert isinstance(ds, MultiRoboCasaGR1Dataset)
    assert ds.normalization_stats_path == str(root / "a" / "meta" / "normalization_stats.npy")
    assert Path(ds.normalization_stats_path).is_file()
    checkpoint = tmp_path / "checkpoint"
    save_normalization_stats(str(checkpoint), ds)
    copied = np.load(checkpoint / "normalization_stats.npy", allow_pickle=True).item()
    assert copied["eef"]["mean"].shape == (20,)


def test_benchmark_fallback_key_order_is_deterministic():
    obs = {"state.z": np.array([3.0]), "state.a": np.array([1.0, 2.0])}
    assert build_state(obs) == [1.0, 2.0, 3.0]

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _DictSpace:
        spaces = {"action.z": _Space((1,)), "action.a": _Space((2,))}

    mapped = action_vector_to_dict([1.0, 2.0, 3.0], _DictSpace())
    np.testing.assert_array_equal(mapped["action.a"], [1.0, 2.0])
    np.testing.assert_array_equal(mapped["action.z"], [3.0])
