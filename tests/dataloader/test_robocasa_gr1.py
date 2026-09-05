from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.bases.lerobot_v3_reader import _read_data_table_cached
from openwam.dataloader.registry import list_registered_datasets
from openwam.dataloader.robocasa_gr1 import (
    EEF33_DIM,
    NORMALIZATION_STATS_FILENAME,
    STATS_SCHEMA_VERSION,
    MultiRoboCasaGR1Dataset,
    RoboCasaGR1Dataset,
    _stats_file_is_compatible,
)
from openwam.dataloader.transforms.builder import build_transforms
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.gr1_kinematics import HAND_DIMS_EEF33, ROT6D_DIMS_EEF33
from openwam.dataloader.utils.normalization import STAT_KEYS, load_stats_file, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation import (
    _compute_global_stats,
    _iter_bucket_arrays,
)
from openwam.deploy.model_loader import _build_normalizer, _UnifyAwareNormalizer
from openwam.train.utils.checkpointing import save_normalization_stats, verify_resume_normalization_stats

EP_LENGTH = 8
HEAD_CAM = "observation.images.ego_view"


def _write_bucket(bucket: Path, *, include_wrist: bool = False) -> None:
    (bucket / "meta" / "episodes").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    (bucket / "videos" / HEAD_CAM / "chunk-000").mkdir(parents=True)
    (bucket / "videos" / HEAD_CAM / "chunk-000" / "file-000.mp4").write_bytes(b"")

    features = {
        HEAD_CAM: {"dtype": "video"},
        "eef_action": {"shape": [EEF33_DIM]},
        "observation.eef_state": {"shape": [EEF33_DIM]},
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

    eef = np.zeros((EP_LENGTH, EEF33_DIM), dtype=np.float32)
    eef[:, 0] = np.linspace(0.0, 0.7, EP_LENGTH)
    eef[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    eef[:, 9:15] = np.linspace(0.0, 1.0, EP_LENGTH)[:, None]
    eef[:, 15] = np.linspace(1.0, 1.7, EP_LENGTH)
    eef[:, 18:24] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    eef[:, 24:30] = np.linspace(1.0, 0.0, EP_LENGTH)[:, None]
    eef[:, 30:33] = np.linspace(-0.2, 0.2, EP_LENGTH)[:, None]
    state = eef.copy()
    state[:, [0, 1, 2, 15, 16, 17]] += 0.1
    state[:, 9:15] += 10.0
    state[:, 24:30] += 20.0
    df = pd.DataFrame(
        {
            "annotation.human.coarse_action": ["pick cup"] * EP_LENGTH,
            "eef_action": list(eef),
            "observation.eef_state": list(state),
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


def _write_tasks_parquet(bucket: Path, prompt: str = "pick cup") -> None:
    pd.DataFrame({"task_index": [0]}, index=[prompt]).to_parquet(bucket / "meta" / "tasks.parquet")


def _dataset(bucket: Path, **overrides) -> RoboCasaGR1Dataset:
    if not (bucket / "meta" / "tasks.parquet").exists():
        _write_tasks_parquet(bucket)
    cfg = {
        "dataset_dir": str(bucket),
        "num_frames": 5,
        "video_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "normalize_mode": None,
    }
    cfg.update(overrides)
    return RoboCasaGR1Dataset.from_config(OmegaConf.create(cfg), split="train")


def _stats_payload(action_stats: dict, state_stats: dict | None = None) -> dict:
    return {
        "eef": action_stats,
        "eef_state": action_stats if state_stats is None else state_stats,
        "robocasa_gr1_stats_schema": STATS_SCHEMA_VERSION,
    }


def _complete_stats_block(
    *,
    min_v: float = 0.0,
    max_v: float = 1.0,
    mean_v: float = 0.0,
    std_v: float = 1.0,
) -> dict:
    """Schema-v2 compatible block with finite ``STAT_KEYS`` vectors of shape ``(33,)``."""
    return {
        "min": np.full(EEF33_DIM, min_v, dtype=np.float32),
        "max": np.full(EEF33_DIM, max_v, dtype=np.float32),
        "mean": np.full(EEF33_DIM, mean_v, dtype=np.float32),
        "std": np.full(EEF33_DIM, std_v, dtype=np.float32),
        "q01": np.full(EEF33_DIM, min_v, dtype=np.float32),
        "q99": np.full(EEF33_DIM, max_v, dtype=np.float32),
    }


def test_registry_includes_robocasa_gr1():
    assert "robocasa_gr1" in list_registered_datasets()


def test_eef_sample_and_missing_wrist_black_slots(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        ds = _dataset(tmp_path)
        sample = ds[0]

    assert ds.action_dim == EEF33_DIM
    assert ds.action_mode == "eef"
    assert sample["action"].shape == (4, EEF33_DIM)
    assert sample["action_mask"].shape == (4, EEF33_DIM)
    assert sample["action_mask"].all()
    assert sample["proprio"].shape == (1, EEF33_DIM)
    assert sample["proprio_mask"].shape == (1, EEF33_DIM)
    assert sample["prompt"] == "pick cup"

    img = sample["video"][0]
    assert img.size == (320, 384)
    assert img.getpixel((10, 10)) == (255, 0, 0)
    assert img.getpixel((10, 300)) == (0, 0, 0)
    assert img.getpixel((250, 300)) == (0, 0, 0)


def test_unify_mode_maps_eef33_to_80_and_masks_unmapped_dims(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        ds = _dataset(
            tmp_path,
            action_mode="eef",
            unify_action=True,
            unify_action_map=["0-8", "10-15", "34-42", "44-49", "68-70"],
        )
        sample = ds[0]

    assert ds.action_dim == 80
    assert sample["action"].shape == (4, 80)
    assert sample["action_mask"].shape == (4, 80)
    assert sample["action_mask"][0].sum().item() == EEF33_DIM
    assert not sample["action_mask"][0, 9]
    assert not sample["action_mask"][0, 16:34].any()
    assert not sample["action_mask"][0, 43]
    assert not sample["action_mask"][0, 50:68].any()
    assert not sample["action_mask"][0, 71:80].any()
    assert sample["action"][0, 0].item() == 0.0
    assert sample["action"][0, 34].item() == 1.0


def test_unify_normalizes_raw_eef_before_mapping(tmp_path: Path):
    _write_bucket(tmp_path)
    action_stats = _complete_stats_block(min_v=0.0, max_v=1.0, mean_v=0.0, std_v=1.0)
    state_stats = {key: value.copy() for key, value in action_stats.items()}
    state_stats["max"][list(HAND_DIMS_EEF33)] = 20.0
    state_stats["q99"][list(HAND_DIMS_EEF33)] = 20.0
    stats_path = tmp_path / "meta" / NORMALIZATION_STATS_FILENAME
    np.save(stats_path, _stats_payload(action_stats, state_stats))

    with _mock_decoder():
        ds = _dataset(
            tmp_path,
            action_mode="eef",
            unify_action=True,
            unify_action_map=["0-8", "10-15", "34-42", "44-49", "68-70"],
            normalize_mode="min-max",
        )
        sample = ds[0]

    # raw EEF left x=0 -> min-max -1, then maps to unified slot 0.
    assert sample["action"][0, 0].item() == -1.0
    # Non-hand state x=0.1 still uses the shared [0, 1] stats: 2*0.1-1 = -0.8.
    np.testing.assert_allclose(sample["proprio"][0, 0].item(), -0.8, atol=1e-6)
    # Hand state=10 uses state-only [0, 20] stats -> 0. The action-only
    # transform would clip it to +1.
    np.testing.assert_allclose(sample["proprio"][0, 10].item(), 0.0, atol=1e-6)
    # raw EEF right x=1 -> min-max +1, then maps to unified slot 34.
    assert sample["action"][0, 34].item() == 1.0
    # Single-bucket mode: the directional source file is the deploy artifact.
    assert ds.normalization_stats_path == str(stats_path)
    assert ds._resolved_stats_path == str(stats_path)
    deploy_stats = np.load(ds.normalization_stats_path, allow_pickle=True).item()
    assert {"eef", "eef_state", "robocasa_gr1_stats_schema"} <= set(deploy_stats)
    assert deploy_stats["eef"]["mean"].shape == (EEF33_DIM,)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef",
                "unify_action": True,
                "unify_action_map": ["0-8", "10-15", "34-42", "44-49", "68-70"],
            }
        }
    )
    from openwam.train.utils.checkpointing import save_normalization_stats

    ckpt_dir = tmp_path / "fake_ckpt"
    ckpt_dir.mkdir()
    save_normalization_stats(str(ckpt_dir), ds)
    normalizer = _build_normalizer(cfg, str(ckpt_dir))
    assert isinstance(normalizer, _UnifyAwareNormalizer)
    raw = ds._raw_action(ds._load_data_table(0, 0).to_pandas())[:4]
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), np.clip(raw, 0, 1), atol=1e-5)
    np.testing.assert_allclose(normalizer.normalize(np.array([[0.1] * EEF33_DIM]))[0, 0], -0.8, atol=1e-6)
    state_probe = np.zeros((1, EEF33_DIM), dtype=np.float32)
    state_probe[:, list(HAND_DIMS_EEF33)] = 10.0
    normalized_probe = normalizer.normalize(state_probe)
    np.testing.assert_allclose(normalized_probe[0, 10], 0.0, atol=1e-6)


def test_eef_normalization_preserves_rot6d_and_changes_xyz_gripper(tmp_path: Path):
    _write_bucket(tmp_path)
    base = {
        "min": np.full(EEF33_DIM, -2.0, dtype=np.float32),
        "max": np.full(EEF33_DIM, 2.0, dtype=np.float32),
        "mean": np.zeros(EEF33_DIM, dtype=np.float32),
        "std": np.full(EEF33_DIM, 2.0, dtype=np.float32),
        "q01": np.full(EEF33_DIM, -2.0, dtype=np.float32),
        "q99": np.full(EEF33_DIM, 2.0, dtype=np.float32),
    }
    pin_rot6d_identity(base, ROT6D_DIMS_EEF33)
    np.save(tmp_path / "meta" / NORMALIZATION_STATS_FILENAME, _stats_payload(base))
    ds = _dataset(
        tmp_path,
        action_mode="eef",
        unify_action=True,
        unify_action_map=["0-8", "10-15", "34-42", "44-49", "68-70"],
        normalize_mode="min-max",
    )
    raw = ds._raw_action(ds._load_data_table(0, 0).to_pandas())
    normalized = ds._normalize_array(raw)
    rot = list(ROT6D_DIMS_EEF33)
    np.testing.assert_allclose(normalized[:, rot], raw[:, rot], atol=1e-7)
    assert not np.allclose(normalized[:, [0, 9, 15, 24, 30]], raw[:, [0, 9, 15, 24, 30]])
    for start in (3, 18):
        first = raw[:, start : start + 3]
        second = raw[:, start + 3 : start + 6]
        np.testing.assert_allclose(np.linalg.norm(first, axis=-1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(second, axis=-1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.sum(first * second, axis=-1), 0.0, atol=1e-6)


def test_joint_mode_is_rejected(tmp_path: Path):
    _write_bucket(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "supports only action_mode='eef'"):
        _dataset(tmp_path, action_mode="joint", unify_action=False)


def test_unify_mode_requires_explicit_map(tmp_path: Path):
    _write_bucket(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "requires an explicit unify_action_map"):
        _dataset(tmp_path, action_mode="eef", unify_action=True, unify_action_map=None)


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
        sample = _dataset(tmp_path)[0]
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
        "q01": np.full(EEF33_DIM, -1.0, dtype=np.float32),
        "q99": np.full(EEF33_DIM, 1.0, dtype=np.float32),
    }
    path = tmp_path / "stats.npy"
    np.save(path, {"eef": raw})
    stats = load_stats_file(path, action_mode="eef", normalize_mode="quantile", dim=EEF33_DIM)
    assert set(stats) == {"min", "max", "mean", "std", "q01", "q99"}
    np.testing.assert_allclose(stats["q01"], -1.0)


def test_stats_stream_exposes_action_and_state_for_global_pool(tmp_path: Path):
    _write_bucket(tmp_path)
    arrays = list(_iter_bucket_arrays(_dataset(tmp_path)))
    assert len(arrays) == 1
    action, state = arrays[0]
    assert action.shape == state.shape == (EP_LENGTH, EEF33_DIM)
    assert action[0, 0] == 0.0
    np.testing.assert_allclose(state[0, 0], 0.1)


def test_stats_split_hand_commands_from_joint_angle_state(tmp_path: Path):
    _write_bucket(tmp_path)
    dataset = _dataset(tmp_path)
    action, state = next(iter(_iter_bucket_arrays(dataset)))
    mode, dim, action_stats, state_stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap=10_000)

    pooled = np.concatenate([action, state], axis=0)
    assert mode == "eef"
    assert dim == EEF33_DIM
    assert action_rows == state_rows == EP_LENGTH
    assert action_stats["num_timesteps"] == state_stats["num_timesteps"] == pooled.shape[0]
    assert action_stats["pool"] == state_stats["pool"] == "action_state_except_hand"
    assert action_stats["hand_pool"] == "action"
    assert state_stats["hand_pool"] == "state"

    hand = list(HAND_DIMS_EEF33)
    non_hand_non_rot = sorted(set(range(EEF33_DIM)) - set(hand) - set(ROT6D_DIMS_EEF33))
    np.testing.assert_allclose(
        np.asarray(action_stats["mean"])[non_hand_non_rot],
        pooled.mean(0)[non_hand_non_rot],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(state_stats["mean"])[non_hand_non_rot],
        pooled.mean(0)[non_hand_non_rot],
        atol=1e-7,
    )
    np.testing.assert_allclose(np.asarray(action_stats["mean"])[hand], action.mean(0)[hand], atol=1e-7)
    np.testing.assert_allclose(np.asarray(state_stats["mean"])[hand], state.mean(0)[hand], atol=1e-7)
    assert not np.allclose(np.asarray(action_stats["mean"])[hand], np.asarray(state_stats["mean"])[hand])


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


def test_color_jitter_changes_pixels_consistently_across_frames():
    source = Image.new("RGB", (32, 32), (80, 140, 200))
    jitter = VideoColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    random.seed(7)
    frames = jitter({"video": [source.copy(), source.copy()]})["video"]
    np.testing.assert_array_equal(np.asarray(frames[0]), np.asarray(frames[1]))
    assert not np.array_equal(np.asarray(frames[0]), np.asarray(source))


def test_robocasa_config_wires_reader_color_jitter():
    cfg = OmegaConf.load("configs/dataloader/robocasa_gr1.yaml")
    assert cfg.color_jitter.brightness == 0.2
    assert "transforms" not in cfg


def test_robocasa_config_has_no_eef_column_knobs():
    cfg = OmegaConf.load("configs/dataloader/robocasa_gr1.yaml")
    assert "eef_action_column" not in cfg
    assert "eef_state_column" not in cfg


def test_robocasa_config_normalizes_min_max_without_stats_path():
    cfg = OmegaConf.load("configs/dataloader/robocasa_gr1.yaml")
    assert cfg.normalize_mode == "min-max"
    assert "normalization_stats_path" not in cfg


def test_multibucket_forwards_generated_deploy_stats(tmp_path: Path):
    root = tmp_path / "root"
    for name in ("a", "b"):
        _write_bucket(root / name)
    source = root / "meta" / NORMALIZATION_STATS_FILENAME
    source.parent.mkdir(parents=True)
    base = np.arange(EEF33_DIM, dtype=np.float32)
    source_stats = {
        "min": base - 1,
        "max": base + 1,
        "mean": base,
        "std": np.ones(EEF33_DIM, dtype=np.float32),
        "q01": base - 1,
        "q99": base + 1,
    }
    pin_rot6d_identity(source_stats, ROT6D_DIMS_EEF33)
    state_stats = {key: np.asarray(value).copy() for key, value in source_stats.items()}
    state_stats["mean"][list(HAND_DIMS_EEF33)] += 10.0
    np.save(source, _stats_payload(source_stats, state_stats))
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(root),
            "num_frames": 5,
            "multiview": True,
            "normalize_mode": "z-score",
        }
    )
    ds = RoboCasaGR1Dataset.from_config(cfg)
    assert isinstance(ds, MultiRoboCasaGR1Dataset)
    # Every bucket normalizes with the one root-level directional file...
    assert {b._resolved_stats_path for b in ds.buckets} == {str(source)}
    # ...and checkpoint saving copies that same file with both directions.
    assert ds.normalization_stats_path == str(source)
    assert Path(ds.normalization_stats_path).is_file()
    checkpoint = tmp_path / "checkpoint"
    save_normalization_stats(str(checkpoint), ds)
    copied = np.load(checkpoint / "normalization_stats.npy", allow_pickle=True).item()
    assert copied["eef"]["mean"].shape == (EEF33_DIM,)
    assert copied["eef_state"]["mean"].shape == (EEF33_DIM,)
    assert copied["robocasa_gr1_stats_schema"] == STATS_SCHEMA_VERSION


def test_stats_are_autobuilt_at_the_fixed_root_path(tmp_path: Path):
    root = tmp_path / "root"
    for name in ("a", "b"):
        _write_bucket(root / name)
    stats_path = root / "meta" / NORMALIZATION_STATS_FILENAME
    assert not stats_path.exists()

    cfg = OmegaConf.create(
        {
            "dataset_dir": str(root),
            "num_frames": 5,
            "multiview": True,
            "normalize_mode": "min-max",
        }
    )
    ds = RoboCasaGR1Dataset.from_config(cfg)
    assert isinstance(ds, MultiRoboCasaGR1Dataset)
    assert stats_path.is_file()
    assert {b._resolved_stats_path for b in ds.buckets} == {str(stats_path)}

    # The auto-built file is exactly what the offline scan pools over both buckets.
    payload = np.load(stats_path, allow_pickle=True).item()
    built = payload["eef"]
    built_state = payload["eef_state"]
    _, _, expected, expected_state, action_rows, state_rows = _compute_global_stats(ds, reservoir_cap=10_000)
    assert payload["robocasa_gr1_stats_schema"] == STATS_SCHEMA_VERSION
    assert built["pool"] == built_state["pool"] == "action_state_except_hand"
    assert action_rows == state_rows == 2 * EP_LENGTH
    np.testing.assert_allclose(np.asarray(built["min"]), np.asarray(expected["min"]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(built["max"]), np.asarray(expected["max"]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(built_state["min"]), np.asarray(expected_state["min"]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(built_state["max"]), np.asarray(expected_state["max"]), atol=1e-7)


def test_corrupt_stats_file_is_replaced_during_autobuild(tmp_path: Path):
    root = tmp_path / "root"
    _write_bucket(root / "a")
    stats_path = root / "meta" / NORMALIZATION_STATS_FILENAME
    stats_path.parent.mkdir(parents=True)
    stats_path.write_bytes(b"truncated")

    cfg = OmegaConf.create(
        {
            "dataset_dir": str(root),
            "num_frames": 5,
            "multiview": True,
            "normalize_mode": "min-max",
        }
    )
    ds = RoboCasaGR1Dataset.from_config(cfg)

    assert isinstance(ds, MultiRoboCasaGR1Dataset)
    payload = np.load(stats_path, allow_pickle=True).item()
    assert payload["robocasa_gr1_stats_schema"] == STATS_SCHEMA_VERSION
    assert {"eef", "eef_state"} <= set(payload)


def test_normalize_mode_defaults_to_min_max_and_ignores_stats_path_key(tmp_path: Path):
    _write_bucket(tmp_path)
    stats = _complete_stats_block(min_v=0.0, max_v=1.0)
    np.save(tmp_path / "meta" / NORMALIZATION_STATS_FILENAME, _stats_payload(stats))
    # A leftover normalization_stats_path is ignored: the key is no longer part
    # of the config surface, and from_config always resolves the fixed path.
    stale = tmp_path / "stale_stats.npy"
    np.save(stale, {"eef": {"min": np.full(EEF33_DIM, -9.0), "max": np.full(EEF33_DIM, 9.0)}})
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(tmp_path),
            "num_frames": 5,
            "video_stride": 1,
            "multiview": True,
            "normalization_stats_path": str(stale),
        }
    )
    ds = RoboCasaGR1Dataset.from_config(cfg)
    assert ds._normalize_mode == "min-max"
    assert ds._resolved_stats_path == str(tmp_path / "meta" / NORMALIZATION_STATS_FILENAME)
    # raw left x=0 with [0, 1] stats -> -1 (the stale [-9, 9] file would give ~0).
    np.testing.assert_allclose(ds._normalize_array(np.zeros((1, EEF33_DIM), np.float32))[0, 0], -1.0, atol=1e-6)


def test_missing_stats_without_from_config_raises(tmp_path: Path):
    _write_bucket(tmp_path)
    with np.testing.assert_raises_regex(FileNotFoundError, "normalization_stats.npy is missing"):
        RoboCasaGR1Dataset(
            dataset_dir=str(tmp_path),
            num_frames=5,
            multiview=True,
            normalize_mode="min-max",
        )


def test_legacy_pooled_hand_stats_are_rejected(tmp_path: Path):
    _write_bucket(tmp_path)
    stats = _complete_stats_block()
    np.save(tmp_path / "meta" / NORMALIZATION_STATS_FILENAME, {"eef": stats})

    with np.testing.assert_raises_regex(ValueError, "legacy or malformed"):
        RoboCasaGR1Dataset(
            dataset_dir=str(tmp_path),
            num_frames=5,
            multiview=True,
            normalize_mode="min-max",
        )


def test_malformed_schema_v2_stats_are_rebuilt(tmp_path: Path):
    """Empty schema-v2 blocks must not count as compatible (KeyError: missing [min, max])."""
    root = tmp_path / "root"
    _write_bucket(root / "a")
    stats_path = root / "meta" / NORMALIZATION_STATS_FILENAME
    stats_path.parent.mkdir(parents=True)
    np.save(
        stats_path,
        {
            "robocasa_gr1_stats_schema": STATS_SCHEMA_VERSION,
            "eef": {},
            "eef_state": {},
        },
    )
    assert not _stats_file_is_compatible(stats_path)

    cfg = OmegaConf.create(
        {
            "dataset_dir": str(root),
            "num_frames": 5,
            "multiview": True,
            "normalize_mode": "min-max",
        }
    )
    ds = RoboCasaGR1Dataset.from_config(cfg)
    payload = np.load(stats_path, allow_pickle=True).item()
    assert payload["robocasa_gr1_stats_schema"] == STATS_SCHEMA_VERSION
    for block_name in ("eef", "eef_state"):
        for key in STAT_KEYS:
            assert np.asarray(payload[block_name][key]).shape == (EEF33_DIM,)
    assert _stats_file_is_compatible(stats_path)
    assert isinstance(ds, MultiRoboCasaGR1Dataset)


def test_schema_v2_rejects_non_numeric_non_finite_and_wrong_shape_vectors(tmp_path: Path):
    stats_path = tmp_path / NORMALIZATION_STATS_FILENAME
    malformed_values = (
        np.full(EEF33_DIM, "not-a-number"),
        np.full(EEF33_DIM, np.nan, dtype=np.float32),
        np.zeros(EEF33_DIM - 1, dtype=np.float32),
        np.zeros(EEF33_DIM, dtype=np.complex64),
    )
    for malformed in malformed_values:
        payload = _stats_payload(_complete_stats_block())
        payload["eef"]["min"] = malformed
        np.save(stats_path, payload)
        assert not _stats_file_is_compatible(stats_path)


def test_unsupported_newer_stats_schema_fails_fast(tmp_path: Path):
    """A future schema must not be overwritten by schema-v2 auto-rebuild."""
    root = tmp_path / "root"
    _write_bucket(root / "a")
    stats_path = root / "meta" / NORMALIZATION_STATS_FILENAME
    stats_path.parent.mkdir(parents=True)
    original = {
        "robocasa_gr1_stats_schema": STATS_SCHEMA_VERSION + 1,
        "eef": _complete_stats_block(),
        "eef_state": _complete_stats_block(),
        "future_marker": "keep-me",
    }
    np.save(stats_path, original)

    cfg = OmegaConf.create(
        {
            "dataset_dir": str(root),
            "num_frames": 5,
            "multiview": True,
            "normalize_mode": "min-max",
        }
    )
    with np.testing.assert_raises_regex(ValueError, "unsupported schema"):
        RoboCasaGR1Dataset.from_config(cfg)
    payload = np.load(stats_path, allow_pickle=True).item()
    assert payload["robocasa_gr1_stats_schema"] == STATS_SCHEMA_VERSION + 1
    assert payload["future_marker"] == "keep-me"


def test_resume_rejects_legacy_checkpoint_stats(tmp_path: Path):
    """Legacy checkpoint artifact must not silently diverge from schema-v2 training stats."""
    root = tmp_path / "root"
    _write_bucket(root / "a")
    dataset_stats = root / "meta" / NORMALIZATION_STATS_FILENAME
    dataset_stats.parent.mkdir(parents=True)
    np.save(dataset_stats, _stats_payload(_complete_stats_block()))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Legacy pooled eef-only artifact retained in the resumed run dir.
    np.save(run_dir / "normalization_stats.npy", {"eef": _complete_stats_block(min_v=-1.0, max_v=1.0)})

    class _Dataset:
        normalization_stats_path = str(dataset_stats)

    with np.testing.assert_raises_regex(ValueError, "finetune_ckpt_path"):
        verify_resume_normalization_stats(str(run_dir), _Dataset())


def test_resume_rejects_mismatched_schema_v2_stats(tmp_path: Path):
    root = tmp_path / "root"
    dataset_stats = root / "meta" / NORMALIZATION_STATS_FILENAME
    dataset_stats.parent.mkdir(parents=True)
    np.save(dataset_stats, _stats_payload(_complete_stats_block(min_v=0.0, max_v=1.0)))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(run_dir / "normalization_stats.npy", _stats_payload(_complete_stats_block(min_v=-2.0, max_v=2.0)))

    class _Dataset:
        normalization_stats_path = str(dataset_stats)

    with np.testing.assert_raises_regex(ValueError, "finetune_ckpt_path"):
        verify_resume_normalization_stats(str(run_dir), _Dataset())


def test_resume_allows_matching_normalization_stats(tmp_path: Path):
    payload = _stats_payload(_complete_stats_block(min_v=-0.5, max_v=1.5))
    dataset_stats = tmp_path / "dataset" / "meta" / NORMALIZATION_STATS_FILENAME
    dataset_stats.parent.mkdir(parents=True)
    np.save(dataset_stats, payload)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(run_dir / "normalization_stats.npy", payload)

    class _Dataset:
        normalization_stats_path = str(dataset_stats)

    verify_resume_normalization_stats(str(run_dir), _Dataset())
