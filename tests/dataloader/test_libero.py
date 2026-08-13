from __future__ import annotations

import json
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

from openwam.dataloader.libero import (
    ACTION7_DIM,
    EEF10_DIM,
    GRIPPER_CONVENTION,
    LIBERO_GRIPPER_WIDTH_OPEN,
    ROT6D_DIMS_EEF10,
    STATE8_DIM,
    LiberoDataset,
    MultiLiberoDataset,
    gripper_cmd_to_open_scale,
    gripper_qpos_to_cmd,
    state8_to_eef10,
)
from openwam.dataloader.registry import build_dataset, list_registered_datasets
from openwam.dataloader.utils.normalization import pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.libero_stats_computation import (
    _compute_global_stats,
    _iter_bucket_arrays,
)
from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec, unmap_from_unify
from openwam.deploy.model_loader import _build_normalizer, _UnifyAwareNormalizer
from openwam.train.utils.checkpointing import save_normalization_stats

EP_LENGTH = 8
HEAD = "observation.images.image"
WRIST = "observation.images.wrist_image"
UNIFY_MAP = ["0-9"]


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
            "observation.state": {"dtype": "float32", "shape": [STATE8_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION7_DIM]},
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
    state = np.zeros((EP_LENGTH, STATE8_DIM), dtype=np.float32)
    state[:, 0:3] = rng.uniform(-0.5, 0.5, size=(EP_LENGTH, 3))
    state[:, 3:6] = rng.uniform(-1.0, 1.0, size=(EP_LENGTH, 3))  # axis-angle
    state[:, 6] = rng.uniform(0.0, 0.04, size=EP_LENGTH)  # Panda finger 1
    state[:, 7] = rng.uniform(-0.04, 0.0, size=EP_LENGTH)  # Panda finger 2
    frame = pd.DataFrame(
        {
            "action": list(rng.uniform(-1, 1, size=(EP_LENGTH, ACTION7_DIM)).astype(np.float32)),
            "observation.state": list(state),
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


def _raw_window(dataset: LiberoDataset) -> pd.DataFrame:
    return dataset._load_data_table(0, 0).to_pandas()


def _unit_stats(scale: float = 2.0) -> dict:
    stats = {
        "min": np.full(EEF10_DIM, -scale, dtype=np.float32),
        "max": np.full(EEF10_DIM, scale, dtype=np.float32),
        "mean": np.zeros(EEF10_DIM, dtype=np.float32),
        "std": np.full(EEF10_DIM, scale, dtype=np.float32),
        "q01": np.full(EEF10_DIM, -scale, dtype=np.float32),
        "q99": np.full(EEF10_DIM, scale, dtype=np.float32),
        "gripper_convention": GRIPPER_CONVENTION,
    }
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF10)
    return stats


def test_registry_and_yaml_include_libero():
    assert "libero" in list_registered_datasets()
    config = OmegaConf.load("configs/dataloader/libero.yaml")
    assert config.type == "libero"
    assert config.action_mode == "eef"
    assert config.unify_action is True
    assert list(config.unify_action_map) == ["0-9"]
    assert "state_stats_mode" not in config
    assert config.color_jitter.brightness == 0.2


def test_state8_to_eef10_rot6d_orthonormal_and_gripper_command():
    rng = np.random.RandomState(3)
    state = rng.uniform(-1, 1, size=(16, STATE8_DIM)).astype(np.float32)
    eef10 = state8_to_eef10(state)
    assert eef10.shape == (16, EEF10_DIM)
    first, second = eef10[:, 3:6], eef10[:, 6:9]
    np.testing.assert_allclose(np.linalg.norm(first, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(second, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(first * second, axis=-1), 0.0, atol=1e-6)
    # Zero rotation -> identity rot6d.
    identity = state8_to_eef10(np.zeros((1, STATE8_DIM), dtype=np.float32))
    np.testing.assert_allclose(identity[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-7)
    # Gripper OPEN-SCALE: fully closed width -> -1, fully open -> +1. This is
    # the opposite of LIBERO's recorded action[6] (+1 = close), which the reader
    # negates via gripper_cmd_to_open_scale.
    assert gripper_qpos_to_cmd(np.array(0.0)) == -1.0
    assert gripper_qpos_to_cmd(np.array(LIBERO_GRIPPER_WIDTH_OPEN)) == 1.0
    assert gripper_qpos_to_cmd(np.array(LIBERO_GRIPPER_WIDTH_OPEN / 2)) == 0.0
    # Out-of-range widths clip to the endpoints rather than extrapolating.
    assert gripper_qpos_to_cmd(np.array(-0.1)) == -1.0
    assert gripper_qpos_to_cmd(np.array(0.2)) == 1.0


def test_gripper_cmd_to_open_scale_negates_recorded_command():
    # LIBERO records a binary +-1 command with +1 = close; the trained channel
    # is +1 = open, so the render is a negation.
    np.testing.assert_allclose(
        gripper_cmd_to_open_scale(np.array([1.0, -1.0, 0.0], dtype=np.float32)),
        [-1.0, 1.0, 0.0],
        atol=1e-7,
    )


def test_libero_sample_uses_next_state_targets_and_eef10_proprio(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path)
        sample = dataset[0]

    assert dataset.action_dim == EEF10_DIM
    assert sample["action"].shape == (4, EEF10_DIM)
    assert sample["proprio"].shape == (1, EEF10_DIM)
    assert sample["proprio_mask"].all()
    assert sample["action_mask"].all()

    win = _raw_window(dataset)
    state = np.stack(win["observation.state"].values)[: dataset._num_frames]
    action7 = np.stack(win["action"].values)[: dataset._num_frames]
    achieved = state8_to_eef10(state)
    got = sample["action"].numpy()
    np.testing.assert_allclose(got[:, 0:9], achieved[1:5, 0:9], atol=1e-6)
    # Action gripper is the recorded command negated into the open-scale.
    np.testing.assert_allclose(got[:, 9], gripper_cmd_to_open_scale(action7[0:4, 6]), atol=1e-6)
    np.testing.assert_allclose(sample["proprio"].numpy()[0], achieved[0], atol=1e-6)

    assert sample["prompt"] == "pick up the red mug"
    assert sample["video"][0].size == (320, 384)
    assert sample["video"][0].getpixel((10, 10)) == (255, 0, 0)
    assert sample["video"][0].getpixel((10, 300)) == (0, 255, 0)
    assert sample["video"][0].getpixel((250, 300)) == (0, 0, 0)


def test_boundary_window_masks_clamped_final_target(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path)
        # Train windows enumerate offsets 0..EP_LENGTH-2; the last one holds a
        # 2-frame window whose single real target is step 0.
        sample = dataset[len(dataset) - 1]
    mask = sample["action_mask"].numpy()
    assert mask[0].all()
    assert not mask[1:].any()


def test_libero_rejects_legacy_native_mode_and_missing_map(tmp_path: Path):
    _write_bucket(tmp_path)
    with pytest.raises(ValueError, match="action_mode='eef'"):
        _dataset(tmp_path, action_mode="libero")
    with pytest.raises(ValueError, match="unify_action_map"):
        _dataset(tmp_path, unify_action=True)


def test_unify_maps_eef10_to_80_and_masks_unmapped_dims(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path, unify_action=True, unify_action_map=UNIFY_MAP)
        sample = dataset[0]
        raw_dataset = _dataset(tmp_path)
        raw_sample = raw_dataset[0]

    assert dataset.action_dim == UNIFY_DIM
    assert sample["action"].shape == (4, UNIFY_DIM)
    dst = parse_unify_spec(UNIFY_MAP, UNIFY_DIM)
    mask = sample["action_mask"].numpy()
    assert mask[:, dst].all()
    off = np.setdiff1d(np.arange(UNIFY_DIM), dst)
    assert not mask[:, off].any()
    np.testing.assert_allclose(
        unmap_from_unify(sample["action"].numpy(), dst), raw_sample["action"].numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        unmap_from_unify(sample["proprio"].numpy(), dst), raw_sample["proprio"].numpy(), atol=1e-6
    )


def test_action_and_state_use_one_global_normalization_stats_block(tmp_path: Path):
    _write_bucket(tmp_path)
    stats_path = tmp_path / "source_stats.npy"
    np.save(stats_path, {"eef": _unit_stats(2.0)})

    with _mock_decoder():
        raw_dataset = _dataset(tmp_path)
        raw_sample = raw_dataset[0]
        dataset = _dataset(
            tmp_path,
            normalize_mode="min-max",
            normalization_stats_path=str(stats_path),
        )
        sample = dataset[0]

    raw_action = raw_sample["action"].numpy()
    raw_proprio = raw_sample["proprio"].numpy()
    got_action = sample["action"].numpy()
    got_proprio = sample["proprio"].numpy()
    pos_grip = [0, 1, 2, 9]
    # The same global [-2, 2] transform applies to both directions.
    np.testing.assert_allclose(got_action[:, pos_grip], raw_action[:, pos_grip] / 2.0, atol=1e-6)
    np.testing.assert_allclose(got_proprio[:, pos_grip], raw_proprio[:, pos_grip] / 2.0, atol=1e-6)
    # rot6d dims are pinned to identity: normalization is a pass-through.
    rot = list(ROT6D_DIMS_EEF10)
    np.testing.assert_allclose(got_action[:, rot], raw_action[:, rot], atol=1e-6)
    np.testing.assert_allclose(got_proprio[:, rot], raw_proprio[:, rot], atol=1e-6)


def test_stats_gripper_convention_guard(tmp_path: Path):
    """A stats file bound to a different gripper direction must not be usable."""
    _write_bucket(tmp_path)

    mismatched = tmp_path / "mismatched_stats.npy"
    stats = _unit_stats(2.0)
    stats["gripper_convention"] = "plus1_closed_minus1_open"
    np.save(mismatched, {"eef": stats})
    with pytest.raises(ValueError, match="gripper_convention"):
        _dataset(tmp_path, normalize_mode="min-max", normalization_stats_path=str(mismatched))

    # No marker at all = pre-flip file. Only `mean` is stale, so min-max is
    # allowed (with a warning) while z-score, which consumes `mean`, is not.
    legacy = tmp_path / "legacy_stats.npy"
    legacy_stats = _unit_stats(2.0)
    del legacy_stats["gripper_convention"]
    np.save(legacy, {"eef": legacy_stats})
    _dataset(tmp_path, normalize_mode="min-max", normalization_stats_path=str(legacy))
    with pytest.raises(ValueError, match="predate"):
        _dataset(tmp_path, normalize_mode="z-score", normalization_stats_path=str(legacy))


def test_generated_stats_record_the_gripper_convention(tmp_path: Path):
    _write_bucket(tmp_path)
    _, _, stats, _, _ = _compute_global_stats(_dataset(tmp_path), reservoir_cap=10_000)
    assert stats["gripper_convention"] == GRIPPER_CONVENTION


def test_missing_default_stats_are_auto_built_once(tmp_path: Path):
    _write_bucket(tmp_path)
    default_path = tmp_path / "meta" / "libero_normalization_stats.npy"
    assert not default_path.exists()
    with _mock_decoder():
        dataset = _dataset(tmp_path, normalize_mode="min-max")
        sample = dataset[0]
        raw_sample = _dataset(tmp_path)[0]

    assert default_path.is_file()
    payload = np.load(default_path, allow_pickle=True).item()
    assert set(payload) == {"eef"}
    smin = np.asarray(payload["eef"]["min"], np.float64)
    smax = np.asarray(payload["eef"]["max"], np.float64)
    rot = list(ROT6D_DIMS_EEF10)
    np.testing.assert_allclose(smin[rot], -1.0)
    np.testing.assert_allclose(smax[rot], 1.0)

    # Normalization is applied against the auto-built pooled stats.
    raw = raw_sample["action"].numpy()
    got = sample["action"].numpy()
    pos_grip = [0, 1, 2, 9]
    expect = 2.0 * (raw[:, pos_grip] - smin[pos_grip]) / np.maximum(smax[pos_grip] - smin[pos_grip], 1e-8) - 1.0
    np.testing.assert_allclose(got[:, pos_grip], expect, atol=1e-5)
    np.testing.assert_allclose(got[:, rot], raw[:, rot], atol=1e-6)  # rot6d passthrough

    # A second construction reuses the file instead of recomputing.
    with _mock_decoder(), patch(
        "openwam.dataloader.utils.stats_computation.libero_stats_computation.build_and_save_libero_stats",
        side_effect=AssertionError("stats must not be recomputed when the default file exists"),
    ):
        _dataset(tmp_path, normalize_mode="min-max")


def test_libero_stats_generate_deploy_artifact_and_roundtrip(tmp_path: Path):
    _write_bucket(tmp_path)
    stats_path = tmp_path / "source_stats.npy"
    np.save(stats_path, {"eef": _unit_stats(2.0)})
    with _mock_decoder():
        dataset = _dataset(
            tmp_path,
            unify_action=True,
            unify_action_map=UNIFY_MAP,
            normalize_mode="min-max",
            normalization_stats_path=str(stats_path),
        )
        sample = dataset[0]

    deploy_path = Path(dataset.normalization_stats_path)
    assert deploy_path == tmp_path / "meta" / "normalization_stats.npy"
    payload = np.load(deploy_path, allow_pickle=True).item()
    assert set(payload) == {"eef"}

    checkpoint = tmp_path / "checkpoint"
    save_normalization_stats(str(checkpoint), dataset)
    normalizer = _build_normalizer(
        OmegaConf.create(
            {
                "dataloader": {
                    "normalize_mode": "min-max",
                    "action_mode": "eef",
                    "unify_action": True,
                    "unify_action_map": UNIFY_MAP,
                }
            }
        ),
        str(checkpoint),
    )
    assert isinstance(normalizer, _UnifyAwareNormalizer)
    raw_target = dataset._raw_action_eef10(_raw_window(dataset)[: dataset._num_frames])[:4]
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), raw_target, atol=1e-5)
    # Deploy proprio path uses the same global stats: x/2 -> 80-D scatter.
    unified = normalizer.normalize(np.full((1, EEF10_DIM), 0.4, dtype=np.float32))
    assert unified.shape == (1, UNIFY_DIM)
    np.testing.assert_allclose(unified[0, 0], 0.2, atol=1e-6)
    np.testing.assert_allclose(unified[0, 3], 0.4, atol=1e-6)  # rot6d pinned identity


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


def test_stats_stream_excludes_clamped_boundary_action_row(tmp_path: Path):
    _write_bucket(tmp_path)
    arrays = list(_iter_bucket_arrays(_dataset(tmp_path)))
    assert len(arrays) == 1
    action, state = arrays[0]
    assert action.shape == (EP_LENGTH - 1, EEF10_DIM)
    assert state.shape == (EP_LENGTH, EEF10_DIM)
    np.testing.assert_allclose(action[:, 0:9], state[1:, 0:9], atol=1e-6)


def test_stats_pool_action_and_state_rows_into_one_global_block(tmp_path: Path):
    _write_bucket(tmp_path)
    dataset = _dataset(tmp_path)
    action, state = next(iter(_iter_bucket_arrays(dataset)))
    mode, dim, stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap=10_000)

    pooled = np.concatenate([action, state], axis=0)
    assert mode == "eef"
    assert dim == EEF10_DIM
    assert action_rows == EP_LENGTH - 1
    assert state_rows == EP_LENGTH
    assert stats["num_timesteps"] == pooled.shape[0]
    assert stats["pool"] == "action_state"
    np.testing.assert_allclose(np.asarray(stats["mean"])[[0, 1, 2, 9]], pooled.mean(0)[[0, 1, 2, 9]])
    np.testing.assert_allclose(np.asarray(stats["min"])[[0, 1, 2, 9]], pooled.min(0)[[0, 1, 2, 9]])


def test_registry_builds_libero_dataset(tmp_path: Path):
    _write_bucket(tmp_path)
    dataset = build_dataset(
        OmegaConf.create(
            {
                "type": "libero",
                "dataset_dir": str(tmp_path),
                "num_frames": 5,
                "height": 384,
                "width": 320,
                "multiview": True,
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
    np.save(stats_path, {"eef": _unit_stats(2.0)})

    dataset = LiberoDataset.from_config(
        OmegaConf.create(
            {
                "dataset_dir": str(root),
                "num_frames": 5,
                "height": 384,
                "width": 320,
                "multiview": True,
                "normalize_mode": "min-max",
                "normalization_stats_path": str(stats_path),
            }
        )
    )
    assert isinstance(dataset, MultiLiberoDataset)
    assert dataset.normalization_stats_path == dataset.buckets[0].normalization_stats_path
    checkpoint = tmp_path / "multibucket_checkpoint"
    save_normalization_stats(str(checkpoint), dataset)
    copied = np.load(checkpoint / "normalization_stats.npy", allow_pickle=True).item()
    assert set(copied) == {"eef"}
