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
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.libero import (
    EEF10_DIM,
    GRIPPER_CONVENTION,
    ROT6D_DIMS_EEF10,
    LiberoDataset,
)
from openwam.dataloader.registry import build_dataset, list_registered_datasets
from openwam.dataloader.transforms.video import VideoColorJitter
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
IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _eef10_fixture(rng: np.random.RandomState, *, action: bool) -> np.ndarray:
    values = np.zeros((EP_LENGTH, EEF10_DIM), dtype=np.float32)
    values[:, 0:3] = rng.uniform(-0.5, 0.5, size=(EP_LENGTH, 3))
    if action:
        values[:, 0:3] += np.array([0.7, -0.2, 0.1], dtype=np.float32)
    values[:, 3:9] = IDENTITY_ROT6D
    values[:, 9] = np.linspace(-1.0, 1.0, EP_LENGTH, dtype=np.float32)
    if action:
        values[:, 9] = values[::-1, 9]
    return values


def _write_bucket(bucket: Path) -> None:
    (bucket / "meta" / "episodes").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    for camera in (HEAD, WRIST):
        video_dir = bucket / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"")

    info = {
        "fps": 20.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            HEAD: {"dtype": "video", "shape": [256, 256, 3]},
            WRIST: {"dtype": "video", "shape": [256, 256, 3]},
            "observation.state": {"dtype": "float32", "shape": [EEF10_DIM]},
            "action": {"dtype": "float32", "shape": [EEF10_DIM]},
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
            "action": list(_eef10_fixture(rng, action=True)),
            "observation.state": list(_eef10_fixture(rng, action=False)),
            "task_index": np.zeros(EP_LENGTH, dtype=np.int64),
        }
    )
    pq.write_table(pa.Table.from_pandas(frame), bucket / "data" / "chunk-000" / "file-000.parquet")


def _set_feature_shape(bucket: Path, feature: str, shape: list[int]) -> None:
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][feature]["shape"] = shape
    info_path.write_text(json.dumps(info), encoding="utf-8")


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


def test_registry_and_default_yaml_use_canonical_libero_dataset():
    registered = list_registered_datasets()
    assert "libero" in registered
    config = OmegaConf.load("configs/dataloader/libero.yaml")
    assert config.type == "libero"
    assert config.dataset_dir == "/path/to/libero"
    assert config.action_mode == "eef"
    assert config.gripper_convention == GRIPPER_CONVENTION
    assert list(config.head_camera_priority) == ["observation.images.image"]
    assert list(config.wrist_camera_priority) == [
        "observation.images.wrist_image",
        "observation.images.image2",
    ]
    assert config.unify_action is True
    assert list(config.unify_action_map) == ["0-9"]
    assert OmegaConf.to_container(config.color_jitter, resolve=True) == {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.0,
    }
    assert "representation" not in config
    assert "gripper_cmd_encoding" not in config
    assert not Path("configs/dataloader/libero_absolute_eef10.yaml").exists()


def test_libero_rejects_non_eef10_disk_contract(tmp_path: Path):
    action_dir = tmp_path / "action7"
    _write_bucket(action_dir)
    _set_feature_shape(action_dir, "action", [7])
    with pytest.raises(ValueError, match=r"action feature must have shape \[10\]"):
        _dataset(action_dir)

    state_dir = tmp_path / "state8"
    _write_bucket(state_dir)
    _set_feature_shape(state_dir, "observation.state", [8])
    with pytest.raises(ValueError, match=r"observation.state feature must have shape \[10\]"):
        _dataset(state_dir)


def test_libero_rejects_historical_multibucket_root(tmp_path: Path):
    root = tmp_path / "root"
    _write_bucket(root / "suite_a")
    with pytest.raises(FileNotFoundError, match="meta/info.json missing"):
        _dataset(root)


def test_libero_consumes_state_and_action_row_aligned(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path)
        sample = dataset[0]

    assert dataset.action_dim == EEF10_DIM
    assert sample["action"].shape == (4, EEF10_DIM)
    assert sample["proprio"].shape == (1, EEF10_DIM)
    assert sample["proprio_mask"].all()
    assert sample["action_mask"].all()

    window = _raw_window(dataset)
    raw_state = np.stack(window["observation.state"].values)
    raw_action = np.stack(window["action"].values)
    np.testing.assert_array_equal(sample["action"].numpy(), raw_action[:4])
    np.testing.assert_array_equal(sample["proprio"].numpy(), raw_state[:1])

    assert sample["prompt"] == "pick up the red mug"
    assert sample["video"][0].size == (320, 384)
    assert sample["video"][0].getpixel((10, 10)) == (255, 0, 0)
    assert sample["video"][0].getpixel((10, 300)) == (0, 255, 0)
    assert sample["video"][0].getpixel((250, 300)) == (0, 0, 0)


def test_color_jitter_excludes_only_the_missing_camera_slot(tmp_path: Path):
    _write_bucket(tmp_path)

    def _decode_with_real_black_pixels(path, frame_indices, h, w):
        color = (255, 0, 0) if HEAD in str(path) else (0, 255, 0)
        frame = Image.new("RGB", (w, h), color)
        if HEAD in str(path):
            # Real black content inside a valid camera must still be jittered;
            # only the structurally missing bottom-right slot is excluded.
            frame.paste((0, 0, 0), (0, 0, w // 4, h // 4))
        return [frame.copy() for _ in frame_indices]

    dataset = _dataset(
        tmp_path,
        color_jitter={"brightness": 0.0, "contrast": 0.2, "saturation": 0.0, "hue": 0.0},
    )
    with (
        patch(
            "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames",
            side_effect=_decode_with_real_black_pixels,
        ),
        patch(
            "openwam.dataloader.transforms.video.random.uniform",
            side_effect=[0.0, -0.2, 0.0],
        ),
    ):
        frame = dataset[0]["video"][0]

    # Missing right-wrist slot remains byte-exact black under contrast < 1.
    assert frame.getpixel((250, 300)) == (0, 0, 0)
    # A black pixel belonging to the valid head view is not mistaken for padding.
    assert frame.getpixel((10, 10)) != (0, 0, 0)
    # Non-black camera content is still augmented.
    assert frame.getpixel((200, 100)) != (255, 0, 0)


def test_zero_exclusion_mask_matches_original_color_jitter_path():
    rng = np.random.default_rng(42)
    source = Image.fromarray(rng.integers(0, 256, size=(65, 97, 3), dtype=np.uint8))
    jitter = VideoColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    random.seed(1234)
    unmasked = jitter({"video": [source.copy()]})["video"][0]
    random.seed(1234)
    zero_mask = Image.new("L", source.size, 0)
    masked = jitter({"video": [source.copy()], "video_jitter_exclusion_masks": [zero_mask]})["video"][0]

    np.testing.assert_array_equal(np.asarray(masked), np.asarray(unmasked))


def test_final_episode_row_is_a_real_supervised_action(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path)
        sample = dataset[len(dataset) - 1]
    raw_action = np.stack(_raw_window(dataset)["action"].values)
    mask = sample["action_mask"].numpy()
    assert mask[0].all()
    assert not mask[1:].any()
    np.testing.assert_array_equal(sample["action"].numpy()[0], raw_action[-1])


def test_libero_rejects_native_mode_and_missing_unify_map(tmp_path: Path):
    _write_bucket(tmp_path)
    with pytest.raises(ValueError, match="action_mode='eef'"):
        _dataset(tmp_path, action_mode="libero")
    with pytest.raises(ValueError, match="unify_action_map"):
        _dataset(tmp_path, unify_action=True)
    with pytest.raises(ValueError, match="gripper_convention"):
        _dataset(tmp_path, gripper_convention="plus1_closed_minus1_open")


def test_unify_maps_eef10_to_80_and_masks_unmapped_dims(tmp_path: Path):
    _write_bucket(tmp_path)
    with _mock_decoder():
        dataset = _dataset(tmp_path, unify_action=True, unify_action_map=UNIFY_MAP)
        sample = dataset[0]
        raw_sample = _dataset(tmp_path)[0]

    assert dataset.action_dim == UNIFY_DIM
    assert sample["action"].shape == (4, UNIFY_DIM)
    dst = parse_unify_spec(UNIFY_MAP, UNIFY_DIM)
    mask = sample["action_mask"].numpy()
    assert mask[:, dst].all()
    off = np.setdiff1d(np.arange(UNIFY_DIM), dst)
    assert not mask[:, off].any()
    np.testing.assert_allclose(unmap_from_unify(sample["action"].numpy(), dst), raw_sample["action"].numpy())
    np.testing.assert_allclose(unmap_from_unify(sample["proprio"].numpy(), dst), raw_sample["proprio"].numpy())


def test_action_and_state_share_one_normalization_block(tmp_path: Path):
    _write_bucket(tmp_path)
    stats_path = tmp_path / "source_stats.npy"
    np.save(stats_path, {"eef": _unit_stats(2.0)})

    with _mock_decoder():
        raw_sample = _dataset(tmp_path)[0]
        sample = _dataset(
            tmp_path,
            normalize_mode="min-max",
            normalization_stats_path=str(stats_path),
        )[0]

    pos_grip = [0, 1, 2, 9]
    rot = list(ROT6D_DIMS_EEF10)
    np.testing.assert_allclose(
        sample["action"].numpy()[:, pos_grip], raw_sample["action"].numpy()[:, pos_grip] / 2, atol=1e-6
    )
    np.testing.assert_allclose(
        sample["proprio"].numpy()[:, pos_grip], raw_sample["proprio"].numpy()[:, pos_grip] / 2, atol=1e-6
    )
    np.testing.assert_allclose(sample["action"].numpy()[:, rot], raw_sample["action"].numpy()[:, rot], atol=1e-6)
    np.testing.assert_allclose(sample["proprio"].numpy()[:, rot], raw_sample["proprio"].numpy()[:, rot], atol=1e-6)


def test_stats_require_canonical_gripper_convention(tmp_path: Path):
    _write_bucket(tmp_path)
    for marker in ("plus1_closed_minus1_open", None):
        path = tmp_path / f"stats-{marker}.npy"
        stats = _unit_stats(2.0)
        if marker is None:
            del stats["gripper_convention"]
        else:
            stats["gripper_convention"] = marker
        np.save(path, {"eef": stats})
        with pytest.raises(ValueError, match="gripper_convention"):
            _dataset(tmp_path, normalize_mode="min-max", normalization_stats_path=str(path))


def test_generated_stats_include_every_row_and_contract_marker(tmp_path: Path):
    _write_bucket(tmp_path)
    dataset = _dataset(tmp_path)
    action, state = next(iter(_iter_bucket_arrays(dataset)))
    mode, dim, stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap=10_000)
    raw = _raw_window(dataset)

    np.testing.assert_array_equal(action, np.stack(raw["action"].values))
    np.testing.assert_array_equal(state, np.stack(raw["observation.state"].values))
    assert mode == "eef"
    assert dim == EEF10_DIM
    assert action_rows == state_rows == EP_LENGTH
    assert stats["num_timesteps"] == 2 * EP_LENGTH
    assert stats["pool"] == "action_state"
    assert stats["gripper_convention"] == GRIPPER_CONVENTION


def test_missing_default_stats_are_built_and_reused(tmp_path: Path):
    _write_bucket(tmp_path)
    default_path = tmp_path / "meta" / "normalization_stats.npy"
    with _mock_decoder():
        dataset = _dataset(tmp_path, normalize_mode="min-max")
        sample = dataset[0]
        raw_sample = _dataset(tmp_path)[0]

    assert Path(dataset.normalization_stats_path) == default_path
    payload = np.load(default_path, allow_pickle=True).item()
    assert payload["eef"]["gripper_convention"] == GRIPPER_CONVENTION
    assert payload["eef"]["action_rows"] == payload["eef"]["state_rows"] == EP_LENGTH
    smin = np.asarray(payload["eef"]["min"], np.float64)
    smax = np.asarray(payload["eef"]["max"], np.float64)
    rot = list(ROT6D_DIMS_EEF10)
    np.testing.assert_allclose(smin[rot], -1.0)
    np.testing.assert_allclose(smax[rot], 1.0)
    pos_grip = [0, 1, 2, 9]
    expected = 2 * (raw_sample["action"].numpy()[:, pos_grip] - smin[pos_grip]) / np.maximum(
        smax[pos_grip] - smin[pos_grip], 1e-8
    ) - 1
    np.testing.assert_allclose(sample["action"].numpy()[:, pos_grip], expected, atol=1e-5)
    np.testing.assert_allclose(sample["action"].numpy()[:, rot], raw_sample["action"].numpy()[:, rot])

    with _mock_decoder(), patch(
        "openwam.dataloader.utils.stats_computation.libero_stats_computation.build_and_save_libero_stats",
        side_effect=AssertionError("stats must be reused"),
    ):
        _dataset(tmp_path, normalize_mode="min-max")


def test_one_stats_file_serves_training_checkpoint_and_deployment(tmp_path: Path):
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
    assert deploy_path == stats_path
    assert not (tmp_path / "meta" / "normalization_stats.npy").exists()
    checkpoint = tmp_path / "checkpoint"
    save_normalization_stats(str(checkpoint), dataset)
    checkpoint_payload = np.load(checkpoint / "normalization_stats.npy", allow_pickle=True).item()
    assert checkpoint_payload["eef"]["gripper_convention"] == GRIPPER_CONVENTION
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


def test_libero_prompt_column_falls_back_to_tasks_parquet(tmp_path: Path):
    fallback_dir = tmp_path / "fallback"
    _write_bucket(fallback_dir)
    _add_prompt_column(fallback_dir, "language_instruction", [""] * EP_LENGTH)
    with _mock_decoder():
        assert _dataset(fallback_dir)[0]["prompt"] == "pick up the red mug"

    direct_dir = tmp_path / "direct"
    _write_bucket(direct_dir)
    _add_prompt_column(direct_dir, "language_instruction", ["lift the crimson cup"] * EP_LENGTH)
    with _mock_decoder():
        assert _dataset(direct_dir)[0]["prompt"] == "lift the crimson cup"


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
