"""End-to-end VLABenchDataset construction over a synthetic LeRobot v3 bucket.

``test_vlabench_offsets.py`` drives ``_add_data_offsets`` in isolation via
``object.__new__`` — deliberately, so the repair can be tested without a bucket.
This file covers everything that only happens once the reader is really
constructed: the euler7 -> EEF10 conversion, normalization, the unify scatter
into slots 0-9 and the resulting ``action_mask``, prompt resolution through
``task_index``, the deploy normalizer artifact, and — the one that matters most —
that a window actually reads the REPAIRED shard rather than the published one.

The bucket carries the upstream metadata bug verbatim: ``dataset_from_index =
length * episode_index`` with ``data/file_index`` mis-assigned in step with it.
Follows the synthetic-bucket + patched-``_decode_video_frames`` pattern of
``test_libero.py``, so no mp4 encoder is needed.
"""

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

from openwam.dataloader.registry import list_registered_datasets
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_ARM10,
    apply_normalization,
    materialize_eef_stats,
    pin_rot6d_identity,
)
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10
from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec
from openwam.dataloader.vlabench import EEF10_DIM, VLABenchDataset

HEAD = "image"
LEFT_WRIST = "wrist_image"
RIGHT_WRIST = "second_image"
CAMERAS = (HEAD, RIGHT_WRIST, LEFT_WRIST)

EULER7_DIM = 7
NUM_FRAMES = 33
T_ACTION = NUM_FRAMES - 1
UNIFY_MAP = ["0-9"]

# The reviewed fixture: 3 episodes packed into 2 shards.
#   ep0 -> file-000 @ 0    (40 rows)
#   ep1 -> file-000 @ 40   (33 rows)
#   ep2 -> file-001 @ 0    (50 rows)
LENGTHS = [40, 33, 50]
SHARD_ROWS = [73, 50]
PROMPTS = ["select the fruit", "place the poker", "add the condiment"]

# actions[:, 0] carries the GLOBAL row index / 1000 so a window's contents
# identify exactly which parquet rows were read.
ROW_MARKER_SCALE = 0.001


def _write_bucket(bucket: Path) -> None:
    """A 3-episode bucket carrying the upstream episode-metadata bug verbatim."""
    (bucket / "meta" / "episodes").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    for camera in CAMERAS:
        video_dir = bucket / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        for ep in range(len(LENGTHS)):
            (video_dir / f"file-{ep:03d}.mp4").write_bytes(b"")

    info = {
        "fps": 10.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            **{cam: {"dtype": "video", "shape": [480, 480, 3]} for cam in CAMERAS},
            "state": {"dtype": "float32", "shape": [EULER7_DIM]},
            "actions": {"dtype": "float32", "shape": [EULER7_DIM]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    lengths = np.asarray(LENGTHS, dtype=np.int64)
    episode_index = np.arange(len(LENGTHS), dtype=np.int64)
    rows = {
        "episode_index": episode_index,
        "length": lengths,
        # THE UPSTREAM BUG, verbatim: as if every episode had the current one's length.
        "dataset_from_index": lengths * episode_index,
        "dataset_to_index": lengths * (episode_index + 1),
        # Mis-assigned in step with it — every episode claims shard 0.
        "data/chunk_index": np.zeros(len(LENGTHS), dtype=np.int64),
        "data/file_index": np.zeros(len(LENGTHS), dtype=np.int64),
    }
    for camera in CAMERAS:
        # Video metadata is sound upstream: one mp4 per episode, correct indices.
        rows[f"videos/{camera}/chunk_index"] = np.zeros(len(LENGTHS), dtype=np.int64)
        rows[f"videos/{camera}/file_index"] = episode_index
        rows[f"videos/{camera}/from_timestamp"] = np.zeros(len(LENGTHS), dtype=np.float64)
        rows[f"videos/{camera}/to_timestamp"] = lengths / 10.0
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows)),
        bucket / "meta" / "episodes" / "chunk-000.parquet",
    )

    pd.DataFrame(
        {"task_index": list(range(len(PROMPTS)))},
        index=pd.Index(PROMPTS, name="task"),
    ).to_parquet(bucket / "meta" / "tasks.parquet")

    # One task_index per episode, laid out in true packing order.
    task_per_row = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(LENGTHS)])
    rng = np.random.RandomState(7)
    global_row = 0
    for shard, n_rows in enumerate(SHARD_ROWS):
        state = rng.uniform(-0.5, 0.5, size=(n_rows, EULER7_DIM)).astype(np.float32)
        actions = rng.uniform(-0.5, 0.5, size=(n_rows, EULER7_DIM)).astype(np.float32)
        actions[:, 0] = (np.arange(global_row, global_row + n_rows) * ROW_MARKER_SCALE).astype(np.float32)
        frame = pd.DataFrame(
            {
                "state": list(state),
                "actions": list(actions),
                "task_index": task_per_row[global_row : global_row + n_rows],
            }
        )
        pq.write_table(pa.Table.from_pandas(frame), bucket / "data" / "chunk-000" / f"file-{shard:03d}.parquet")
        global_row += n_rows

    _write_stats(bucket)


def _write_stats(bucket: Path) -> None:
    """``meta/vlabench_normalization_stats.npy`` in the flat stats schema.

    rot6d dims are pinned to identity, matching what
    ``vlabench_stats_computation`` writes (and the real release: dims 3:9 are
    +-1). Without the pin the reader warns that normalization would distort the
    rotation representation — a fixture that skips it is not representative.
    """
    arrays = {
        "min": np.full(EEF10_DIM, -2.0, dtype=np.float32),
        "max": np.full(EEF10_DIM, 2.0, dtype=np.float32),
        "mean": np.zeros(EEF10_DIM, dtype=np.float32),
        "std": np.ones(EEF10_DIM, dtype=np.float32),
        "q01": np.full(EEF10_DIM, -2.0, dtype=np.float32),
        "q99": np.full(EEF10_DIM, 2.0, dtype=np.float32),
    }
    pin_rot6d_identity(arrays, ROT6D_DIMS_ARM10)
    stats = {"n_samples": int(sum(SHARD_ROWS)) * 2}
    stats.update({k: v.tolist() for k, v in arrays.items()})
    np.save(bucket / "meta" / "vlabench_normalization_stats.npy", stats, allow_pickle=True)


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), (17, 17, 17)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _dataset(bucket: Path, **overrides) -> VLABenchDataset:
    config = {
        "dataset_dir": str(bucket),
        "num_frames": NUM_FRAMES,
        "video_stride": 4,
        "window_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "normalize_mode": "min-max",
        "unify_action": True,
        "unify_action_map": UNIFY_MAP,
    }
    config.update(overrides)
    return VLABenchDataset.from_config(OmegaConf.create(config), split="train")


@pytest.fixture
def bucket(tmp_path: Path) -> Path:
    _write_bucket(tmp_path)
    return tmp_path


def _stats_for(bucket: Path) -> dict:
    raw = np.load(bucket / "meta" / "vlabench_normalization_stats.npy", allow_pickle=True).item()
    return materialize_eef_stats(raw, "min-max", dim=EEF10_DIM, strict_minmax=True)


def test_registry_and_yaml_agree(bucket: Path):
    assert "vlabench" in list_registered_datasets()
    config = OmegaConf.load("configs/dataloader/vlabench.yaml")
    assert config.type == "vlabench"
    assert config.unify_action is True
    assert list(config.unify_action_map) == UNIFY_MAP


def test_construction_repairs_the_published_shard_assignment(bucket: Path):
    with _mock_decoder():
        dataset = _dataset(bucket)

    eps = dataset._eps_df.sort_values("episode_index")
    assert list(eps["data/file_index"]) == [0, 0, 1]
    assert list(eps["data/chunk_index"]) == [0, 0, 0]
    assert list(eps["_data_row_offset"]) == [0, 40, 0]
    # Windows: one per labeled step, so sum(lengths).
    assert len(dataset) == sum(LENGTHS)


def test_window_reads_the_repaired_shard_not_the_published_one(bucket: Path):
    """Episode 2 is the decisive case: published metadata sends it to file-000."""
    with _mock_decoder():
        dataset = _dataset(bucket, normalize_mode=None)
        first_window_of_ep2 = int(dataset._cum_n_starts[2])
        sample = dataset[first_window_of_ep2]

    # Episode 2 occupies global rows 73..122, i.e. file-001 rows 0..49.
    expected_markers = np.arange(73, 73 + T_ACTION) * ROW_MARKER_SCALE
    got = sample["action"].numpy()[:, 0]
    np.testing.assert_allclose(got, expected_markers, atol=1e-5)
    # Had the published (wrong) file_index been trusted, the window would have
    # started at global row 100 inside file-000, which only holds 73 rows.
    assert not np.allclose(got[0], 100 * ROW_MARKER_SCALE)


def test_unify_scatters_eef10_into_slots_0_9_and_masks_the_rest(bucket: Path):
    with _mock_decoder():
        dataset = _dataset(bucket)
        sample = dataset[0]
        raw_window = dataset._load_data_table(0, 0).to_pandas().iloc[:T_ACTION]

    assert dataset.action_dim == UNIFY_DIM
    action = sample["action"].numpy()
    assert action.shape == (T_ACTION, UNIFY_DIM)

    dst = parse_unify_spec(UNIFY_MAP, UNIFY_DIM)
    expected = apply_normalization(
        euler7_action_to_arm10(np.stack(raw_window["actions"].values).astype(np.float32)),
        _stats_for(bucket),
        "min-max",
    )
    np.testing.assert_allclose(action[:, dst], expected, atol=1e-6)

    off = np.setdiff1d(np.arange(UNIFY_DIM), dst)
    assert np.count_nonzero(action[:, off]) == 0

    mask = sample["action_mask"].numpy()
    assert mask.shape == (T_ACTION, UNIFY_DIM)
    assert mask[:, dst].all()
    assert not mask[:, off].any()


def test_proprio_is_one_scattered_step(bucket: Path):
    with _mock_decoder():
        dataset = _dataset(bucket)
        sample = dataset[0]
        raw_window = dataset._load_data_table(0, 0).to_pandas()

    proprio = sample["proprio"].numpy()
    assert proprio.shape == (1, UNIFY_DIM)
    dst = parse_unify_spec(UNIFY_MAP, UNIFY_DIM)
    expected = apply_normalization(
        euler7_action_to_arm10(np.stack(raw_window["state"].values[:1]).astype(np.float32)),
        _stats_for(bucket),
        "min-max",
    )
    np.testing.assert_allclose(proprio[:, dst], expected, atol=1e-6)
    off = np.setdiff1d(np.arange(UNIFY_DIM), dst)
    assert np.count_nonzero(proprio[:, off]) == 0


def test_prompt_resolves_through_task_index(bucket: Path):
    with _mock_decoder():
        dataset = _dataset(bucket)
        for ep, prompt in enumerate(PROMPTS):
            sample = dataset[int(dataset._cum_n_starts[ep])]
            assert sample["prompt"] == prompt


def test_load_stats_writes_the_deploy_normalizer_artifact(bucket: Path):
    artifact = bucket / "meta" / "normalization_stats.npy"
    assert not artifact.exists()

    with _mock_decoder():
        _dataset(bucket)

    assert artifact.exists()
    saved = np.load(artifact, allow_pickle=True).item()
    # The deploy server reads this key from `dataloader.action_mode`.
    assert VLABenchDataset.DEPLOY_ACTION_MODE in saved
    block = saved[VLABenchDataset.DEPLOY_ACTION_MODE]
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        assert np.asarray(block[key]).shape == (EEF10_DIM,)


def test_rot6d_dims_are_normalization_passthrough(bucket: Path):
    """min-max must not rescale the rotation representation per-dim."""
    probe = np.zeros((1, EEF10_DIM), dtype=np.float32)
    probe[0, ROT6D_DIMS_ARM10] = 0.42
    out = apply_normalization(probe.copy(), _stats_for(bucket), "min-max")
    np.testing.assert_allclose(out[0, ROT6D_DIMS_ARM10], 0.42, atol=1e-6)


def test_multiview_canvas_is_the_l_shape_at_configured_size(bucket: Path):
    with _mock_decoder():
        dataset = _dataset(bucket)
        sample = dataset[0]
    frames = sample["video"]
    # video_stride=4 over a 33-frame window -> ceil(33/4) sampled frames.
    assert len(frames) == len(range(0, NUM_FRAMES, 4))
    for frame in frames:
        assert frame.size == (320, 384)  # PIL is (W, H)


def test_shards_are_ordered_numerically_not_lexicographically(tmp_path: Path):
    """Shard order must come from the parsed ints, not the path string.

    ``file-10`` sorts BEFORE ``file-9`` lexicographically, so a string-ordered
    walk builds the cumulative row offsets against the wrong packing order and
    every episode after the first lands in the wrong shard. The shipped release
    hides this behind ``chunks_size: 1000`` (file_index never exceeds 3 digits),
    which is an upstream layout detail this reader neither reads nor enforces.
    """
    _write_bucket(tmp_path)
    data_dir = tmp_path / "data" / "chunk-000"
    # 999 / 1000 is the first pair where the "{file_index:03d}" path template
    # stops zero-padding, so the on-disk names are what a real writer would emit.
    (data_dir / "file-000.parquet").rename(data_dir / "file-999.parquet")
    (data_dir / "file-001.parquet").rename(data_dir / "file-1000.parquet")

    with _mock_decoder():
        dataset = _dataset(tmp_path, normalize_mode=None)
        eps = dataset._eps_df.sort_values("episode_index")
        assert list(eps["data/file_index"]) == [999, 999, 1000]
        assert list(eps["_data_row_offset"]) == [0, 40, 0]
        sample = dataset[int(dataset._cum_n_starts[2])]

    expected = np.arange(73, 73 + T_ACTION) * ROW_MARKER_SCALE
    np.testing.assert_allclose(sample["action"].numpy()[:, 0], expected, atol=1e-5)
