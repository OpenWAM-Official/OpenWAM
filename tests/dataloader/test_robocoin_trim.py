"""Regression tests for RoboCOIN's CSV-driven dead-frame trimming.

The end-to-end case uses a synthetic LeRobot v3 bucket whose action payload
and decoded pixels both encode their source row/frame index.  This catches a
trim offset being applied to parquet but not video (or vice versa).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from openwam.dataloader.robocoin import (
    ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS,
    RoboCOINDataset,
    _load_trim_spec,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError

CAM = "observation.images.cam_head_rgb"
TRIM_COLUMNS = (
    "dataset",
    "episode_index",
    "total_frames",
    "trim_head_to",
    "trim_tail_from",
)


def _write_trim_csv(path: Path, rows: list[dict], columns=TRIM_COLUMNS) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _trim_row(**overrides) -> dict:
    row = {
        "dataset": "bucket",
        "episode_index": 0,
        "total_frames": 10,
        "trim_head_to": 2,
        "trim_tail_from": 8,
    }
    row.update(overrides)
    return row


def _episodes(lengths: list[int]) -> pd.DataFrame:
    starts = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    return pd.DataFrame(
        {
            "episode_index": np.arange(len(lengths), dtype=np.int64),
            "length": lengths,
            "_data_row_offset": starts,
            f"_video_frame_offset/{CAM}": starts,
            "_video_frame_offset/second_camera": starts + 100,
        }
    )


def _filter(tmp_path: Path, eps: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    trim_csv = _write_trim_csv(tmp_path / "trim.csv", rows)
    reader = RoboCOINDataset.__new__(RoboCOINDataset)
    reader._trim_csv = trim_csv
    reader._dataset_id = "bucket"
    reader._fps = 30.0
    return reader._filter_episodes(eps)


@pytest.fixture(autouse=True)
def _isolate_trim_cache():
    # Immutable trim snapshots are shared by many bucket readers.
    # Tests reuse tmp_path names across separate pytest processes, so isolate it.
    from openwam.dataloader import robocoin

    robocoin._TRIM_SNAPSHOT_CACHE.clear()
    yield
    robocoin._TRIM_SNAPSHOT_CACHE.clear()


class TestLoadTrimSpec:
    def test_parses_empty_head_or_tail_and_skips_noop_rows(self, tmp_path):
        path = _write_trim_csv(
            tmp_path / "trim.csv",
            [
                {
                    "dataset": "bucket-a",
                    "episode_index": 0,
                    "total_frames": 10,
                    "trim_head_to": 2,
                    "trim_tail_from": "",
                },
                {
                    "dataset": "bucket-a",
                    "episode_index": 1,
                    "total_frames": 12,
                    "trim_head_to": "",
                    "trim_tail_from": 9,
                },
                {
                    "dataset": "bucket-a",
                    "episode_index": 2,
                    "total_frames": 8,
                    "trim_head_to": "",
                    "trim_tail_from": "",
                },
                {
                    "dataset": "bucket-b",
                    "episode_index": 3,
                    "total_frames": 6,
                    "trim_head_to": 1,
                    "trim_tail_from": 5,
                },
            ],
        )

        assert _load_trim_spec(path) == {
            "bucket-a": {0: (2, None, 10), 1: (0, 9, 12)},
            "bucket-b": {3: (1, 5, 6)},
        }

    def test_explicit_missing_path_fails_fast(self, tmp_path):
        with pytest.raises((OSError, ValueError)):
            _load_trim_spec(tmp_path / "does-not-exist.csv")

    def test_empty_string_is_not_the_disabled_value(self):
        # Only trim_csv=None is an opt-out; a configured empty path is invalid.
        with pytest.raises((OSError, ValueError)):
            _load_trim_spec("")

    @pytest.mark.parametrize("missing", TRIM_COLUMNS)
    def test_missing_required_column_fails_fast(self, tmp_path, missing):
        columns = tuple(c for c in TRIM_COLUMNS if c != missing)
        row = {
            "dataset": "bucket",
            "episode_index": 0,
            "total_frames": 10,
            "trim_head_to": 2,
            "trim_tail_from": 8,
        }
        path = _write_trim_csv(tmp_path / f"missing-{missing}.csv", [row], columns)
        with pytest.raises(ValueError):
            _load_trim_spec(path)

    def test_empty_total_frames_fails_fast(self, tmp_path):
        path = _write_trim_csv(
            tmp_path / "empty-total.csv",
            [
                {
                    "dataset": "bucket",
                    "episode_index": 0,
                    "total_frames": "",
                    "trim_head_to": 2,
                    "trim_tail_from": 8,
                }
            ],
        )
        with pytest.raises(ValueError):
            _load_trim_spec(path)

    def test_failed_parse_is_not_cached(self, tmp_path):
        from openwam.dataloader import robocoin

        path = _write_trim_csv(
            tmp_path / "repairable.csv",
            [_trim_row()],
            columns=tuple(c for c in TRIM_COLUMNS if c != "total_frames"),
        )
        with pytest.raises(ValueError):
            _load_trim_spec(path)
        assert str(path) not in robocoin._TRIM_SNAPSHOT_CACHE

        _write_trim_csv(path, [_trim_row()])
        assert _load_trim_spec(path) == {"bucket": {0: (2, 8, 10)}}

    def test_cache_refreshes_when_same_path_is_replaced(self, tmp_path):
        path = _write_trim_csv(tmp_path / "replaceable.csv", [_trim_row(trim_head_to=2)])
        assert _load_trim_spec(path) == {"bucket": {0: (2, 8, 10)}}

        replacement = _write_trim_csv(tmp_path / "replacement.csv", [_trim_row(trim_head_to=3)])
        replacement.replace(path)
        assert _load_trim_spec(path) == {"bucket": {0: (3, 8, 10)}}

    @pytest.mark.parametrize(
        "overrides",
        [
            {"trim_head_to": -1, "trim_tail_from": ""},
            {"trim_head_to": 11, "trim_tail_from": ""},
            {"trim_head_to": "", "trim_tail_from": -1},
            {"trim_head_to": "", "trim_tail_from": 11},
            {"trim_head_to": 8, "trim_tail_from": 3},
        ],
        ids=["negative-head", "head-past-total", "negative-tail", "tail-past-total", "head-after-tail"],
    )
    def test_invalid_trim_bounds_fail_fast(self, tmp_path, overrides):
        path = _write_trim_csv(tmp_path / "invalid-bounds.csv", [_trim_row(**overrides)])
        with pytest.raises(ValueError):
            _load_trim_spec(path)

    def test_duplicate_dataset_episode_key_fails_fast(self, tmp_path):
        path = _write_trim_csv(
            tmp_path / "duplicate.csv",
            [_trim_row(trim_head_to=1), _trim_row(trim_head_to=2)],
        )
        with pytest.raises(ValueError):
            _load_trim_spec(path)


class TestFilterEpisodes:
    def test_head_tail_and_both_update_all_offsets(self, tmp_path):
        eps = _episodes([10, 10, 10])
        out = _filter(
            tmp_path,
            eps,
            [
                {
                    "dataset": "bucket",
                    "episode_index": 0,
                    "total_frames": 10,
                    "trim_head_to": 2,
                    "trim_tail_from": "",
                },
                {
                    "dataset": "bucket",
                    "episode_index": 1,
                    "total_frames": 10,
                    "trim_head_to": "",
                    "trim_tail_from": 7,
                },
                {
                    "dataset": "bucket",
                    "episode_index": 2,
                    "total_frames": 10,
                    "trim_head_to": 2,
                    "trim_tail_from": 8,
                },
            ],
        )

        assert out["length"].tolist() == [8, 7, 6]
        assert out["_data_row_offset"].tolist() == [2, 10, 22]
        assert out[f"_video_frame_offset/{CAM}"].tolist() == [2, 10, 22]
        assert out["_video_frame_offset/second_camera"].tolist() == [102, 110, 122]

    def test_one_stale_entry_fails_closed_with_data_contract_error(self, tmp_path):
        eps = _episodes([10, 10])
        with pytest.raises(DataContractError, match="[Ss][Tt][Aa][Ll][Ee]"):
            _filter(
                tmp_path,
                eps,
                [
                    {
                        "dataset": "bucket",
                        "episode_index": 0,
                        "total_frames": 999,
                        "trim_head_to": 1,
                        "trim_tail_from": 9,
                    },
                    {
                        # This simulates an undetectable same-length collision.
                        # Rejecting the bucket prevents this entry being applied.
                        "dataset": "bucket",
                        "episode_index": 1,
                        "total_frames": 10,
                        "trim_head_to": 2,
                        "trim_tail_from": 8,
                    },
                ],
            )

    def test_zero_span_trim_drops_episode(self, tmp_path):
        eps = _episodes([10, 10])
        out = _filter(
            tmp_path,
            eps,
            [
                {
                    "dataset": "bucket",
                    "episode_index": 0,
                    "total_frames": 10,
                    "trim_head_to": 10,
                    "trim_tail_from": "",
                }
            ],
        )

        assert out["episode_index"].tolist() == [1]
        assert out["_data_row_offset"].tolist() == [10]
        assert out[f"_video_frame_offset/{CAM}"].tolist() == [10]


def _make_bucket(bucket: Path, lengths: list[int]) -> Path:
    meta = bucket / "meta"
    (meta / "episodes").mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "robot_type": "test_robot",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    CAM: {"dtype": "video"},
                    "eef_sim_pose_action": {},
                    "gripper_open_scale_action": {},
                    "eef_sim_pose_state": {},
                    "gripper_open_scale_state": {},
                },
            }
        )
    )

    rows = []
    start = 0
    for ep, length in enumerate(lengths):
        rows.append(
            {
                "episode_index": ep,
                "length": length,
                "dataset_from_index": start,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{CAM}/chunk_index": 0,
                f"videos/{CAM}/file_index": 0,
            }
        )
        start += length
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), meta / "episodes" / "chunk-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["do the thing"], name="task")).to_parquet(meta / "tasks.parquet")

    source_index = np.arange(start, dtype=np.float32)
    eef = np.zeros((start, 12), dtype=np.float32)
    eef[:, 0] = source_index
    episode_index = np.concatenate([np.full(length, ep, dtype=np.int64) for ep, length in enumerate(lengths)])
    frame_index = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "task_index": np.zeros(start, dtype=np.int64),
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "eef_sim_pose_action": list(eef),
                    "gripper_open_scale_action": list(np.zeros((start, 2), dtype=np.float32)),
                    "eef_sim_pose_state": list(eef),
                    "gripper_open_scale_state": list(np.zeros((start, 2), dtype=np.float32)),
                }
            )
        ),
        data_dir / "file-000.parquet",
    )
    return bucket


def _set_info_splits(bucket: Path, splits: dict) -> None:
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = splits
    info_path.write_text(json.dumps(info))


@pytest.fixture
def patch_decode(monkeypatch):
    def fake_decode(path, frame_indices, height, width):
        del path
        frames = []
        for frame_index in frame_indices:
            image = Image.new("RGB", (width, height), (0, 0, 0))
            image.putpixel((0, 0), (int(frame_index), 0, 0))
            frames.append(image)
        return frames

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


def _decoded_index(image: Image.Image) -> int:
    return image.getpixel((0, 0))[0]


def test_root_from_config_preloads_bad_trim_csv_before_bucket_workers(tmp_path):
    root = tmp_path / "root"
    _make_bucket(root / "bucket-a", [8])
    _make_bucket(root / "bucket-b", [8])
    bad_csv = _write_trim_csv(
        tmp_path / "bad.csv",
        [_trim_row()],
        columns=tuple(c for c in TRIM_COLUMNS if c != "total_frames"),
    )

    # build_multibucket deliberately tolerates an individual corrupt bucket.
    # A shared trim_csv is global configuration, so it must be validated before
    # entering those workers and preserve the actionable parse exception.
    with pytest.raises(ValueError, match="trim_csv"):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(bad_csv)})


def test_root_mode_trim_csv_with_no_bucket_key_overlap_fails_fast(tmp_path):
    root = tmp_path / "root"
    _make_bucket(root / "bucket-a", [8])
    _make_bucket(root / "bucket-b", [8])
    trim_csv = _write_trim_csv(
        tmp_path / "wrong-root.csv",
        [
            _trim_row(
                dataset="bucket-from-another-root",
                total_frames=8,
                trim_head_to=1,
                trim_tail_from=7,
            )
        ],
    )

    with pytest.raises(ValueError):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})


def test_root_mode_trim_csv_with_partial_bucket_key_overlap_is_allowed(tmp_path):
    root = tmp_path / "root"
    _make_bucket(root / "bucket-a", [8])
    _make_bucket(root / "bucket-b", [8])
    trim_csv = _write_trim_csv(
        tmp_path / "partial-root.csv",
        [
            _trim_row(dataset="bucket-a", total_frames=8, trim_head_to=1, trim_tail_from=7),
            _trim_row(dataset="retired-bucket", total_frames=8, trim_head_to=2, trim_tail_from=6),
        ],
    )

    ds = RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})
    lengths_by_bucket = {bucket._dataset_id: bucket._eps_df["length"].tolist() for bucket in ds._buckets}
    assert lengths_by_bucket == {"bucket-a": [6], "bucket-b": [8]}


def test_root_mode_drops_whole_bucket_exclusion_before_fanout(tmp_path):
    root = tmp_path / "root"
    kept = _make_bucket(root / "kept-bucket", [40])
    excluded_name = next(iter(ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS))
    _make_bucket(root / excluded_name, [40])

    ds = RoboCOINDataset.from_config({"dataset_dir": str(root), "normalize_mode": None})

    assert [bucket._dataset_dir for bucket in ds._buckets] == [kept]


def test_direct_reader_rejects_whole_bucket_exclusion(tmp_path):
    excluded_name = next(iter(ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS))
    bucket = _make_bucket(tmp_path / excluded_name, [40])

    with pytest.raises(DataContractError, match="excluded as a whole"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)


def test_excluded_bucket_does_not_count_as_trim_csv_overlap(tmp_path):
    root = tmp_path / "root"
    _make_bucket(root / "kept-bucket", [40])
    excluded_name = next(iter(ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS))
    _make_bucket(root / excluded_name, [40])
    trim_csv = _write_trim_csv(
        tmp_path / "excluded-only.csv",
        [
            _trim_row(
                dataset=excluded_name,
                total_frames=40,
                trim_head_to=1,
                trim_tail_from=39,
            )
        ],
    )

    with pytest.raises(ValueError, match="silently no-op"):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})


def test_reader_rejects_noncanonical_numeric_data_shard(tmp_path):
    bucket = _make_bucket(tmp_path / "bucket", [8])
    canonical = bucket / "data" / "chunk-000" / "file-000.parquet"
    canonical.rename(canonical.with_name("file-0.parquet"))

    with pytest.raises(DataContractError, match="non-canonical numeric data shard"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)


def test_reader_rejects_non_string_data_path_template(tmp_path):
    bucket = _make_bucket(tmp_path / "bucket", [8])
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = None
    info_path.write_text(json.dumps(info))

    with pytest.raises(DataContractError, match="data_path must be a non-empty string"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)


def test_reader_rejects_malformed_string_data_path_template(tmp_path):
    bucket = _make_bucket(tmp_path / "bucket", [8])
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = "data/chunk-{chunk_index.foo}/file-{file_index}.parquet"
    info_path.write_text(json.dumps(info))

    with pytest.raises(DataContractError, match="invalid info.json data_path template"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)


def test_root_mode_pins_one_snapshot_for_all_bucket_constructors(tmp_path, monkeypatch):
    from openwam.dataloader import robocoin

    root = tmp_path / "root"
    _make_bucket(root / "bucket-a", [10])
    _make_bucket(root / "bucket-b", [10])
    trim_csv = _write_trim_csv(
        tmp_path / "trim.csv",
        [_trim_row(dataset="bucket-a")],
    )
    original_load = robocoin._load_trim_snapshot
    loads = []

    def record_load(path):
        snapshot = original_load(path)
        loads.append(snapshot)
        return snapshot

    monkeypatch.setattr(robocoin, "_load_trim_snapshot", record_load)
    ds = RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})

    assert len(loads) == 1
    assert {bucket._dataset_id: bucket._eps_df["length"].tolist() for bucket in ds._buckets} == {
        "bucket-a": [6],
        "bucket-b": [10],
    }
    assert all(bucket._trim_snapshot is None for bucket in ds._buckets)


def test_root_mode_rejects_csv_change_during_fanout_without_normalization(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _make_bucket(root / "bucket", [10])
    trim_csv = _write_trim_csv(tmp_path / "trim.csv", [_trim_row()])
    replacement = _write_trim_csv(
        tmp_path / "replacement.csv",
        [_trim_row(trim_head_to=3, trim_tail_from=7)],
    )
    original_add_offsets = RoboCOINDataset._add_data_offsets

    def add_offsets_then_replace(self, eps):
        original_add_offsets(self, eps)
        replacement.replace(trim_csv)

    monkeypatch.setattr(RoboCOINDataset, "_add_data_offsets", add_offsets_then_replace)
    with pytest.raises(DataContractError, match="changed while from_config reader construction"):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})


def test_root_mode_unknown_episode_id_fails_closed_against_full_manifest(tmp_path):
    root = tmp_path / "root"
    bucket = _make_bucket(root / "bucket", [8, 8])
    _set_info_splits(bucket, {"train": "0:1", "val": "1:2"})
    trim_csv = _write_trim_csv(
        tmp_path / "unknown-episode.csv",
        [
            _trim_row(
                dataset="bucket",
                episode_index=99,
                total_frames=8,
                trim_head_to=1,
                trim_tail_from=7,
            )
        ],
    )

    with pytest.raises(DataContractError, match="[Uu]nknown.*episode"):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})


def test_root_mode_trim_entry_outside_selected_split_is_not_unknown(tmp_path):
    root = tmp_path / "root"
    bucket = _make_bucket(root / "bucket", [8, 8])
    _set_info_splits(bucket, {"train": "0:1", "val": "1:2"})
    trim_csv = _write_trim_csv(
        tmp_path / "val-episode.csv",
        [
            _trim_row(
                dataset="bucket",
                episode_index=1,
                total_frames=8,
                trim_head_to=1,
                trim_tail_from=7,
            )
        ],
    )

    ds = RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})
    assert [bucket._dataset_id for bucket in ds._buckets] == ["bucket"]
    assert ds._buckets[0]._eps_df["episode_index"].tolist() == [0]
    assert ds._buckets[0]._eps_df["length"].tolist() == [8]


def test_root_mode_stale_trim_entry_outside_selected_split_fails_closed(tmp_path):
    root = tmp_path / "root"
    bucket = _make_bucket(root / "bucket", [8, 8])
    _set_info_splits(bucket, {"train": "0:1", "val": "1:2"})
    trim_csv = _write_trim_csv(
        tmp_path / "stale-val-episode.csv",
        [
            _trim_row(
                dataset="bucket",
                episode_index=1,
                total_frames=9,
                trim_head_to=1,
                trim_tail_from=7,
            )
        ],
    )

    with pytest.raises(DataContractError, match="[Ss][Tt][Aa][Ll][Ee]"):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})


def test_root_mode_stale_bucket_fails_closed_globally(tmp_path):
    root = tmp_path / "root"
    _make_bucket(root / "bucket-a", [8])
    _make_bucket(root / "bucket-b", [8])
    trim_csv = _write_trim_csv(
        tmp_path / "stale.csv",
        [
            {
                "dataset": "bucket-a",
                "episode_index": 0,
                "total_frames": 9,
                "trim_head_to": 1,
                "trim_tail_from": 7,
            }
        ],
    )

    with pytest.raises(DataContractError, match="[Ss][Tt][Aa][Ll][Ee]"):
        RoboCOINDataset.from_config({"dataset_dir": str(root), "trim_csv": str(trim_csv), "normalize_mode": None})


def test_trimmed_video_and_action_remain_aligned_end_to_end(tmp_path, patch_decode):
    lengths = [8, 8, 8, 8]
    bucket = _make_bucket(tmp_path / "bucket", lengths)
    trim_csv = _write_trim_csv(
        tmp_path / "trim.csv",
        [
            {
                "dataset": bucket.name,
                "episode_index": 0,
                "total_frames": 8,
                "trim_head_to": 2,
                "trim_tail_from": "",
            },
            {
                "dataset": bucket.name,
                "episode_index": 1,
                "total_frames": 8,
                "trim_head_to": "",
                "trim_tail_from": 6,
            },
            {
                "dataset": bucket.name,
                "episode_index": 2,
                "total_frames": 8,
                "trim_head_to": 2,
                "trim_tail_from": 6,
            },
            {
                "dataset": bucket.name,
                "episode_index": 3,
                "total_frames": 8,
                "trim_head_to": 8,
                "trim_tail_from": "",
            },
        ],
    )
    ds = RoboCOINDataset(
        dataset_dir=str(bucket),
        num_frames=5,
        video_stride=1,
        height=32,
        width=32,
        multiview=False,
        normalize_mode=None,
        trim_csv=trim_csv,
    )

    assert ds._eps_df["episode_index"].tolist() == [0, 1, 2]
    assert ds._eps_df["length"].tolist() == [6, 6, 4]
    assert ds._ep_data_row_offset.tolist() == [2, 8, 18]
    assert ds._ep_video_frame_offsets[CAM].tolist() == [2, 8, 18]
    assert len(ds) == 16

    seen = set()
    for idx in range(len(ds)):
        sample = ds[idx]
        action_rows = sample["action"][sample["action_mask"].any(dim=1), 0].to(torch.int64).tolist()
        video_rows = [
            _decoded_index(frame) for frame, valid in zip(sample["video"], sample["video_mask"].tolist()) if valid
        ]
        assert action_rows == video_rows[: len(action_rows)]
        seen.update(video_rows)

    assert seen == set(range(2, 8)) | set(range(8, 14)) | set(range(18, 22))
