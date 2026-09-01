"""Public integrity scanner tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import scan_dataset  # noqa: E402


class FakeLeaf:
    def __init__(self, episodes=(10, 20), failures=()):
        self._dataset_dir = "/public/dataset"
        self._cum_n_starts = np.asarray([0, 2, 5], dtype=np.int64)
        self._eps_df = _EpisodeTable(episodes)
        self._failures = set(failures)

    def __len__(self):
        return int(self._cum_n_starts[-1])

    def _getitem_impl(self, index):
        if index in self._failures:
            raise RuntimeError(f"bad window {index}")
        return {"ok": True}


class _EpisodeColumn:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.int64)

    def to_numpy(self):
        return self._values


class _EpisodeRows:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return {"episode_index": self._values[index]}


class _EpisodeTable:
    def __init__(self, values):
        self._values = list(values)
        self.iloc = _EpisodeRows(self._values)

    def __getitem__(self, key):
        assert key == "episode_index"
        return _EpisodeColumn(self._values)


def test_leaf_episode_identity_and_last_window_lookup():
    leaf = FakeLeaf()
    assert scan_dataset._leaf_episode_key(leaf, 0) == 10
    assert scan_dataset._leaf_episode_key(leaf, 2) == 20
    assert scan_dataset._leaf_locate_episode(leaf, 10) == 1
    assert scan_dataset._leaf_locate_episode(leaf, 20) == 4
    assert scan_dataset._leaf_locate_episode(leaf, 999) is None


def test_scan_plan_is_lazy_strided_and_sharded():
    class Ten:
        def __len__(self):
            return 10

    plan = scan_dataset._ScanPlan([Ten()], sample_stride=2, shard=1, num_shards=2, limit=None)
    assert len(plan) == 2
    assert [plan.resolve(i) for i in range(len(plan))] == [(0, 2), (0, 6)]


def test_episode_plan_uses_highest_window_per_episode():
    leaf = FakeLeaf()
    plan = scan_dataset._EpisodePlan([leaf], shard=0, num_shards=1, limit=None)
    assert [plan.resolve(i) for i in range(len(plan))] == [(0, 1), (0, 4)]


def test_scan_dataset_records_direct_decode_failure():
    leaf = FakeLeaf(failures={4})
    plan = scan_dataset._EpisodePlan([leaf], shard=0, num_shards=1, limit=None)
    wrapped = scan_dataset._ScanDataset([leaf], plan)
    assert wrapped[0]["ok"] is True
    failed = wrapped[1]
    assert failed["ok"] is False
    assert failed["key"] == 20
    assert "RuntimeError" in failed["err"]


def test_scan_rejects_mixture_before_build(monkeypatch):
    monkeypatch.setattr(scan_dataset, "_peek_config_type", lambda _name: "mixture")
    monkeypatch.setattr(
        scan_dataset,
        "_build_dataset",
        lambda _name: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    assert scan_dataset.cmd_scan(argparse.Namespace(config="mixture")) == 2


@pytest.mark.parametrize("shard,total", [(1, 1), (4, 4), (-1, 2), (0, 0)])
def test_invalid_shard_fails_loud(shard, total):
    with pytest.raises(SystemExit, match="--shard must satisfy"):
        scan_dataset._validate_shard(argparse.Namespace(shard=shard, num_shards=total))


def test_lazy_verify_indices_support_limit_and_negative_index():
    values = scan_dataset._ShardedIndices(1_000_000_000, shard=2, num_shards=7, limit=4)
    assert list(values) == [2, 9, 16, 23]
    assert values[-1] == 23
    with pytest.raises(IndexError):
        _ = values[4]


def test_reprobe_uses_episode_identity_after_index_drift(monkeypatch):
    leaf = FakeLeaf(episodes=(30, 40), failures={1})
    target = "/public/dataset/meta/excluded_episodes.json"
    monkeypatch.setattr(scan_dataset, "_leaf_info", lambda _leaf: ("lerobot", target))
    record = {
        "kind": "lerobot",
        "target": target,
        "key": 30,
        "local": 4,  # currently points to episode 40; key relocation must find index 1
        "err": "recorded",
    }
    assert scan_dataset._filter_reprobe_against_leaves([record], [leaf], tries=2) == [record]


def test_reprobe_drops_transient_and_keeps_gpu_file_record(monkeypatch):
    leaf = FakeLeaf()
    target = "/public/dataset/meta/excluded_episodes.json"
    monkeypatch.setattr(scan_dataset, "_leaf_info", lambda _leaf: ("lerobot", target))
    transient = {"kind": "lerobot", "target": target, "key": 10, "local": 1, "err": "timeout"}
    file_level = {
        "kind": "lerobot",
        "target": target,
        "key": 20,
        "local": -1,
        "err": "decode_error: broken",
    }
    kept = scan_dataset._filter_reprobe_against_leaves([transient, file_level], [leaf], tries=3)
    assert kept == [file_level]


def test_read_failures_skips_only_truncated_json_line(tmp_path):
    out = tmp_path / "scan"
    out.mkdir()
    (out / "failures.shard0-of-1.jsonl").write_text(
        json.dumps({"kind": "lerobot", "target": "/t", "key": 1}) + "\n{"
    )
    assert scan_dataset._read_all_failures(out) == [{"kind": "lerobot", "target": "/t", "key": 1}]


def test_gpu_scanner_validates_shard_before_environment_checks(monkeypatch):
    path = _SCRIPTS / "gpu_decode_scan.py"
    spec = importlib.util.spec_from_file_location("gpu_decode_scan_under_test", path)
    gpu_scan = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gpu_scan)

    def too_far(*_args, **_kwargs):
        raise AssertionError("validation must happen first")

    monkeypatch.setattr(gpu_scan, "_check_ffmpeg_cuda", too_far)
    monkeypatch.setattr(gpu_scan, "_detect_num_gpus", too_far)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu_decode_scan.py", "--config", "robocoin", "--shard", "1", "--num-shards", "1"],
    )
    with pytest.raises(SystemExit, match="--shard must satisfy"):
        gpu_scan.main()
