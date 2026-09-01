"""Emit-path and DROID exclusion ownership tests."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import scan_dataset  # noqa: E402

from openwam.dataloader.oxe_droid import (  # noqa: E402
    DROID_PROMPT_EXCLUSION_KEY,
    DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY,
)


def test_group_failures_deduplicates_and_surfaces_unknown():
    records = [
        {"kind": "lerobot", "target": "/a/meta/excluded_episodes.json", "key": 5},
        {"kind": "lerobot", "target": "/a/meta/excluded_episodes.json", "key": 5},
        {"kind": "unsupported", "target": "/x", "key": 0},
        {"kind": "unsupported", "target": "/x", "key": 1},
    ]
    grouped, unknown = scan_dataset._group_failures(records)
    assert grouped == {"/a/meta/excluded_episodes.json": {5}}
    assert unknown == {"unsupported": 2}


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "{}",
        '{"episode_indices": "bad"}',
        '{"episode_indices": [1, -1]}',
        '{"episode_indices": [1, true]}',
    ],
)
def test_existing_exclusion_malformed_or_wrong_shape_fails_loud(tmp_path, payload):
    target = tmp_path / "excluded_episodes.json"
    target.write_text(payload)
    with pytest.raises(SystemExit):
        scan_dataset._read_lerobot_excluded(str(target))


def test_missing_exclusion_file_starts_empty(tmp_path):
    assert scan_dataset._read_lerobot_excluded(str(tmp_path / "missing.json")) == set()


def test_merge_preserves_droid_namespace_and_records_independent_overlap():
    prompt = {
        "episode_indices": [2],
        DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY: [3],
        "latest_scan": {"keep": "unchanged"},
    }
    payload = {
        "episode_indices": [2, 3],
        "reason": "keep",
        DROID_PROMPT_EXCLUSION_KEY: prompt,
    }
    merged, existing, new = scan_dataset._merge_lerobot_payload(payload, {2, 4})
    assert existing == {2, 3}
    assert new == {4}
    assert merged["episode_indices"] == [2, 3, 4]
    assert merged["reason"] == "keep"
    assert merged[DROID_PROMPT_EXCLUSION_KEY]["episode_indices"] == [2]
    assert merged[DROID_PROMPT_EXCLUSION_KEY][DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY] == [2, 3, 4]
    assert merged[DROID_PROMPT_EXCLUSION_KEY]["latest_scan"] == {"keep": "unchanged"}


def test_merge_rejects_inconsistent_droid_ownership():
    payload = {
        "episode_indices": [2],
        DROID_PROMPT_EXCLUSION_KEY: {
            "episode_indices": [2],
            DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY: [3],
        },
    }
    with pytest.raises(ValueError, match="must equal the union"):
        scan_dataset._merge_lerobot_payload(payload, {4})


def test_exclusion_ratio_guard_uses_pre_exclusion_population():
    planned = [{"target": "/t", "n_final": 6, "existing": 3, "n_new": 3}]
    assert scan_dataset._exclusion_ratio_violations(planned, {"/t": 100}, 0.05) == [
        ("/t", 6, 100, 0.06)
    ]
    assert scan_dataset._exclusion_ratio_violations(
        [{"target": "/t", "n_final": 90, "existing": 90, "n_new": 0}],
        {"/t": 100},
        0.05,
    ) == []


def _write_failures(out_dir: Path, target: Path, episodes: list[int]) -> None:
    out_dir.mkdir(parents=True)
    records = [
        {
            "kind": "lerobot",
            "target": str(target),
            "key": episode,
            "local": -1,
            "err": "decode_error: broken",
        }
        for episode in episodes
    ]
    (out_dir / "failures.shard0-of-1.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def test_emit_rereads_and_unions_inside_lock(tmp_path, monkeypatch):
    scan_root = tmp_path / "scan_out"
    out_dir = scan_root / "dataset"
    target = tmp_path / "dataset" / "meta" / "excluded_episodes.json"
    _write_failures(out_dir, target, [2])
    real_lock = scan_dataset.locked_exclusion_files

    @contextmanager
    def competing_publish(paths):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"episode_indices": [1]}))
        with real_lock(paths):
            yield

    monkeypatch.setattr(scan_dataset, "locked_exclusion_files", competing_publish)
    monkeypatch.setattr(scan_dataset, "_build_leaves", lambda _config: None)

    args = type(
        "Args",
        (),
        {
            "config": "dataset",
            "out_dir": str(scan_root),
            "dry_run": False,
            "max_exclude_frac": 0.05,
            "force": False,
            "reprobe": False,
        },
    )()
    assert scan_dataset.cmd_emit(args) == 0
    assert json.loads(target.read_text())["episode_indices"] == [1, 2]


def test_emit_guard_aborts_before_over_exclusion_publish(tmp_path, monkeypatch):
    scan_root = tmp_path / "scan_out"
    out_dir = scan_root / "dataset"
    target = tmp_path / "dataset" / "meta" / "excluded_episodes.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"episode_indices": [0, 1, 2]}))
    _write_failures(out_dir, target, [3, 4, 5])
    monkeypatch.setattr(scan_dataset, "_build_leaves", lambda _config: [object()])
    monkeypatch.setattr(scan_dataset, "_target_episode_totals", lambda _leaves: {str(target): 100})

    args = type(
        "Args",
        (),
        {
            "config": "dataset",
            "out_dir": str(scan_root),
            "dry_run": False,
            "max_exclude_frac": 0.05,
            "force": False,
            "reprobe": False,
        },
    )()
    assert scan_dataset.cmd_emit(args) == 3
    assert json.loads(target.read_text())["episode_indices"] == [0, 1, 2]


def test_emit_aborts_unknown_kind_without_writing(tmp_path, monkeypatch):
    out_dir = tmp_path / "scan_out" / "dataset"
    out_dir.mkdir(parents=True)
    (out_dir / "failures.shard0-of-1.jsonl").write_text(
        json.dumps({"kind": "unsupported", "target": "/x", "key": 0}) + "\n"
    )
    monkeypatch.setattr(scan_dataset, "_build_leaves", lambda _config: None)
    args = type(
        "Args",
        (),
        {
            "config": "dataset",
            "out_dir": str(tmp_path / "scan_out"),
            "dry_run": True,
            "max_exclude_frac": 0.05,
            "force": False,
            "reprobe": False,
        },
    )()
    with pytest.raises(SystemExit, match="unknown kind"):
        scan_dataset.cmd_emit(args)
