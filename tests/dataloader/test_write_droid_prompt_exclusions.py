from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from openwam.dataloader.oxe_droid import (
    DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
    DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION,
    DROID_PROMPT_INPUTS_DIGEST_KEY,
    OxeDroidDataset,
    load_droid_prompt_exclusions,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "write_droid_prompt_exclusions.py"
_FALLBACK_COL = OxeDroidDataset.PROMPT_FALLBACK_COLS[0]


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("write_droid_prompt_exclusions_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_bucket(
    tmp_path: Path,
    *,
    task_text: str,
    data_task_index: int,
    fallback: list[str],
    episode_index: int = 0,
) -> Path:
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "episodes").mkdir()
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "total_frames": len(fallback),
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )
    pd.DataFrame(
        {
            "episode_index": [episode_index],
            "length": [len(fallback)],
            "dataset_from_index": [0],
            "data/chunk_index": [0],
            "data/file_index": [0],
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000.parquet")
    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([task_text], name="task"),
    ).to_parquet(root / "meta" / "tasks.parquet")

    rows = {
        "episode_index": [episode_index] * len(fallback),
        "task_index": [data_task_index] * len(fallback),
    }
    for col in OxeDroidDataset.PROMPT_FALLBACK_COLS:
        rows[col] = fallback if col == _FALLBACK_COL else [""] * len(fallback)
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    return root


def _run_generator(root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--dataset-dir", str(root), "--workers", "1", *extra_args],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("task_text", "data_task_index", "fallback", "error_text"),
    [
        ("", 0, [""] * 70 + ["valid fallback"], "PARTIALLY unresolvable"),
        ("known task", 12345, ["valid fallback"], "absent from tasks.parquet"),
    ],
    ids=["partial-episode", "missing-task-index"],
)
def test_invariant_violation_fails_without_replacing_artifact(
    tmp_path,
    task_text,
    data_task_index,
    fallback,
    error_text,
):
    root = _write_bucket(
        tmp_path,
        task_text=task_text,
        data_task_index=data_task_index,
        fallback=fallback,
    )
    out = root / "meta" / "excluded_episodes.json"
    original = b'{"episode_indices":[999],"reason":"truncated video"}\n'
    out.write_bytes(original)

    result = _run_generator(root)

    assert result.returncode != 0
    assert error_text in result.stderr
    assert out.read_bytes() == original


def test_existing_exclusions_are_unioned_and_provenance_is_preserved(tmp_path):
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=[""])
    out = root / "meta" / "excluded_episodes.json"
    previous = {
        "episode_indices": [999],
        "reason": "truncated video",
        "scanner": {"run_id": "video-scan-1"},
    }
    out.write_text(json.dumps(previous))

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["episode_indices"] == [0, 999]
    assert payload["reason"] == previous["reason"]
    assert payload["scanner"] == previous["scanner"]
    assert payload["droid_prompt_exclusions"]["episode_indices"] == [0]
    assert payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == [999]
    assert payload["droid_prompt_exclusions"]["schema_version"] == DROID_PROMPT_EXCLUSION_SCHEMA_VERSION
    assert payload["droid_prompt_exclusions"]["fallback_chain"] == list(OxeDroidDataset.PROMPT_FALLBACK_COLS)
    assert payload["droid_prompt_exclusions"]["latest_scan"]["episode_indices"] == [0]
    digest = payload["droid_prompt_exclusions"]["latest_scan"][DROID_PROMPT_INPUTS_DIGEST_KEY]
    assert digest["algorithm"] == "sha256"
    assert digest["format_version"] == DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION
    assert len(digest["value"]) == 64


def test_rerun_replaces_prompt_owned_exclusions_and_preserves_unrelated_ones(tmp_path):
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=[""], episode_index=0)
    out = root / "meta" / "excluded_episodes.json"
    out.write_text(json.dumps({"episode_indices": [999], "reason": "truncated video"}))

    first = _run_generator(root)
    assert first.returncode == 0, first.stderr

    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["repaired task"], name="task"),
    ).to_parquet(root / "meta" / "tasks.parquet")

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["episode_indices"] == [999]
    assert payload["reason"] == "truncated video"
    assert payload["droid_prompt_exclusions"]["episode_indices"] == []
    assert payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == [999]
    assert payload["droid_prompt_exclusions"]["latest_scan"]["episode_indices"] == []


def test_rerun_preserves_overlapping_independent_exclusion(tmp_path):
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=[""], episode_index=0)
    out = root / "meta" / "excluded_episodes.json"
    # Episode 0 is already excluded by a different scanner before the prompt
    # scan discovers the same episode is also prompt-bad.
    out.write_text(json.dumps({"episode_indices": [0], "reason": "truncated video"}))

    first = _run_generator(root)
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(out.read_text())
    assert first_payload["droid_prompt_exclusions"]["episode_indices"] == [0]
    assert first_payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == [0]

    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["repaired task"], name="task"),
    ).to_parquet(root / "meta" / "tasks.parquet")

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["episode_indices"] == [0]
    assert payload["droid_prompt_exclusions"]["episode_indices"] == []
    assert payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == [0]


def test_legacy_prompt_provenance_refuses_unsafe_rescan(tmp_path):
    root = _write_bucket(tmp_path, task_text="repaired task", data_task_index=0, fallback=[""])
    out = root / "meta" / "excluded_episodes.json"
    previous = {
        "episode_indices": [0],
        "droid_prompt_exclusions": {
            "episode_indices": [0],
            "latest_scan": {"episode_indices": [0]},
        },
    }
    out.write_text(json.dumps(previous))

    result = _run_generator(root)

    assert result.returncode != 0
    assert "independent ownership" in result.stderr
    assert json.loads(out.read_text()) == previous


def test_original_generator_artifact_migrates_as_prompt_owned(tmp_path):
    root = _write_bucket(tmp_path, task_text="repaired task", data_task_index=0, fallback=[""])
    out = root / "meta" / "excluded_episodes.json"
    old_reason = (
        "episodes whose prompt cannot be resolved from tasks.parquet or any "
        "PROMPT_FALLBACK_COLS entry on any row; generated by "
        "scripts/write_droid_prompt_exclusions.py"
    )
    out.write_text(
        json.dumps(
            {
                "reason": old_reason,
                "generated": "2026-08-01",
                "episode_indices": [0],
                "stats": {
                    "rows_scanned": 1,
                    "unresolved_rows": 1,
                    "episodes_all_unresolved": 1,
                    "episodes_partially_unresolved": 0,
                    "fallback_chain": list(OxeDroidDataset.PROMPT_FALLBACK_COLS),
                },
            }
        )
    )

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["episode_indices"] == []
    assert payload["droid_prompt_exclusions"]["episode_indices"] == []
    assert payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == []
    assert "stats" not in payload


def test_info_row_count_mismatch_does_not_publish(tmp_path):
    root = _write_bucket(tmp_path, task_text="task", data_task_index=0, fallback=[""])
    (root / "meta" / "info.json").write_text(json.dumps({"total_frames": 2}))
    out = root / "meta" / "excluded_episodes.json"
    original = b'{"episode_indices":[999],"reason":"truncated video"}\n'
    out.write_bytes(original)

    result = _run_generator(root)

    assert result.returncode != 0
    assert "episodes manifest addresses 1 rows" in result.stderr
    assert out.read_bytes() == original


def test_generator_refuses_rglob_backup_when_manifest_target_is_missing(tmp_path):
    root = _write_bucket(tmp_path, task_text="task", data_task_index=0, fallback=[""])
    out = root / "meta" / "excluded_episodes.json"
    original = b'{"episode_indices":[999],"reason":"truncated video"}\n'
    out.write_bytes(original)
    (root / "data" / "chunk-000" / "file-000.parquet").rename(root / "data" / "chunk-000" / "backup.parquet")

    result = _run_generator(root)

    assert result.returncode != 0
    assert "file-000.parquet" in result.stderr
    assert out.read_bytes() == original


def test_generator_ignores_unreferenced_backup_parquet(tmp_path):
    root = _write_bucket(tmp_path, task_text="task", data_task_index=0, fallback=[""])
    backup = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    backup["episode_index"] = 999
    backup.to_parquet(root / "data" / "chunk-000" / "backup.parquet")

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    payload = json.loads((root / "meta" / "excluded_episodes.json").read_text())
    assert payload["episode_indices"] == []
    assert payload["droid_prompt_exclusions"]["latest_scan"]["stats"]["rows_scanned"] == 1


def test_generator_honors_nondefault_info_data_path(tmp_path):
    root = _write_bucket(tmp_path, task_text="task", data_task_index=0, fallback=[""])
    custom_dir = root / "custom"
    custom_dir.mkdir()
    (root / "data" / "chunk-000" / "file-000.parquet").rename(custom_dir / "shard-0.parquet")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = "custom/shard-{file_index}.parquet"
    info_path.write_text(json.dumps(info))

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    assert json.loads((root / "meta" / "excluded_episodes.json").read_text())["episode_indices"] == []


def test_generator_rejects_manifest_episode_mapping_that_disagrees_with_rows(tmp_path):
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=["", "valid fallback"])
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    frame_data = pd.read_parquet(data_path)
    frame_data["episode_index"] = [0, 1]
    frame_data.to_parquet(data_path)
    pd.DataFrame(
        {
            "episode_index": [1, 0],
            "length": [1, 1],
            "dataset_from_index": [0, 1],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000.parquet")
    out = root / "meta" / "excluded_episodes.json"
    original = b'{"episode_indices":[999],"reason":"truncated video"}\n'
    out.write_bytes(original)

    result = _run_generator(root)

    assert result.returncode != 0
    assert "episode_index" in result.stderr
    assert out.read_bytes() == original


def test_publish_reloads_and_merges_artifact_changed_during_scan(tmp_path, monkeypatch):
    generator = _load_generator_module()
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=[""])
    out = root / "meta" / "excluded_episodes.json"
    out.write_text(json.dumps({"episode_indices": [1], "reason": "video scan"}))

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _ImmediateExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return _ImmediateFuture(function(*args))

    @contextmanager
    def _inject_concurrent_update(_paths):
        out.write_text(json.dumps({"episode_indices": [1, 2], "reason": "video scan"}))
        yield

    monkeypatch.setattr(generator, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(generator, "as_completed", lambda futures: futures)
    monkeypatch.setattr(generator, "locked_exclusion_files", _inject_concurrent_update)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--dataset-dir", str(root), "--workers", "1"])

    generator.main()

    payload = json.loads(out.read_text())
    assert payload["episode_indices"] == [0, 1, 2]
    assert payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == [1, 2]


def test_prompt_inputs_digest_is_stable_across_worker_completion_order(tmp_path):
    root = _write_bucket(tmp_path, task_text="task", data_task_index=0, fallback=[""], episode_index=0)
    first_shard = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    first_shard["episode_index"] = 1
    first_shard.to_parquet(root / "data" / "chunk-000" / "file-001.parquet")
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "length": [1, 1],
            "dataset_from_index": [0, 1],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 1],
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000.parquet")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 2
    info_path.write_text(json.dumps(info))

    first = _run_generator(root)
    assert first.returncode == 0, first.stderr
    out = root / "meta" / "excluded_episodes.json"
    first_digest = json.loads(out.read_text())["droid_prompt_exclusions"]["latest_scan"][DROID_PROMPT_INPUTS_DIGEST_KEY]

    second = _run_generator(root, "--workers", "2")
    assert second.returncode == 0, second.stderr
    second_digest = json.loads(out.read_text())["droid_prompt_exclusions"]["latest_scan"][
        DROID_PROMPT_INPUTS_DIGEST_KEY
    ]

    assert second_digest == first_digest
    _, canonical = load_droid_prompt_exclusions(root)
    assert canonical == set()


@pytest.mark.parametrize(
    "original",
    [
        b"{",
        b"[]",
        b'{"reason":"missing canonical key"}',
        b'{"episode_indices":[true]}',
        b'{"episode_indices":[],"droid_prompt_exclusions":{}}',
        b'{"episode_indices":[],"droid_prompt_exclusions":{"episode_indices":[5]}}',
    ],
)
def test_malformed_existing_artifact_is_not_replaced(tmp_path, original):
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=[""])
    out = root / "meta" / "excluded_episodes.json"
    out.write_bytes(original)

    result = _run_generator(root)

    assert result.returncode != 0
    assert "malformed" in result.stderr
    assert out.read_bytes() == original
