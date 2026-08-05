from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from openwam.dataloader.oxe_droid import OxeDroidDataset

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
    (root / "data" / "chunk-000").mkdir(parents=True)
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
    assert payload["droid_prompt_exclusions"]["schema_version"] == 1
    assert payload["droid_prompt_exclusions"]["fallback_chain"] == list(OxeDroidDataset.PROMPT_FALLBACK_COLS)
    assert payload["droid_prompt_exclusions"]["latest_scan"]["episode_indices"] == [0]


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
    assert "prompt scan covered 1 rows" in result.stderr
    assert out.read_bytes() == original


def test_publish_refuses_to_overwrite_artifact_changed_during_scan(tmp_path):
    generator = _load_generator_module()
    out = tmp_path / "excluded_episodes.json"
    preflight = b'{"episode_indices":[1]}'
    newer = b'{"episode_indices":[1,2]}'
    out.write_bytes(newer)

    with pytest.raises(SystemExit, match="changed during the prompt scan"):
        generator._atomic_write_text(out, '{"episode_indices":[3]}', expected=preflight)

    assert out.read_bytes() == newer
    assert list(tmp_path.glob("*.tmp")) == []


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
