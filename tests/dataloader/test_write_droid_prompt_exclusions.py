from __future__ import annotations

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
    assert payload["droid_prompt_exclusions"]["latest_scan"]["episode_indices"] == [0]


def test_rerun_accumulates_prompt_provenance(tmp_path):
    root = _write_bucket(tmp_path, task_text="", data_task_index=0, fallback=[""], episode_index=0)
    out = root / "meta" / "excluded_episodes.json"
    out.write_text(
        json.dumps(
            {
                "episode_indices": [5, 999],
                "droid_prompt_exclusions": {
                    "episode_indices": [5],
                    "source_note": "previous prompt scan",
                },
            }
        )
    )

    result = _run_generator(root)

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["episode_indices"] == [0, 5, 999]
    assert payload["droid_prompt_exclusions"]["episode_indices"] == [0, 5]
    assert payload["droid_prompt_exclusions"]["source_note"] == "previous prompt scan"
    assert payload["droid_prompt_exclusions"]["latest_scan"]["episode_indices"] == [0]


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
