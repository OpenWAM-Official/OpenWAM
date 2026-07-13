"""Unit tests for benchmarks/behavior/gen_task_prompts.py (no dataset needed).

The generator joins OmniGibson's task_id→activity_name table to the dataset's
training sentences through the annotation ``task_name`` field. These tests pin
the two properties the bridge relies on: the join runs on names (never on
task_id ↔ task-chunk order), and every inconsistency fails fast instead of
falling back to activity-name prompts.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.behavior.gen_task_prompts import build_task_prompts, normalize_activity_name

CHUNKS_SIZE = 10000
RADIO = "Turn on the radio receiver that's on the table in the living room."
MEAT = "Open the kitchen cabinet, take out the two hinged jars, and can the meat."


def _write_dataset(root, *, chunks, episodes):
    """chunks: {chunk_id: task_name}; episodes: [(episode_index, prompt)]."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"chunks_size": CHUNKS_SIZE}))
    with open(meta / "episodes.jsonl", "w") as f:
        for ep_idx, prompt in episodes:
            f.write(json.dumps({"episode_index": ep_idx, "tasks": [prompt] if prompt else []}) + "\n")
    for chunk_id, task_name in chunks.items():
        chunk_dir = root / "annotations" / f"task-{chunk_id:04d}"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / f"episode_{chunk_id * CHUNKS_SIZE:08d}.json").write_text(
            json.dumps({"task_name": task_name, "skill_annotation": []})
        )
    return str(root)


@pytest.fixture()
def dataset(tmp_path):
    return _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio", 1: "can meat"},
        episodes=[(0, RADIO), (10, RADIO), (1 * CHUNKS_SIZE, MEAT)],
    )


def test_joins_on_name_not_on_index_order(dataset):
    # task_ids deliberately unrelated to the chunk numbering.
    out = build_task_prompts(dataset, {7: "turning_on_radio", 5: "can_meat"})
    assert out == {7: RADIO, 5: MEAT}


def test_normalization_tolerates_case_and_spacing(dataset):
    out = build_task_prompts(dataset, {0: "Turning_On_Radio", 1: "can  meat"})
    assert out == {0: RADIO, 1: MEAT}


def test_unmatched_activity_name_fails_fast(dataset):
    with pytest.raises(ValueError, match="no matching annotation task_name"):
        build_task_prompts(dataset, {0: "turning_on_radio", 1: "no_such_task"})


def test_missing_annotation_chunk_fails_fast(tmp_path):
    ds = _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio"},  # chunk 1 has episodes but no annotations
        episodes=[(0, RADIO), (1 * CHUNKS_SIZE, MEAT)],
    )
    with pytest.raises(FileNotFoundError, match="annotations download is incomplete"):
        build_task_prompts(ds, {0: "turning_on_radio"})


def test_conflicting_prompts_within_chunk_fail_fast(tmp_path):
    ds = _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio"},
        episodes=[(0, RADIO), (10, MEAT)],  # same chunk, different sentences
    )
    with pytest.raises(ValueError, match="differing prompts"):
        build_task_prompts(ds, {0: "turning_on_radio"})


def test_empty_episode_prompt_fails_fast(tmp_path):
    ds = _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio"},
        episodes=[(0, "")],
    )
    with pytest.raises(ValueError, match="empty 'tasks' prompt"):
        build_task_prompts(ds, {0: "turning_on_radio"})


def test_normalize_activity_name():
    assert normalize_activity_name("Turning_On__Radio ") == "turning on radio"
