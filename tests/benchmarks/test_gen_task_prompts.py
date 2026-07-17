"""Unit tests for benchmarks/behavior/gen_task_prompts.py (no dataset needed).

The generator joins OmniGibson's task_id→activity_name table to the dataset's
training sentences through the annotation ``task_name`` field. These tests pin
the properties the bridge relies on: the join runs on names (never on
task_id ↔ task-chunk order), the annotation directory numbering is
cross-checked against the episode indices in the filenames (a permuted
numbering must fail, not silently permute the mapping), and every
inconsistency fails fast instead of falling back to activity-name prompts.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.behavior.gen_task_prompts import (
    build_task_prompts,
    normalize_activity_name,
    underscored_task_ids,
)

CHUNKS_SIZE = 10000
RADIO = "Turn on the radio receiver that's on the table in the living room."
MEAT = "Open the kitchen cabinet, take out the two hinged jars, and can the meat."


def _write_annotation(root, chunk_id, ep_idx, task_name):
    chunk_dir = root / "annotations" / f"task-{chunk_id:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / f"episode_{ep_idx:08d}.json").write_text(json.dumps({"task_name": task_name, "skill_annotation": []}))


def _write_metainfo(root, ep_idx, activity_name, *, dir_chunk):
    meta_dir = root / "meta" / "episodes" / f"task-{dir_chunk:04d}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"task": {"activity_name": activity_name}})
    (meta_dir / f"episode_{ep_idx:08d}.json").write_text(json.dumps({"config": config}))


def _write_dataset(root, *, chunks, episodes, chunks_size=CHUNKS_SIZE):
    """chunks: {chunk_id: task_name}; episodes: [(episode_index, prompt)]."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"chunks_size": chunks_size}))
    with open(meta / "episodes.jsonl", "w") as f:
        for ep_idx, prompt in episodes:
            f.write(json.dumps({"episode_index": ep_idx, "tasks": [prompt] if prompt else []}) + "\n")
    for chunk_id, task_name in chunks.items():
        _write_annotation(root, chunk_id, chunk_id * chunks_size, task_name)
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


def test_misnumbered_annotation_dir_fails_fast(tmp_path):
    """Dir numbering ↔ filename-episode chunking disagreement must abort, not permute."""
    ds = _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio", 1: "can meat"},
        episodes=[(0, RADIO), (1 * CHUNKS_SIZE, MEAT)],
    )
    # Swap the two annotation directory names: contents (filenames + task_name)
    # now claim the other chunk. All pre-existing checks would pass and the
    # output would be a permuted mapping — the filename cross-check must trip.
    a = tmp_path / "ds" / "annotations" / "task-0000"
    b = tmp_path / "ds" / "annotations" / "task-0001"
    tmp = tmp_path / "ds" / "annotations" / "task-tmp"
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)
    with pytest.raises(ValueError, match="numbering and the episodes.jsonl chunking disagree"):
        build_task_prompts(ds, {0: "turning_on_radio", 1: "can_meat"})


def test_non_directory_task_entries_ignored(dataset, tmp_path):
    (tmp_path / "ds" / "annotations" / "task-0000.bak").write_text("junk")
    out = build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})
    assert out == {0: RADIO, 1: MEAT}


def test_non_numeric_task_dir_fails_with_path(dataset, tmp_path):
    (tmp_path / "ds" / "annotations" / "task-abc").mkdir()
    with pytest.raises(ValueError, match=r"task-abc.*numeric chunk id"):
        build_task_prompts(dataset, {0: "turning_on_radio"})


def test_unparseable_annotation_filename_fails_with_path(dataset, tmp_path):
    (tmp_path / "ds" / "annotations" / "task-0000" / "notes.json").write_text("{}")
    with pytest.raises(ValueError, match=r"notes\.json.*episode_<index>\.json"):
        build_task_prompts(dataset, {0: "turning_on_radio"})


def test_matched_name_without_episodes_gets_distinct_error(tmp_path):
    """Annotation chunk exists, name joins, but episodes.jsonl lacks the chunk."""
    ds = _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio", 1: "can meat"},
        episodes=[(0, RADIO)],  # no chunk-1 episodes
    )
    with pytest.raises(ValueError, match="episodes.jsonl has no episodes for the chunk"):
        build_task_prompts(ds, {0: "turning_on_radio", 1: "can_meat"})


def test_underscored_task_ids_flags_rewrites():
    assert underscored_task_ids({0: "plain sentence", 1: "has_underscore", 2: "b_c"}) == [1, 2]
    assert underscored_task_ids({0: "clean"}) == []


def test_chunks_size_read_from_info_json(tmp_path):
    """A regression hardcoding 10000 must fail on a dataset with another chunks_size."""
    ds = _write_dataset(
        tmp_path / "ds",
        chunks={0: "turning on radio", 1: "can meat"},
        episodes=[(3, RADIO), (7 + 3, MEAT)],  # chunk = index // 7
        chunks_size=7,
    )
    out = build_task_prompts(ds, {0: "turning_on_radio", 1: "can_meat"})
    assert out == {0: RADIO, 1: MEAT}


def test_multi_file_chunk_misplaced_later_file_fails(dataset, tmp_path):
    """The filename cross-check must cover every file, not just files[0]."""
    _write_annotation(tmp_path / "ds", 0, 1 * CHUNKS_SIZE + 20, "turning on radio")  # chunk-1 episode in dir 0
    with pytest.raises(ValueError, match="numbering and the episodes.jsonl chunking disagree"):
        build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})


def test_multi_file_chunk_conflicting_task_name_fails(dataset, tmp_path):
    """task_name must be unanimous across the chunk — a stale files[0] cannot win."""
    _write_annotation(tmp_path / "ds", 0, 20, "can meat")
    with pytest.raises(ValueError, match="conflicting task_name"):
        build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})


def test_duplicate_chunk_ids_fail(dataset, tmp_path):
    """task-01 and task-0001 both parse to chunk 1 — must abort, not last-write-win."""
    dup = tmp_path / "ds" / "annotations" / "task-01"
    dup.mkdir()
    (dup / f"episode_{CHUNKS_SIZE + 40:08d}.json").write_text(json.dumps({"task_name": "can meat"}))
    with pytest.raises(ValueError, match="duplicate annotation chunk id 1"):
        build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})


def test_empty_task_name_fails(tmp_path):
    ds = _write_dataset(tmp_path / "ds", chunks={0: "  "}, episodes=[(0, RADIO)])
    with pytest.raises(ValueError, match="empty 'task_name'"):
        build_task_prompts(ds, {0: "turning_on_radio"})


def test_metainfo_cross_check_passes_when_consistent(dataset, tmp_path, capsys):
    _write_metainfo(tmp_path / "ds", 0, "turning_on_radio", dir_chunk=0)
    out = build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})
    assert out == {0: RADIO, 1: MEAT}
    assert "metainfo cross-check: 1/2 chunks validated" in capsys.readouterr().out


def test_metainfo_cross_check_mismatch_fails(dataset, tmp_path):
    _write_metainfo(tmp_path / "ds", 0, "can_meat", dir_chunk=0)  # wrong label for chunk 0
    with pytest.raises(ValueError, match="disagrees with metainfo activity_name"):
        build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})


def test_metainfo_chunk_from_filename_not_dirname(dataset, tmp_path):
    """A renumbered meta/episodes dir must not misdirect the cross-check."""
    # File is a chunk-0 episode with chunk 0's correct label, but sits in a dir named task-0001.
    _write_metainfo(tmp_path / "ds", 0, "turning_on_radio", dir_chunk=1)
    out = build_task_prompts(dataset, {0: "turning_on_radio", 1: "can_meat"})
    assert out == {0: RADIO, 1: MEAT}


def test_main_round_trips_into_bridge_loader(dataset, tmp_path, monkeypatch, capsys):
    """main() output must reload through the bridge's _load_task_names unchanged."""
    from benchmarks.behavior import gen_task_prompts
    from benchmarks.behavior.openwam2behavior_bridge import _load_task_names

    acts = tmp_path / "acts.json"
    acts.write_text(json.dumps({"0": "turning_on_radio", "1": "can_meat"}))
    out_path = tmp_path / "task_prompts.json"
    monkeypatch.setattr(
        "sys.argv",
        ["gen_task_prompts", "--dataset-dir", dataset, "--activity-names", str(acts), "--output", str(out_path)],
    )
    gen_task_prompts.main()
    assert _load_task_names(str(out_path)) == {0: RADIO, 1: MEAT}
    assert "wrote 2 task prompts" in capsys.readouterr().out


def test_main_warns_on_underscored_prompt(tmp_path, monkeypatch, capsys):
    underscored = "Plug in the USB_C cable."
    ds = _write_dataset(tmp_path / "ds", chunks={0: "plug cable"}, episodes=[(0, underscored)])
    acts = tmp_path / "acts.json"
    acts.write_text(json.dumps({"0": "plug_cable"}))
    out_path = tmp_path / "task_prompts.json"
    monkeypatch.setattr(
        "sys.argv",
        ["gen_task_prompts", "--dataset-dir", ds, "--activity-names", str(acts), "--output", str(out_path)],
    )
    from benchmarks.behavior import gen_task_prompts

    gen_task_prompts.main()
    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "task_id(s) [0]" in captured.err
    assert json.loads(out_path.read_text()) == {"0": underscored}
