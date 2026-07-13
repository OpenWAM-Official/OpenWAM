"""Generate the bridge's ``--task-names`` JSON with training-verbatim prompts.

The eval wire carries only ``task_id``; the bridge synthesizes the language
prompt from a ``task_id → string`` JSON. Feeding it the de-underscored
OmniGibson activity name ("turning on radio") hands a language-conditioned
checkpoint text it never saw in training — the model was trained on the
dataset's per-episode ``tasks[0]`` sentences ("Turn on the radio receiver
that's on the table in the living room."). This script rewrites the mapping's
values to those sentences, joining through the dataset's own files:

  eval side   task_id → activity_name     ``--activity-names`` JSON, generated
              on the sim box from
              ``omnigibson.learning.utils.eval_utils.TASK_INDICES_TO_NAMES``
  data side   task chunk → task_name      ``annotations/task-XXXX/*.json``
              task chunk → prompt         ``meta/episodes.jsonl`` (``tasks[0]``)

The join key is the normalized activity name, NOT the task_id order, so
nothing rests on OmniGibson's task indices matching the dataset's task-chunk
numbering. On the data side no single labeling source is trusted blindly:

* the annotation directory numbering is cross-checked against the episode
  index embedded in EVERY annotation filename (``episode_index //
  chunks_size`` must equal the directory number);
* the ``task_name`` field is read from EVERY annotation JSON of a chunk and
  must be unanimous (a stale first file cannot mislabel the chunk);
* where ``meta/episodes/task-XXXX/`` metainfo is present locally, its
  ``config.task.activity_name`` — an independent, simulator-authored label —
  must agree with the annotation ``task_name`` (chunk derived from the
  metainfo filename, not its directory name).

Every inconsistency fails fast with the offending items listed; the script
never falls back to activity-name prompts. Reading all ~10k annotation JSONs
takes a few seconds — this is a run-once generator, not a hot path.

Usage::

    python -m benchmarks.behavior.gen_task_prompts \
        --dataset-dir /path/to/behaviour-1k \
        --activity-names task_names.json \
        --output task_prompts.json

Pass the result to ``run_bridge.sh --task-names task_prompts.json``. The
bridge's ``.replace("_", " ")`` is a no-op on the sentences (none contain an
underscore — the generator warns if a future dataset breaks this), so they
reach the model verbatim.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

_EP_PREFIX = "episode_"


def normalize_activity_name(name: str) -> str:
    """Join-key normalization: de-underscore, collapse whitespace, lowercase."""
    return " ".join(str(name).replace("_", " ").lower().split())


def underscored_task_ids(task_prompts: Dict[int, str]) -> List[int]:
    """task_ids whose prompt the bridge's ``replace("_", " ")`` would rewrite."""
    return sorted(tid for tid, prompt in task_prompts.items() if "_" in prompt)


def _read_chunks_size(dataset_dir: str) -> int:
    with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
        return int(json.load(f)["chunks_size"])


def _to_ascii_int(text: str) -> int:
    """Strict decimal parse: plain ASCII digits only (no '_' grouping, no unicode digits)."""
    if not (text.isascii() and text.isdigit()):
        raise ValueError(f"not a plain decimal integer: {text!r}")
    return int(text)


def _parse_chunk_id(chunk_dir: str) -> int:
    suffix = os.path.basename(chunk_dir).split("-", 1)[1]
    try:
        return _to_ascii_int(suffix)
    except ValueError:
        raise ValueError(f"annotation dir {chunk_dir!r} does not end in a numeric chunk id") from None


def _parse_episode_index(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith(_EP_PREFIX):
        try:
            return _to_ascii_int(stem[len(_EP_PREFIX) :])
        except ValueError:
            pass
    raise ValueError(f"annotation file {path!r} is not named {_EP_PREFIX}<index>.json")


def _load_chunk_task_names(dataset_dir: str, chunks_size: int) -> Dict[int, str]:
    """Map task chunk id → the unanimous annotation ``task_name`` of the chunk.

    Trust nothing singly: the directory number must match the chunk derived
    from every contained filename's episode index, chunk ids must be unique
    across directories, and the ``task_name`` field must be identical across
    all JSONs of the chunk.
    """
    entries = sorted(glob.glob(os.path.join(glob.escape(dataset_dir), "annotations", "task-*")))
    chunk_dirs = [p for p in entries if os.path.isdir(p)]
    if not chunk_dirs:
        raise FileNotFoundError(
            f"no annotations/task-* directories under {dataset_dir!r}. The "
            "annotations/ folder ships in the same HF repo as the demos "
            "(behavior-1k/2025-challenge-demos) — download it first."
        )
    names: Dict[int, str] = {}
    for chunk_dir in chunk_dirs:
        chunk = _parse_chunk_id(chunk_dir)
        if chunk in names:
            raise ValueError(
                f"duplicate annotation chunk id {chunk}: {chunk_dir!r} collides with an earlier "
                "task-* dir (differently padded numbering?) — refusing to pick one."
            )
        files = sorted(glob.glob(os.path.join(glob.escape(chunk_dir), "*.json")))
        if not files:
            raise FileNotFoundError(f"annotation dir {chunk_dir!r} has no episode JSONs")
        task_name = None
        for fp in files:
            ep_chunk = _parse_episode_index(fp) // chunks_size
            if ep_chunk != chunk:
                raise ValueError(
                    f"annotation dir {chunk_dir!r} is numbered {chunk} but contains {fp!r} whose "
                    f"episode index falls in chunk {ep_chunk} (chunks_size={chunks_size}); the "
                    "directory numbering and the episodes.jsonl chunking disagree — refusing to join."
                )
            with open(fp) as f:
                this_name = str(json.load(f).get("task_name", "")).strip()
            if not this_name:
                raise ValueError(f"{fp!r} has an empty 'task_name'")
            if task_name is None:
                task_name = this_name
            elif this_name != task_name:
                raise ValueError(
                    f"annotation dir {chunk_dir!r} has conflicting task_name values "
                    f"({task_name!r} vs {this_name!r} in {fp!r}); refusing to pick one."
                )
        names[chunk] = task_name
    return names


def _cross_check_metainfo(dataset_dir: str, chunk_task_names: Dict[int, str], chunks_size: int) -> Tuple[int, int]:
    """Validate annotation task_names against ``meta/episodes`` where present.

    ``meta/episodes/task-XXXX/episode_*.json`` carries the simulator-authored
    ``config.task.activity_name`` — a labeling source independent of the
    ``annotations/`` folder. For every chunk that has such metainfo locally
    (the folder is large and often partially downloaded), the two labels must
    agree under join normalization. Returns ``(checked, total)`` chunk counts.
    """
    meta_dirs = sorted(glob.glob(os.path.join(glob.escape(dataset_dir), "meta", "episodes", "task-*")))
    checked = 0
    for meta_dir in meta_dirs:
        if not os.path.isdir(meta_dir):
            continue
        files = sorted(glob.glob(os.path.join(glob.escape(meta_dir), "*.json")))
        if not files:
            continue
        # Chunk from the metainfo FILENAME (dir numbering not trusted here either).
        chunk = _parse_episode_index(files[0]) // chunks_size
        if chunk not in chunk_task_names:
            continue
        with open(files[0]) as f:
            config = json.load(f).get("config")
        if isinstance(config, str):
            config = json.loads(config)
        activity = str((config or {}).get("task", {}).get("activity_name", "")).strip()
        if not activity:
            continue
        if normalize_activity_name(activity) != normalize_activity_name(chunk_task_names[chunk]):
            raise ValueError(
                f"chunk {chunk}: annotation task_name {chunk_task_names[chunk]!r} disagrees with "
                f"metainfo activity_name {activity!r} ({files[0]!r}); the annotations/ and "
                "meta/episodes labels are inconsistent — refusing to join."
            )
        checked += 1
    return checked, len(chunk_task_names)


def _load_chunk_prompts(dataset_dir: str, chunks_size: int) -> Dict[int, str]:
    """Map task chunk id → the training prompt (``tasks[0]``) shared by its episodes."""
    prompts: Dict[int, str] = {}
    episodes_path = os.path.join(dataset_dir, "meta", "episodes.jsonl")
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            chunk = int(ep["episode_index"]) // chunks_size
            tasks = ep.get("tasks") or []
            prompt = str(tasks[0]).strip() if tasks else ""
            if not prompt:
                raise ValueError(f"episode {ep['episode_index']} in {episodes_path!r} has an empty 'tasks' prompt")
            if prompts.setdefault(chunk, prompt) != prompt:
                raise ValueError(
                    f"task chunk {chunk} has episodes with differing prompts "
                    f"({prompts[chunk]!r} vs {prompt!r}); refusing to pick one"
                )
    return prompts


def build_task_prompts(dataset_dir: str, activity_names: Dict[int, str]) -> Dict[int, str]:
    """Return ``task_id → training-verbatim prompt`` for the bridge's --task-names."""
    chunks_size = _read_chunks_size(dataset_dir)
    chunk_task_names = _load_chunk_task_names(dataset_dir, chunks_size)
    checked, total = _cross_check_metainfo(dataset_dir, chunk_task_names, chunks_size)
    print(f"metainfo cross-check: {checked}/{total} chunks validated against meta/episodes activity_name")
    chunk_prompts = _load_chunk_prompts(dataset_dir, chunks_size)

    missing_annotations = sorted(set(chunk_prompts) - set(chunk_task_names))
    if missing_annotations:
        raise FileNotFoundError(
            f"episodes.jsonl covers task chunks {missing_annotations} but annotations/ has no "
            "matching task-XXXX dir(s) — the local annotations download is incomplete."
        )

    name_to_chunk: Dict[str, int] = {}
    for chunk, task_name in chunk_task_names.items():
        key = normalize_activity_name(task_name)
        if key in name_to_chunk:
            raise ValueError(
                f"task_name {task_name!r} normalizes identically for chunks "
                f"{name_to_chunk[key]} and {chunk}; the name join is ambiguous"
            )
        name_to_chunk[key] = chunk

    out: Dict[int, str] = {}
    unmatched_names = []
    matched_without_episodes = []
    for task_id, activity in sorted(activity_names.items()):
        chunk = name_to_chunk.get(normalize_activity_name(activity))
        if chunk is None:
            unmatched_names.append((task_id, activity))
        elif chunk not in chunk_prompts:
            matched_without_episodes.append((task_id, activity, chunk))
        else:
            out[task_id] = chunk_prompts[chunk]
    if unmatched_names:
        raise ValueError(
            f"{len(unmatched_names)} activity name(s) have no matching annotation task_name: "
            f"{unmatched_names}. Dataset annotations and TASK_INDICES_TO_NAMES disagree — "
            "resolve before eval, do not fall back to activity-name prompts."
        )
    if matched_without_episodes:
        raise ValueError(
            f"{len(matched_without_episodes)} activity name(s) matched an annotation task_name but "
            f"episodes.jsonl has no episodes for the chunk: {matched_without_episodes}. "
            "The annotations/ and meta/episodes.jsonl in --dataset-dir are out of sync "
            "(incomplete or mixed-version episodes.jsonl)."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="LeRobot dataset root (contains meta/ and annotations/)")
    parser.add_argument(
        "--activity-names",
        required=True,
        help="JSON mapping task_id→activity_name (from eval_utils.TASK_INDICES_TO_NAMES)",
    )
    parser.add_argument("--output", required=True, help="where to write the task_id→prompt JSON")
    args = parser.parse_args()

    with open(args.activity_names) as f:
        activity_names = {int(k): str(v) for k, v in json.load(f).items()}
    task_prompts = build_task_prompts(args.dataset_dir, activity_names)
    underscored = underscored_task_ids(task_prompts)
    if underscored:
        print(
            f"WARNING: prompt(s) for task_id(s) {underscored} contain '_', which the bridge "
            "rewrites to spaces — the model will NOT receive them verbatim.",
            file=sys.stderr,
        )
    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in sorted(task_prompts.items())}, f, indent=2)
    print(f"wrote {len(task_prompts)} task prompts → {args.output}")


if __name__ == "__main__":
    main()
