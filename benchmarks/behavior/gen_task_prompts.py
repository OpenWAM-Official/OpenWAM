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
numbering. Every step fails fast: a missing annotation dir, an activity name
with no matching ``task_name``, or episodes of one chunk disagreeing on their
prompt all abort with the offending items listed.

Usage::

    python -m benchmarks.behavior.gen_task_prompts \
        --dataset-dir /path/to/datasets/behaviour-1k \
        --activity-names task_names.json \
        --output task_prompts.json

Pass the result to ``run_bridge.sh --task-names task_prompts.json``. The
bridge's ``.replace("_", " ")`` is a no-op on the sentences (none contain an
underscore), so they reach the model verbatim.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict


def normalize_activity_name(name: str) -> str:
    """Join-key normalization: de-underscore, collapse whitespace, lowercase."""
    return " ".join(str(name).replace("_", " ").lower().split())


def _load_chunk_task_names(dataset_dir: str) -> Dict[int, str]:
    """Map task chunk id → annotation ``task_name`` (one JSON sampled per chunk)."""
    chunk_dirs = sorted(glob.glob(os.path.join(dataset_dir, "annotations", "task-*")))
    if not chunk_dirs:
        raise FileNotFoundError(
            f"no annotations/task-* directories under {dataset_dir!r}. The "
            "annotations/ folder ships in the same HF repo as the demos "
            "(behavior-1k/2025-challenge-demos) — download it first."
        )
    names: Dict[int, str] = {}
    for chunk_dir in chunk_dirs:
        chunk = int(os.path.basename(chunk_dir).split("-", 1)[1])
        files = sorted(glob.glob(os.path.join(chunk_dir, "*.json")))
        if not files:
            raise FileNotFoundError(f"annotation dir {chunk_dir!r} has no episode JSONs")
        with open(files[0]) as f:
            task_name = str(json.load(f).get("task_name", "")).strip()
        if not task_name:
            raise ValueError(f"{files[0]!r} has an empty 'task_name'")
        names[chunk] = task_name
    return names


def _load_chunk_prompts(dataset_dir: str) -> Dict[int, str]:
    """Map task chunk id → the training prompt (``tasks[0]``) shared by its episodes."""
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path) as f:
        chunks_size = int(json.load(f)["chunks_size"])
    prompts: Dict[int, str] = {}
    episodes_path = os.path.join(dataset_dir, "meta", "episodes.jsonl")
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            chunk = int(ep["episode_index"]) // chunks_size
            tasks = ep.get("tasks") or []
            prompt = str(tasks[0]).strip() if tasks else ""
            if not prompt:
                raise ValueError(
                    f"episode {ep['episode_index']} in {episodes_path!r} has an empty 'tasks' prompt"
                )
            if prompts.setdefault(chunk, prompt) != prompt:
                raise ValueError(
                    f"task chunk {chunk} has episodes with differing prompts "
                    f"({prompts[chunk]!r} vs {prompt!r}); refusing to pick one"
                )
    return prompts


def build_task_prompts(dataset_dir: str, activity_names: Dict[int, str]) -> Dict[int, str]:
    """Return ``task_id → training-verbatim prompt`` for the bridge's --task-names."""
    chunk_task_names = _load_chunk_task_names(dataset_dir)
    chunk_prompts = _load_chunk_prompts(dataset_dir)

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
    unmatched = []
    for task_id, activity in sorted(activity_names.items()):
        chunk = name_to_chunk.get(normalize_activity_name(activity))
        if chunk is None or chunk not in chunk_prompts:
            unmatched.append((task_id, activity))
            continue
        out[task_id] = chunk_prompts[chunk]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} activity name(s) have no matching annotation task_name: "
            f"{unmatched}. Dataset annotations and TASK_INDICES_TO_NAMES disagree — "
            "resolve before eval, do not fall back to activity-name prompts."
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
    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in sorted(task_prompts.items())}, f, indent=2)
    print(f"wrote {len(task_prompts)} task prompts → {args.output}")


if __name__ == "__main__":
    main()
