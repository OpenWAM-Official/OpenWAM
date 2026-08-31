"""Normalization statistics for canonical compact RoboCasa365 state19/action15 v3 data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.robocasa365 import (
    ACTION_DIM,
    ACTION_STATS_KEY,
    STATE_DIM,
    STATE_STATS_KEY,
    _task_from_source_prefix,
)
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import atomic_save_stats_npy
from scripts.convert_robocasa365_compact_v3 import (
    StatsAccumulator,
    _stats_payload,
)


def _iter_arrays(data_root: str, task_name: str | None = None):
    root = Path(data_root)
    episodes = load_episodes_parquet(root)
    episodes["_offset"] = compute_file_local_offsets(episodes, "data/chunk_index", "data/file_index")
    if task_name is not None:
        episodes = episodes[episodes["source_prefix"].map(_task_from_source_prefix) == task_name].reset_index(drop=True)
    if episodes.empty:
        raise FileNotFoundError(f"No episodes for task {task_name!r} under {root}")
    import json

    with (root / "meta" / "info.json").open(encoding="utf-8") as handle:
        template = json.load(handle)["data_path"]
    for (chunk, file_index), group in episodes.groupby(["data/chunk_index", "data/file_index"], sort=False):
        path = root / template.format(chunk_index=int(chunk), file_index=int(file_index))
        frame = pd.read_parquet(path, columns=["observation.state", "action"])
        states, actions = frame["observation.state"].values, frame["action"].values
        # A whole-repository scan covers every row exactly once.  Yield the
        # shard in one batch instead of slicing it into thousands of episodes.
        if task_name is None:
            expected = int(group["length"].sum())
            if expected != len(frame):
                raise ValueError(f"episode lengths do not cover {path}: {expected} != {len(frame)}")
            yield np.stack(states).astype(np.float32), np.stack(actions).astype(np.float32)
            continue
        for _, row in group.iterrows():
            offset, length = int(row["_offset"]), int(row["length"])
            yield (
                np.stack(states[offset : offset + length]).astype(np.float32),
                np.stack(actions[offset : offset + length]).astype(np.float32),
            )


def _compute(items, label: str) -> dict:
    action_acc = StatsAccumulator(ACTION_DIM, seed=71)
    state_acc = StatsAccumulator(STATE_DIM, seed=73)
    total = 0
    for index, (state, action) in enumerate(items, 1):
        state_acc.update(state)
        action_acc.update(action)
        total += state.shape[0]
        if index % 10 == 0:
            print(f"  [{label}] {index} batches, {total:,} rows", flush=True)
    return _stats_payload(action_acc, state_acc)


def compute_normalization_stats(data_root: str, task_name: str | None = None, **_unused) -> dict:
    return _compute(_iter_arrays(data_root, task_name), "stats")


def compute_multitask_stats(roots: list, **_unused) -> dict:
    def items():
        for task_name, repo in roots:
            yield from _iter_arrays(repo, task_name)

    return _compute(items(), f"multitask/{len(roots)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", nargs="+", help="one or more compact RoboCasa365 v3 repositories")
    parser.add_argument("--task")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    if args.task is not None and len(args.data_root) != 1:
        parser.error("--task can only be used with one data_root")
    if len(args.data_root) == 1:
        output = args.output or os.path.join(args.data_root[0], f"{args.task or 'robocasa365'}_compact_stats.npy")
        payload = compute_normalization_stats(args.data_root[0], args.task)
    else:
        if args.output is None:
            parser.error("--output is required when scanning multiple data roots")
        output = args.output
        payload = compute_multitask_stats([(None, root) for root in args.data_root])
    if ACTION_STATS_KEY not in payload or STATE_STATS_KEY not in payload:
        raise AssertionError("internal compact stats schema error")
    atomic_save_stats_npy(output, payload)
    print(f"Saved stats -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
