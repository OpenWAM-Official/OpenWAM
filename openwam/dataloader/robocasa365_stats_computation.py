"""Per-task action-normalization stats for RoboCasa365 (single-arm EEF; computed
at 10-D, persisted at 20-D).

Simplified single-arm dual of ``robotwin_stats_computation.py``. RoboCasa365 has
exactly one action representation (the state-derived single-arm EEF), so there is
no joint/eef split and no multitask-checkpoint sharding — just: iterate a bucket's
episodes, assemble the raw 10-D arm pose from ``observation.state`` (the same
``state_to_arm10`` the reader uses), and reduce to ``mean/std/min/max/q01/q99``.

Output schema (``.npy``, ``allow_pickle``)::

    {"eef": {mean, std, min, max, q01, q99}, "num_timesteps": int}   # 20-D vectors
    #   (arm10 left = real stats, right half = neutral: mean0/std1/min-1/max1/q01-1/q99 1;
    #    reduced at 10-D internally, then left-padded to 20-D by _expand_stats_to_20d before
    #    persist so the deploy normalizer can invert the model's 20-D action)

``RoboCasa365Dataset`` auto-computes this on first use when ``normalize_mode`` is
set and no stats file exists; run :func:`main` to precompute.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.robocasa365 import (
    STATS_DIM,
    _expand_stats_to_20d,
    _task_from_source_prefix,
    state_to_arm10,
)
from openwam.dataloader.transforms.normalize import compute_extended_stats
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet


def atomic_save_stats_npy(path: str, stats: dict) -> None:
    """Write the stats ``.npy`` atomically (tmp + ``os.replace``).

    ``np.save`` creates the destination at open time but fills it afterwards, so a
    polling consumer can see a half-written file; serialize to a sibling tmp path
    and rename it into place instead. (Identical contract to robotwin's.)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    np.save(tmp_path, stats, allow_pickle=True)
    actual_tmp = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npy"
    os.replace(actual_tmp, path)


# LeRobot ``action`` base command dims (mobile): [0:3] x/y/yaw vel, [3] torso, [4] control_mode.
_ACTION_BASE = slice(0, 5)
BASE_ACTION_DIM = 5


def _iter_episode_arrays(data_root: str, include_base: bool = False, task_name: str | None = None):
    """Yield ``(arm10, base5)`` per episode from a v3 aggregated repo.

    ``arm10`` = raw ``(T, 10)`` arm pose from ``observation.state`` (always). ``base5`` = raw
    ``(T, 5)`` base command from the ``action`` field ([x/y/yaw vel, torso, control_mode]) when
    ``include_base`` else ``None``. When ``task_name`` is given the episode table is filtered to that
    task's ``source_prefix`` (v3 packs all tasks into one repo). Each aggregated shard is read once
    and sliced per episode via its file-local row offset (mirrors the reader).
    """
    with open(os.path.join(data_root, "meta", "info.json")) as f:
        data_tmpl = json.load(f)["data_path"]
    eps = load_episodes_parquet(Path(data_root))
    # Offsets over the FULL table BEFORE the task filter (a task not first in its shard must read at
    # its true file-local offset, past the other tasks' rows in the same file). Mirrors the reader.
    eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
    if task_name is not None:
        eps = eps[eps["source_prefix"].map(_task_from_source_prefix) == task_name].reset_index(drop=True)
    if len(eps) == 0:
        raise FileNotFoundError(f"No episodes for task {task_name!r} under {data_root}")
    cols = ["observation.state"] + (["action"] if include_base else [])
    for (chunk, file), grp in eps.groupby(["data/chunk_index", "data/file_index"], sort=False):
        path = os.path.join(data_root, data_tmpl.format(chunk_index=int(chunk), file_index=int(file)))
        df = pd.read_parquet(path, columns=cols)
        st = df["observation.state"].values
        ac = df["action"].values if include_base else None
        for _, r in grp.iterrows():
            o, length = int(r["_data_row_offset"]), int(r["length"])
            arm10 = state_to_arm10(np.stack(st[o : o + length]).astype(np.float32))
            base5 = np.stack(ac[o : o + length]).astype(np.float32)[:, _ACTION_BASE] if include_base else None
            yield arm10, base5


def _base_stats_block(base_chunks: list) -> dict:
    """5-D base command stats, with a min==max / std==0 guard so constant dims (e.g. control_mode
    all -1 or torso all 0 in fixed-base buckets) don't divide-by-zero at min-max/z-score time —
    a constant dim then normalizes to a constant that still round-trips through denormalize."""
    b = compute_extended_stats(base_chunks)
    out = {k: np.asarray(b[k], np.float32).reshape(-1).copy() for k in ("mean", "std", "min", "max", "q01", "q99")}
    degenerate = (out["max"] - out["min"]) < 1e-6
    out["max"] = np.where(degenerate, out["min"] + 1.0, out["max"]).astype(np.float32)
    out["std"] = np.where(out["std"] < 1e-6, 1.0, out["std"]).astype(np.float32)
    if out["mean"].shape[0] != BASE_ACTION_DIM:
        raise ValueError(f"base stats dim {out['mean'].shape[0]} != {BASE_ACTION_DIM}")
    return out


def compute_normalization_stats(data_root: str, include_base: bool = False, task_name: str | None = None) -> dict:
    """Compute single-arm 10-D EEF stats (+ optional 5-D base command stats) for one task.

    Returns ``{"eef": {...20-D...}, "num_timesteps": int}``, plus ``"base": {...5-D...}`` when
    ``include_base`` (mobile: base command read from the LeRobot ``action`` field). ``task_name``
    filters the v3 aggregated repo to one task (see :func:`_iter_episode_arrays`).
    """
    arm_chunks, base_chunks, total = [], [], 0
    for i, (arm10, base5) in enumerate(_iter_episode_arrays(data_root, include_base, task_name)):
        arm_chunks.append(arm10)
        total += arm10.shape[0]
        if include_base:
            base_chunks.append(base5)
        if (i + 1) % 100 == 0:
            print(f"  [stats] {i + 1} episodes, {total} timesteps so far")
    if not arm_chunks:
        raise ValueError(f"No timesteps accumulated from {data_root}")
    eef = compute_extended_stats(arm_chunks)
    if len(eef["mean"]) != STATS_DIM:
        raise ValueError(f"computed arm dim {len(eef['mean'])} != {STATS_DIM}")
    print(f"  [stats] done: {total} timesteps over {len(arm_chunks)} episodes, dim={STATS_DIM}"
          f"{' +base5' if include_base else ''}")
    # Persist arm at the full 20-D action dim (left=arm, right=neutral) so the deploy normalizer
    # can invert the model's 20-D output (the 10-D file was the deploy-break).
    out = {"eef": _expand_stats_to_20d(eef), "num_timesteps": int(total)}
    if include_base:
        out["base"] = _base_stats_block(base_chunks)
    return out


def compute_multitask_stats(roots: list, include_base: bool = False) -> dict:
    """Compute ONE shared 10-D EEF (+ optional 5-D base) stats across several tasks.

    ``roots`` is ``[(task_name, repo), ...]`` (v3: every entry shares the same aggregated repo, one
    per selected task). Mirrors robotwin's multi-task shared-stats contract: every task in a
    multi-task run must train in the SAME normalized space, so stats are pooled over all tasks. Same
    schema as the single-task path (plus a ``"base"`` block when ``include_base``).
    """
    arm_chunks, base_chunks, total = [], [], 0
    for task_name, repo in roots:
        n0 = total
        for arm10, base5 in _iter_episode_arrays(repo, include_base, task_name):
            arm_chunks.append(arm10)
            total += arm10.shape[0]
            if include_base:
                base_chunks.append(base5)
        print(f"  [multitask-stats] {task_name}: +{total - n0} timesteps (running {total})")
    if not arm_chunks:
        raise ValueError(f"No timesteps accumulated from {len(roots)} tasks")
    eef = compute_extended_stats(arm_chunks)
    if len(eef["mean"]) != STATS_DIM:
        raise ValueError(f"computed arm dim {len(eef['mean'])} != {STATS_DIM}")
    print(f"  [multitask-stats] done: {total} timesteps over {len(roots)} tasks, dim={STATS_DIM}"
          f"{' +base5' if include_base else ''}")
    out = {"eef": _expand_stats_to_20d(eef), "num_timesteps": int(total)}
    if include_base:
        out["base"] = _base_stats_block(base_chunks)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute RoboCasa365 per-task EEF normalization stats")
    ap.add_argument("data_root", help="A task's lerobot/ bucket (has meta/info.json + meta/episodes.jsonl)")
    ap.add_argument("-o", "--output", default=None, help="Output .npy (default: {data_root}/{task}_eef_stats.npy)")
    args = ap.parse_args()

    out = args.output
    if out is None:
        task = os.path.basename(os.path.dirname(os.path.dirname(args.data_root.rstrip("/")))) or "task"
        out = os.path.join(args.data_root, f"{task}_eef_stats.npy")
    stats = compute_normalization_stats(args.data_root)
    atomic_save_stats_npy(out, stats)
    print(f"Saved stats -> {out}")


if __name__ == "__main__":
    main()
