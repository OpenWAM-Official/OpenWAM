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

import numpy as np
import pandas as pd

from openwam.dataloader.robocasa365 import STATS_DIM, _expand_stats_to_20d, state_to_arm10
from openwam.dataloader.transforms.normalize import compute_extended_stats


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


def _iter_episode_arm10(data_root: str):
    """Yield each episode's raw ``(T, 10)`` arm pose from a v2.1 bucket."""
    with open(os.path.join(data_root, "meta", "info.json")) as f:
        info = json.load(f)
    data_tmpl = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))
    with open(os.path.join(data_root, "meta", "episodes.jsonl")) as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    if not episodes:
        raise FileNotFoundError(f"No episodes in {data_root}/meta/episodes.jsonl")
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        path = os.path.join(
            data_root, data_tmpl.format(episode_chunk=ep_idx // chunks_size, episode_index=ep_idx)
        )
        state = np.stack(pd.read_parquet(path, columns=["observation.state"])["observation.state"].values)
        yield state_to_arm10(state.astype(np.float32))


def compute_normalization_stats(data_root: str) -> dict:
    """Compute single-arm 10-D EEF stats for one RoboCasa365 task bucket.

    Returns ``{"eef": {mean, std, min, max, q01, q99}, "num_timesteps": int}``.
    """
    chunks = []
    total = 0
    for i, arm10 in enumerate(_iter_episode_arm10(data_root)):
        chunks.append(arm10)
        total += arm10.shape[0]
        if (i + 1) % 100 == 0:
            print(f"  [stats] {i + 1} episodes, {total} timesteps so far")
    if not chunks:
        raise ValueError(f"No timesteps accumulated from {data_root}")
    eef = compute_extended_stats(chunks)
    if len(eef["mean"]) != STATS_DIM:
        raise ValueError(f"computed arm dim {len(eef['mean'])} != {STATS_DIM}")
    print(f"  [stats] done: {total} timesteps over {len(chunks)} episodes, dim={STATS_DIM}")
    # Persist at the full 20-D action dim (left=arm, right=neutral) so the deploy normalizer
    # can invert the model's 20-D output (the 10-D file was the deploy-break).
    return {"eef": _expand_stats_to_20d(eef), "num_timesteps": int(total)}


def compute_multitask_stats(data_roots: list[str]) -> dict:
    """Compute ONE shared 10-D EEF stats across several task buckets.

    Mirrors robotwin's multi-task shared-stats contract: every task in a
    multi-task run must train in the SAME normalized space, so stats are pooled
    over all buckets (not computed per-task). Returns the same
    ``{"eef": {...}, "num_timesteps": int}`` schema as the single-task path.
    """
    chunks = []
    total = 0
    for dr in data_roots:
        n0 = total
        for arm10 in _iter_episode_arm10(dr):
            chunks.append(arm10)
            total += arm10.shape[0]
        print(f"  [multitask-stats] {os.path.basename(os.path.dirname(os.path.dirname(dr.rstrip('/'))))}: "
              f"+{total - n0} timesteps (running {total})")
    if not chunks:
        raise ValueError(f"No timesteps accumulated from {len(data_roots)} buckets")
    eef = compute_extended_stats(chunks)
    if len(eef["mean"]) != STATS_DIM:
        raise ValueError(f"computed arm dim {len(eef['mean'])} != {STATS_DIM}")
    print(f"  [multitask-stats] done: {total} timesteps over {len(data_roots)} buckets, dim={STATS_DIM}")
    # Persist at the full 20-D action dim (left=arm, right=neutral) so the deploy normalizer
    # can invert the model's 20-D output (the 10-D file was the deploy-break).
    return {"eef": _expand_stats_to_20d(eef), "num_timesteps": int(total)}


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
