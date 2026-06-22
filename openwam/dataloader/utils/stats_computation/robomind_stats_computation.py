#!/usr/bin/env python3
"""Compute per-robot-type unified 20-D EEF stats for RoboMIND datasets.

Mirrors ``robocoin_stats_computation``. For each ``robot_type`` (pooled
across benchmarks — e.g. franka_3rgb_b10/b11/b12 share one stats file), streams
all parquet files and pulls BOTH the ``action`` and ``observation.state`` EEF
columns. Each row is converted to the canonical 20-D EEF schema with the SAME
``robomind_raw_to_20d`` the reader uses (so stats and reader are bit-identical),
and the two streams are pooled before accumulating mean/std/min/max + q01/q99.

The embodiment's raw EEF layout (``single_euler`` / ``single_quat`` /
``dual_euler``) is resolved from each bucket's ``info.json`` camera set via
``resolve_robomind_layout`` — identical to the reader's ``_resolve_cameras``.

For single-arm embodiments the right half [10:20] of every 20-D point is zero,
so its stats are degenerate (std/range 0). That is fine: the reader's
normalizer floors every scale at NORM_EPS, mapping those dims to a constant,
and they are masked out by LEFT_ARM_DIM_MASK and never reach the loss.

Results are written to:
    {dataset_dir}/meta/stats_{robot_type}.json

with the schema:
    {
      "eef": {
        "mean": [...20], "std": [...20], "min": [...20], "max": [...20],
        "q01": [...20], "q99": [...20],   # for quantile normalization
        "num_timesteps": <total rows across action+state>,
        "num_datasets": <n>, "num_files": <m>,
        "robot_type": "<name>", "eef_kind": "<kind>",
        "pool": "action+state"
      }
    }

mean/std/min/max are exact (streamed over every row); q01/q99 are estimated
from a bounded uniform reservoir sample.

Usage:
    python -m openwam.dataloader.utils.stats_computation.robomind_stats_computation --dataset_dir /path/to/robomind
    python -m openwam.dataloader.utils.stats_computation.robomind_stats_computation --dataset_dir <root> --robot_type franka_panda_3rgb
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# Reuse the reader's accumulator (exact mean/std/min/max + reservoir q01/q99)
# and the reader's raw→20-D converter, so stats and reader never diverge.
from openwam.dataloader.robomind import resolve_robomind_layout, robomind_raw_to_20d
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import (
    RESERVOIR_CAP,
    Accumulator,
)

_NEEDED_COLS = ["observation.state", "action"]


def discover_datasets_by_robot_type(root: str) -> dict:
    """Group buckets by robot_type; also stash each group's eef_kind.

    Returns ``{robot_type: {"dirs": [...], "eef_kind": "<kind>"}}``.
    """
    groups: dict = {}
    for name in sorted(os.listdir(root)):
        info_path = os.path.join(root, name, "meta", "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path) as f:
            info = json.load(f)
        rtype = info.get("robot_type", "unknown")
        _, _, _, kind = resolve_robomind_layout(info.get("features", {}))
        g = groups.setdefault(rtype, {"dirs": [], "eef_kind": kind})
        g["dirs"].append(os.path.join(root, name))
        if g["eef_kind"] != kind:
            raise ValueError(
                f"robot_type {rtype!r} maps to inconsistent eef_kind "
                f"({g['eef_kind']!r} vs {kind!r}) across buckets — check info.json camera sets."
            )
    return groups


def compute_stats_for_robot_type(rtype: str, kind: str, dataset_dirs: list) -> dict:
    """Pool action+state 20-D EEF over all buckets of one robot_type."""
    acc = Accumulator(dim=20)
    total_files = 0
    for ds_dir in dataset_dirs:
        data_dir = os.path.join(ds_dir, "data")
        if not os.path.isdir(data_dir):
            continue
        for chunk in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk)
            if not os.path.isdir(chunk_path):
                continue
            for fname in sorted(os.listdir(chunk_path)):
                if not fname.endswith(".parquet"):
                    continue
                fpath = os.path.join(chunk_path, fname)
                try:
                    df = pd.read_parquet(fpath, columns=_NEEDED_COLS)
                    action_raw = np.stack(df["action"].values).astype(np.float32)
                    state_raw = np.stack(df["observation.state"].values).astype(np.float32)
                    action_20d = robomind_raw_to_20d(action_raw, kind)
                    state_20d = robomind_raw_to_20d(state_raw, kind)
                    pooled = np.concatenate([action_20d, state_20d], axis=0)
                    acc.update_batch(pooled)
                    total_files += 1
                except Exception as e:
                    print(f"  Warning: skipping {fpath}: {e}")
    stats = acc.finalize()
    stats["num_timesteps"] = int(acc.count)
    stats["num_datasets"] = len(dataset_dirs)
    stats["num_files"] = total_files
    stats["robot_type"] = rtype
    stats["eef_kind"] = kind
    stats["pool"] = "action+state"
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="RoboMIND root (subdirs = per-embodiment buckets)")
    parser.add_argument("--robot_type", default=None, help="Compute stats for a single robot_type only")
    args = parser.parse_args()

    groups = discover_datasets_by_robot_type(args.dataset_dir)
    print(f"Found {len(groups)} robot types: {sorted(groups.keys())} (reservoir_cap={RESERVOIR_CAP:,})")

    out_dir = os.path.join(args.dataset_dir, "meta")
    os.makedirs(out_dir, exist_ok=True)

    for rtype in sorted(groups.keys()):
        if args.robot_type and rtype != args.robot_type:
            continue
        g = groups[rtype]
        print(f"\n{'=' * 60}")
        print(f"Computing stats for {rtype} (kind={g['eef_kind']}, {len(g['dirs'])} buckets)...")
        stats = compute_stats_for_robot_type(rtype, g["eef_kind"], g["dirs"])

        out_path = os.path.join(out_dir, f"stats_{rtype}.json")
        with open(out_path, "w") as f:
            json.dump({"eef": stats}, f, indent=2)

        print(f"  timesteps: {stats['num_timesteps']:,}")
        print(f"  mean[:5]: {stats['mean'][:5]}")
        print(f"  q01[:5]:  {stats['q01'][:5]}")
        print(f"  q99[:5]:  {stats['q99'][:5]}")
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
