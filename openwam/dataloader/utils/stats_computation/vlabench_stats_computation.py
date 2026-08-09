"""Compute EEF stats for the VLABench primitive finetune dataset.

Scans every data parquet shard, converts the 7-D
``[x, y, z, roll, pitch, yaw, gripper]`` state and action streams to 10-D EEF
(``pos(3) + rot6d(6) + grip(1)``), and writes a merged
``min / max / mean / std / q01 / q99`` summary to
``{dataset_dir}/meta/eef_stats.json``.

state and action are stacked into a single ``(N, 10)`` matrix so one set of
parameters governs both streams — matching the OXE convention and what
``VLABenchDataset`` normalizes against.

The rot6d dims (3:9) are pinned to identity by default
(:func:`~openwam.dataloader.utils.normalization.pin_rot6d_identity`) so
normalization is a pass-through on the rotation representation under every
mode. Without the pin, min-max would rescale each rot6d component
independently and distort rotations; the reader warns loudly when it loads a
stats file that predates the pin.

Usage:
    python -m openwam.dataloader.utils.stats_computation.vlabench_stats_computation \
        --dataset-dir /path/to/vlabench_primitive_ft_lerobot_video
    ... --dry-run          # print the table, write nothing
    ... --no-rot6d-identity
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.utils.normalization import ROT6D_DIMS_ARM10, pin_rot6d_identity
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("vlabench_stats_computation")

STATE_COL = "state"
ACTION_COL = "actions"
EULER7_DIM = 7


def _load_euler7(path: Path, column: str) -> np.ndarray:
    """Load one parquet shard's ``(N, 7)`` euler7 column."""
    table = pq.read_table(path, memory_map=True, columns=[column])
    arr = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EULER7_DIM:
        raise ValueError(f"{path.name}: column {column!r} must be (N, {EULER7_DIM}), got {arr.shape}")
    return arr


def compute_dataset_stats(dataset_dir: Path, rot6d_identity: bool = True) -> Tuple[dict, int, int]:
    """Walk the data parquets, convert to 10-D EEF, aggregate stats.

    Returns:
        ``(stats_dict, n_state_samples, n_action_samples)``
    """
    parquet_paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards under {dataset_dir}/data")
    logger.info("VLABench: scanning %d parquet shards under %s/data", len(parquet_paths), dataset_dir)

    state_arrs: List[np.ndarray] = []
    action_arrs: List[np.ndarray] = []
    for i, p in enumerate(parquet_paths, start=1):
        state_arrs.append(euler7_action_to_arm10(_load_euler7(p, STATE_COL)))
        action_arrs.append(euler7_action_to_arm10(_load_euler7(p, ACTION_COL)))
        if i % 100 == 0 or i == len(parquet_paths):
            logger.info("  processed %d/%d shards", i, len(parquet_paths))

    state_all = np.concatenate(state_arrs, axis=0)
    action_all = np.concatenate(action_arrs, axis=0)
    n_state = int(len(state_all))
    n_action = int(len(action_all))
    merged = np.concatenate([state_all, action_all], axis=0)
    logger.info(
        "VLABench: merged %d state + %d action rows = %d total samples for stats",
        n_state,
        n_action,
        len(merged),
    )

    stats = {
        "n_samples": int(len(merged)),
        "n_state_samples": n_state,
        "n_action_samples": n_action,
        "min": merged.min(axis=0).astype(np.float64).tolist(),
        "max": merged.max(axis=0).astype(np.float64).tolist(),
        "mean": merged.mean(axis=0).astype(np.float64).tolist(),
        "std": merged.std(axis=0).astype(np.float64).tolist(),
        "q01": np.quantile(merged, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(merged, 0.99, axis=0).astype(np.float64).tolist(),
    }
    if rot6d_identity:
        pin_rot6d_identity(stats, ROT6D_DIMS_ARM10)
    return stats, n_state, n_action


def _print_stats_table(stats: dict) -> None:
    """Per-dim summary for human eyeballing."""
    dim_names = ["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grip"]
    print(f"\n  VLABench  (n_samples={stats['n_samples']:,})")
    print(f"  {'dim':<6} {'min':>10} {'max':>10} {'q01':>10} {'q99':>10} {'mean':>10} {'std':>10}")
    for i, dn in enumerate(dim_names):
        print(
            f"  {dn:<6} "
            f"{stats['min'][i]:>10.3f} "
            f"{stats['max'][i]:>10.3f} "
            f"{stats['q01'][i]:>10.3f} "
            f"{stats['q99'][i]:>10.3f} "
            f"{stats['mean'][i]:>10.3f} "
            f"{stats['std'][i]:>10.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="VLABench LeRobot v3 dataset root (the dir holding data/ and meta/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats but do not write meta/eef_stats.json",
    )
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="Disable pinning rot6d stats to identity (rot6d would then be per-dim normalized "
        "like pos/gripper — generally undesirable; see pin_rot6d_identity).",
    )
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    if not ds_dir.is_dir():
        parser.error(f"dataset dir does not exist: {ds_dir}")

    stats, _n_state, _n_action = compute_dataset_stats(
        ds_dir, rot6d_identity=not args.no_rot6d_identity
    )
    _print_stats_table(stats)
    if args.dry_run:
        logger.info("--dry-run, no file written")
        return
    out_path = ds_dir / "meta" / "eef_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
