"""Normalization statistics for the converted native-action LIBERO dataset.

Scans every ``data/**/*.parquet`` row of an already-converted EEF10 LIBERO
dataset (columns ``action`` / ``observation.state``) and writes the same
payload the retired EEF10 converter used to emit alongside the conversion:
separate action / state blocks with exact moments, min/max, quantiles, and the
rot6d dims (3:9) pinned to identity.

Default output (the libero.yaml convention, auto-built by the reader on first
use — rank 0 computes, other ranks wait):

    <dataset_dir>/meta/libero_normalization_stats.npy

CLI:
    python -m openwam.dataloader.utils.stats_computation.libero_stats_computation \\
        --dataset-dir /path/to/benchmark_data/libero
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.libero import (
    ACTION_STATS_KEY,
    GRIPPER_CONVENTION,
    NORMALIZATION_STATS_FILENAME,
    OUTPUT_REPRESENTATION,
    STATE_STATS_KEY,
)

ACTION_COLUMN = "action"
STATE_COLUMN = "observation.state"
ROT6D_DIMS = tuple(range(3, 9))


def _feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"statistics require a non-empty 2-D array, got {values.shape}")
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _pin_rot6d_identity(stats: dict) -> None:
    identity = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}
    for key, value in identity.items():
        for dim in ROT6D_DIMS:
            stats[key][dim] = value


def _load_columns(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    parquet_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {dataset_dir / 'data'}")
    actions, states = [], []
    for path in parquet_files:
        table = pq.read_table(path, columns=[ACTION_COLUMN, STATE_COLUMN])
        actions.append(np.stack(table.column(ACTION_COLUMN).to_pylist()).astype(np.float32))
        states.append(np.stack(table.column(STATE_COLUMN).to_pylist()).astype(np.float32))
    return np.concatenate(actions, axis=0), np.concatenate(states, axis=0)


def compute_libero_stats(dataset_dir: str | Path) -> dict:
    """Full-corpus scan of the converted dataset's EEF10 action/state columns."""
    action_all, state_all = _load_columns(Path(dataset_dir))
    action_stats = _feature_stats(action_all)
    state_stats = _feature_stats(state_all)
    _pin_rot6d_identity(action_stats)
    _pin_rot6d_identity(state_stats)
    action_stats.update(
        {
            "num_timesteps": int(action_all.shape[0]),
            "pool": "action_only",
            "gripper_convention": GRIPPER_CONVENTION,
            "representation": OUTPUT_REPRESENTATION,
            "action_semantics": "native normalized LIBERO delta command",
        }
    )
    state_stats.update(
        {
            "num_timesteps": int(state_all.shape[0]),
            "pool": "state_only",
            "gripper_convention": GRIPPER_CONVENTION,
            "representation": OUTPUT_REPRESENTATION,
            "state_semantics": "achieved EEF pose",
        }
    )
    return {ACTION_STATS_KEY: action_stats, STATE_STATS_KEY: state_stats}


def build_and_save_libero_stats(dataset_dir: str | Path, output: str | Path | None = None) -> Path:
    """Compute and atomically write the stats payload; returns the output path."""
    dataset_dir = Path(dataset_dir)
    out_path = Path(output) if output else dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_libero_stats(dataset_dir)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_name, out_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Converted native-action LIBERO dataset root")
    parser.add_argument(
        "--output",
        default=None,
        help=f".npy path (default <dataset_dir>/meta/{NORMALIZATION_STATS_FILENAME})",
    )
    args = parser.parse_args()
    out_path = build_and_save_libero_stats(args.dataset_dir, args.output)
    payload = np.load(out_path, allow_pickle=True).item()
    print(f"wrote {out_path}")
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        print(f"  {key}: num_timesteps={block['num_timesteps']}, mean[:3]={np.round(block['mean'][:3], 5).tolist()}")


if __name__ == "__main__":
    main()
