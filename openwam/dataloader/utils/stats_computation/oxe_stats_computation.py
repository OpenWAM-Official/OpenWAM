"""Public implementation. Dataset-specific audit notes were removed."""


















from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.utils.eef import assert_unit_quaternion
from openwam.dataloader.utils.oxe_schema import (
    bcz_state_to_arm10,
    droid_state_to_arm10,
    euler7_action_to_arm10,
    fractal_state_to_arm10,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("oxe_stats_computation")



SCHEMA: Dict[str, Dict] = {
    "BC-Z": {
        "state_cols": ["observation.state"],
        "action_cols": ["action"],
        "state_fn": "bcz_state",
        "action_fn": "euler7_action",
    },
    "Bridge": {
        "state_cols": ["observation.state"],
        "action_cols": ["action"],
        "state_fn": "bcz_state",
        "action_fn": "euler7_action",
    },
    "Fractal": {
        "state_cols": ["observation.state"],
        "action_cols": ["action"],
        "state_fn": "fractal_state",
        "action_fn": "euler7_action",
    },
    "DROID": {

        "state_cols": [
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
        ],


        "action_cols": ["action.original"],
        "state_fn": "droid_state",
        "action_fn": "euler7_action",
    },
}










ROT6D_DIMS = (3, 4, 5, 6, 7, 8)


def _pin_rot6d_identity(stats: dict) -> None:
    """Public implementation. Dataset-specific audit notes were removed."""






    ident = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}
    for key, val in ident.items():
        for i in ROT6D_DIMS:
            stats[key][i] = val


def _convert_state(rows: Dict[str, np.ndarray], state_fn: str) -> np.ndarray:
    if state_fn == "bcz_state":
        return bcz_state_to_arm10(rows["observation.state"])
    if state_fn == "fractal_state":
        quat = rows["observation.state"][:, 3:7]
        assert_unit_quaternion(quat, tol=0.05, sample_n=min(64, len(quat)))
        return fractal_state_to_arm10(rows["observation.state"])
    if state_fn == "droid_state":
        return droid_state_to_arm10(
            rows["observation.state.cartesian_position"],
            rows["observation.state.gripper_position"],
        )
    raise ValueError(f"unknown state_fn={state_fn}")


def _convert_action(rows: Dict[str, np.ndarray], action_fn: str) -> np.ndarray:
    if action_fn == "euler7_action":

        return euler7_action_to_arm10(rows[list(rows.keys())[0]])
    raise ValueError(f"unknown action_fn={action_fn}")


def _load_shard(path: Path, cols: List[str]) -> Dict[str, np.ndarray]:
    """Public implementation. Dataset-specific audit notes were removed."""
    table = pq.read_table(path, memory_map=True, columns=cols)
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        col_data = table.column(c).to_pylist()

        if col_data and not isinstance(col_data[0], (list, np.ndarray)):
            out[c] = np.asarray(col_data, dtype=np.float32).reshape(-1, 1)
        else:
            out[c] = np.asarray(col_data, dtype=np.float32)
    return out


def compute_dataset_stats(
    dataset_dir: Path, dataset_name: str, rot6d_identity: bool = True
) -> Tuple[dict, int, int]:
    """Public implementation. Dataset-specific audit notes were removed."""




    spec = SCHEMA[dataset_name]
    parquet_paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards under {dataset_dir}/data")
    logger.info("%s: scanning %d parquet shards under %s/data", dataset_name, len(parquet_paths), dataset_dir)

    state_arrs: List[np.ndarray] = []
    action_arrs: List[np.ndarray] = []
    for i, p in enumerate(parquet_paths, start=1):
        state_rows = _load_shard(p, spec["state_cols"])
        action_rows = _load_shard(p, spec["action_cols"])
        state10 = _convert_state(state_rows, spec["state_fn"])
        action10 = _convert_action(action_rows, spec["action_fn"])
        state_arrs.append(state10)
        action_arrs.append(action10)
        if i % 50 == 0 or i == len(parquet_paths):
            logger.info("  %s: processed %d/%d shards", dataset_name, i, len(parquet_paths))

    state_all = np.concatenate(state_arrs, axis=0)
    action_all = np.concatenate(action_arrs, axis=0)
    n_state = int(len(state_all))
    n_action = int(len(action_all))

    merged = np.concatenate([state_all, action_all], axis=0)
    logger.info(
        "%s: merged %d state + %d action rows = %d total samples for stats",
        dataset_name,
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

        _pin_rot6d_identity(stats)
    return stats, n_state, n_action


def _print_stats_table(stats: dict, name: str) -> None:
    """Public implementation. Dataset-specific audit notes were removed."""
    dim_names = ["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grip"]
    print(f"\n  {name}  (n_samples={stats['n_samples']:,})")
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
        "--root",
        type=str,
        default="/path/to/OXE",
        help="OXE dataset root (each dataset is in {root}/<name>-Dataset/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(SCHEMA.keys()),
        help="Single dataset to process (mutually exclusive with --all)",
    )
    parser.add_argument("--all", action="store_true", help="Process all 4 OXE datasets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats but do not write meta/eef_stats.json",
    )
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="Disable pinning rot6d stats to identity (rot6d would then be per-dim normalized "
        "like pos/gripper — generally undesirable; see _pin_rot6d_identity).",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("must specify either --dataset NAME or --all")
    targets = list(SCHEMA.keys()) if args.all else [args.dataset]
    root = Path(args.root)
    for name in targets:
        ds_dir = root / f"{name}-Dataset"
        if not ds_dir.is_dir():
            logger.warning("%s: directory %s missing, skipping", name, ds_dir)
            continue
        stats, n_state, n_action = compute_dataset_stats(
            ds_dir, name, rot6d_identity=not args.no_rot6d_identity
        )
        _print_stats_table(stats, name)
        if not args.dry_run:
            out_path = ds_dir / "meta" / "eef_stats.json"
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info("%s: wrote %s", name, out_path)
        else:
            logger.info("%s: --dry-run, no file written", name)


if __name__ == "__main__":
    main()
