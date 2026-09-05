"""Compute per-dataset 10-D EEF normalization stats for OXE readers.

Each schema converts state and action to ``xyz(3) + rot6d(6) + grip(1)`` and
pools the two streams into ``meta/eef_stats.json``.  For DROID, the scan uses
the arm-side wrist/gripper-mount streams, applies the canonical prompt
exclusions, inverts gripper closedness, and records data-population provenance.
Rot6d dimensions are pinned to identity before output.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.oxe_droid import (
    DROID_DATA_POPULATION_DIGEST_KEY,
    DROID_EEF_STATS_CONTRACT,
    DROID_EEF_STATS_CONTRACT_KEY,
    DROID_STATS_POPULATION_KEY,
    load_droid_prompt_exclusions,
    resolve_droid_stats_population,
)
from openwam.dataloader.utils.eef import assert_unit_quaternion
from openwam.dataloader.utils.lerobotv3 import (
    digest_lerobot_v3_data_population,
    parse_info_json,
    read_lerobot_v3_population_shard,
    resolve_lerobot_v3_data_population,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_ARM10, pin_rot6d_identity
from openwam.dataloader.utils.oxe_schema import (
    bcz_state_to_arm10,
    droid_euler7_to_arm10,
    droid_pose6_closedness_to_arm10,
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
        "state_cols": ["other_information.observation_gripper_pose6d", "state"],
        "action_cols": ["other_information.action_wrist_pose"],
        "state_fn": "droid_gripper_pose6",
        "action_fn": "droid_euler7",
    },
}


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
    if state_fn == "euler7_state":
        return euler7_action_to_arm10(rows[list(rows.keys())[0]])
    if state_fn == "droid_euler7":
        return droid_euler7_to_arm10(rows[list(rows.keys())[0]])
    if state_fn == "droid_gripper_pose6":
        return droid_pose6_closedness_to_arm10(
            rows["other_information.observation_gripper_pose6d"],
            rows["state"][:, 6:7],
        )
    raise ValueError(f"unknown state_fn={state_fn}")


def _convert_action(rows: Dict[str, np.ndarray], action_fn: str) -> np.ndarray:
    if action_fn == "euler7_action":
        return euler7_action_to_arm10(rows[list(rows.keys())[0]])
    if action_fn == "droid_euler7":
        return droid_euler7_to_arm10(rows[list(rows.keys())[0]])
    raise ValueError(f"unknown action_fn={action_fn}")


def _table_rows(table: pa.Table, cols: List[str]) -> Dict[str, np.ndarray]:
    """Convert selected Arrow columns to stacked numpy rows."""
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        col_data = table.column(c).to_pylist()

        if col_data and not isinstance(col_data[0], (list, np.ndarray)):
            out[c] = np.asarray(col_data, dtype=np.float32).reshape(-1, 1)
        else:
            out[c] = np.asarray(col_data, dtype=np.float32)
    return out


def _load_shard(path: Path, cols: List[str]) -> Dict[str, np.ndarray]:
    """Load one parquet shard, returning a dict {col: ndarray-of-stacked-rows}."""
    return _table_rows(pq.read_table(path, memory_map=True, columns=cols), cols)


def compute_dataset_stats(
    dataset_dir: Path,
    dataset_name: str,
    rot6d_identity: bool = True,
    *,
    split: str = "train",
) -> Tuple[dict, int, int]:
    """Walk a dataset's data parquets, convert to 10-D EEF, aggregate stats.

    Returns:
        (stats_dict, n_state_samples, n_action_samples)
    """

    spec = SCHEMA[dataset_name]
    excluded_episode_indices: set[int] = set()
    droid_population = None
    droid_stats_population = None
    droid_stats_episode_indices = None
    if dataset_name == "DROID":
        info = parse_info_json(dataset_dir)
        droid_population = resolve_lerobot_v3_data_population(dataset_dir, info=info)
        _, excluded_episode_indices = load_droid_prompt_exclusions(
            dataset_dir,
            population=droid_population,
        )
        selected, droid_stats_population = resolve_droid_stats_population(
            droid_population,
            info,
            excluded_episode_indices,
            split=split,
        )
        droid_stats_episode_indices = selected["episode_index"].to_numpy(dtype=np.int64, copy=False)
        shard_inputs = list(droid_population.shards)
        logger.info("%s: scanning %d manifest-addressed data shards", dataset_name, len(shard_inputs))
    else:
        shard_inputs = sorted((dataset_dir / "data").rglob("*.parquet"))
        if not shard_inputs:
            raise FileNotFoundError(f"No parquet shards under {dataset_dir}/data")
        logger.info("%s: scanning %d parquet shards under %s/data", dataset_name, len(shard_inputs), dataset_dir)

    state_arrs: List[np.ndarray] = []
    action_arrs: List[np.ndarray] = []
    for i, shard_input in enumerate(shard_inputs, start=1):
        if droid_population is not None:
            columns = list(dict.fromkeys(["episode_index", *spec["state_cols"], *spec["action_cols"]]))
            table = read_lerobot_v3_population_shard(dataset_dir, shard_input, columns)
            episode_indices = table.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False)
            keep = np.isin(episode_indices, droid_stats_episode_indices)
            if not keep.any():
                continue
            table = table.filter(pa.array(keep))
            state_rows = _table_rows(table, spec["state_cols"])
            action_rows = _table_rows(table, spec["action_cols"])
        else:
            p = shard_input
            state_rows = _load_shard(p, spec["state_cols"])
            action_rows = _load_shard(p, spec["action_cols"])
        if not state_rows or not action_rows:
            continue
        state10 = _convert_state(state_rows, spec["state_fn"])
        action10 = _convert_action(action_rows, spec["action_fn"])
        state_arrs.append(state10)
        action_arrs.append(action10)
        if i % 50 == 0 or i == len(shard_inputs):
            logger.info("  %s: processed %d/%d shards", dataset_name, i, len(shard_inputs))

    if not state_arrs:
        raise ValueError(f"{dataset_name}: every parquet row is excluded; cannot compute stats")
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
        "mean": merged.mean(axis=0, dtype=np.float64).tolist(),
        "std": merged.std(axis=0, dtype=np.float64).tolist(),
        "q01": np.quantile(merged, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(merged, 0.99, axis=0).astype(np.float64).tolist(),
    }
    if dataset_name == "DROID":
        if n_state != droid_stats_population["num_rows"] or n_action != droid_stats_population["num_rows"]:
            raise ValueError(
                f"DROID stats scan selected {n_state} state/{n_action} action rows but the "
                f"effective train population declares {droid_stats_population['num_rows']} rows"
            )
        stats["excluded_episode_indices"] = sorted(excluded_episode_indices)
        stats[DROID_DATA_POPULATION_DIGEST_KEY] = digest_lerobot_v3_data_population(droid_population)
        stats[DROID_STATS_POPULATION_KEY] = droid_stats_population
        stats[DROID_EEF_STATS_CONTRACT_KEY] = dict(DROID_EEF_STATS_CONTRACT)
    if rot6d_identity:
        pin_rot6d_identity(stats, ROT6D_DIMS_ARM10)
    return stats, n_state, n_action


def _print_stats_table(stats: dict, name: str) -> None:
    """Per-dim summary for human eyeballing."""
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
        "--dataset-dir",
        type=str,
        default=None,
        help="Bucket path to scan, overriding {root}/{dataset}-Dataset. Requires --dataset "
        "(the schema to read it with) and is incompatible with --all. Use when a bucket lives "
        "outside the OXE root, e.g. --dataset DROID --dataset-dir /path/to/pretrain_dataset/Droid",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats but do not write meta/eef_stats.json",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="info.json split used for DROID normalization statistics (default: train)",
    )
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="Disable pinning rot6d stats to identity (rot6d would then be per-dim normalized "
        "like pos/gripper — generally undesirable; see pin_rot6d_identity).",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("must specify either --dataset NAME or --all")
    if args.dataset_dir and args.all:
        parser.error("--dataset-dir applies to a single bucket; use --dataset NAME, not --all")
    targets = list(SCHEMA.keys()) if args.all else [args.dataset]
    root = Path(args.root)
    for name in targets:
        ds_dir = Path(args.dataset_dir) if args.dataset_dir else root / f"{name}-Dataset"
        if not ds_dir.is_dir():
            logger.warning("%s: directory %s missing, skipping", name, ds_dir)
            continue
        stats, n_state, n_action = compute_dataset_stats(
            ds_dir,
            name,
            rot6d_identity=not args.no_rot6d_identity,
            split=args.split,
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
