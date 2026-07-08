#!/usr/bin/env python3
"""Public implementation. Dataset-specific audit notes were removed."""

















































import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator




_COLS = {
    "action.ee_base": 18,
    "action.gripper": 2,
    "action.dex": 12,
    "action.robot_velocity": 3,
    "observation.state.ee_base": 18,
    "observation.state.gripper": 2,
    "observation.state.dex": 12,
    "observation.state.robot_velocity": 3,
}

OUT_FILENAME = "stats_pooled.json"


def compute_bucket(bucket_dir: str) -> dict:
    """Public implementation. Dataset-specific audit notes were removed."""
    accs = {c: Accumulator(dim=d) for c, d in _COLS.items()}
    files = sorted(glob.glob(os.path.join(bucket_dir, "data", "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no data parquet under {bucket_dir}/data")
    n_files = 0
    n_time = 0
    for fp in files:
        present = set(pq.ParquetFile(fp).schema_arrow.names)
        cols = [c for c in _COLS if c in present]
        df = pd.read_parquet(fp, columns=cols)
        for c in cols:
            arr = np.stack(df[c].values).astype(np.float32)
            accs[c].update_batch(arr)
        n_files += 1
        n_time += len(df)
    out = {}
    for c, acc in accs.items():
        s = acc.finalize()
        s["num_timesteps"] = int(acc.count)
        out[c] = s
    out["num_files"] = n_files
    out["num_timesteps"] = n_time
    return out


def _process_one(bucket_dir: str):
    """Public implementation. Dataset-specific audit notes were removed."""
    name = os.path.basename(bucket_dir.rstrip("/"))
    try:
        result = compute_bucket(bucket_dir)
        out_path = os.path.join(bucket_dir, "meta", OUT_FILENAME)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return name, {"num_timesteps": result["num_timesteps"], "num_files": result["num_files"], "path": out_path}
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}


def discover_buckets(root: str) -> list:
    """Public implementation. Dataset-specific audit notes were removed."""
    return sorted(
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "meta", "info.json"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--bucket", default=None, help="Compute for a single bucket id/name only.")
    parser.add_argument("--workers", type=int, default=1, help="Process pool size across buckets (default 1).")
    args = parser.parse_args()

    if args.bucket:
        buckets = [os.path.join(args.dataset_dir, args.bucket)]
    else:
        buckets = discover_buckets(args.dataset_dir)
    print(f"Computing pooled stats for {len(buckets)} bucket(s) with {args.workers} worker(s)...")

    done = 0
    failed = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process_one, b): b for b in buckets}
            for fut in as_completed(futs):
                name, info = fut.result()
                done += 1
                if "error" in info:
                    failed.append((name, info["error"]))
                    print(f"[{done}/{len(buckets)}] {name}: FAILED — {info['error']}")
                else:
                    print(f"[{done}/{len(buckets)}] {name}: {info['num_timesteps']:,} rows, {info['num_files']} files")
    else:
        for b in buckets:
            name, info = _process_one(b)
            done += 1
            if "error" in info:
                failed.append((name, info["error"]))
                print(f"[{done}/{len(buckets)}] {name}: FAILED — {info['error']}")
            else:
                print(f"[{done}/{len(buckets)}] {name}: {info['num_timesteps']:,} rows, {info['num_files']} files")

    print(f"\nDone: {done - len(failed)}/{len(buckets)} buckets written ({OUT_FILENAME}).")
    if failed:
        print(f"{len(failed)} failed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
