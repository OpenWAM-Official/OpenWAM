#!/usr/bin/env python3
"""Public implementation. Dataset-specific audit notes were removed."""









































import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.agibotworld import _DEX_BUCKET_IDS, _bucket_has_base_motion
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator



_COL_WIDTH = {
    "action.ee_base": 18, "observation.state.ee_base": 18,
    "action.gripper": 2, "observation.state.gripper": 2,
    "action.dex": 12, "observation.state.dex": 12,
    "action.robot_velocity": 3, "observation.state.robot_velocity": 3,
}

OUT_FILENAME = "stats_g2a.json"
WORKER_CAP = 200_000
GLOBAL_CAP = 1_000_000


def _columns_for(name: str, moving: bool):
    """Public implementation. Dataset-specific audit notes were removed."""
    is_dex = name in _DEX_BUCKET_IDS
    cols = ["action.ee_base", "observation.state.ee_base"]
    if is_dex:
        cols += ["action.dex", "observation.state.dex"]
    else:
        cols += ["action.gripper", "observation.state.gripper"]
    if moving:
        cols += ["action.robot_velocity", "observation.state.robot_velocity"]
    return cols


def _partial_bucket(bucket_dir: str):
    """Public implementation. Dataset-specific audit notes were removed."""
    name = os.path.basename(bucket_dir.rstrip("/"))
    moving = _bucket_has_base_motion(bucket_dir)
    want = {c: _COL_WIDTH[c] for c in _columns_for(name, moving)}
    accs = {c: Accumulator(dim=d, reservoir_cap=WORKER_CAP) for c, d in want.items()}
    files = sorted(glob.glob(os.path.join(bucket_dir, "data", "chunk-*", "*.parquet")))
    if not files:
        return name, {}
    for fp in files:
        present = set(pq.ParquetFile(fp).schema_arrow.names)
        cols = [c for c in want if c in present]
        if not cols:
            continue
        df = pd.read_parquet(fp, columns=cols)
        if not len(df):
            continue
        for c in cols:
            accs[c].update_batch(np.stack(df[c].values).astype(np.float32))
    out = {}
    for c, acc in accs.items():
        if acc.count == 0:
            continue
        out[c] = {
            "count": int(acc.count),
            "mean": acc.mean.astype(np.float64),
            "m2": acc.m2.astype(np.float64),
            "min": acc.min_val.astype(np.float64),
            "max": acc.max_val.astype(np.float64),
            "res": acc._res[: acc._res_n].astype(np.float32),
        }
    return name, out


def _merge_reservoir(r1, n1, r2, n2, rng):
    """Public implementation. Dataset-specific audit notes were removed."""

    if len(r1) == 0:
        return r2[:GLOBAL_CAP].copy()
    if len(r2) == 0:
        return r1[:GLOBAL_CAP].copy()
    size = min(GLOBAL_CAP, n1 + n2)
    from_g = rng.random(size) < (n1 / (n1 + n2))
    idx1 = rng.integers(0, len(r1), size)
    idx2 = rng.integers(0, len(r2), size)
    out = np.where(from_g[:, None], r1[idx1], r2[idx2])
    return out.astype(np.float32)


def _merge_into(g: dict, partial: dict, rng):
    """Public implementation. Dataset-specific audit notes were removed."""
    for c, p in partial.items():
        if c not in g:
            g[c] = {k: p[k] for k in ("count", "mean", "m2", "min", "max", "res")}
            continue
        s = g[c]
        n1, n2 = s["count"], p["count"]
        n = n1 + n2
        delta = p["mean"] - s["mean"]
        mean = s["mean"] + delta * n2 / n
        m2 = s["m2"] + p["m2"] + delta * delta * n1 * n2 / n
        g[c] = {
            "count": n,
            "mean": mean,
            "m2": m2,
            "min": np.minimum(s["min"], p["min"]),
            "max": np.maximum(s["max"], p["max"]),
            "res": _merge_reservoir(s["res"], n1, p["res"], n2, rng),
        }


def _finalize(g: dict, contrib: dict):
    """Public implementation. Dataset-specific audit notes were removed."""
    out = {}
    for c, s in g.items():
        std = np.sqrt(s["m2"] / max(s["count"], 1))
        std = np.where(std < 1e-8, 1.0, std)
        res = s["res"]
        q01 = np.quantile(res, 0.01, axis=0)
        q99 = np.quantile(res, 0.99, axis=0)
        out[c] = {
            "mean": s["mean"].astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": s["min"].astype(np.float32).tolist(),
            "max": s["max"].astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
            "num_timesteps": int(s["count"]),
            "num_buckets": int(contrib.get(c, 0)),
        }
    return out


def discover_buckets(root: str) -> list:
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "meta", "info.json"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    buckets = discover_buckets(args.dataset_dir)
    print(f"Unified g2a stats over {len(buckets)} buckets with {args.workers} workers...", flush=True)
    rng = np.random.default_rng(args.seed)
    g: dict = {}
    contrib: dict = {}
    contributed, skipped, failed = [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:



        futures = {b: pool.submit(_partial_bucket, b) for b in buckets}
        for b in buckets:
            name = os.path.basename(b.rstrip("/"))
            done += 1
            try:

                _, partial = futures[b].result()
            except Exception as e:
                failed.append(name)
                print(f"[{done}/{len(buckets)}] {name}: FAILED — {type(e).__name__}: {e}", flush=True)
                continue
            if not partial:



                skipped.append(name)
                print(f"[{done}/{len(buckets)}] {name}: SKIPPED — no data rows found", flush=True)
                continue
            contributed.append(name)
            _merge_into(g, partial, rng)
            for c in partial:
                contrib[c] = contrib.get(c, 0) + 1
            if done % 20 == 0 or done == len(buckets):
                print(f"[{done}/{len(buckets)}] merged {name}: cols={sorted(partial)}", flush=True)

    if not contributed:
        raise RuntimeError(f"No bucket contributed any data under {args.dataset_dir}; nothing to write.")

    result = {
        "robot_type": "g2a",
        "num_buckets": len(contributed),
        "num_skipped": len(skipped),
        "num_failed": len(failed),
    }
    result.update(_finalize(g, contrib))

    out_dir = os.path.join(args.dataset_dir, "meta")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUT_FILENAME)
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_path)

    print("\n" + "=" * 60, flush=True)
    print(f"Wrote {out_path}", flush=True)
    print(f"  contributed {len(contributed)} / skipped {len(skipped)} / failed {len(failed)} "
          f"of {len(buckets)} buckets", flush=True)
    if skipped:
        print(f"  SKIPPED (no data): {sorted(skipped)}", flush=True)
    if failed:
        print(f"  FAILED: {sorted(failed)}", flush=True)
    for c in sorted(k for k in result if isinstance(result[k], dict)):
        s = result[c]
        print(f"  {c:34s} n={s['num_timesteps']:>12,}  buckets={s['num_buckets']:>3}  "
              f"q01[0]={s['q01'][0]:+.3f} q99[0]={s['q99'][0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
