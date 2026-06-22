#!/usr/bin/env python3
"""Public implementation. Dataset-specific audit notes were removed."""







































import argparse
import json
import os

import numpy as np
import pandas as pd

from openwam.dataloader.robocoin import _eef14_to_eef20





RESERVOIR_CAP = 1_000_000


class Accumulator:
    """Public implementation. Dataset-specific audit notes were removed."""

    def __init__(self, dim: int = 20, reservoir_cap: int = RESERVOIR_CAP, seed: int = 0):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

        self.cap = int(reservoir_cap)
        self.rng = np.random.RandomState(seed)
        self._res = np.empty((self.cap, dim), dtype=np.float32)
        self._res_n = 0
        self._res_seen = 0

    def update(self, batch: np.ndarray):
        """Public implementation. Dataset-specific audit notes were removed."""
        for i in range(len(batch)):
            x = batch[i].astype(np.float64)
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
            self.min_val = np.minimum(self.min_val, x)
            self.max_val = np.maximum(self.max_val, x)
        self._reservoir_add(np.asarray(batch, dtype=np.float32))

    def update_batch(self, batch: np.ndarray):
        """Public implementation. Dataset-specific audit notes were removed."""
        n = len(batch)
        if n == 0:
            return
        self._reservoir_add(np.asarray(batch, dtype=np.float32))
        batch = batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_var * n
            self.min_val = batch_min
            self.max_val = batch_max
            self.count = n
        else:
            total = self.count + n
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * n / total
            self.m2 = self.m2 + batch_var * n + delta**2 * self.count * n / total
            self.mean = new_mean
            self.count = total
            self.min_val = np.minimum(self.min_val, batch_min)
            self.max_val = np.maximum(self.max_val, batch_max)

    def _reservoir_add(self, batch: np.ndarray):
        """Public implementation. Dataset-specific audit notes were removed."""
        n = len(batch)
        if n == 0:
            return

        if self._res_n < self.cap:
            take = min(self.cap - self._res_n, n)
            self._res[self._res_n : self._res_n + take] = batch[:take]
            self._res_n += take
            self._res_seen += take
            batch = batch[take:]
            if len(batch) == 0:
                return

        m = len(batch)
        t = self._res_seen + np.arange(m)
        p = self.rng.randint(0, t + 1)
        keep = p < self.cap
        self._res[p[keep]] = batch[keep]
        self._res_seen += m

    def finalize(self):
        std = np.sqrt(self.m2 / max(self.count, 1))
        std = np.where(std < 1e-8, 1.0, std)
        if self._res_n > 0:
            res = self._res[: self._res_n]
            q01 = np.quantile(res, 0.01, axis=0)
            q99 = np.quantile(res, 0.99, axis=0)
        else:
            q01 = self.min_val
            q99 = self.max_val
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": self.min_val.astype(np.float32).tolist(),
            "max": self.max_val.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }


def discover_datasets_by_robot_type(root: str) -> dict:
    """Public implementation. Dataset-specific audit notes were removed."""
    groups = {}
    for name in sorted(os.listdir(root)):
        info_path = os.path.join(root, name, "meta", "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path) as f:
            info = json.load(f)
        rtype = info.get("robot_type", "unknown")
        groups.setdefault(rtype, []).append(os.path.join(root, name))
    return groups


_NEEDED_COLS = [
    "eef_sim_pose_action",
    "gripper_open_scale_action",
    "eef_sim_pose_state",
    "gripper_open_scale_state",
]


def compute_stats_for_robot_type(rtype: str, dataset_dirs: list) -> dict:
    """Public implementation. Dataset-specific audit notes were removed."""






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
                    eef_a = np.stack(df["eef_sim_pose_action"].values).astype(np.float32)
                    grip_a = np.stack(df["gripper_open_scale_action"].values).astype(np.float32)
                    action_20d = _eef14_to_eef20(eef_a, grip_a)

                    eef_s = np.stack(df["eef_sim_pose_state"].values).astype(np.float32)
                    grip_s = np.stack(df["gripper_open_scale_state"].values).astype(np.float32)
                    state_20d = _eef14_to_eef20(eef_s, grip_s)

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
    stats["pool"] = "action+state"
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--robot_type", default=None, help="Compute stats for a single robot type only")
    args = parser.parse_args()

    groups = discover_datasets_by_robot_type(args.dataset_dir)
    print(f"Found {len(groups)} robot types: {sorted(groups.keys())}")

    out_dir = os.path.join(args.dataset_dir, "meta")
    os.makedirs(out_dir, exist_ok=True)

    for rtype in sorted(groups.keys()):
        if args.robot_type and rtype != args.robot_type:
            continue
        ds_list = groups[rtype]
        print(f"\n{'=' * 60}")
        print(f"Computing stats for {rtype} ({len(ds_list)} datasets)...")
        stats = compute_stats_for_robot_type(rtype, ds_list)

        out_path = os.path.join(out_dir, f"stats_{rtype}.json")
        with open(out_path, "w") as f:
            json.dump({"eef": stats}, f, indent=2)

        print(f"  timesteps: {stats['num_timesteps']:,}")
        print(f"  mean[:5]: {stats['mean'][:5]}")
        print(f"  std[:5]:  {stats['std'][:5]}")
        print(f"  min[:5]:  {stats['min'][:5]}")
        print(f"  max[:5]:  {stats['max'][:5]}")
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
