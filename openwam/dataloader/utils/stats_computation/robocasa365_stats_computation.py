"""Normalization statistics for canonical compact RoboCasa365 state19/action15 v3 data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from openwam.dataloader.robocasa365 import (
    ACTION_DIM,
    ACTION_STATS_KEY,
    STATE_DIM,
    STATE_STATS_KEY,
    _task_from_source_prefix,
)
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import atomic_save_stats_npy

# Inlined from the retired RoboCasa365 compact-v3 converter so the stats
# tool stays self-contained.
REPRESENTATION = "robocasa365_compact_native_delta_eef_v1"


class StatsAccumulator:
    """Exact streaming moments/range plus a bounded deterministic quantile sample."""

    def __init__(self, dim: int, *, seed: int, sample_limit: int = 250_000):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.minimum = np.full(dim, np.inf, np.float64)
        self.maximum = np.full(dim, -np.inf, np.float64)
        self._sample = np.empty((0, dim), np.float32)
        self._priority = np.empty((0,), np.float64)
        self._sample_limit = sample_limit
        self._rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, np.float32)
        if values.ndim != 2 or values.shape[1] != self.dim or values.shape[0] == 0:
            raise ValueError(f"stats expected nonempty (N,{self.dim}), got {values.shape}")
        batch = values.astype(np.float64)
        n = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + n
            self.mean += delta * n / total
            self.m2 += batch_m2 + np.square(delta) * self.count * n / total
        self.count += n
        self.minimum = np.minimum(self.minimum, batch.min(axis=0))
        self.maximum = np.maximum(self.maximum, batch.max(axis=0))

        take = min(n, 10_000)
        indices = self._rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        sample = values[indices]
        priority = self._rng.random(take)
        self._sample = np.concatenate([self._sample, sample], axis=0)
        self._priority = np.concatenate([self._priority, priority])
        if self._sample.shape[0] > self._sample_limit:
            keep = np.argpartition(self._priority, -self._sample_limit)[-self._sample_limit :]
            self._sample = self._sample[keep]
            self._priority = self._priority[keep]

    def merge(self, other: "StatsAccumulator") -> None:
        if other.dim != self.dim or other.count == 0:
            if other.dim != self.dim:
                raise ValueError("cannot merge stats with different dimensions")
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            self.minimum = other.minimum.copy()
            self.maximum = other.maximum.copy()
        else:
            total = self.count + other.count
            delta = other.mean - self.mean
            self.m2 += other.m2 + np.square(delta) * self.count * other.count / total
            self.mean += delta * other.count / total
            self.count = total
            self.minimum = np.minimum(self.minimum, other.minimum)
            self.maximum = np.maximum(self.maximum, other.maximum)
        self._sample = np.concatenate([self._sample, other._sample], axis=0)
        self._priority = np.concatenate([self._priority, other._priority])
        if self._sample.shape[0] > self._sample_limit:
            keep = np.argpartition(self._priority, -self._sample_limit)[-self._sample_limit :]
            self._sample = self._sample[keep]
            self._priority = self._priority[keep]

    def finish(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot finish empty stats")
        std = np.sqrt(self.m2 / self.count)
        q01, q99 = np.quantile(self._sample.astype(np.float64), [0.01, 0.99], axis=0)
        return {
            "mean": self.mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
        }


def _pin_dims(stats: dict[str, np.ndarray], dims: Iterable[int]) -> None:
    identity = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}
    for key, value in identity.items():
        stats[key][list(dims)] = value


def _stats_payload(
    action_acc: StatsAccumulator,
    state_acc: StatsAccumulator,
) -> dict:
    action = action_acc.finish()
    state = state_acc.finish()

    # Rot6D is a geometric representation rather than six independent scalar
    # quantities.  Preserve it exactly under every supported normalization mode.
    _pin_dims(action, range(3, 9))
    _pin_dims(state, (*range(3, 9), *range(13, 19)))
    action.update(
        {
            "representation": REPRESENTATION,
            "gripper_convention": "minus1_closed_plus1_open",
            "native_delta_rotation_scale_applied": False,
            "normalization_scope": {
                "native_delta_xyz_gripper": "action_only",
                "native_delta_rot6d": "identity",
                "base_vx_vy_vyaw_torso_mode": "action_only",
            },
            "rot6d_identity_dims": list(range(3, 9)),
        }
    )
    state.update(
        {
            "representation": REPRESENTATION,
            "normalization_scope": {
                "eef_xyz_gripper": "state_only",
                "eef_rot6d": "identity",
                "base_xyz": "state_only",
                "base_rot6d": "identity",
            },
            "rot6d_identity_dims": [*range(3, 9), *range(13, 19)],
        }
    )
    return {
        ACTION_STATS_KEY: action,
        STATE_STATS_KEY: state,
        "num_timesteps": int(action_acc.count),
        "quantiles": f"deterministic priority sample <= {action_acc._sample_limit} rows",
    }


def _iter_arrays(data_root: str, task_name: str | None = None):
    root = Path(data_root)
    episodes = load_episodes_parquet(root)
    episodes["_offset"] = compute_file_local_offsets(episodes, "data/chunk_index", "data/file_index")
    if task_name is not None:
        episodes = episodes[episodes["source_prefix"].map(_task_from_source_prefix) == task_name].reset_index(drop=True)
    if episodes.empty:
        raise FileNotFoundError(f"No episodes for task {task_name!r} under {root}")
    import json

    with (root / "meta" / "info.json").open(encoding="utf-8") as handle:
        template = json.load(handle)["data_path"]
    for (chunk, file_index), group in episodes.groupby(["data/chunk_index", "data/file_index"], sort=False):
        path = root / template.format(chunk_index=int(chunk), file_index=int(file_index))
        frame = pd.read_parquet(path, columns=["observation.state", "action"])
        states, actions = frame["observation.state"].values, frame["action"].values
        # A whole-repository scan covers every row exactly once.  Yield the
        # shard in one batch instead of slicing it into thousands of episodes.
        if task_name is None:
            expected = int(group["length"].sum())
            if expected != len(frame):
                raise ValueError(f"episode lengths do not cover {path}: {expected} != {len(frame)}")
            yield np.stack(states).astype(np.float32), np.stack(actions).astype(np.float32)
            continue
        for _, row in group.iterrows():
            offset, length = int(row["_offset"]), int(row["length"])
            yield (
                np.stack(states[offset : offset + length]).astype(np.float32),
                np.stack(actions[offset : offset + length]).astype(np.float32),
            )


def _compute(items, label: str) -> dict:
    action_acc = StatsAccumulator(ACTION_DIM, seed=71)
    state_acc = StatsAccumulator(STATE_DIM, seed=73)
    total = 0
    for index, (state, action) in enumerate(items, 1):
        state_acc.update(state)
        action_acc.update(action)
        total += state.shape[0]
        if index % 10 == 0:
            print(f"  [{label}] {index} batches, {total:,} rows", flush=True)
    return _stats_payload(action_acc, state_acc)


def compute_normalization_stats(data_root: str, task_name: str | None = None, **_unused) -> dict:
    return _compute(_iter_arrays(data_root, task_name), "stats")


def compute_multitask_stats(roots: list, **_unused) -> dict:
    def items():
        for task_name, repo in roots:
            yield from _iter_arrays(repo, task_name)

    return _compute(items(), f"multitask/{len(roots)}")


def build_and_save_robocasa365_stats(data_roots, output) -> str:
    """Pooled stats over one or more compact repos, written atomically."""
    roots = [data_roots] if isinstance(data_roots, (str, Path)) else list(data_roots)
    payload = compute_multitask_stats([(None, str(root)) for root in roots])
    os.makedirs(os.path.dirname(str(output)) or ".", exist_ok=True)
    atomic_save_stats_npy(str(output), payload)
    return str(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", nargs="+", help="one or more compact RoboCasa365 v3 repositories")
    parser.add_argument("--task")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    if args.task is not None and len(args.data_root) != 1:
        parser.error("--task can only be used with one data_root")
    if args.task is not None:
        # Single-task debugging aid: no canonical location for task-level stats.
        if args.output is None:
            parser.error("--output is required with --task (single-task stats have no canonical location)")
        output = args.output
        payload = compute_normalization_stats(args.data_root[0], args.task)
    else:
        output = args.output or os.path.join(args.data_root[0], "meta", "robocasa365_normalization_stats.npy")
        payload = compute_multitask_stats([(None, root) for root in args.data_root])
    os.makedirs(os.path.dirname(str(output)) or ".", exist_ok=True)
    if ACTION_STATS_KEY not in payload or STATE_STATS_KEY not in payload:
        raise AssertionError("internal compact stats schema error")
    atomic_save_stats_npy(output, payload)
    print(f"Saved stats -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
