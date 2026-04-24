"""Compute per-task action normalization stats for LeRobot v3 format datasets.

Reads parquet data, applies any action transform (e.g. 40-D → 20-D EEF),
and saves ``eef_stats.json`` to ``{task_root}/meta/``.  On the next training
run, ``LeRobot3Dataset`` auto-loads the file — no ``action_stats_path`` needed.

Supported datasets
------------------
- AgiBotWorld 2026 (``--type agibot``):  40-D raw → 20-D EEF via quat→rot6d.
- Galaxea Open World (``--type galaxea``): stats written from raw action fields;
  ``meta/stats.json`` already covers these, so this script is usually not
  needed for Galaxea unless you want per-dataset aggregated stats.

Output
------
``{task_root}/meta/eef_stats.json`` containing::

    {
        "action": {"mean": [...], "std": [...], "min": [...], "max": [...]},
        "num_timesteps": int,
    }

The key ``"action"`` matches the ``action_fields`` expected by
``_load_action_stats`` for AgiBotWorld (single-column action).
For Galaxea (multi-column), ``action_fields`` keys are written separately.

Usage
-----
    # AgiBot — all tasks (writes eef_stats.json into every task subfolder)
    python -m openwam.dataloader.lerobot_v3_stats_computation \\
        --type agibot \\
        --dataset_dir /path/to/dataset \\
        --quat_convention xyzw

    # AgiBot — single task
    python -m openwam.dataloader.lerobot_v3_stats_computation \\
        --type agibot \\
        --task_root /path/to/dataset \\
        --quat_convention xyzw

    # Galaxea — all tasks
    python -m openwam.dataloader.lerobot_v3_stats_computation \\
        --type galaxea \\
        --dataset_dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Online accumulator — ported from lerobot/src/lerobot/datasets/compute_stats.py
# (Apache-2.0, Copyright 2024 The HuggingFace Inc. team)
#
# Improvements over a naive buffer approach:
#   - mean/std: weighted parallel accumulation (no full buffer)
#   - min/max:  running online update (no full buffer)
#   - quantiles: adaptive histogram approximation (no full buffer)
# ---------------------------------------------------------------------------

_DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
_NUM_QUANTILE_BINS = 5000


class _Accumulator:
    """Online statistics accumulator aligned with lerobot's RunningQuantileStats."""

    def __init__(self, dim: int):
        self.dim = dim
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None

    def update(self, data: np.ndarray) -> None:
        """data: (T, dim) float array."""
        if data.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got shape {data.shape}")
        num_elements = len(data)

        if self._count == 0:
            self._mean = np.mean(data, axis=0)
            self._mean_of_squares = np.mean(data ** 2, axis=0)
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0)
            self._histograms = [np.zeros(_NUM_QUANTILE_BINS) for _ in range(self.dim)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, _NUM_QUANTILE_BINS + 1)
                for i in range(self.dim)
            ]
        else:
            new_min = np.min(data, axis=0)
            new_max = np.max(data, axis=0)
            range_expanded = np.any(new_max > self._max) or np.any(new_min < self._min)
            self._min = np.minimum(self._min, new_min)
            self._max = np.maximum(self._max, new_max)
            if range_expanded:
                self._adjust_histograms()

        self._count += num_elements
        self._mean += (np.mean(data, axis=0) - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (np.mean(data ** 2, axis=0) - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(data)

    def _adjust_histograms(self) -> None:
        for i in range(self.dim):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            padding = max((self._max[i] - self._min[i]) * 1e-10, 1e-10)
            new_edges = np.linspace(self._min[i] - padding, self._max[i] + padding, _NUM_QUANTILE_BINS + 1)
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(_NUM_QUANTILE_BINS)
            for center, cnt in zip(old_centers, old_hist):
                if cnt > 0:
                    idx = max(0, min(int(np.searchsorted(new_edges, center)) - 1, _NUM_QUANTILE_BINS - 1))
                    new_hist[idx] += cnt
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, data: np.ndarray) -> None:
        for i in range(self.dim):
            hist, _ = np.histogram(data[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantile(self, q: float) -> np.ndarray:
        target = q * self._count
        result = []
        for hist, edges in zip(self._histograms, self._bin_edges):
            cumsum = np.cumsum(hist)
            idx = int(np.searchsorted(cumsum, target))
            if idx == 0:
                result.append(float(edges[0]))
            elif idx >= len(cumsum):
                result.append(float(edges[-1]))
            else:
                count_before = cumsum[idx - 1]
                count_in_bin = cumsum[idx] - count_before
                fraction = (target - count_before) / count_in_bin if count_in_bin > 0 else 0.0
                result.append(float(edges[idx] + fraction * (edges[idx + 1] - edges[idx])))
        return np.array(result, dtype=np.float32)

    @property
    def count(self) -> int:
        return self._count

    def finalize(self) -> dict:
        if self._count == 0:
            raise ValueError("No data accumulated")
        var = np.maximum(self._mean_of_squares - self._mean ** 2, 0.0)
        std = np.maximum(np.sqrt(var), 1e-3)
        stats = {
            "mean": self._mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": self._min.astype(np.float32).tolist(),
            "max": self._max.astype(np.float32).tolist(),
            "count": int(self._count),
        }
        for q in _DEFAULT_QUANTILES:
            key = f"q{int(q * 100):02d}"
            stats[key] = self._compute_quantile(q).tolist()
        return stats


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def _load_task_parquets(task_root: str, show_progress: bool = False, desc: str = ""):
    """Yield DataFrames from all data parquet files in a LeRobot v3 task root."""
    import pandas as pd

    data_dir = Path(task_root) / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    iterable = parquet_files
    if show_progress:
        try:
            from tqdm import tqdm
            iterable = tqdm(parquet_files, desc=desc or "  reading parquets", unit="file", leave=False)
        except ImportError:
            pass

    for pf in iterable:
        yield pd.read_parquet(pf)


def _compute_task_stats(
    task_root: str,
    action_fields: List[str],
    field_dims: List[int],
    action_transform: Optional[Callable] = None,
    action_out_dim: Optional[int] = None,
    show_progress: bool = False,
) -> Tuple[dict, int]:
    """Compute action stats for a single LeRobot v3 task root.

    Returns (stats_dict, num_timesteps).  stats_dict keys match action_fields.
    If action_transform is set, a single merged key ``"action"`` is written
    for the transformed output instead.
    """
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info_path = os.path.join(task_root, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    composer = ActionComposer(action_fields, info["features"])

    if action_transform is not None:
        out_dim = action_out_dim or composer.action_dim
        acc = _Accumulator(out_dim)
    else:
        accs = {field: _Accumulator(dim) for field, dim in zip(action_fields, field_dims)}

    total = 0
    task_name = os.path.basename(task_root)
    for df in _load_task_parquets(task_root, show_progress=show_progress, desc=f"  {task_name}"):
        raw = composer.extract(df)  # (T, raw_dim)
        if action_transform is not None:
            data = action_transform(raw)  # (T, out_dim)
            acc.update(data)
            total += len(data)
        else:
            offset = 0
            for field, dim in zip(action_fields, field_dims):
                chunk = raw[:, offset:offset + dim]
                accs[field].update(chunk)
                offset += dim
            total += len(raw)

    if action_transform is not None:
        stats = {"action": acc.finalize()}
    else:
        stats = {field: accs[field].finalize() for field in action_fields}

    return stats, total


def _save_stats(task_root: str, stats: dict, num_timesteps: int) -> str:
    out = {"num_timesteps": num_timesteps}
    out.update(stats)
    out_path = os.path.join(task_root, "meta", "eef_stats.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# AgiBot
# ---------------------------------------------------------------------------

_AGIBOT_ACTION_FIELDS = ["action"]
_AGIBOT_ACTION_DIM_RAW = 40
_AGIBOT_ACTION_DIM_EEF = 20


def compute_agibot_task_stats(task_root: str, quat_convention: str = "xyzw") -> Tuple[dict, int]:
    from openwam.dataloader.agibot import _make_eef_transform

    transform = _make_eef_transform(quat_convention)
    return _compute_task_stats(
        task_root,
        action_fields=_AGIBOT_ACTION_FIELDS,
        field_dims=[_AGIBOT_ACTION_DIM_RAW],
        action_transform=transform,
        action_out_dim=_AGIBOT_ACTION_DIM_EEF,
    )


def _run_tasks(
    task_roots: List[Tuple[str, str]],
    compute_fn: Callable[[str], Tuple[dict, int]],
    desc: str,
) -> None:
    """Shared task loop with tqdm progress bar.

    Args:
        task_roots:  List of (task_name, task_root_path) pairs.
        compute_fn:  Callable(task_root) → (stats_dict, num_timesteps).
        desc:        Label shown in the tqdm bar.
    """
    try:
        from tqdm import tqdm
        task_iter = tqdm(task_roots, desc=desc, unit="task")
        _use_tqdm = True
    except ImportError:
        task_iter = task_roots
        _use_tqdm = False

    for name, root in task_iter:
        if _use_tqdm:
            task_iter.set_postfix_str(name[-50:] if len(name) > 50 else name)
        try:
            stats, n = compute_fn(root)
            _save_stats(root, stats, n)
        except Exception as e:
            msg = f"ERROR {name}: {e}, skipping"
            if _use_tqdm:
                task_iter.write(msg)
            else:
                print(msg)


def run_agibot(dataset_dir: str, quat_convention: str = "xyzw", task_root: Optional[str] = None):
    """Compute and save EEF stats for AgiBot tasks."""
    if task_root:
        task_roots = [(os.path.basename(task_root), task_root)]
    else:
        from openwam.dataloader.agibot import discover_agibot_tasks
        task_roots = discover_agibot_tasks(dataset_dir)

    if not task_roots:
        raise FileNotFoundError(f"No AgiBot tasks found under {dataset_dir}")

    _run_tasks(
        task_roots,
        compute_fn=lambda root: compute_agibot_task_stats(root, quat_convention),
        desc="EEF stats",
    )


# ---------------------------------------------------------------------------
# Galaxea
# ---------------------------------------------------------------------------

_GALAXEA_ACTION_FIELDS = [
    "action.left_arm",
    "action.right_arm",
    "action.left_gripper",
    "action.right_gripper",
]
_GALAXEA_ACTION_FIELDS_EEF = _GALAXEA_ACTION_FIELDS + ["action.torso.velocities"]


def run_galaxea(dataset_dir: str, action_format: str = "eef", task_root: Optional[str] = None):
    """Compute and save action stats for Galaxea tasks.

    Note: ``meta/stats.json`` already contains per-field stats for Galaxea.
    This script writes an aggregated ``eef_stats.json`` so the auto-resolve
    path in ``LeRobot3Dataset`` can pick it up without scanning the original
    stats.json field-by-field.
    """
    from openwam.dataloader.galaxea import discover_galaxea_tasks

    if task_root:
        task_roots = [(os.path.basename(task_root), task_root)]
    else:
        task_roots = discover_galaxea_tasks(dataset_dir)

    if not task_roots:
        raise FileNotFoundError(f"No Galaxea tasks found under {dataset_dir}")

    fields = _GALAXEA_ACTION_FIELDS_EEF if action_format == "eef" else _GALAXEA_ACTION_FIELDS

    import json as _json
    # Read field dims from a representative info.json
    sample_info_path = os.path.join(task_roots[0][1], "meta", "info.json")
    with open(sample_info_path) as f:
        sample_info = _json.load(f)
    field_dims = [sample_info["features"][field]["shape"][0] for field in fields]

    _run_tasks(
        task_roots,
        compute_fn=lambda root: _compute_task_stats(root, fields, field_dims),
        desc="EEF stats",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--type", choices=["agibot", "galaxea"], required=True,
        help="Dataset type"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Top-level dataset directory (multi-task mode)"
    )
    parser.add_argument(
        "--task_root", type=str, default=None,
        help="Single task root directory (single-task mode)"
    )
    parser.add_argument(
        "--quat_convention", type=str, default="xyzw", choices=["wxyz", "xyzw"],
        help="[AgiBot] Quaternion convention in raw parquet data (default: xyzw)"
    )
    parser.add_argument(
        "--action_format", type=str, default="eef", choices=["full", "eef"],
        help="[Galaxea] Action format (default: eef)"
    )
    args = parser.parse_args()

    if args.dataset_dir is None and args.task_root is None:
        parser.error("provide --dataset_dir (multi-task) or --task_root (single-task)")

    if args.type == "agibot":
        run_agibot(args.dataset_dir, args.quat_convention, args.task_root)
    else:
        run_galaxea(args.dataset_dir, args.action_format, args.task_root)


if __name__ == "__main__":
    main()
