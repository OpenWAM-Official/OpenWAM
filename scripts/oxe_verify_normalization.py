"""Generate min-max / z-score / quantile normalization comparison report for OXE.

Reads each OXE dataset's ``meta/eef_stats.json`` (produced by
``oxe_compute_stats.py``), samples a subset of state+action rows, applies
all three normalization modes, and reports:

    * Aggregate utilization: fraction of normalized values landing in
      [-0.9, 0.9] / [-1, 1] under each mode, plus the fraction clipped to
      the boundary under quantile.
    * Per-dim post-normalization std — the metric that actually separates
      the modes. The aggregate utilization numbers hide the min-max
      pathology because they average across well-behaved and collapsed
      dims; the per-dim std table exposes it (e.g. RT-1 action_y collapses
      to std≈0.008 under min-max vs ≈0.43 under quantile, because a single
      max=22.09 outlier squashes 98% of values into a 1.6%-wide band).

Note: z-score is unbounded (no clip), so its [-1,1] coverage is just the
fraction within ±1σ and is not directly comparable to the bounded modes;
read it alongside the per-dim std table.

Outputs:
    * stdout + summary.md: markdown summary (aggregate + per-dim std tables)
    * PNG histograms saved under --out-dir (one per dataset × mode)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.oxe_schema import (
    bcz_state_to_arm10,
    droid_state_to_arm10,
    euler7_action_to_arm10,
    rt1_state_to_arm10,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("oxe_verify_normalization")

# Sample at most this many rows per stream to keep the report fast.
MAX_SAMPLES = 200_000

SAMPLER: Dict[str, Dict] = {
    "BC-Z": {
        "state_cols": ["observation.state"],
        "action_cols": ["action"],
        "state_fn": "bcz_state",
        "action_fn": "euler7",
    },
    "Bridge": {
        "state_cols": ["observation.state"],
        "action_cols": ["action"],
        "state_fn": "bcz_state",
        "action_fn": "euler7",
    },
    "RT-1": {
        "state_cols": ["observation.state"],
        "action_cols": ["action"],
        "state_fn": "rt1_state",
        "action_fn": "euler7",
    },
    "DROID": {
        "state_cols": ["observation.state.cartesian_position", "observation.state.gripper_position"],
        "action_cols": ["action.original"],
        "state_fn": "droid_state",
        "action_fn": "euler7",
    },
}


def _convert_state(rows: Dict[str, np.ndarray], fn: str) -> np.ndarray:
    if fn == "bcz_state":
        return bcz_state_to_arm10(rows["observation.state"])
    if fn == "rt1_state":
        return rt1_state_to_arm10(rows["observation.state"])
    if fn == "droid_state":
        return droid_state_to_arm10(
            rows["observation.state.cartesian_position"],
            rows["observation.state.gripper_position"],
        )
    raise ValueError(fn)


def _convert_action(rows: Dict[str, np.ndarray], fn: str) -> np.ndarray:
    if fn == "euler7":
        return euler7_action_to_arm10(rows[list(rows.keys())[0]])
    raise ValueError(fn)


def _load_shard(path: Path, cols: List[str]) -> Dict[str, np.ndarray]:
    table = pq.read_table(path, memory_map=True, columns=cols)
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        col_data = table.column(c).to_pylist()
        if col_data and not isinstance(col_data[0], (list, np.ndarray)):
            out[c] = np.asarray(col_data, dtype=np.float32).reshape(-1, 1)
        else:
            out[c] = np.asarray(col_data, dtype=np.float32)
    return out


def sample_arm10(dataset_dir: Path, dataset_name: str, max_rows: int) -> np.ndarray:
    spec = SAMPLER[dataset_name]
    paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    rng = np.random.RandomState(42)
    rng.shuffle(paths)
    state_arrs: List[np.ndarray] = []
    action_arrs: List[np.ndarray] = []
    n = 0
    for p in paths:
        state10 = _convert_state(_load_shard(p, spec["state_cols"]), spec["state_fn"])
        action10 = _convert_action(_load_shard(p, spec["action_cols"]), spec["action_fn"])
        state_arrs.append(state10)
        action_arrs.append(action10)
        n += len(state10) + len(action10)
        if n >= max_rows:
            break
    sample = np.concatenate([np.concatenate(state_arrs), np.concatenate(action_arrs)], axis=0)
    if len(sample) > max_rows:
        idx = rng.choice(len(sample), max_rows, replace=False)
        sample = sample[idx]
    return sample.astype(np.float32)


def load_stats(stats_path: Path) -> dict:
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        "min": np.array(raw["min"], dtype=np.float32),
        "max": np.array(raw["max"], dtype=np.float32),
        "mean": np.array(raw["mean"], dtype=np.float32),
        "std": np.array(raw["std"], dtype=np.float32),
        "q01": np.array(raw["q01"], dtype=np.float32),
        "q99": np.array(raw["q99"], dtype=np.float32),
    }


def compute_metrics(normalized: np.ndarray, mode: str) -> dict:
    """Per-mode normalization quality metrics."""
    n_total = normalized.size
    in_unit = (np.abs(normalized) <= 1.0 + 1e-6).sum()
    in_safe = (np.abs(normalized) <= 0.9 + 1e-6).sum()
    # Clipping fraction (only meaningful for quantile mode)
    if mode == "quantile":
        clipped = (np.abs(normalized) >= 1.0 - 1e-6).sum()
        clipped_frac = clipped / n_total
    else:
        # For min-max, by definition values outside [min, max] would map outside [-1, 1].
        # Count those.
        outside = (np.abs(normalized) > 1.0 + 1e-6).sum()
        clipped_frac = outside / n_total
    return {
        "in_unit_frac": float(in_unit / n_total),
        "in_safe_frac": float(in_safe / n_total),
        "clipped_frac": float(clipped_frac),
        "per_dim_min": normalized.min(axis=0).tolist(),
        "per_dim_max": normalized.max(axis=0).tolist(),
        "per_dim_mean": normalized.mean(axis=0).tolist(),
        "per_dim_std": normalized.std(axis=0).tolist(),
    }


def save_histograms(sample: np.ndarray, name: str, mode: str, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping histogram PNG")
        return
    # z-score is unbounded → widen the window so the tails are visible;
    # min-max / quantile live in [-1, 1].
    xlim = (-4.0, 4.0) if mode == "z-score" else (-1.5, 1.5)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    dim_names = ["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grip"]
    for i, ax in enumerate(axes.flatten()):
        ax.hist(sample[:, i], bins=80, color="steelblue", alpha=0.85)
        ax.axvline(-1.0, color="red", linewidth=0.8)
        ax.axvline(1.0, color="red", linewidth=0.8)
        ax.set_title(dim_names[i], fontsize=10)
        ax.set_xlim(*xlim)
    fig.suptitle(f"{name} / {mode} normalization (n={len(sample)})", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{name.lower()}_{mode}.png"
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    logger.info("  saved %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/path/to/OXE")
    parser.add_argument("--out-dir", type=str, default=str(Path.home() / "oxe_samples" / "norm_report"))
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    modes = ["min-max", "z-score", "quantile"]
    dim_names = ["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grip"]
    # results[name][mode] = metrics dict
    results: Dict[str, Dict[str, dict]] = {}
    for name in SAMPLER:
        ds_dir = root / f"{name}-Dataset"
        stats_path = ds_dir / "meta" / "eef_stats.json"
        if not stats_path.exists():
            logger.warning("%s: stats file %s missing — run oxe_compute_stats.py first", name, stats_path)
            continue
        logger.info("==== %s ====", name)
        stats = load_stats(stats_path)
        sample = sample_arm10(ds_dir, name, args.max_samples)
        logger.info("  sampled %d rows; computing metrics", len(sample))

        results[name] = {}
        for mode in modes:
            normed = apply_normalization(sample.copy(), stats, mode)
            results[name][mode] = compute_metrics(normed, mode)
            save_histograms(normed, name, mode, out_dir)

    # ── Build the markdown report once, then print + persist it. ──
    lines: List[str] = []
    lines.append("# OXE normalization comparison (min-max / z-score / quantile)")
    lines.append("")
    lines.append("## Aggregate utilization")
    lines.append("")
    lines.append("`mean_dim_std` = mean over the 10 EEF dims of the post-normalization std")
    lines.append("(higher = more usable signal). `clip%` is only meaningful for quantile")
    lines.append("(fraction clipped to ±1); z-score is unbounded so its in[-1,1]% is just")
    lines.append("the fraction within ±1σ.")
    lines.append("")
    lines.append("| dataset | mode      | mean_dim_std | in[-0.9,0.9]% | in[-1,1]% | clip% |")
    lines.append("|---------|-----------|-------------:|--------------:|----------:|------:|")
    for name in results:
        for mode in modes:
            m = results[name][mode]
            mean_std = float(np.mean(m["per_dim_std"]))
            clip = f"{m['clipped_frac'] * 100:>5.2f}" if mode == "quantile" else "    —"
            lines.append(
                f"| {name:<7} | {mode:<9} | "
                f"{mean_std:>12.4f} | "
                f"{m['in_safe_frac'] * 100:>12.2f}% | "
                f"{m['in_unit_frac'] * 100:>8.2f}% | "
                f"{clip}% |"
            )
    lines.append("")
    lines.append("## Per-dim post-normalization std (the metric that separates the modes)")
    lines.append("")
    lines.append("Higher = more usable signal. Where min-max is much smaller than quantile,")
    lines.append("an outlier has collapsed that dim's dynamic range under min-max.")
    for name in results:
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| dim   | min-max | z-score | quantile | quantile/min-max |")
        lines.append("|-------|--------:|--------:|---------:|-----------------:|")
        std_mm = results[name]["min-max"]["per_dim_std"]
        std_zs = results[name]["z-score"]["per_dim_std"]
        std_q = results[name]["quantile"]["per_dim_std"]
        for i, dn in enumerate(dim_names):
            ratio = std_q[i] / std_mm[i] if std_mm[i] > 1e-9 else float("inf")
            lines.append(f"| {dn:<5} | {std_mm[i]:>7.4f} | {std_zs[i]:>7.4f} | {std_q[i]:>8.4f} | {ratio:>16.1f} |")

    report = "\n".join(lines)
    print()
    print(report)
    print()
    print(f"Histograms saved to: {out_dir}")

    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(report + "\n")
    logger.info("wrote markdown summary to %s", md_path)


if __name__ == "__main__":
    main()
