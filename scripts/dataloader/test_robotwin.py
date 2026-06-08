#!/usr/bin/env python3
"""Manual smoke test for the RoboTwin dataloader.

Reads ``configs/dataloader/robotwin.yaml`` (with optional CLI overrides),
builds the dataset, samples a few items, prints tensor shapes, and dumps
per-frame PNG images plus the prompt into an output directory for visual
inspection.

Examples
--------
    # Use the yaml as-is (uses dataset_dir baked into the config)
    python scripts/dataloader/test_robotwin.py

    # Probe 5 samples from the val split
    python scripts/dataloader/test_robotwin.py --split val --num-samples 5

    # Override any yaml key via OmegaConf dotlist (after --)
    python scripts/dataloader/test_robotwin.py -- num_frames=9 video_stride=4 \
            dataset_dir=/path/to/RoboTwin2.0/dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataloader" / "robotwin.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dataset_probe"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to dataloader yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Dataset split to probe.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to probe (evenly spaced across the dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Where to save per-sample frame PNGs (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit indices to probe; overrides --num-samples.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. 'num_frames=9 video_stride=4'.",
    )
    return parser.parse_args()


def describe(name: str, value) -> str:
    if isinstance(value, torch.Tensor):
        body = f"Tensor shape={tuple(value.shape)} dtype={value.dtype}"
        if value.dtype == torch.bool:
            body += f" true_count={int(value.sum().item())}/{value.numel()}"
        elif value.numel() <= 8:
            body += f" values={value.tolist()}"
        else:
            body += f" min={value.float().min().item():.4f} max={value.float().max().item():.4f}"
        return f"{name}: {body}"
    if isinstance(value, np.ndarray):
        return f"{name}: ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], Image.Image):
            w, h = value[0].size
            return f"{name}: List[PIL.Image] len={len(value)} each_size=({w}x{h}) mode={value[0].mode}"
        return f"{name}: list len={len(value)}"
    if isinstance(value, Image.Image):
        return f"{name}: PIL.Image size={value.size} mode={value.mode}"
    if isinstance(value, str):
        trimmed = value if len(value) < 200 else value[:200] + "..."
        return f"{name}: str[{len(value)}] {trimmed!r}"
    return f"{name}: {type(value).__name__} value={value!r}"


def pick_indices(total: int, requested: int) -> list:
    if total <= 0:
        return []
    n = max(1, min(requested, total))
    if n == 1:
        return [0]
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def save_sample(sample: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sample["video"]
    for fi, frame in enumerate(frames):
        frame.save(out_dir / f"frame_{fi:02d}.png")

    prompt = sample.get("prompt", "")
    (out_dir / "prompt.txt").write_text(str(prompt))

    # Dump key tensors as .npy for later inspection
    np.save(out_dir / "action.npy", sample["action"].numpy())
    np.save(out_dir / "proprio.npy", sample["proprio"].numpy())
    np.save(out_dir / "action_mask.npy", sample["action_mask"].numpy())
    np.save(out_dir / "video_mask.npy", sample["video_mask"].numpy())


def _find_subdataset(dataset, idx):
    """Map a global idx onto the underlying single-task RoboTwinDataset.

    Works for both ``RoboTwinDataset`` (returned as-is) and
    ``MultiTaskRoboTwinDataset`` (binary-search through cumulative lengths).
    """
    subs = getattr(dataset, "_sub_datasets", None)
    if not subs:
        return dataset, idx
    cum = dataset._cumulative_lengths
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if idx < cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    local_idx = idx if lo == 0 else idx - cum[lo - 1]
    return subs[lo], local_idx


def save_raw_per_camera(dataset, idx: int, out_dir: Path) -> None:
    """Dump the un-composed per-camera frames for the same window as ``dataset[idx]``.

    Reaches into ``RoboTwinDataset`` internals (``_window_index``,
    ``_episode_files``, ``_read_camera_frames``) — fine for a debug script.
    Output layout::

        out_dir/raw_per_camera/<camera_name>/frame_00.png ...

    No-op for single-view datasets (the assembled image already shows the
    only camera).
    """
    sub, local_idx = _find_subdataset(dataset, idx)
    if not getattr(sub, "multiview", False):
        return

    if getattr(sub, "_val_samples", None) is not None:
        ep_idx, start = sub._val_samples[local_idx]
    else:
        ep_idx, start = sub._window_index[local_idx]
    path = sub._episode_files[ep_idx]
    ep_len = sub._episode_lengths[ep_idx]
    raw_end = min(start + sub._raw_window_len, ep_len)

    raw_root = out_dir / "raw_per_camera"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "_meta.txt").write_text(
        f"episode_path={path}\nep_idx={ep_idx} start={start} raw_end={raw_end} "
        f"ep_len={ep_len}\nvideo_sample_indices={sub._video_sample_indices}\n"
        f"cameras={list(sub.cameras)}\ncamera_layout={sub.camera_layout}\n"
    )

    with h5py.File(path, "r") as f:
        for cam in sub.cameras:
            cam_dir = raw_root / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            try:
                frames = sub._read_camera_frames(f, cam, start, raw_end)
            except KeyError:
                (cam_dir / "MISSING.txt").write_text(f"camera '{cam}' not present in {path}\n")
                continue
            sampled = [frames[i] for i in sub._video_sample_indices if i < len(frames)]
            for fi, frame in enumerate(sampled):
                frame.save(cam_dir / f"frame_{fi:02d}.png")


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg.merge_with(OmegaConf.from_dotlist(args.overrides))

    print("=" * 72)
    print(f"Config: {args.config}")
    print(f"Split:  {args.split}")
    print("-" * 72)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 72)

    dataset = build_dataset(cfg, split=args.split)
    print()
    print(f"Built {type(dataset).__name__}")
    print(f"  len(dataset)   = {len(dataset)}")
    print(f"  action_dim     = {dataset.action_dim}")

    # --- Normalization summary -------------------------------------------------
    norm_mode = OmegaConf.select(cfg, "normalize_mode", default=None)
    action_mode = OmegaConf.select(cfg, "action_mode", default="joint")
    print(f"  normalize_mode = {norm_mode!r}")
    print(f"  action_mode    = {action_mode!r}")
    stats = getattr(dataset, "normalization_stats", None)
    if stats is None:
        print("  normalization_stats   = None (normalization disabled or stats not loaded)")
        expected_bounds = None
    else:
        stats_mean = np.asarray(stats["mean"])
        stats_std = np.asarray(stats["std"])
        stats_min = np.asarray(stats["min"])
        stats_max = np.asarray(stats["max"])
        print(f"  normalization_stats[{action_mode}]:")
        print(f"    mean[{stats_mean.shape}] range=[{stats_mean.min():.4f}, {stats_mean.max():.4f}]")
        print(f"    std[{stats_std.shape}]  range=[{stats_std.min():.4f}, {stats_std.max():.4f}]")
        print(f"    min[{stats_min.shape}]  range=[{stats_min.min():.4f}, {stats_min.max():.4f}]")
        print(f"    max[{stats_max.shape}]  range=[{stats_max.min():.4f}, {stats_max.max():.4f}]")
        if norm_mode == "min-max":
            expected_bounds = (-1.0 - 1e-4, 1.0 + 1e-4)
        elif norm_mode == "z-score":
            expected_bounds = None  # unbounded; report mean/std instead
        else:
            expected_bounds = None

    if len(dataset) == 0:
        print("Dataset is empty. Nothing to sample.")
        return 1

    if args.indices is not None:
        indices = [i for i in args.indices if 0 <= i < len(dataset)]
        if not indices:
            print("No valid indices in --indices; falling back to evenly-spaced.")
            indices = pick_indices(len(dataset), args.num_samples)
    else:
        indices = pick_indices(len(dataset), args.num_samples)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nSampling {len(indices)} items → {out_root}")

    action_ranges = []  # (min, max, mean, std) collected across samples for summary
    oob_report = []  # out-of-bounds violations when expected_bounds is set

    for pos, idx in enumerate(indices):
        sample = dataset[idx]
        print("\n" + "-" * 72)
        print(f"sample {pos + 1}/{len(indices)}   idx={idx}")
        print("-" * 72)
        for key in (
            "video",
            "action",
            "action_mask",
            "video_mask",
            "proprio",
            "proprio_mask",
            "prompt",
            "episode_index",
            "episode_path",
            "start_frame",
            "end_frame",
            "episode_length",
            "task_name",
            "active_arm",
        ):
            if key in sample:
                print("  " + describe(key, sample[key]))

        # --- Per-sample normalization range -----------------------------------
        act = sample["action"].numpy()
        prop = sample["proprio"].numpy()
        a_min, a_max = float(act.min()), float(act.max())
        a_mean, a_std = float(act.mean()), float(act.std())
        p_min, p_max = float(prop.min()), float(prop.max())
        print(
            f"  normalize check: action min/max=[{a_min:+.4f}, {a_max:+.4f}] "
            f"mean={a_mean:+.4f} std={a_std:.4f} | "
            f"proprio min/max=[{p_min:+.4f}, {p_max:+.4f}]"
        )
        action_ranges.append((a_min, a_max, a_mean, a_std))
        if expected_bounds is not None:
            lo, hi = expected_bounds
            if a_min < lo or a_max > hi or p_min < lo or p_max > hi:
                oob_report.append((idx, a_min, a_max, p_min, p_max))

        sample_dir = out_root / f"sample_{pos:02d}_idx{idx:06d}"
        save_sample(sample, sample_dir)
        save_raw_per_camera(dataset, idx, sample_dir)
        print(f"  saved → {sample_dir}")

    # --- Aggregate normalization report ---------------------------------------
    print("\n" + "=" * 72)
    print("Normalization summary across sampled items")
    print("-" * 72)
    if action_ranges:
        all_min = min(r[0] for r in action_ranges)
        all_max = max(r[1] for r in action_ranges)
        mean_of_means = float(np.mean([r[2] for r in action_ranges]))
        mean_of_stds = float(np.mean([r[3] for r in action_ranges]))
        print(f"  action global min/max : [{all_min:+.4f}, {all_max:+.4f}]")
        print(f"  action mean-of-means  : {mean_of_means:+.4f}")
        print(f"  action mean-of-stds   : {mean_of_stds:.4f}")
        if norm_mode == "min-max":
            verdict = "OK" if not oob_report else f"FAIL ({len(oob_report)} samples out of [-1, 1])"
            print(f"  min-max bound check   : {verdict}")
            for idx_, amn, amx, pmn, pmx in oob_report[:5]:
                print(f"    - idx={idx_} action=[{amn:+.4f}, {amx:+.4f}] proprio=[{pmn:+.4f}, {pmx:+.4f}]")
        elif norm_mode == "z-score":
            # Expect mean≈0, std≈1 when the sampled window covers a representative slice.
            print(f"  z-score expectation   : mean≈0, std≈1  (observed {mean_of_means:+.3f}, {mean_of_stds:.3f})")

    # --- Denormalize roundtrip ------------------------------------------------
    # MultiTaskRoboTwinDataset delegates denormalize_action to its first sub-dataset,
    # so we reach through to the active ActionNormalizer to run normalize(denorm(x))
    # and confirm it recovers the original normalized tensor.
    denorm_fn = getattr(dataset, "denormalize_action", None)
    normalizer = getattr(dataset, "_action_normalizer", None)
    if normalizer is None:
        subs = getattr(dataset, "_sub_datasets", None)
        if subs:
            normalizer = getattr(subs[0], "_action_normalizer", None)
    if denorm_fn is not None and normalizer is not None:
        last_sample = dataset[indices[-1]]
        normalized = last_sample["action"].numpy()
        recovered = np.asarray(denorm_fn(normalized))
        reback = normalizer.normalize(recovered)
        err = float(np.max(np.abs(reback - normalized)))
        status = "OK" if err < 1e-4 else f"HIGH ERROR ({err:.2e})"
        print(f"  denorm→norm roundtrip : max|err|={err:.2e} → {status}")
        print(f"    normalized range   : [{normalized.min():+.4f}, {normalized.max():+.4f}]")
        print(f"    denormalized range : [{recovered.min():+.4f}, {recovered.max():+.4f}]")
    else:
        print("  denorm is pass-through (no active normalizer on dataset or sub-datasets)")
    print("=" * 72)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
