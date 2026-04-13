#!/usr/bin/env python3
"""Smoke-test the RoboTwin dataloader via the dataset registry.

Loads a small number of samples from the configured dataset and prints
shapes, value ranges, and gripper statistics. Optionally saves sample
frames to disk for visual inspection.

Uses ``build_dataset`` from the registry so the construction path is
identical to what the training script uses.

Usage:
    # Default: multi-task, eef, both variants, multiview
    python scripts/dataloader/robotwin.py

    # Save sample frames to disk
    python scripts/dataloader/robotwin.py --save-frames

    # Single-task
    python scripts/dataloader/robotwin.py dataloader.task_name=adjust_bottle

    # Joint mode instead of eef
    python scripts/dataloader/robotwin.py dataloader.action_mode=joint

    # Single variant only
    python scripts/dataloader/robotwin.py dataloader.variant=clean_50

    # Single-view mode
    python scripts/dataloader/robotwin.py dataloader.multiview=false

    # Check more samples
    python scripts/dataloader/robotwin.py --num-samples 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Parse our own flags BEFORE Hydra sees sys.argv
# ---------------------------------------------------------------------------


def _split_argv():
    """Separate our custom flags from Hydra overrides."""
    ours, hydra = [], []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--save-frames",):
            ours.append(arg)
        elif arg in ("--num-samples",) and i < len(sys.argv) - 1:
            ours.extend([arg, sys.argv[i + 1]])
            skip_next = True
        elif arg.startswith("--num-samples="):
            ours.append(arg)
        else:
            hydra.append(arg)
    return ours, hydra


_our_argv, _hydra_argv = _split_argv()

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--save-frames", action="store_true", default=False)
_parser.add_argument("--num-samples", type=int, default=3)
_flags = _parser.parse_args(_our_argv)

# Restore sys.argv for Hydra (only script name + hydra overrides)
sys.argv = [sys.argv[0]] + _hydra_argv

# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

# EEF 20D layout: [left_xyz(3), left_rot6d(6), left_grip(1),
#                  right_xyz(3), right_rot6d(6), right_grip(1)]
_EEF_LABELS = (
    ["left_xyz"] * 3 + ["left_rot6d"] * 6 + ["left_grip"] + ["right_xyz"] * 3 + ["right_rot6d"] * 6 + ["right_grip"]
)

# Joint 14D layout: [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]
_JOINT_LABELS = ["left_arm"] * 6 + ["left_grip"] + ["right_arm"] * 6 + ["right_grip"]


def _describe_action_ranges(actions: torch.Tensor, action_mode: str):
    """Print per-channel value ranges for the action tensor."""
    a = actions.numpy()
    labels = _EEF_LABELS if action_mode == "eef" else _JOINT_LABELS

    # Group consecutive channels with the same label
    groups = []
    i = 0
    while i < min(a.shape[-1], len(labels)):
        label = labels[i]
        j = i + 1
        while j < min(a.shape[-1], len(labels)) and labels[j] == label:
            j += 1
        groups.append((label, i, j, a[:, i:j]))
        i = j

    print("  Action ranges:")
    for label, start, end, cols in groups:
        idx_str = f"[{start}]" if end - start == 1 else f"[{start}:{end}]"
        uniq = np.unique(cols)
        if len(uniq) <= 4:
            vals_str = "{" + ", ".join(f"{v:.1f}" for v in sorted(uniq)) + "}"
            print(f"    {label:15s} {idx_str:8s}  values={vals_str}")
        else:
            print(f"    {label:15s} {idx_str:8s}  min={cols.min():+.4f}  max={cols.max():+.4f}")


def _save_frame(image, path: str):
    """Save a PIL Image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print(f"  Saved: {path}")


def _print_sample(sample, idx: int, action_mode: str, save_frames: bool, out_dir: str):
    """Print detailed info for a single sample."""
    video = sample["video"]
    action = sample["action"]
    action_mask = sample["action_mask"]
    prompt = sample["prompt"]

    n_frames = len(video)
    frame0 = video[0]
    w, h = frame0.size

    print(f"  video:        {n_frames} x PIL({w}x{h} {frame0.mode})")
    print(f"  action:       {action.shape}  dtype={action.dtype}")
    print(f"  action_mask:  {action_mask.shape}  ({action_mask.sum().item()} valid)")
    print(f'  prompt:       "{prompt[:80]}{"..." if len(prompt) > 80 else ""}"')

    # Metadata
    for key in ["task_name", "episode_index", "start_frame", "end_frame", "active_arm"]:
        if key in sample:
            print(f"  {key:15s} {sample[key]}")

    # Action ranges
    _describe_action_ranges(action, action_mode)

    # VACE reference
    vace_ref = sample.get("vace_reference_image")
    if vace_ref:
        ref0 = vace_ref[0]
        print(f"  vace_ref:     {len(vace_ref)} x PIL({ref0.size[0]}x{ref0.size[1]})")

    # Save frames
    if save_frames:
        _save_frame(frame0, os.path.join(out_dir, f"sample_{idx}_frame_first.png"))
        _save_frame(video[n_frames // 2], os.path.join(out_dir, f"sample_{idx}_frame_mid.png"))
        _save_frame(video[-1], os.path.join(out_dir, f"sample_{idx}_frame_last.png"))


def _check_dataloader_batch(dataset):
    """Verify DataLoader batching works correctly."""
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        result = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)
            else:
                result[key] = vals
        return result

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))

    print("--- DataLoader batch check (batch_size=2) ---")
    print(f"  batch['action']:       {batch['action'].shape}")
    print(f"  batch['action_mask']:  {batch['action_mask'].shape}")
    print(f"  batch['video']:        {len(batch['video'])} items, each {len(batch['video'][0])} frames")
    print(f"  batch['prompt']:       {batch['prompt']}")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig):
    d = cfg.dataloader
    action_mode = d.get("action_mode", "eef")
    task_name = d.get("task_name", None)

    print("=" * 60)
    print("RoboTwin Dataloader Smoke Test (via Registry)")
    print("=" * 60)
    print()
    print("Config:")
    print(f"  dataset_dir:  {d.dataset_dir}")
    print(f"  mode:         {'single-task (' + task_name + ')' if task_name else 'multi-task (all tasks)'}")
    print(f"  robot:        {d.robot}")
    print(f"  variant:      {d.variant}")
    print(f"  action_mode:  {action_mode} ({'20D' if action_mode == 'eef' else '14D'})")
    print(f"  num_frames:   {d.num_frames}")
    print(f"  resolution:   {d.height}x{d.width}")
    print(f"  multiview:    {d.multiview}")
    print()

    # Build dataset via registry
    from openwam.dataloader.registry import build_dataset

    print("Building dataset via registry...")
    t0 = time.time()
    dataset = build_dataset(d, split="train")
    build_time = time.time() - t0

    print(f"\nDataset ready ({build_time:.1f}s)")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Action dim:    {dataset.action_dim}")
    print(f"  Action stats:  {'loaded' if dataset.action_stats else 'None'}")
    print()

    # Sample and inspect
    num_samples = min(_flags.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    out_dir = os.path.join("outputs", "dataloader", "robotwin", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    for i, idx in enumerate(indices):
        print(f"--- Sample {i} (index={idx}) ---")
        t0 = time.time()
        sample = dataset[idx]
        print(f"  load_time:    {time.time() - t0:.3f}s")
        _print_sample(sample, i, action_mode, _flags.save_frames, out_dir)
        print()

    # Batch check
    _check_dataloader_batch(dataset)
    print()

    print("=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
