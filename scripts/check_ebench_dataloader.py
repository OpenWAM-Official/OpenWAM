#!/usr/bin/env python3
"""Quick EBench dataloader sanity check.

This script instantiates ``EBenchDataset`` directly, fetches a few samples, and
checks the OpenWAM-facing tensor contracts without constructing the model.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from openwam.dataloader.ebench import EBENCH80_DIM_MASK, EBenchDataset, discover_ebench_buckets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/path/to/data_lake/EBench-Dataset")
    parser.add_argument(
        "--buckets",
        nargs="*",
        default=["simple_pnp/task1", "teleop_tasks/peg_in_hole"],
        help="Relative EBench buckets to check.",
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--normalize-mode", default="z-score", choices=["z-score", "min-max", "null"])
    parser.add_argument("--stats-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    normalize_mode = None if args.normalize_mode == "null" else args.normalize_mode
    stats_path = args.stats_path
    if normalize_mode is not None and stats_path is None:
        stats_path = str(Path(tempfile.mkdtemp(prefix="openwam_ebench_check_")) / "ebench80_stats.npy")
    buckets = discover_ebench_buckets(args.dataset_dir, buckets=args.buckets)
    print(f"Found {len(buckets)} EBench bucket(s): {[str(p.relative_to(args.dataset_dir)) for p in buckets]}")

    ds = EBenchDataset.from_config(
        {
            "type": "ebench",
            "dataset_dir": args.dataset_dir,
            "buckets": [str(bucket.relative_to(Path(args.dataset_dir))) for bucket in buckets],
            "groups": None,
            "num_frames": args.num_frames,
            "video_stride": args.video_stride,
            "height": args.height,
            "width": args.width,
            "normalize_mode": normalize_mode,
            "normalization_stats_path": stats_path,
        },
        split="train",
    )
    print(f"Dataset windows={len(ds)} action_dim={ds.action_dim} stats_path={stats_path}")

    readers = ds.buckets if hasattr(ds, "buckets") else [ds]
    for reader in readers:
        print(f"\nBucket {Path(reader._dataset_id)}: windows={len(reader)} action_dim={reader.action_dim}")
        for i in range(min(args.samples, len(reader))):
            sample = reader[i]
            action = sample["action"].numpy()
            action_mask = sample["action_mask"].numpy()
            proprio = sample["proprio"].numpy()
            proprio_mask = sample["proprio_mask"].numpy()
            assert action.shape == (args.num_frames - 1, 80), action.shape
            assert action_mask.shape == action.shape, action_mask.shape
            assert proprio.shape == (1, 80), proprio.shape
            assert proprio_mask.shape == (1, 80), proprio_mask.shape
            assert action_mask[:1].sum() == int(EBENCH80_DIM_MASK.sum()), action_mask[:1].sum()
            assert np.all(action[:, ~EBENCH80_DIM_MASK] == 0.0)
            assert len(sample["video"]) == len(sample["video_mask"])
            print(
                f"  sample {i}: action={action.shape}, mask_valid={int(action_mask.sum())}, "
                f"proprio={proprio.shape}, video_frames={len(sample['video'])}, prompt={sample['prompt'][:96]!r}"
            )


if __name__ == "__main__":
    main()
