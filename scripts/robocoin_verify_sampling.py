#!/usr/bin/env python3
"""Sampling verification for RoboCOIN dataloader.

Samples one item from every dataset and performs deep numerical validation.
Saves frame images + metadata JSON for manual review.

Usage:
    python scripts/robocoin_verify_sampling.py \
        --dataset_dir /path/to/RoboCOIN \
        --output_dir /path/to/robocoin_verify \
        --multiview true
"""

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openwam.dataloader.robocoin import RoboCOINDataset


def verify_one_dataset(
    ds_dir: str,
    output_dir: str,
    multiview: bool,
    dataset_root: str,
) -> dict:
    """Verify a single dataset, return a summary dict."""
    name = os.path.basename(ds_dir)
    result = {
        "dataset": name,
        "status": "FAIL",
        "robot_type": "",
        "head_camera": "",
        "n_episodes": 0,
        "n_windows": 0,
        "errors": [],
    }
    out_path = os.path.join(output_dir, name)
    os.makedirs(out_path, exist_ok=True)

    try:
        ds = RoboCOINDataset(
            dataset_dir=ds_dir,
            num_frames=33,
            video_stride=4,
            window_stride=1,
            height=384 if multiview else 256,
            width=320,
            split="train",
            val_ratio=0.0,
            seed=42,
            multiview=multiview,
            dataset_root=dataset_root,
        )
        result["robot_type"] = ds.robot_type
        result["head_camera"] = ds._head_camera
        result["n_episodes"] = len(ds._eps_df)
        result["n_windows"] = len(ds)

        if len(ds) == 0:
            result["errors"].append("empty dataset (0 windows)")
            return result

        # --- Sample at index 0 ---
        sample = ds[0]

        # Save frame image
        if sample["video"]:
            sample["video"][0].save(os.path.join(out_path, "frame_0.png"))

        # --- Numerical validations ---
        action = sample["action"].numpy()
        action_mask = sample["action_mask"].numpy()
        video_mask = sample["video_mask"].numpy()
        proprio = sample["proprio"].numpy()
        proprio_mask = sample["proprio_mask"].numpy()

        # Check 1: action shape
        assert action.shape == (32, 20), f"action shape {action.shape} != (32, 20)"

        # Check 2: action values in reasonable range for valid positions
        n_valid = int(action_mask.sum())
        if n_valid > 0:
            valid_action = action[:n_valid]
            pos_l = valid_action[:, 0:3]
            pos_r = valid_action[:, 10:13]
            rot6d_l = valid_action[:, 3:9]
            rot6d_r = valid_action[:, 13:19]
            grip_l = valid_action[:, 9:10]
            grip_r = valid_action[:, 19:20]

            # pos should be in reasonable range (meters)
            if np.abs(pos_l).max() > 10.0:
                result["errors"].append(f"L_pos out of range: max={np.abs(pos_l).max():.3f}")
            if np.abs(pos_r).max() > 10.0:
                result["errors"].append(f"R_pos out of range: max={np.abs(pos_r).max():.3f}")

            # rot6d should be in [-1, 1] (columns of rotation matrix)
            if np.abs(rot6d_l).max() > 1.5:
                result["errors"].append(f"L_rot6d out of range: max={np.abs(rot6d_l).max():.3f}")
            if np.abs(rot6d_r).max() > 1.5:
                result["errors"].append(f"R_rot6d out of range: max={np.abs(rot6d_r).max():.3f}")

            # gripper should be in [0, 1] range (roughly)
            if grip_l.min() < -0.5 or grip_l.max() > 1.5:
                result["errors"].append(f"L_grip range: [{grip_l.min():.3f}, {grip_l.max():.3f}]")
            if grip_r.min() < -0.5 or grip_r.max() > 1.5:
                result["errors"].append(f"R_grip range: [{grip_r.min():.3f}, {grip_r.max():.3f}]")

            # Check rot6d orthogonality: col0 · col1 ≈ 0, |col0| ≈ 1, |col1| ≈ 1
            col0 = valid_action[0, 3:6]
            col1 = valid_action[0, 6:9]
            dot = np.dot(col0, col1)
            norm0 = np.linalg.norm(col0)
            norm1 = np.linalg.norm(col1)
            if abs(dot) > 0.1:
                result["errors"].append(f"L rot6d not orthogonal: dot={dot:.4f}")
            if abs(norm0 - 1.0) > 0.1:
                result["errors"].append(f"L rot6d col0 not unit: norm={norm0:.4f}")
            if abs(norm1 - 1.0) > 0.1:
                result["errors"].append(f"L rot6d col1 not unit: norm={norm1:.4f}")

        # Check 3: action_mask correctness
        assert action_mask.dtype == bool, f"action_mask dtype {action_mask.dtype}"
        # Padded positions should have zero action
        if n_valid < 32:
            padded_action = action[n_valid:]
            if np.any(padded_action != 0):
                result["errors"].append("padded action positions are not zero")

        # Check 4: video_mask
        n_video_real = int(video_mask.sum())
        assert video_mask.shape == (9,), f"video_mask shape {video_mask.shape}"

        # Check 5: proprio
        assert proprio.shape == (1, 20), f"proprio shape {proprio.shape}"
        assert proprio_mask.shape == (1,), f"proprio_mask shape {proprio_mask.shape}"
        assert proprio_mask[0], "proprio_mask should be True"

        # --- Also sample the LAST window (boundary test) ---
        last_idx = len(ds) - 1
        last_sample = ds[last_idx]
        last_action_mask = last_sample["action_mask"].numpy()
        last_video_mask = last_sample["video_mask"].numpy()

        # Save metadata
        metadata = {
            "robot_type": ds.robot_type,
            "head_camera": ds._head_camera,
            "left_wrist_camera": ds._left_wrist_camera,
            "right_wrist_camera": ds._right_wrist_camera,
            "n_episodes": len(ds._eps_df),
            "n_windows": len(ds),
            "prompt": sample["prompt"],
            "action_shape": list(action.shape),
            "proprio_shape": list(proprio.shape),
            "action_first_3_steps": action[:3].tolist() if n_valid >= 3 else action[:n_valid].tolist(),
            "proprio_values": proprio[0].tolist(),
            "action_mask_sum": n_valid,
            "video_mask_sum": n_video_real,
            "video_mask": video_mask.tolist(),
            "action_mask_first10": action_mask[:10].tolist(),
            "last_window_action_mask_sum": int(last_action_mask.sum()),
            "last_window_video_mask": last_video_mask.tolist(),
        }

        with open(os.path.join(out_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if not result["errors"]:
            result["status"] = "OK"

    except Exception as e:
        result["errors"].append(f"Exception: {str(e)}")
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--multiview", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all datasets
    root = args.dataset_dir
    datasets = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "meta", "info.json")))
    print(f"Found {len(datasets)} datasets to verify")

    # Run verification (sequential for clearer logging, parallel for speed)
    results = []
    for i, name in enumerate(datasets):
        ds_dir = os.path.join(root, name)
        print(f"[{i + 1}/{len(datasets)}] Verifying {name}...")
        r = verify_one_dataset(ds_dir, args.output_dir, args.multiview, root)
        results.append(r)
        status = r["status"]
        errors = "; ".join(r["errors"]) if r["errors"] else ""
        print(f"  -> {status} (robot={r['robot_type']}, cam={r['head_camera']}, wins={r['n_windows']}) {errors}")

    # Write summary CSV
    csv_path = os.path.join(args.output_dir, "verify_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "status",
                "robot_type",
                "head_camera",
                "n_episodes",
                "n_windows",
                "errors",
            ],
        )
        writer.writeheader()
        for r in results:
            r_out = dict(r)
            r_out["errors"] = "; ".join(r_out["errors"])
            writer.writerow(r_out)

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {ok} OK, {fail} FAIL out of {len(results)} datasets")
    if fail > 0:
        print("Failed datasets:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['dataset']}: {'; '.join(r['errors'])}")


if __name__ == "__main__":
    main()
