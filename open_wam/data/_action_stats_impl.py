"""
Compute global action normalization stats (mean, std, min, max) from episode HDF5 files.

Uses a memory-efficient single-pass algorithm: accumulates running_sum and
running_sum_sq in float64 precision, then computes mean and std at the end.

Supports two action modes via --action_mode:
  joint (default)  — reads joint_action/vector (T, 14|16)
  eef              — reads endpose/ keys, converts to 20D EEF representation

Mode auto-detection:
  --data_root            → single-task stats
  --dataset_dir          → multi-task stats (aggregate across all training tasks)

Multi-variant support via --variant:
  --variant both   (aggregate across clean_50 + randomized_500)

Usage:
    python -m open_wam.data.action_stats --data_root /path/to/episodes
    python -m open_wam.data.action_stats --dataset_dir /path/to/dataset --robot arx-x5
    python -m open_wam.data.action_stats --dataset_dir /path/to/dataset --robot arx-x5 \\
        --variant both --action_mode eef
"""

import argparse
import glob
import os

import h5py
import numpy as np


def _read_eef_actions_from_file(f) -> np.ndarray:
    """Read endpose keys from an open HDF5 file and assemble 20D EEF actions.

    Layout: [left_xyz(3), left_rot6d(6), left_grip(1),
             right_xyz(3), right_rot6d(6), right_grip(1)]
    Gripper: 1 = closed, 0 = open (inverted from raw HDF5).
    """
    from open_wam.data.transforms.rotation import quat_xyzw_to_rotation_6d

    left_ep = f["endpose/left_endpose"][()].astype(np.float64)  # (T, 7)
    right_ep = f["endpose/right_endpose"][()].astype(np.float64)
    left_grip = 1.0 - f["endpose/left_gripper"][()].astype(np.float64)
    right_grip = 1.0 - f["endpose/right_gripper"][()].astype(np.float64)

    left = np.concatenate(
        [
            left_ep[:, :3],
            quat_xyzw_to_rotation_6d(left_ep[:, 3:]).astype(np.float64),
            left_grip[:, None] if left_grip.ndim == 1 else left_grip,
        ],
        axis=-1,
    )
    right = np.concatenate(
        [
            right_ep[:, :3],
            quat_xyzw_to_rotation_6d(right_ep[:, 3:]).astype(np.float64),
            right_grip[:, None] if right_grip.ndim == 1 else right_grip,
        ],
        axis=-1,
    )
    return np.concatenate([left, right], axis=-1)  # (T, 20)


def compute_action_stats(data_root: str, action_mode: str = "joint") -> dict:
    """Compute global mean and std of actions across all episodes.

    Args:
        data_root: Directory containing episode HDF5 files (RoboTwin format).
        action_mode: ``"joint"`` or ``"eef"``.

    Returns:
        dict with mean, std, min, max, q01, q99.
    """
    return _compute_robotwin_stats(data_root, action_mode=action_mode)


def _compute_robotwin_stats(data_root: str, action_mode: str = "joint") -> dict:
    """Compute extended stats for RoboTwin format.

    Returns dict with mean, std, min, max, q01, q99 (backward compatible).
    """
    pattern = os.path.join(data_root, "episode*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_root}")

    # Auto-detect action dim from first file
    with h5py.File(files[0], "r") as f:
        if action_mode == "eef":
            action_dim = 20
        else:
            action_dim = f["joint_action/vector"].shape[1]
    print(f"  action_mode={action_mode}, action_dim={action_dim} from {os.path.basename(files[0])}")

    all_actions = []
    running_sum = np.zeros(action_dim, dtype=np.float64)
    running_sum_sq = np.zeros(action_dim, dtype=np.float64)
    total_count = 0

    for i, path in enumerate(files):
        try:
            with h5py.File(path, "r") as f:
                if action_mode == "eef":
                    if "endpose/left_endpose" not in f:
                        print(f"  [{i + 1}/{len(files)}] {os.path.basename(path)}: no endpose keys, skipping")
                        continue
                    actions = _read_eef_actions_from_file(f)
                else:
                    if "joint_action/vector" not in f:
                        print(f"  [{i + 1}/{len(files)}] {os.path.basename(path)}: no joint_action/vector, skipping")
                        continue
                    actions = f["joint_action/vector"][:].astype(np.float64)
        except Exception as e:
            print(f"  [{i + 1}/{len(files)}] {os.path.basename(path)}: error {e}, skipping")
            continue

        T = actions.shape[0]
        running_sum += actions.sum(axis=0)
        running_sum_sq += (actions**2).sum(axis=0)
        total_count += T
        all_actions.append(actions)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  [{i + 1}/{len(files)}] processed, total timesteps: {total_count}")

    if total_count == 0:
        raise ValueError("No action data found in any episode")

    mean = running_sum / total_count
    variance = running_sum_sq / total_count - mean**2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-3)

    concatenated = np.concatenate(all_actions, axis=0)

    return {
        "mean": mean,
        "std": std,
        "min": concatenated.min(axis=0),
        "max": concatenated.max(axis=0),
        "q01": np.percentile(concatenated, 1, axis=0),
        "q99": np.percentile(concatenated, 99, axis=0),
    }


def compute_multitask_robotwin_stats(
    dataset_dir: str,
    robot: str,
    variant: str = "clean_50",
    tasks: list = None,
    action_mode: str = "joint",
) -> dict:
    """Compute action stats across multiple RoboTwin tasks.

    Args:
        dataset_dir: Top-level dataset directory (e.g. /path/to/robotwin_2_0/dataset).
        robot: Robot name (e.g. "aloha-agilex").
        variant: ``"clean_50"``, ``"randomized_500"``, or ``"both"``
            (aggregates across clean_50 + randomized_500).
        tasks: List of task names.  If None, uses configured training tasks.
        action_mode: ``"joint"`` or ``"eef"``.

    Returns:
        dict with mean, std, min, max, q01, q99.
    """
    from open_wam.data._robotwin_impl import discover_robotwin_roots

    # Resolve variant(s)
    if variant == "both":
        variant_list = ["clean_50", "randomized_500"]
    else:
        variant_list = [variant]

    # Discover task roots across one or more variants
    task_roots = []
    for v in variant_list:
        task_roots.extend(discover_robotwin_roots(dataset_dir, robot, v, tasks))

    if not task_roots:
        raise FileNotFoundError(f"No task data found in {dataset_dir} for robot={robot}, variant={variant}")

    print(f"Computing stats across {len(task_roots)} task-variant pairs for {robot}, action_mode={action_mode}")

    def _read_actions(f):
        if action_mode == "eef":
            if "endpose/left_endpose" not in f:
                return None
            return _read_eef_actions_from_file(f)
        else:
            if "joint_action/vector" not in f:
                return None
            return f["joint_action/vector"][:].astype(np.float64)

    running_sum = None
    running_sum_sq = None
    total_count = 0

    for task_idx, (task_name, data_root) in enumerate(task_roots):
        pattern = os.path.join(data_root, "episode*.hdf5")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [{task_idx + 1}/{len(task_roots)}] {task_name}: no episodes, skipping")
            continue

        task_count = 0
        for path in files:
            try:
                with h5py.File(path, "r") as f:
                    actions = _read_actions(f)
            except Exception:
                continue
            if actions is None:
                continue

            if running_sum is None:
                action_dim = actions.shape[1]
                running_sum = np.zeros(action_dim, dtype=np.float64)
                running_sum_sq = np.zeros(action_dim, dtype=np.float64)

            running_sum += actions.sum(axis=0)
            running_sum_sq += (actions**2).sum(axis=0)
            total_count += actions.shape[0]
            task_count += actions.shape[0]

        print(f"  [{task_idx + 1}/{len(task_roots)}] {task_name}: {len(files)} episodes, {task_count} timesteps")

    if total_count == 0:
        raise ValueError("No action data found in any task")

    mean = running_sum / total_count
    variance = running_sum_sq / total_count - mean**2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-3)

    # Compute percentile stats — requires collecting all actions (second pass)
    all_actions_list = []
    for _, data_root in task_roots:
        pattern = os.path.join(data_root, "episode*.hdf5")
        for path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(path, "r") as f:
                    actions = _read_actions(f)
                    if actions is not None:
                        all_actions_list.append(actions)
            except Exception:
                continue

    if all_actions_list:
        concatenated = np.concatenate(all_actions_list, axis=0)
        stats_min = concatenated.min(axis=0)
        stats_max = concatenated.max(axis=0)
        q01 = np.percentile(concatenated, 1, axis=0)
        q99 = np.percentile(concatenated, 99, axis=0)
    else:
        stats_min = mean - 3 * std
        stats_max = mean + 3 * std
        q01 = mean - 2.326 * std
        q99 = mean + 2.326 * std

    print(f"\nTotal: {total_count} timesteps across {len(task_roots)} task-variant pairs")
    return {
        "mean": mean,
        "std": std,
        "min": stats_min,
        "max": stats_max,
        "q01": q01,
        "q99": q99,
    }


def parse_tasks_file(tasks_file: str) -> list:
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            tasks.append(line.split()[0])
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Compute action normalization stats from episode HDF5 files.")
    parser.add_argument(
        "--data_root", type=str, default=None, help="Directory containing episode HDF5 files (for single-task mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for action_stats.npy (default: <data_root>/action_stats.npy)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="robotwin",
        choices=["robotwin", "robotwin_multitask"],
        help="HDF5 format: robotwin (default). 'robotwin_multitask' is a backward-compat "
        "alias — use --dataset_dir without --data_root for multi-task mode.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Top-level RoboTwin dataset directory (for multi-task mode)",
    )
    parser.add_argument("--robot", type=str, default="aloha-agilex", help="Robot name")
    parser.add_argument(
        "--variant",
        type=str,
        default="clean_50",
        help='Data variant: "clean_50", "randomized_500", or "both" (merge both)',
    )
    parser.add_argument("--tasks_file", type=str, default=None, help="Optional file listing task names to include")
    parser.add_argument(
        "--action_mode",
        type=str,
        default="joint",
        choices=["joint", "eef"],
        help="Action mode: joint (joint_action/vector) or eef (endpose/ → 20D)",
    )
    args = parser.parse_args()

    tasks = parse_tasks_file(args.tasks_file) if args.tasks_file else None

    # Auto-detect mode: --dataset_dir without --data_root → multi-task
    # --data_root → single-task, --format robotwin_multitask → backward compat
    is_multitask = (args.format == "robotwin_multitask") or (args.dataset_dir and not args.data_root)

    if is_multitask:
        if not args.dataset_dir:
            parser.error("--dataset_dir is required for multi-task mode")
        output_path = args.output or os.path.join(args.dataset_dir, f"{args.robot}_action_stats.npy")
        print(
            f"Computing multi-task action stats: {args.dataset_dir} "
            f"(robot={args.robot}, variant={args.variant}, action_mode={args.action_mode})"
        )
        stats = compute_multitask_robotwin_stats(
            args.dataset_dir,
            args.robot,
            variant=args.variant,
            tasks=tasks,
            action_mode=args.action_mode,
        )
    else:
        if not args.data_root:
            parser.error("--data_root is required for single-task mode")
        output_path = args.output or os.path.join(args.data_root, "action_stats.npy")
        print(f"Computing action stats from: {args.data_root} (action_mode={args.action_mode})")
        stats = compute_action_stats(args.data_root, action_mode=args.action_mode)

    print(f"\nResults ({stats['mean'].shape[0]}D actions, saved to {output_path}):")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")

    np.save(output_path, stats)
    print("Done.")


if __name__ == "__main__":
    main()
