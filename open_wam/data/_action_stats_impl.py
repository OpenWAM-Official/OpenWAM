"""
Compute global action normalization stats (mean, std) from episode HDF5 files.

Uses a memory-efficient single-pass algorithm: accumulates running_sum and
running_sum_sq in float64 precision, then computes mean and std at the end.

Supports two modes via --format:
  robotwin           — joint_action/vector (T, 14|16) -> 14/16D joint actions
  robotwin_multitask — aggregate stats across all training tasks

Usage:
    python compute_action_stats.py --data_root /path/to/episodes
    python compute_action_stats.py --data_root /path/to/episodes --output action_stats.npy
    python compute_action_stats.py --format robotwin_multitask --dataset_dir /path/to/dataset --robot arx-x5
"""

import argparse
import glob
import os

import h5py
import numpy as np


def compute_action_stats(data_root: str) -> dict:
    """Compute global mean and std of actions across all episodes.

    Args:
        data_root: Directory containing episode HDF5 files (RoboTwin format).

    Returns:
        dict with "mean" (D,) float64 and "std" (D,) float64.
    """
    return _compute_robotwin_stats(data_root)


def _compute_robotwin_stats(data_root: str) -> dict:
    """Compute extended stats for RoboTwin format: joint_action/vector (T, 14|16).

    Returns dict with mean, std, min, max, q01, q99 (backward compatible).
    """
    # RoboTwin uses episode0.hdf5 (no underscore)
    pattern = os.path.join(data_root, "episode*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_root}")

    # Auto-detect action dim from first file
    with h5py.File(files[0], "r") as f:
        action_dim = f["joint_action/vector"].shape[1]
    print(f"  Auto-detected action_dim={action_dim} from {os.path.basename(files[0])}")

    all_actions = []
    running_sum = np.zeros(action_dim, dtype=np.float64)
    running_sum_sq = np.zeros(action_dim, dtype=np.float64)
    total_count = 0

    for i, path in enumerate(files):
        try:
            with h5py.File(path, "r") as f:
                if "joint_action/vector" not in f:
                    print(f"  [{i+1}/{len(files)}] {os.path.basename(path)}: "
                          f"no joint_action/vector, skipping")
                    continue
                actions = f["joint_action/vector"][:].astype(np.float64)  # (T, D)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {os.path.basename(path)}: error {e}, skipping")
            continue

        T = actions.shape[0]
        running_sum += actions.sum(axis=0)
        running_sum_sq += (actions ** 2).sum(axis=0)
        total_count += T
        all_actions.append(actions)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] processed, total timesteps: {total_count}")

    if total_count == 0:
        raise ValueError("No action data found in any episode")

    mean = running_sum / total_count
    variance = running_sum_sq / total_count - mean ** 2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-3)

    # Compute percentile stats for q99 normalization
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
) -> dict:
    """Compute action stats across multiple RoboTwin tasks.

    Args:
        dataset_dir: Top-level dataset directory (e.g. /path/to/robotwin_2_0/dataset).
        robot: Robot name (e.g. "aloha-agilex").
        variant: "clean_50" or "randomized_500".
        tasks: List of task names.  If None, uses configured training tasks.

    Returns:
        dict with "mean" (D,) float64 and "std" (D,) float64.
    """
    from open_wam.data._robotwin_impl import discover_robotwin_roots, ROBOTWIN_TRAIN_TASKS

    task_roots = discover_robotwin_roots(dataset_dir, robot, variant, tasks)
    if not task_roots:
        raise FileNotFoundError(
            f"No task data found in {dataset_dir} for robot={robot}, variant={variant}"
        )

    print(f"Computing stats across {len(task_roots)} tasks for {robot}/{variant}")

    running_sum = None
    running_sum_sq = None
    total_count = 0

    for task_idx, (task_name, data_root) in enumerate(task_roots):
        pattern = os.path.join(data_root, "episode*.hdf5")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [{task_idx+1}/{len(task_roots)}] {task_name}: no episodes, skipping")
            continue

        task_count = 0
        for path in files:
            try:
                with h5py.File(path, "r") as f:
                    if "joint_action/vector" not in f:
                        continue
                    actions = f["joint_action/vector"][:].astype(np.float64)
            except Exception:
                continue

            if running_sum is None:
                action_dim = actions.shape[1]
                running_sum = np.zeros(action_dim, dtype=np.float64)
                running_sum_sq = np.zeros(action_dim, dtype=np.float64)

            running_sum += actions.sum(axis=0)
            running_sum_sq += (actions ** 2).sum(axis=0)
            total_count += actions.shape[0]
            task_count += actions.shape[0]

        print(f"  [{task_idx+1}/{len(task_roots)}] {task_name}: "
              f"{len(files)} episodes, {task_count} timesteps")

    if total_count == 0:
        raise ValueError("No action data found in any task")

    mean = running_sum / total_count
    variance = running_sum_sq / total_count - mean ** 2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-3)

    # Compute percentile stats — requires collecting all actions
    # For very large datasets this may use significant memory;
    # consider streaming percentile estimation for production use.
    all_actions_list = []
    for _, data_root in task_roots:
        pattern = os.path.join(data_root, "episode*.hdf5")
        for path in sorted(glob.glob(pattern)):
            try:
                with h5py.File(path, "r") as f:
                    if "joint_action/vector" not in f:
                        continue
                    all_actions_list.append(f["joint_action/vector"][:].astype(np.float64))
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
        q01 = mean - 2.326 * std  # ~1st percentile of normal
        q99 = mean + 2.326 * std

    print(f"\nTotal: {total_count} timesteps across {len(task_roots)} tasks")
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
    parser = argparse.ArgumentParser(
        description="Compute action normalization stats from episode HDF5 files."
    )
    parser.add_argument("--data_root", type=str, default=None,
                        help="Directory containing episode HDF5 files "
                             "(for single-task mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for action_stats.npy "
                             "(default: <data_root>/action_stats.npy)")
    parser.add_argument("--format", type=str, default="robotwin",
                        choices=["robotwin", "robotwin_multitask"],
                        help="HDF5 format: robotwin (joint_action/vector, default), "
                             "or robotwin_multitask (aggregate across all training tasks)")
    # Multi-task options
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Top-level RoboTwin dataset directory "
                             "(required for --format robotwin_multitask)")
    parser.add_argument("--robot", type=str, default="aloha-agilex",
                        help="Robot name (for --format robotwin_multitask)")
    parser.add_argument("--variant", type=str, default="clean_50",
                        help="Data variant (for --format robotwin_multitask)")
    parser.add_argument("--tasks_file", type=str, default=None,
                        help="Optional file listing task names to include")
    args = parser.parse_args()

    tasks = parse_tasks_file(args.tasks_file) if args.tasks_file else None

    if args.format == "robotwin_multitask":
        if not args.dataset_dir:
            parser.error("--dataset_dir is required for --format robotwin_multitask")
        output_path = args.output or os.path.join(
            args.dataset_dir, f"{args.robot}_action_stats.npy"
        )
        print(f"Computing multi-task action stats: {args.dataset_dir} "
              f"(robot={args.robot}, variant={args.variant})")
        if args.variant in {"train", "val"}:
            if tasks is None:
                parser.error("--tasks_file is required when --variant is train or val")
            roots = []
            for task in tasks:
                data_root = os.path.join(args.dataset_dir, task, args.robot, args.variant)
                if os.path.isdir(data_root):
                    roots.append((task, data_root))
            if not roots:
                raise FileNotFoundError(
                    f"No split-layout task data found in {args.dataset_dir} "
                    f"for robot={args.robot}, variant={args.variant}"
                )

            print(f"Computing stats across {len(roots)} tasks for {args.robot}/{args.variant}")
            running_sum = None
            running_sum_sq = None
            total_count = 0
            for task_idx, (task_name, data_root) in enumerate(roots):
                pattern = os.path.join(data_root, "episode*.hdf5")
                files = sorted(glob.glob(pattern))
                if not files:
                    print(f"  [{task_idx+1}/{len(roots)}] {task_name}: no episodes, skipping")
                    continue

                task_count = 0
                for path in files:
                    try:
                        with h5py.File(path, "r") as f:
                            if "joint_action/vector" not in f:
                                continue
                            actions = f["joint_action/vector"][:].astype(np.float64)
                    except Exception:
                        continue

                    if running_sum is None:
                        action_dim = actions.shape[1]
                        running_sum = np.zeros(action_dim, dtype=np.float64)
                        running_sum_sq = np.zeros(action_dim, dtype=np.float64)

                    running_sum += actions.sum(axis=0)
                    running_sum_sq += (actions ** 2).sum(axis=0)
                    total_count += actions.shape[0]
                    task_count += actions.shape[0]

                print(f"  [{task_idx+1}/{len(roots)}] {task_name}: "
                      f"{len(files)} episodes, {task_count} timesteps")

            if total_count == 0:
                raise ValueError("No action data found in any split-layout task")

            mean = running_sum / total_count
            variance = running_sum_sq / total_count - mean ** 2
            variance = np.maximum(variance, 0.0)
            std = np.sqrt(variance)
            std = np.maximum(std, 1e-3)
            stats = {"mean": mean, "std": std}
        else:
            stats = compute_multitask_robotwin_stats(
                args.dataset_dir, args.robot, args.variant, tasks
            )
    else:
        if not args.data_root:
            parser.error("--data_root is required for single-task formats")
        output_path = args.output or os.path.join(args.data_root, "action_stats.npy")
        print(f"Computing action stats from: {args.data_root}")
        stats = compute_action_stats(args.data_root)

    print(f"\nResults ({stats['mean'].shape[0]}D actions, "
          f"saved to {output_path}):")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")

    np.save(output_path, stats)
    print("Done.")


if __name__ == "__main__":
    main()
