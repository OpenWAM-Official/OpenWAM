"""
Unzip and organize RoboTwin 2.0 data for VAM training.

RoboTwin 2.0 stores data as zip archives at:
    {src}/{task}/{robot}_{variant}_{count}.zip

This script extracts them to:
    {dst}/{task}/{robot}_{variant}_{count}/data/episode*.hdf5

Usage:
    # Extract all tasks for aloha-agilex clean_50
    python prepare_robotwin.py \
        --src /path/to/robotwin_2_0/dataset \
        --dst data/robotwin \
        --robot aloha-agilex \
        --variant clean_50

    # Extract specific tasks only
    python prepare_robotwin.py \
        --src /path/to/robotwin_2_0/dataset \
        --dst data/robotwin \
        --robot aloha-agilex \
        --variant clean_50 \
        --tasks adjust_bottle,beat_block_hammer

    # List available tasks and robots
    python prepare_robotwin.py \
        --src /path/to/robotwin_2_0/dataset \
        --list
"""

import argparse
import glob
import os
import zipfile

import h5py


def list_available(src: str):
    """List all available tasks, robots, and variants in the source directory."""
    tasks = sorted(
        d for d in os.listdir(src)
        if os.path.isdir(os.path.join(src, d)) and not d.startswith(".")
    )
    print(f"Source: {src}")
    print(f"Tasks ({len(tasks)}):")
    for task in tasks:
        task_dir = os.path.join(src, task)
        zips = sorted(glob.glob(os.path.join(task_dir, "*.zip")))
        zip_names = [os.path.basename(z).replace(".zip", "") for z in zips]
        print(f"  {task}: {', '.join(zip_names) if zip_names else '(no zips)'}")


def extract_task(src: str, dst: str, task: str, robot: str, variant: str):
    """Extract a single task zip for a given robot/variant.

    Returns (dst_dir, num_episodes, action_dim) or None if zip not found.
    """
    task_dir = os.path.join(src, task)
    # Find matching zip: {robot}_{variant}.zip or {robot}_{variant}_{count}.zip
    pattern = os.path.join(task_dir, f"{robot}_{variant}*.zip")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  WARNING: No zip matching {robot}_{variant}* in {task_dir}, skipping")
        return None

    zip_path = matches[0]
    zip_name = os.path.basename(zip_path).replace(".zip", "")
    dst_dir = os.path.join(dst, task, zip_name, "data")
    os.makedirs(dst_dir, exist_ok=True)

    # Check if already extracted
    existing = glob.glob(os.path.join(dst_dir, "episode*.hdf5"))
    if existing:
        print(f"  {task}/{zip_name}: already extracted ({len(existing)} episodes)")
        # Probe action dim
        with h5py.File(existing[0], "r") as f:
            action_dim = f["joint_action/vector"].shape[1]
        return dst_dir, len(existing), action_dim

    # Extract
    print(f"  Extracting {zip_path} -> {dst_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # RoboTwin zips may have nested directories; we want episode*.hdf5 at dst_dir level
        members = zf.namelist()
        hdf5_members = [m for m in members if m.endswith(".hdf5")]
        if not hdf5_members:
            print(f"    WARNING: No HDF5 files in {zip_path}")
            return None

        for member in hdf5_members:
            # Extract to dst_dir, flattening any subdirectory structure
            basename = os.path.basename(member)
            target_path = os.path.join(dst_dir, basename)
            with zf.open(member) as src_f, open(target_path, "wb") as dst_f:
                dst_f.write(src_f.read())

    episodes = sorted(glob.glob(os.path.join(dst_dir, "episode*.hdf5")))
    if not episodes:
        print(f"    WARNING: No episodes after extraction")
        return None

    # Probe action dim
    with h5py.File(episodes[0], "r") as f:
        action_dim = f["joint_action/vector"].shape[1]

    print(f"    Extracted {len(episodes)} episodes, action_dim={action_dim}")
    return dst_dir, len(episodes), action_dim


def main():
    parser = argparse.ArgumentParser(
        description="Unzip and organize RoboTwin 2.0 data for VAM training."
    )
    parser.add_argument("--src", type=str, required=True,
                        help="Source directory containing RoboTwin dataset "
                             "(e.g. /path/to/robotwin_2_0/dataset)")
    parser.add_argument("--dst", type=str, default="data/robotwin",
                        help="Destination directory for extracted data")
    parser.add_argument("--robot", type=str, default="aloha-agilex",
                        help="Robot name (e.g. aloha-agilex, piper, arx-x5, ur5, franka)")
    parser.add_argument("--variant", type=str, default="clean_50",
                        help="Dataset variant (e.g. clean_50, clean_500)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names. If not specified, "
                             "extracts all tasks found in src.")
    parser.add_argument("--list", action="store_true",
                        help="List available tasks and exit")
    args = parser.parse_args()

    if args.list:
        list_available(args.src)
        return

    # Discover tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = sorted(
            d for d in os.listdir(args.src)
            if os.path.isdir(os.path.join(args.src, d)) and not d.startswith(".")
        )

    print(f"Preparing RoboTwin 2.0 data:")
    print(f"  Source: {args.src}")
    print(f"  Dest:   {args.dst}")
    print(f"  Robot:  {args.robot}")
    print(f"  Variant: {args.variant}")
    print(f"  Tasks:  {len(tasks)}")
    print()

    results = []
    total_episodes = 0
    for task in tasks:
        result = extract_task(args.src, args.dst, task, args.robot, args.variant)
        if result:
            dst_dir, n_eps, action_dim = result
            results.append((task, dst_dir, n_eps, action_dim))
            total_episodes += n_eps

    print(f"\nSummary:")
    print(f"  Tasks extracted: {len(results)}/{len(tasks)}")
    print(f"  Total episodes: {total_episodes}")
    if results:
        print(f"  Action dim: {results[0][2]}")
        print(f"\nExtracted paths:")
        for task, dst_dir, n_eps, action_dim in results:
            print(f"  {task}: {dst_dir} ({n_eps} episodes, {action_dim}D)")

    print("\nNext steps:")
    print(f"  1. Compute action stats:")
    print(f"     python compute_action_stats.py --data_root {results[0][1] if results else '<path>'} "
          f"--format robotwin --output {args.dst}/{args.robot}_action_stats.npy")
    print(f"  2. Train:")
    print(f"     bash train_s3_robotwin.sh")


if __name__ == "__main__":
    main()
