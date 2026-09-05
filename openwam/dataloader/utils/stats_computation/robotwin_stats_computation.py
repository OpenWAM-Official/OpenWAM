"""Compute per-episode action normalization stats for RoboTwin data.

The output is a single ``.npy`` file containing BOTH joint and eef stats:

    {
        "joint": {"mean", "std", "min", "max", "q01", "q99"},   # shape (14,)
        "eef":   {"mean", "std", "min", "max", "q01", "q99"},   # shape (20,)
        "num_timesteps": int,
    }

Running once produces stats for both action modes — downstream loading picks
whichever sub-dict matches ``action_mode`` at dataset-construction time.

Usage
-----
    # yaml-driven (preferred)
    python -m openwam.dataloader.utils.stats_computation.robotwin_stats_computation \
        --config configs/dataloader/robotwin.yaml

    # single-task
    python -m openwam.dataloader.utils.stats_computation.robotwin_stats_computation \
        --data_root /path/to/task/{embodiment}_{variant}/data

    # multi-task
    python -m openwam.dataloader.utils.stats_computation.robotwin_stats_computation \
        --dataset_dir /path/to/RoboTwin2.0/dataset --embodiment aloha-agilex --variant both
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
from typing import Optional

import h5py
import numpy as np

# Per-mode action dim is fixed by the HDF5 layout
_JOINT_ACTION_DIM = 14  # aloha-agilex qpos vector
_EEF_ACTION_DIM = 20  # [xyz(3) + rot6d(6) + grip(1)] x 2 arms
_MODES = ("joint", "eef")

_SHARD_DIR_NAME = "shards_v1"


# ---------------------------------------------------------------------------
# Atomic IO helpers (used both for shards and the final stats .npy)
# ---------------------------------------------------------------------------


def _atomic_save_npz(path: str, **arrays) -> None:
    """Write a compressed NPZ via tmp + os.replace (atomic on POSIX)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_save_stats_npy(path: str, stats: dict) -> None:
    """Write the final stats .npy atomically.

    ``np.save`` is *not* atomic — the destination file is created at
    ``open(..., "wb")`` time but only filled in afterwards, so a polling
    consumer (e.g. another distributed rank) can observe a half-written
    file via ``os.path.exists``. We instead serialize to a sibling tmp path
    and ``os.replace`` it into place.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    np.save(tmp_path, stats, allow_pickle=True)
    # ``np.save`` may append a ``.npy`` extension if the target path doesn't
    # already end with one; move the actually-produced file.
    actual_tmp = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npy"
    os.replace(actual_tmp, path)


# ---------------------------------------------------------------------------
# Partial checkpoint directory + shard helpers
# ---------------------------------------------------------------------------


def _partial_stats_dir(checkpoint_path: str) -> str:
    return f"{checkpoint_path}.partial"


def cleanup_partial_stats_checkpoint(checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return
    partial_dir = _partial_stats_dir(checkpoint_path)
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)


def _shard_path_for_root(checkpoint_path: str, data_root: str) -> str:
    """Deterministic shard path for one task ``data_root``.

    The absolute ``data_root`` is the whole cache key: expanding/shrinking the
    task list naturally reuses or ignores shards by recomputing this path for
    the *current* task_roots. ``shards_v1`` is the shard-format namespace; bump
    it if the NPZ payload schema ever changes.
    """
    canonical = os.path.abspath(data_root)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_partial_stats_dir(checkpoint_path), _SHARD_DIR_NAME, f"{digest}.npz")


def _read_joint_actions(f) -> Optional[np.ndarray]:
    if "joint_action/vector" not in f:
        return None
    return f["joint_action/vector"][()].astype(np.float64)


def _read_eef_actions(f) -> Optional[np.ndarray]:
    if "endpose/left_endpose" not in f:
        return None
    # Local import avoids a hard dependency when only joint is needed in tests
    from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d

    left_ep = f["endpose/left_endpose"][()].astype(np.float64)
    right_ep = f["endpose/right_endpose"][()].astype(np.float64)
    left_grip = f["endpose/left_gripper"][()].astype(np.float64)
    right_grip = f["endpose/right_gripper"][()].astype(np.float64)

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


class _ModeAccumulator:
    """Online accumulator for one action modality."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.running_sum = np.zeros(action_dim, dtype=np.float64)
        self.running_sum_sq = np.zeros(action_dim, dtype=np.float64)
        self.total_count = 0
        self._buffers: list[np.ndarray] = []

    def update(self, actions: np.ndarray) -> None:
        if actions is None:
            return
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got shape {actions.shape}")
        self.running_sum += actions.sum(axis=0)
        self.running_sum_sq += (actions**2).sum(axis=0)
        self.total_count += actions.shape[0]
        self._buffers.append(actions)

    def finalize(self) -> dict:
        if self.total_count == 0:
            raise ValueError("No data accumulated for this mode")
        mean = self.running_sum / self.total_count
        variance = np.maximum(self.running_sum_sq / self.total_count - mean**2, 0.0)
        std = np.maximum(np.sqrt(variance), 1e-3)

        concatenated = np.concatenate(self._buffers, axis=0)
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": concatenated.min(axis=0).astype(np.float32),
            "max": concatenated.max(axis=0).astype(np.float32),
            "q01": np.percentile(concatenated, 1, axis=0).astype(np.float32),
            "q99": np.percentile(concatenated, 99, axis=0).astype(np.float32),
        }


def _iter_episode_files(data_root: str) -> list[str]:
    pattern = os.path.join(data_root, "episode*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_root}")
    return files


def _accumulate_from_files(
    files: list[str],
    joint_acc: _ModeAccumulator,
    eef_acc: _ModeAccumulator,
    label: str = "",
) -> int:
    """Run a single file pass, feeding BOTH accumulators."""
    total = 0
    for i, path in enumerate(files):
        try:
            with h5py.File(path, "r") as f:
                joint_actions = _read_joint_actions(f)
                eef_actions = _read_eef_actions(f)
        except Exception as e:
            print(f"  [{label}][{i + 1}/{len(files)}] {os.path.basename(path)}: error {e}, skipping")
            continue

        if joint_actions is not None:
            joint_acc.update(joint_actions)
            total = max(total, joint_acc.total_count)
        if eef_actions is not None:
            eef_acc.update(eef_actions)
            total = max(total, eef_acc.total_count)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  [{label}][{i + 1}/{len(files)}] processed; total timesteps so far: {total}")
    return total


def _collect_actions_from_files(
    files: list[str],
    label: str = "",
    prior_total: int = 0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Collect one task-root's actions into raw arrays for shard persistence.

    Mirrors :func:`_accumulate_from_files` but returns the concatenated joint
    and eef arrays so the caller can persist them to a per-task NPZ shard
    (vs. folding them straight into a global accumulator). ``prior_total`` is
    purely cosmetic — it makes the running progress log reflect already-
    persisted task-roots when resuming.
    """
    joint_chunks: list[np.ndarray] = []
    eef_chunks: list[np.ndarray] = []
    joint_total = 0
    eef_total = 0

    for i, path in enumerate(files):
        try:
            with h5py.File(path, "r") as f:
                joint_actions = _read_joint_actions(f)
                eef_actions = _read_eef_actions(f)
        except Exception as e:
            print(f"  [{label}][{i + 1}/{len(files)}] {os.path.basename(path)}: error {e}, skipping")
            continue

        if joint_actions is not None:
            joint_chunks.append(joint_actions)
            joint_total += joint_actions.shape[0]
        if eef_actions is not None:
            eef_chunks.append(eef_actions)
            eef_total += eef_actions.shape[0]

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            total = prior_total + max(joint_total, eef_total)
            print(f"  [{label}][{i + 1}/{len(files)}] processed; total timesteps so far: {total}")

    joint = np.concatenate(joint_chunks, axis=0) if joint_chunks else None
    eef = np.concatenate(eef_chunks, axis=0) if eef_chunks else None
    return joint, eef, max(joint_total, eef_total)


def _rebuild_stats_from_shards(
    *,
    task_roots: list[tuple[str, str]],
    checkpoint_path: str,
) -> dict:
    """Aggregate the current task_roots' deterministic shards into stats."""
    partial_dir = _partial_stats_dir(checkpoint_path)
    joint_acc = _ModeAccumulator(_JOINT_ACTION_DIM)
    eef_acc = _ModeAccumulator(_EEF_ACTION_DIM)

    for _, data_root in task_roots:
        shard_path = _shard_path_for_root(checkpoint_path, data_root)
        if not os.path.exists(shard_path):
            continue
        with np.load(shard_path, allow_pickle=False) as payload:
            joint = payload["joint"]
            eef = payload["eef"]
        if joint.size > 0:
            joint_acc.update(joint.astype(np.float64, copy=False))
        if eef.size > 0:
            eef_acc.update(eef.astype(np.float64, copy=False))

    result: dict = {
        "num_timesteps": int(max(joint_acc.total_count, eef_acc.total_count)),
    }
    if joint_acc.total_count == 0 and eef_acc.total_count == 0:
        raise FileNotFoundError(
            f"Cannot rebuild stats from partial checkpoint at {partial_dir}: no shards matched the current task_roots."
        )
    if joint_acc.total_count > 0:
        result["joint"] = joint_acc.finalize()
    else:
        print("  WARNING: joint stats are empty; no joint_action/vector seen.")
    if eef_acc.total_count > 0:
        result["eef"] = _pin_eef_rot6d_identity(eef_acc.finalize())
    else:
        print("  WARNING: eef stats are empty; no endpose/* seen.")
    return result


_EEF_ROT6D_DIMS = (*range(3, 9), *range(13, 19))


def _pin_eef_rot6d_identity(stats: dict) -> dict:
    """Pin the per-arm rot6d dims to identity (the repo-wide convention).

    Normalization must be a pass-through on the rotation representation:
    min-max would otherwise rescale each rot6d component independently and
    distort rotations. Matches the historical shipped stats files.
    """
    identity = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}
    for key, value in identity.items():
        arr = np.asarray(stats[key], dtype=np.float64).copy()
        arr[list(_EEF_ROT6D_DIMS)] = value
        stats[key] = arr
    return stats


def compute_normalization_stats(data_root: str) -> dict:
    """Compute stats for a single-task directory, covering joint + eef.

    Args:
        data_root: Directory containing ``episode*.hdf5`` files.

    Returns:
        Nested dict ``{"joint": {...}, "eef": {...}, "num_timesteps": int}``.
        A mode key is omitted if the corresponding HDF5 fields were absent in
        every episode.
    """
    files = _iter_episode_files(data_root)
    joint_acc = _ModeAccumulator(_JOINT_ACTION_DIM)
    eef_acc = _ModeAccumulator(_EEF_ACTION_DIM)

    print(f"Computing joint+eef stats from {len(files)} episodes in {data_root}")
    total = _accumulate_from_files(files, joint_acc, eef_acc)

    result: dict = {"num_timesteps": int(max(joint_acc.total_count, eef_acc.total_count, total))}
    if joint_acc.total_count > 0:
        result["joint"] = joint_acc.finalize()
    else:
        print("  WARNING: no joint_action/vector found in any episode; joint stats omitted.")
    if eef_acc.total_count > 0:
        result["eef"] = _pin_eef_rot6d_identity(eef_acc.finalize())
    else:
        print("  WARNING: no endpose/* found in any episode; eef stats omitted.")
    return result


def compute_multitask_robotwin_stats(
    dataset_dir: str,
    embodiment: str,
    variant: str = "clean_50",
    tasks: Optional[list] = None,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Compute joint+eef stats aggregated across many task/variant roots.

    Args:
        dataset_dir: Top-level RoboTwin dataset directory.
        embodiment: Robot embodiment name.
        variant: ``"clean_50"``, ``"randomized_500"``, or ``"both"``.
        tasks: Optional internal task restriction. Defaults to every task
            discovered on disk.
        checkpoint_path: When set, the function persists a per-task NPZ
            shard into ``<checkpoint_path>.partial/shards_v1/`` after each
            task-root is processed. A subsequent call recomputes the shard
            path from the current ``data_root`` and skips it when the shard
            already exists.

    Returns:
        Nested dict identical in shape to :func:`compute_normalization_stats`.
    """
    from openwam.dataloader.robotwin import discover_robotwin_roots

    variant_list = ["clean_50", "randomized_500"] if variant == "both" else [variant]

    task_roots = []
    for v in variant_list:
        task_roots.extend(discover_robotwin_roots(dataset_dir, embodiment, v, tasks))

    if not task_roots:
        raise FileNotFoundError(f"No task data found in {dataset_dir} for embodiment={embodiment}, variant={variant}")

    print(f"Computing joint+eef stats across {len(task_roots)} task-variant pairs for embodiment={embodiment}")

    if checkpoint_path is not None:
        return _compute_multitask_with_checkpoint(
            task_roots=task_roots,
            checkpoint_path=checkpoint_path,
        )

    joint_acc = _ModeAccumulator(_JOINT_ACTION_DIM)
    eef_acc = _ModeAccumulator(_EEF_ACTION_DIM)

    for task_idx, (task_name, data_root) in enumerate(task_roots):
        label = f"{task_idx + 1}/{len(task_roots)} {task_name}"
        try:
            files = _iter_episode_files(data_root)
        except FileNotFoundError:
            print(f"  [{label}] no episodes, skipping")
            continue
        _accumulate_from_files(files, joint_acc, eef_acc, label=label)

    result: dict = {
        "num_timesteps": int(max(joint_acc.total_count, eef_acc.total_count)),
    }
    if joint_acc.total_count > 0:
        result["joint"] = joint_acc.finalize()
    else:
        print("  WARNING: joint stats are empty; no joint_action/vector seen.")
    if eef_acc.total_count > 0:
        result["eef"] = _pin_eef_rot6d_identity(eef_acc.finalize())
    else:
        print("  WARNING: eef stats are empty; no endpose/* seen.")

    return result


def _compute_multitask_with_checkpoint(
    *,
    task_roots: list[tuple[str, str]],
    checkpoint_path: str,
) -> dict:
    """Resumable variant: deterministic shard-per-task, then rebuild stats.

    Split out from :func:`compute_multitask_robotwin_stats` so the
    non-checkpoint path stays a clean linear accumulation and the resumable
    path can isolate its shard IO.
    """
    partial_dir = _partial_stats_dir(checkpoint_path)
    os.makedirs(partial_dir, exist_ok=True)
    processed_total = 0
    existing_shards = {
        os.path.abspath(data_root): _shard_path_for_root(checkpoint_path, data_root)
        for _, data_root in task_roots
        if os.path.exists(_shard_path_for_root(checkpoint_path, data_root))
    }
    if existing_shards:
        print(
            f"Resuming multi-task stats from partial checkpoint: "
            f"{len(existing_shards)}/{len(task_roots)} task-variant pairs already "
            f"persisted at {partial_dir}"
        )

    for task_idx, (task_name, data_root) in enumerate(task_roots):
        label = f"{task_idx + 1}/{len(task_roots)} {task_name}"
        shard_path = _shard_path_for_root(checkpoint_path, data_root)
        if os.path.exists(shard_path):
            print(f"  [{label}] already checkpointed, skipping")
            continue
        try:
            files = _iter_episode_files(data_root)
        except FileNotFoundError:
            print(f"  [{label}] no episodes, skipping")
            continue

        joint, eef, local_total = _collect_actions_from_files(files, label=label, prior_total=processed_total)
        processed_total += local_total

        _atomic_save_npz(
            shard_path,
            joint=joint if joint is not None else np.empty((0, _JOINT_ACTION_DIM), dtype=np.float64),
            eef=eef if eef is not None else np.empty((0, _EEF_ACTION_DIM), dtype=np.float64),
        )

    return _rebuild_stats_from_shards(task_roots=task_roots, checkpoint_path=checkpoint_path)


def parse_tasks_file(tasks_file: str) -> list:
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            tasks.append(line.split()[0])
    return tasks


def _load_yaml_config(path: str) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def _print_summary(stats: dict) -> None:
    for mode in _MODES:
        sub = stats.get(mode)
        if sub is None:
            continue
        print(f"\n[{mode}] dim={sub['mean'].shape[0]}")
        print(f"  mean: {np.round(sub['mean'], 4)}")
        print(f"  std:  {np.round(sub['std'], 4)}")
        print(f"  min:  {np.round(sub['min'], 4)}")
        print(f"  max:  {np.round(sub['max'], 4)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a dataloader yaml (e.g. configs/dataloader/robotwin.yaml). "
        "CLI flags below override values read from it.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Single-task mode: directory with episode*.hdf5 files",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Multi-task mode: top-level RoboTwin dataset directory",
    )
    parser.add_argument("--embodiment", type=str, default=None, help="Robot name")
    parser.add_argument("--variant", type=str, default=None, help='"clean_50" | "randomized_500" | "both"')
    parser.add_argument(
        "--tasks_file", type=str, default=None, help="Optional file listing tasks to include (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy path (required for --data_root single-task mode; default "
        "<dataset_dir>/meta/robotwin_<variant>_normalization_stats.npy for multi-task)",
    )
    args = parser.parse_args()

    # Load yaml first, then let CLI flags override
    cfg: dict = {}
    if args.config:
        cfg = _load_yaml_config(args.config)

    dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
    data_root = args.data_root  # data_root is not a yaml concept, CLI-only
    embodiment = args.embodiment or cfg.get("embodiment", "aloha-agilex")
    variant = args.variant or cfg.get("variant", "clean_50")
    tasks = parse_tasks_file(args.tasks_file) if args.tasks_file else None
    output = args.output

    if data_root:
        # Single-task CLI (debugging aid): stats have no canonical
        # per-task location any more, so the caller must name the output.
        if not output:
            parser.error("--output is required with --data_root (single-task stats have no canonical location)")
        resolved_output = output
        print(f"Single-task stats from: {data_root}")
        stats = compute_normalization_stats(data_root)
    else:
        if not dataset_dir:
            parser.error("either --data_root or --dataset_dir / --config providing one is required")
        resolved_output = output or os.path.join(dataset_dir, "meta", f"robotwin_{variant}_normalization_stats.npy")
        print(f"Multi-task stats from: {dataset_dir} (embodiment={embodiment}, variant={variant})")
        stats = compute_multitask_robotwin_stats(
            dataset_dir=dataset_dir,
            embodiment=embodiment,
            variant=variant,
            tasks=tasks,
            checkpoint_path=resolved_output,
        )

    _print_summary(stats)
    os.makedirs(os.path.dirname(resolved_output) or ".", exist_ok=True)
    atomic_save_stats_npy(resolved_output, stats)
    cleanup_partial_stats_checkpoint(resolved_output)
    print(f"\nSaved stats ({stats.get('num_timesteps', 0)} timesteps) to {resolved_output}")


if __name__ == "__main__":
    main()
