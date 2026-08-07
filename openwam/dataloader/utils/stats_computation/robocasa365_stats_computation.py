"""Per-task action-normalization stats for RoboCasa365 (single-arm EEF; computed at 10-D, persisted
at 20-D, or the combined 25-D ``eef_base`` [arm20, base5] when mobile).

Simplified single-arm dual of ``robotwin_stats_computation.py``, reading the v3.0 aggregated repo:
iterate the episodes (single task via ``task_name``, or all tasks pooled — see
:func:`compute_multitask_stats`), assemble the raw 10-D arm pose from ``observation.state`` (the same
``state_to_arm10`` the reader uses), reduce to ``mean/std/min/max/q01/q99``, and — when
``include_base`` — also the 5-D base command from the LeRobot ``action`` field, then concatenate the
two into ONE 25-D block so the whole ``[arm20, base5]`` vector normalizes with a single stats block
(deploy gathers 80->25 and un-normalizes with it, no base special-casing).

Output schema (``.npy``, ``allow_pickle``)::

    {"eef": {mean, std, min, max, q01, q99}, "num_timesteps": int}            # 20-D (non-mobile)
    #   arm10 left = real stats with the rot6d dims (3:9) pinned to identity (normalization is a
    #   pass-through on rot6d under every mode — only pos + gripper normalize), right half = neutral
    #   (mean0/std1/min-1/max1/q01-1/q99 1); reduced at 10-D internally, then left-padded to 20-D by
    #   _expand_stats_to_20d before persist.
    {"eef_base": {...25-D...}, "num_timesteps": int}                          # 25-D (include_base)
    #   eef_base = concat(eef20, base5); base5 = the RoboCasa-native action command stats
    #   [x_vel, y_vel, yaw_vel, torso, control_mode] with a constant-dim guard (torso is 0 across the
    #   whole dataset → its min==max is nudged to avoid divide-by-zero at normalize time).

The PROPRIO base velocity does NOT get its own stats block: the reader rescales it into the action's
command space (A′, see robocasa365._BASE_VEL_PHYS_MAX) so it shares the base5 command stats. The
``base_proprio="global_pose"`` proprio DOES: pose is meters/unit-circle, not command space, so
``include_base`` also emits ``eef_base_pose_proprio`` = concat(eef20, pose5 stats) where pose5 =
``[x, y, sin(yaw), cos(yaw), 0]`` from ``observation.state`` (sin/cos dims pinned to the unit-circle
[-1, +1] range; the spare dim 4 gets the constant-dim guard). Velocity-proprio ckpts ignore the
extra block, so one stats run serves both modes.

``RoboCasa365Dataset`` auto-computes this on first use when ``normalize_mode`` is set and no stats file
exists; run :func:`main` to precompute.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.robocasa365 import (
    _MOBILE_STATS_KEY,
    BASE_ACTION_DIM,
    RAW_MOBILE_DIM,
    STATS_DIM,
    _expand_stats_to_20d,
    _task_from_source_prefix,
    state_to_arm10,
)
from openwam.dataloader.transforms.normalize import compute_extended_stats
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.normalization import ROT6D_DIMS_ARM10, STAT_KEYS, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import atomic_save_stats_npy

# LeRobot ``action`` base command dims (mobile): [0:3] x/y/yaw vel, [3] torso, [4] control_mode.
_ACTION_BASE = slice(0, 5)


def _planar_base_pose5(states: np.ndarray) -> np.ndarray:
    """``(T, 5)`` planar world base pose ``[x, y, sin(yaw), cos(yaw), 0]`` from the 16-D state.

    State layout: ``[0:3]`` base_position (world), ``[3:7]`` base_rotation (world quat xyzw). The
    same quantity the reader's ``base_proprio="global_pose"`` proprio carries (its stats source).
    """
    x, y = states[:, 0], states[:, 1]
    qx, qy, qz, qw = states[:, 3], states[:, 4], states[:, 5], states[:, 6]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.stack([x, y, np.sin(yaw), np.cos(yaw), np.zeros_like(x)], axis=-1).astype(np.float32)


def _iter_episode_arrays(data_root: str, include_base: bool = False, task_name: str | None = None):
    """Yield ``(arm10, base5, pose5)`` per episode from a v3 aggregated repo.

    ``arm10`` = raw ``(T, 10)`` arm pose from ``observation.state`` (always). ``base5`` = raw
    ``(T, 5)`` base command from the ``action`` field when ``include_base`` else ``None``; ``pose5``
    = the planar world base pose from the state (``include_base`` only, see :func:`_planar_base_pose5`).
    ``task_name`` filters the v3 aggregated repo to one task. Each aggregated shard is read once and
    sliced per episode via its file-local row offset.
    """
    with open(os.path.join(data_root, "meta", "info.json")) as f:
        data_tmpl = json.load(f)["data_path"]
    eps = load_episodes_parquet(Path(data_root))
    # Offsets over the FULL table BEFORE the task filter (a task not first in its shard must read at
    # its true file-local offset, past the other tasks' rows in the same file). Mirrors the reader.
    eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
    if task_name is not None:
        eps = eps[eps["source_prefix"].map(_task_from_source_prefix) == task_name].reset_index(drop=True)
    if len(eps) == 0:
        raise FileNotFoundError(f"No episodes for task {task_name!r} under {data_root}")
    cols = ["observation.state"] + (["action"] if include_base else [])
    for (chunk, file), grp in eps.groupby(["data/chunk_index", "data/file_index"], sort=False):
        path = os.path.join(data_root, data_tmpl.format(chunk_index=int(chunk), file_index=int(file)))
        df = pd.read_parquet(path, columns=cols)
        st = df["observation.state"].values
        ac = df["action"].values if include_base else None
        for _, r in grp.iterrows():
            o, length = int(r["_data_row_offset"]), int(r["length"])
            states = np.stack(st[o : o + length]).astype(np.float32)  # (T, 16)
            arm10 = state_to_arm10(states)
            base5 = np.stack(ac[o : o + length]).astype(np.float32)[:, _ACTION_BASE] if include_base else None
            pose5 = _planar_base_pose5(states) if include_base else None
            yield arm10, base5, pose5


def _base_stats_block(base_chunks: list) -> dict:
    """5-D base command stats, with a min==max / std==0 guard so constant dims (torso is 0 across the
    whole dataset; control_mode all -1 in fixed-base buckets) don't divide-by-zero at normalize time —
    a constant dim then normalizes to a constant that still round-trips through denormalize."""
    b = compute_extended_stats(base_chunks)
    out = {k: np.asarray(b[k], np.float32).reshape(-1).copy() for k in STAT_KEYS}
    degenerate = (out["max"] - out["min"]) < 1e-6
    out["max"] = np.where(degenerate, out["min"] + 1.0, out["max"]).astype(np.float32)
    out["std"] = np.where(out["std"] < 1e-6, 1.0, out["std"]).astype(np.float32)
    if out["mean"].shape[0] != BASE_ACTION_DIM:
        raise ValueError(f"base stats dim {out['mean'].shape[0]} != {BASE_ACTION_DIM}")
    return out


def _pin_gripper_stats(eef10: dict) -> dict:
    """Pin the gripper dim (dim 9 of the 10-D arm) normalize range to the [-1, +1] COMMAND space.

    The gripper lives in the pretrain open-scale (-1=close, +1=open: ACTION = the NEGATED recorded
    binary command; PROPRIO = the achieved width rendered into [-1, +1], open->+1), so its min/max
    are pinned to [-1, +1] instead of the achieved-width data range. Then the model's ±1 output
    de-normalizes to EXACTLY ±1 (the deploy bridge confident-closes at < -0.5), and normalizing the
    rendered proprio is the identity. mean/std are set command-neutral (0/1) for the z-score path.
    The symmetric pin is polarity-invariant, so no stats regeneration was needed for the flip."""
    out = {k: np.array(eef10[k], dtype=np.float32) for k in eef10}
    g = STATS_DIM - 1  # dim 9 = gripper
    out["min"][g], out["max"][g], out["q01"][g], out["q99"][g] = -1.0, 1.0, -1.0, 1.0
    out["mean"][g], out["std"][g] = 0.0, 1.0
    return out


def _pose_stats_block(pose_chunks: list) -> dict:
    """5-D planar base pose stats ``[x, y, sin, cos, 0]`` for the ``eef_base_pose_proprio`` block.

    sin/cos dims are pinned to the unit-circle [-1, +1] range (representation bound, not the data
    range — same philosophy as the gripper pin); the spare dim 4 (constant 0) gets the same
    degenerate-dim guard as the base command block."""
    p = compute_extended_stats(pose_chunks)
    out = {k: np.asarray(p[k], np.float32).reshape(-1).copy() for k in STAT_KEYS}
    if out["mean"].shape[0] != BASE_ACTION_DIM:
        raise ValueError(f"pose stats dim {out['mean'].shape[0]} != {BASE_ACTION_DIM}")
    for d in (2, 3):  # sin(yaw), cos(yaw)
        out["min"][d], out["max"][d], out["q01"][d], out["q99"][d] = -1.0, 1.0, -1.0, 1.0
        out["mean"][d], out["std"][d] = 0.0, 1.0
    degenerate = (out["max"] - out["min"]) < 1e-6
    out["max"] = np.where(degenerate, out["min"] + 1.0, out["max"]).astype(np.float32)
    out["std"] = np.where(out["std"] < 1e-6, 1.0, out["std"]).astype(np.float32)
    return out


def _finish(arm_chunks: list, base_chunks: list, pose_chunks: list, total: int, include_base: bool, label: str) -> dict:
    """Reduce accumulated arm (+ base command + base pose) chunks to the persisted stats dict.

    Non-mobile → ``{"eef": 20-D}``; mobile → ``{"eef_base": 25-D, "eef_base_pose_proprio": 25-D}``
    (both concat the same 20-D arm block with, respectively, the 5-D base COMMAND stats — the shared
    action/velocity-proprio block — and the 5-D planar base POSE stats for the
    ``base_proprio="global_pose"`` proprio). The arm block is computed at 10-D (STATS_DIM) then
    left-padded to the 20-D bimanual schema; the gripper dim is pinned to the [-1, +1] command range
    (see _pin_gripper_stats)."""
    if not arm_chunks:
        raise ValueError(f"No timesteps accumulated ({label})")
    eef10 = compute_extended_stats(arm_chunks)
    if len(eef10["mean"]) != STATS_DIM:
        raise ValueError(f"computed arm dim {len(eef10['mean'])} != {STATS_DIM}")
    # Make normalization a no-op on the rot6d dims (see pin_rot6d_identity): only pos + gripper
    # normalize; the rotation representation reaches the model unchanged.
    pin_rot6d_identity(eef10, ROT6D_DIMS_ARM10)
    eef20 = _expand_stats_to_20d(_pin_gripper_stats(eef10))
    print(f"  [{label}] done: {total} timesteps, dim=20{' +base5 (combined eef_base)' if include_base else ''}")
    if not include_base:
        return {"eef": eef20, "num_timesteps": int(total)}
    base5 = _base_stats_block(base_chunks)
    combined = {k: np.concatenate([eef20[k], base5[k]]).astype(np.float32) for k in STAT_KEYS}
    if combined["mean"].shape[0] != RAW_MOBILE_DIM:
        raise ValueError(f"combined {_MOBILE_STATS_KEY} dim {combined['mean'].shape[0]} != {RAW_MOBILE_DIM}")
    pose5 = _pose_stats_block(pose_chunks)
    pose_combined = {k: np.concatenate([eef20[k], pose5[k]]).astype(np.float32) for k in STAT_KEYS}
    return {
        _MOBILE_STATS_KEY: combined,
        "eef_base_pose_proprio": pose_combined,  # robocasa365._PROPRIO_POSE_STATS_KEY
        "num_timesteps": int(total),
    }


def compute_normalization_stats(data_root: str, include_base: bool = False, task_name: str | None = None) -> dict:
    """Compute single-arm 20-D EEF stats (+ optional 5-D base command → combined 25-D ``eef_base``).

    ``task_name`` filters the v3 aggregated repo to one task (see :func:`_iter_episode_arrays`).
    """
    arm_chunks, base_chunks, pose_chunks, total = [], [], [], 0
    for i, (arm10, base5, pose5) in enumerate(_iter_episode_arrays(data_root, include_base, task_name)):
        arm_chunks.append(arm10)
        total += arm10.shape[0]
        if include_base:
            base_chunks.append(base5)
            pose_chunks.append(pose5)
        if (i + 1) % 100 == 0:
            print(f"  [stats] {i + 1} episodes, {total} timesteps so far")
    return _finish(arm_chunks, base_chunks, pose_chunks, total, include_base, "stats")


def compute_multitask_stats(roots: list, include_base: bool = False) -> dict:
    """Compute ONE shared 20-D EEF (+ optional 5-D base → combined 25-D ``eef_base``) across tasks.

    ``roots`` is ``[(task_name, repo), ...]`` (one per selected task, paired with its repo). Mirrors
    robotwin's multi-task shared-stats contract: every task in a multi-task run must train in the SAME
    normalized space, so stats are pooled over all tasks. Same schema as the single-task path.
    """
    arm_chunks, base_chunks, pose_chunks, total = [], [], [], 0
    for task_name, repo in roots:
        n0 = total
        for arm10, base5, pose5 in _iter_episode_arrays(repo, include_base, task_name):
            arm_chunks.append(arm10)
            total += arm10.shape[0]
            if include_base:
                base_chunks.append(base5)
                pose_chunks.append(pose5)
        print(f"  [multitask-stats] {task_name}: +{total - n0} timesteps (running {total})")
    return _finish(arm_chunks, base_chunks, pose_chunks, total, include_base, f"multitask-stats over {len(roots)} tasks")


def main():
    ap = argparse.ArgumentParser(
        description="Precompute RoboCasa365 EEF (+ optional combined 25-D eef_base) normalization stats"
    )
    ap.add_argument("data_root", help="A v3.0 aggregated RoboCasa365 repo (meta/info.json + meta/episodes/*.parquet)")
    ap.add_argument("--task", default=None, help="Filter the repo to one task (source_prefix); default: all tasks")
    ap.add_argument(
        "--mobile-base",
        action="store_true",
        help="Also read the base command and emit the combined 25-D 'eef_base' block",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .npy (default: {data_root}/{task|robocasa365}_{eef|eefbase}_stats.npy)",
    )
    args = ap.parse_args()

    out = args.output
    if out is None:
        tag = "eefbase" if args.mobile_base else "eef"
        out = os.path.join(args.data_root, f"{args.task or 'robocasa365'}_{tag}_stats.npy")
    stats = compute_normalization_stats(args.data_root, include_base=args.mobile_base, task_name=args.task)
    atomic_save_stats_npy(out, stats)
    print(f"Saved stats -> {out}")


if __name__ == "__main__":
    main()
