#!/usr/bin/env python3
"""Compute action-normalization stats for the BEHAVIOR-1K dataset (all action modes).

BEHAVIOR-1K (2025 challenge demos, robot R1Pro) is a single-robot LeRobot v2.1
dataset, so — unlike :mod:`robocoin_stats_computation` — there is no per-robot
grouping: one ``meta/stats_R1Pro.json`` is written for the whole dataset (named
by ``info.json``'s ``robot_type``, which the reader hardcodes).

The reader (:class:`~openwam.dataloader.behavior.BehaviorDataset`) emits a raw
27-D EEF vector ``[L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6, R_grip1, base3,
trunk4]`` (eef/unified modes) or a raw 23-D joint vector ``[L_arm7, L_grip1,
R_arm7, R_grip1, base3, trunk4]`` (joint mode). We write **four** stats blocks —
``eef``/``base_vel``/``trunk`` cover the eef·unified modes (raw 27) and
``arm_joint`` covers joint mode (raw 23 = ``arm_joint16 + base_vel3 + trunk4``);
``base_vel`` and ``trunk`` are shared (same native columns):

  * ``eef``       — 20-D ``[L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6, R_grip1]``.
                    Same layout RoboCOIN normalizes, so we **reuse its
                    ``Accumulator`` and ``_pin_rot6d_identity`` verbatim**: the 12
                    rot6d dims (3:9 / 13:19) are pinned to identity so
                    normalization is a pass-through on the rotation manifold
                    (pos / gripper keep real stats). Pass ``--no-rot6d-identity``
                    to disable.
  * ``base_vel``  — 3-D ``[vx, vy, vyaw]`` base velocity. Pooled: action = local-frame
                    command (``action[0:3]``), proprio = RAW world-frame ``base_qvel``
                    (Larchenko base mode). Real stats — NOT pinned (a genuine velocity).
  * ``trunk``     — 4-D absolute torso joint targets (native ``action[3:7]``). Like
                    ``base_vel``, a BEHAVIOR-specific block with **real** stats —
                    NOT pinned (genuine joint angles).
  * ``arm_joint`` — 16-D ``[L_arm7, L_grip1, R_arm7, R_grip1]`` native
                    JointController setpoints (``action[7:14]/[14]/[15:22]/[22]``),
                    the joint-mode arm block. **Real** stats, NOT pinned (no rot6d).

Every row contributes TWO points per block — one from the ACTION stream (the
target: gripper/base/trunk commands, native arm setpoints) and one from the
PROPRIO stream (the achieved state the reader renders: gripper open-scale, base-
frame velocity, trunk qpos, arm qpos). We pool both into one accumulator, exactly
as RoboCOIN pools ``*_action`` + ``*_state`` (``pool: action+state``). This is the
correction that makes the shared stats actually serve both: the proprio gripper
is a continuous open-scale while the action gripper is a binary ±1 command — NOT
the same marginal — so normalizing proprio with action-only stats (as this script
did before) would mis-scale it. The eef POSE dims are identical in both streams
(both = ``eef(state)``), so pooling them is a harmless duplicate; only the
gripper/base/trunk/arm dims genuinely differ and need the union.

To guarantee zero layout drift, the action-stream vectors are built with the
reader's own ``_state_to_eef18`` / ``_assemble_raw`` / ``_assemble_arm_joint`` and
the proprio-stream vectors with the reader's own ``_state_to_raw_proprio_eef`` /
``_state_to_raw_proprio_joint`` — the stats are computed over the exact numbers the
reader feeds the model as action and as proprio (pre-scatter, pre-normalization).

Output schema (``meta/stats_R1Pro.json``)::

    {
      "eef":       {"mean":[..20], "std":[..20], "min":[..20], "max":[..20],
                    "q01":[..20], "q99":[..20], "num_timesteps":N, "num_files":M,
                    "robot_type":"R1Pro", "rot6d_identity":true},
      "base_vel":  {"mean":[..3], ..., "q01":[..3], "q99":[..3],
                    "num_timesteps":N, "layout":"vx,vy,vyaw"},
      "trunk":     {"mean":[..4], ..., "q01":[..4], "q99":[..4],
                    "num_timesteps":N, "layout":"torso_joint_abs"},
      "arm_joint": {"mean":[..16], ..., "q01":[..16], "q99":[..16],
                    "num_timesteps":N, "layout":"L_arm7,L_grip1,R_arm7,R_grip1"}
    }

mean/std/min/max are exact (streamed over every row); q01/q99 come from a bounded
uniform reservoir sample (see :class:`Accumulator`).

Usage:
    python -m openwam.dataloader.utils.stats_computation.behavior_stats_computation \
        --dataset_dir /path/to/datasets/behaviour-1k
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Reuse the reader's own EEF/base assembly + offsets so the stats are computed
# over byte-identical numbers to what the reader emits (no layout drift).
from openwam.dataloader.behavior import (
    _ACT_BASE,
    _ACT_LARM,
    _ACT_LGRIP,
    _ACT_RARM,
    _ACT_RGRIP,
    _ACT_TRUNK,
    _ARM_JOINT_DIM,
    _BASE_DIM,
    _EEF_DIM,
    _TRUNK_DIM,
    _assemble_arm_joint,
    _assemble_raw,
    _state_to_eef18,
    _state_to_raw_proprio_eef,
    _state_to_raw_proprio_joint,
)

# Reuse RoboCOIN's online accumulator + rot6d-identity pin VERBATIM (dev-aligned:
# one definition of the EEF stats machinery and the rot6d convention).
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import (
    Accumulator,
    _pin_rot6d_identity,
)

_NEEDED_COLS = ["observation.state", "action"]


def _iter_episode_parquets(dataset_dir: Path):
    """Yield every ``data/task-*/episode_*.parquet`` on disk (sorted, partial-download safe)."""
    data_dir = dataset_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} does not exist — download the dataset first.")
    for task_dir in sorted(data_dir.glob("task-*")):
        if not task_dir.is_dir():
            continue
        for fpath in sorted(task_dir.glob("episode_*.parquet")):
            yield fpath


def _rows_to_blocks(state: np.ndarray, action: np.ndarray):
    """``(T,256)`` state + ``(T,23)`` action → ``(T,20)`` eef + ``(T,3)`` base + ``(T,4)`` trunk.

    Built via the reader's own helpers: eef18 = ``_state_to_eef18`` (state quats),
    then ``_assemble_raw`` interleaves the gripper commands + base velocity + trunk
    joints into the canonical raw-27 layout, which splits cleanly as ``[:20]`` (eef)
    / ``[20:23]`` (base) / ``[23:27]`` (trunk). This is exactly the pre-normalization
    vector the reader scatters.
    """
    eef18 = _state_to_eef18(state)
    l_grip = action[:, _ACT_LGRIP : _ACT_LGRIP + 1]
    r_grip = action[:, _ACT_RGRIP : _ACT_RGRIP + 1]
    base = action[:, _ACT_BASE]
    trunk = action[:, _ACT_TRUNK]
    raw = _assemble_raw(eef18, l_grip, r_grip, base, trunk)
    e, b = _EEF_DIM, _BASE_DIM
    return raw[:, :e], raw[:, e : e + b], raw[:, e + b : e + b + _TRUNK_DIM]


def _rows_to_arm_joint(action: np.ndarray) -> np.ndarray:
    """``(T,23)`` native action → ``(T,16)`` joint-mode arm block
    ``[L_arm7, L_grip1, R_arm7, R_grip1]`` (via the reader's own ``_assemble_arm_joint``
    so the stats are computed over byte-identical numbers to what joint mode feeds)."""
    return _assemble_arm_joint(
        action[:, _ACT_LARM],
        action[:, _ACT_LGRIP : _ACT_LGRIP + 1],
        action[:, _ACT_RARM],
        action[:, _ACT_RGRIP : _ACT_RGRIP + 1],
    )


def _proprio_rows_to_blocks(state: np.ndarray):
    """``(T,256)`` state → the PROPRIO stream's ``(T,20)`` eef + ``(T,3)`` base +
    ``(T,4)`` trunk + ``(T,16)`` arm-joint, using the reader's own achieved-state
    renderers so the stats see the exact numbers ``_proprio_20d`` emits.

    Proprio differs from action on the gripper (open-scale vs ±1 cmd), base
    (achieved raw WORLD-frame velocity vs local-frame cmd) and arm (achieved qpos vs setpoint); the
    eef POSE dims are identical to the action stream (both ``eef(state)``)."""
    p_eef27 = _state_to_raw_proprio_eef(state)  # [eef20, base3, trunk4]
    p_joint23 = _state_to_raw_proprio_joint(state)  # [arm16, base3, trunk4]
    e, b = _EEF_DIM, _BASE_DIM
    p_eef20 = p_eef27[:, :e]
    p_base3 = p_eef27[:, e : e + b]
    p_trunk4 = p_eef27[:, e + b : e + b + _TRUNK_DIM]
    p_arm16 = p_joint23[:, :_ARM_JOINT_DIM]
    return p_eef20, p_base3, p_trunk4, p_arm16


def compute_behavior_stats(dataset_dir: Path, rot6d_identity: bool = True) -> dict:
    """Stream every episode parquet → eef(20) + base_vel(3) + trunk(4) + arm_joint(16) stats.

    ``eef``/``base_vel``/``trunk`` serve the eef/unified modes (raw 27); ``arm_joint``
    is the joint-mode arm block (raw 23 = arm_joint16 + base_vel3 + trunk4). base_vel
    and trunk are shared by both modes (same native columns) so they are computed once.
    """
    eef_acc = Accumulator(dim=_EEF_DIM)
    base_acc = Accumulator(dim=_BASE_DIM)
    trunk_acc = Accumulator(dim=_TRUNK_DIM)
    arm_acc = Accumulator(dim=_ARM_JOINT_DIM)
    n_files = 0

    for fpath in _iter_episode_parquets(dataset_dir):
        try:
            df = pq.read_table(fpath, columns=_NEEDED_COLS).to_pandas()
            state = np.stack(df["observation.state"].values).astype(np.float32)
            action = np.stack(df["action"].values).astype(np.float32)
            # ACTION stream (targets) + PROPRIO stream (achieved state the reader
            # renders) — pool both into each accumulator, mirroring RoboCOIN's
            # action+state pooling so the shared stats cover the proprio gripper /
            # base / arm marginals too (not just the action command's).
            a_eef20, a_base3, a_trunk4 = _rows_to_blocks(state, action)
            a_arm16 = _rows_to_arm_joint(action)
            p_eef20, p_base3, p_trunk4, p_arm16 = _proprio_rows_to_blocks(state)
            eef_acc.update_batch(np.concatenate([a_eef20, p_eef20], axis=0))
            base_acc.update_batch(np.concatenate([a_base3, p_base3], axis=0))
            trunk_acc.update_batch(np.concatenate([a_trunk4, p_trunk4], axis=0))
            arm_acc.update_batch(np.concatenate([a_arm16, p_arm16], axis=0))
            n_files += 1
        except Exception as e:  # noqa: BLE001 — skip a corrupt shard, keep going
            print(f"  Warning: skipping {fpath}: {e}")

    if eef_acc.count == 0:
        raise RuntimeError(f"no usable episode parquet under {dataset_dir}/data — nothing to compute stats from.")

    eef = eef_acc.finalize()
    if rot6d_identity:
        # Identity on the 12 rot6d dims (3:9 / 13:19); pos + gripper keep real stats.
        _pin_rot6d_identity(eef)
    eef["num_timesteps"] = int(eef_acc.count)
    eef["num_files"] = n_files
    eef["robot_type"] = "R1Pro"
    eef["rot6d_identity"] = bool(rot6d_identity)
    eef["pool"] = "action+proprio"  # self-documents the pooled marginal (== RoboCOIN)

    base = base_acc.finalize()  # NOT pinned — real base-velocity stats
    base["num_timesteps"] = int(base_acc.count)
    base["num_files"] = n_files
    base["layout"] = "vx,vy,vyaw"
    base["pool"] = "action+proprio"

    trunk = trunk_acc.finalize()  # NOT pinned — real torso-joint stats
    trunk["num_timesteps"] = int(trunk_acc.count)
    trunk["num_files"] = n_files
    trunk["layout"] = "torso_joint_abs"
    trunk["pool"] = "action+proprio"

    arm = arm_acc.finalize()  # NOT pinned — real arm-joint stats (no rot6d in joint mode)
    arm["num_timesteps"] = int(arm_acc.count)
    arm["num_files"] = n_files
    arm["layout"] = "L_arm7,L_grip1,R_arm7,R_grip1"
    arm["pool"] = "action+proprio"

    return {"eef": eef, "base_vel": base, "trunk": trunk, "arm_joint": arm}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, help="BEHAVIOR-1K LeRobot root (has meta/info.json)")
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="Disable pinning rot6d stats to identity (rot6d would then be per-dim normalized; "
        "generally undesirable — see _pin_rot6d_identity).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.is_file():
        rtype = str(json.load(open(info_path)).get("robot_type", "R1Pro"))
        if rtype != "R1Pro":
            print(f"  Warning: info.json robot_type={rtype!r} (expected 'R1Pro'); the reader loads stats_R1Pro.json.")

    result = compute_behavior_stats(dataset_dir, rot6d_identity=not args.no_rot6d_identity)

    out_dir = dataset_dir / "meta"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "stats_R1Pro.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    eef, base, trunk, arm = result["eef"], result["base_vel"], result["trunk"], result["arm_joint"]
    print(f"\nBEHAVIOR-1K stats ({eef['num_files']} episode files):")
    print(f"  eef timesteps: {eef['num_timesteps']:,}")
    print(f"  eef pos  mean[:3]: {[round(x, 4) for x in eef['mean'][:3]]}")
    print(f"  eef grip mean[9],[19]: {round(eef['mean'][9], 4)}, {round(eef['mean'][19], 4)}")
    print(f"  eef rot6d pinned identity: {eef['rot6d_identity']}")
    print(f"  base_vel mean: {[round(x, 5) for x in base['mean']]}")
    print(f"  base_vel q01/q99: {[round(x, 4) for x in base['q01']]} / {[round(x, 4) for x in base['q99']]}")
    print(f"  trunk    mean: {[round(x, 4) for x in trunk['mean']]}")
    print(f"  arm_joint L_arm mean[:7]: {[round(x, 4) for x in arm['mean'][:7]]}")
    print(f"  arm_joint grip mean[7],[15]: {round(arm['mean'][7], 4)}, {round(arm['mean'][15], 4)}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
