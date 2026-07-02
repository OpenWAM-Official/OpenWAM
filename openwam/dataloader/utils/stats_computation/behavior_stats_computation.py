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
  * ``base_vel``  — 3-D ``[vx, vy, vyaw]`` base-frame velocity (Larchenko's mobile
                    base design). A BEHAVIOR-specific block with **real** stats —
                    NOT pinned (it's a genuine velocity, not a rotation basis).
  * ``trunk``     — 4-D absolute torso joint targets (native ``action[3:7]``). Like
                    ``base_vel``, a BEHAVIOR-specific block with **real** stats —
                    NOT pinned (genuine joint angles).
  * ``arm_joint`` — 16-D ``[L_arm7, L_grip1, R_arm7, R_grip1]`` native
                    JointController setpoints (``action[7:14]/[14]/[15:22]/[22]``),
                    the joint-mode arm block. **Real** stats, NOT pinned (no rot6d).

We compute TWO independent stat sets — ACTION (the target: gripper/base/trunk
commands, native arm setpoints) and PROPRIO (the achieved state the reader renders:
gripper open-scale, WORLD-frame base velocity, trunk qpos, arm qpos). Following the
1st-place Larchenko solution, proprio and action are a different physical quantity
per slot (the proprio gripper is a continuous open-scale vs the binary ±1 command;
the proprio base is world-frame velocity vs the local-frame command) and are
normalized SEPARATELY — a single shared set would mis-scale one. So the two are NOT
pooled: each stream gets its own eef/base_vel/trunk/arm_joint stats.

To guarantee zero layout drift, the action-stream vectors are built with the
reader's own ``_state_to_eef18`` / ``_assemble_raw`` / ``_assemble_arm_joint`` and
the proprio-stream vectors with the reader's own ``_state_to_raw_proprio_eef`` /
``_state_to_raw_proprio_joint`` — the stats are computed over the exact numbers the
reader feeds the model as action and as proprio (pre-scatter, pre-normalization).

Output schema (``meta/stats_R1Pro.json``) — ACTION blocks at top level, PROPRIO
blocks nested under ``"proprio"`` (the reader reads action from top level, proprio
from ``["proprio"]``)::

    {
      # ACTION blocks (top level); each: mean/std/min/max/q01/q99 + num_timesteps/num_files
      "eef":       {..20, "robot_type":"R1Pro", "rot6d_identity":true, "stream":"action"},
      "base_vel":  {..3,  "layout":"vx,vy,vyaw",              "stream":"action"},
      "trunk":     {..4,  "layout":"torso_joint_abs",         "stream":"action"},
      "arm_joint": {..16, "layout":"L_arm7,L_grip1,R_arm7,R_grip1", "stream":"action"},
      # PROPRIO blocks (achieved-state distribution), same block names
      "proprio":   {"eef":{..20,"rot6d_identity":true,"stream":"proprio"},
                    "base_vel":{..3}, "trunk":{..4}, "arm_joint":{..16}}
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
    (achieved WORLD-frame velocity vs local-frame cmd) and arm (achieved qpos vs
    setpoint); the eef POSE dims share the action distribution (both ``eef(state)``)."""
    p_eef27 = _state_to_raw_proprio_eef(state)  # [eef20, base3, trunk4]
    p_joint23 = _state_to_raw_proprio_joint(state)  # [arm16, base3, trunk4]
    e, b = _EEF_DIM, _BASE_DIM
    p_eef20 = p_eef27[:, :e]
    p_base3 = p_eef27[:, e : e + b]
    p_trunk4 = p_eef27[:, e + b : e + b + _TRUNK_DIM]
    p_arm16 = p_joint23[:, :_ARM_JOINT_DIM]
    return p_eef20, p_base3, p_trunk4, p_arm16


class _BlockAccumulators:
    """The four raw blocks (eef20 / base3 / trunk4 / arm_joint16) for one stream."""

    def __init__(self):
        self.eef = Accumulator(dim=_EEF_DIM)
        self.base = Accumulator(dim=_BASE_DIM)
        self.trunk = Accumulator(dim=_TRUNK_DIM)
        self.arm = Accumulator(dim=_ARM_JOINT_DIM)

    def update(self, eef20, base3, trunk4, arm16):
        self.eef.update_batch(eef20)
        self.base.update_batch(base3)
        self.trunk.update_batch(trunk4)
        self.arm.update_batch(arm16)

    def finalize(self, n_files: int, rot6d_identity: bool, stream: str) -> dict:
        eef = self.eef.finalize()
        if rot6d_identity:
            # Identity on the 12 rot6d dims (3:9 / 13:19); pos + gripper keep real stats.
            _pin_rot6d_identity(eef)
        eef.update({"num_files": n_files, "robot_type": "R1Pro", "rot6d_identity": bool(rot6d_identity)})
        base = self.base.finalize()  # NOT pinned — real base-velocity stats
        base["layout"] = "vx,vy,vyaw"
        trunk = self.trunk.finalize()  # NOT pinned — real torso-joint stats
        trunk["layout"] = "torso_joint_abs"
        arm = self.arm.finalize()  # NOT pinned — real arm-joint stats (no rot6d in joint mode)
        arm["layout"] = "L_arm7,L_grip1,R_arm7,R_grip1"
        for blk, acc in ((eef, self.eef), (base, self.base), (trunk, self.trunk), (arm, self.arm)):
            blk["num_timesteps"] = int(acc.count)
            blk["stream"] = stream  # "action" | "proprio" — self-documents which distribution
        return {"eef": eef, "base_vel": base, "trunk": trunk, "arm_joint": arm}


def compute_behavior_stats(dataset_dir: Path, rot6d_identity: bool = True) -> dict:
    """Stream every episode parquet → SEPARATE action + proprio stat sets.

    Following the 1st-place Larchenko solution, proprio (achieved state) and the action
    (command) are DIFFERENT quantities and are normalized with different stats, so we
    compute two independent sets. Each set has eef(20)/base_vel(3)/trunk(4)/arm_joint(16)
    blocks. Output schema: the ACTION blocks at top level + the PROPRIO blocks nested
    under ``"proprio"`` (what the reader's ``_load_stats`` reads back)::

        {"eef":.., "base_vel":.., "trunk":.., "arm_joint":..,          # ACTION
         "proprio": {"eef":.., "base_vel":.., "trunk":.., "arm_joint":..}}

    The action stream's base is the local-frame command; the proprio stream's base is the
    raw WORLD-frame base_qvel — hence the separate stats (a shared set would mis-scale one).
    """
    action_acc = _BlockAccumulators()
    proprio_acc = _BlockAccumulators()
    n_files = 0

    for fpath in _iter_episode_parquets(dataset_dir):
        try:
            df = pq.read_table(fpath, columns=_NEEDED_COLS).to_pandas()
            state = np.stack(df["observation.state"].values).astype(np.float32)
            action = np.stack(df["action"].values).astype(np.float32)
            a_eef20, a_base3, a_trunk4 = _rows_to_blocks(state, action)
            action_acc.update(a_eef20, a_base3, a_trunk4, _rows_to_arm_joint(action))
            proprio_acc.update(*_proprio_rows_to_blocks(state))
            n_files += 1
        except Exception as e:  # noqa: BLE001 — skip a corrupt shard, keep going
            print(f"  Warning: skipping {fpath}: {e}")

    if action_acc.eef.count == 0:
        raise RuntimeError(f"no usable episode parquet under {dataset_dir}/data — nothing to compute stats from.")

    result = action_acc.finalize(n_files, rot6d_identity, stream="action")
    result["proprio"] = proprio_acc.finalize(n_files, rot6d_identity, stream="proprio")
    return result


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

    eef, base, arm = result["eef"], result["base_vel"], result["arm_joint"]
    p_eef, p_base = result["proprio"]["eef"], result["proprio"]["base_vel"]
    print(f"\nBEHAVIOR-1K stats ({eef['num_files']} episode files); separate action + proprio sets:")
    print(f"  eef timesteps: {eef['num_timesteps']:,}")
    print(f"  [action]  eef pos mean[:3]: {[round(x, 4) for x in eef['mean'][:3]]}")
    print(f"  [action]  eef grip mean[9],[19]: {round(eef['mean'][9], 4)}, {round(eef['mean'][19], 4)}")
    print(f"  [action]  base_vel (local cmd) mean: {[round(x, 5) for x in base['mean']]}")
    print(f"  [proprio] eef grip mean[9],[19]: {round(p_eef['mean'][9], 4)}, {round(p_eef['mean'][19], 4)}")
    print(f"  [proprio] base_vel (WORLD raw) mean: {[round(x, 5) for x in p_base['mean']]}")
    print(f"  [proprio] base_vel q01/q99: {[round(x, 4) for x in p_base['q01']]} / {[round(x, 4) for x in p_base['q99']]}")
    print(f"  eef rot6d pinned identity: {eef['rot6d_identity']} (both streams)")
    print(f"  arm_joint L_arm mean[:7]: {[round(x, 4) for x in arm['mean'][:7]]}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
