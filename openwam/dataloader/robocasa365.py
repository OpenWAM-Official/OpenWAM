"""RoboCasa365 dataloader — full task set, single-arm 20-D EEF + mobile base (LeRobot v3.0).

Bespoke ``BaseDataset`` that reads the RAW downloaded RoboCasa365 LeRobot **v3.0**
aggregated repo directly (aggregated ``data/chunk-*/file-*.parquet`` +
``videos/<cam>/chunk-*/file-*.mp4`` + ``meta/episodes/*.parquet``). Shares the v3 IO
helpers (``load_episodes_parquet`` / ``compute_file_local_offsets`` /
``decode_video_frames``) with the ``LeRobotV3Reader`` benches, but is NOT a subclass
of it: that base class's "EEF read from an action column" archetype doesn't fit
RoboCasa's state-derived arm + mobile base + benchmark-deploy stats, so RoboCasa
stays a ``BaseDataset`` sibling (a faithful dual of ``robotwin.py``: same window
enumeration, multiview L-shape, per-task stats, ``Single`` + ``Multi`` pair).

v3 packs ALL tasks into one aggregated repo; each episode is tagged by
``source_prefix`` (``<split>/<atomic|composite>/<Task>/<date>``). A single-task reader
(``task_name`` given) filters the episode table to that task; the multi-task wrapper
discovers/pools tasks from the same repo. ``dataset_dir`` points at the repo exactly as
downloaded — **no migration / conversion / overlay**.

Single-arm, so the numeric EEF path is borrowed from the OXE single-arm readers
(``single_arm_20d`` / ``LEFT_ARM_DIM_MASK`` / 10-D ``eef_stats``): the canonical
20-D bimanual schema is filled LEFT-only, right 10 zero-padded + masked.

20-D EEF definition (action & proprio share ONE definition and ONE stats, like
robotwin's endpose): both are the **full base-relative** single-arm end-effector pose taken from
``observation.state``.

  ⚠️ Terminology: elsewhere this is loosely called the "absolute EEF pose", where "absolute" means
  ONLY "a FULL pose, not the env's per-step OSC *delta*" (axis 1). It does NOT mean world-frame
  (axis 2): the pose is expressed in the robot **base frame** — the state field
  ``end_effector_position_relative`` is ``robot0_base_to_eef_pos`` (eef IN base coordinates), and the
  OSC controller runs with ``input_ref_frame="base"``. So it is base-relative (scene/base-position
  invariant), NOT world-absolute. The eval bridge converts this full pose to the env's OSC delta.

::

    arm10 = [eef_pos_rel(3) + eef_rot_rel(quat->rot6d, 6) + gripper(1)]   # base-frame, full pose
          -> single-arm LEFT 10 of the canonical 20-D EEF.
    proprio = eef20d[0:1]      # current pose
    action  = eef20d[1:T]      # future-pose trajectory (model predicts full base-relative poses)

Mobile base (``mobile_base=True``): the base command is folded INTO the raw pre-unify vector, so
action and proprio share one 25-D layout ``[arm20, base5]`` and ONE unify map (like BEHAVIOR's
RAW-27) — base is no longer a bypass channel with its own stats/deploy special-casing::

    raw25 = [ arm20 (single-arm EEF, left real + right zero) , base5 ]
    action  base5 = [x_vel, y_vel, yaw_vel, torso, control_mode]   (RoboCasa-native command, raw)
    proprio base5 = [vx, vy, vyaw, 0(masked), 0(masked)]           (base_proprio="velocity", historical)
                  | [x, y, sin(yaw), cos(yaw), 0(masked)]          (base_proprio="global_pose")

The action base command is passed direct-to-env at eval (arm bridged, base raw). The proprio base
block depends on ``base_proprio``: "velocity" (historical; absent config key) = finite-diff of the
``observation.state`` base pose rescaled into the action's command space (``× fps /
_BASE_VEL_PHYS_MAX``) so it shares the action's base stats; "global_pose" = the world planar base
pose (its own ``eef_base_pose_proprio`` stats block — meters can't share command stats). torso
(constant 0 across the whole dataset) + control_mode have no achieved value, so those proprio slots
are masked. ``binary_action_dims`` (e.g. ``[9, 24]``: gripper + control_mode) keeps those two-point
±1 action targets raw and has deploy snap the decoded output back to exact ±1.
``unify_action`` scatters the whole 25-D via ``["0-9", "34-43", "68-72"]``; deploy gathers 80->25 and
un-normalizes with the single ``eef_base`` stats block. mobile_base and unify_action are decoupled
(non-unify emits the raw 25-D directly). See ``docs/plans/robocasa365-unify-raw-vector-refactor.md``.

``observation.state`` layout (16-D, from meta/modality.json)::

    base_position(0:3) + base_rotation(3:7) + eef_pos_rel(7:10)
    + eef_rot_rel(10:14, quat xyzw) + gripper_qpos(14:16)

Cameras (2-view mapping): head=``robot0_agentview_left``,
left_wrist=``robot0_eye_in_hand``, right_wrist=None -> black (single arm, no 2nd
wrist), composed into the L-shape via ``assemble_multiview_layout``.
"""

from __future__ import annotations

import functools
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import (
    assemble_multiview_layout,
    crop_and_resize,
    format_prompt_for_inference,
)
from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    assemble_single_arm_left,
    build_action_mask_2d,
    build_proprio_mask_2d,
)
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec, unmap_from_unify
from openwam.dataloader.utils.video_io import decode_video_frames

# 2-view mapping (also what the deploy server composes).
HEAD_CAMERA = "observation.images.robot0_agentview_left"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"
STATS_DIM = ARM10_DIM  # the per-arm stats are COMPUTED at 10-D, then expanded to 20-D for persist
_MISSING_RIGHT = "__missing_right_wrist__"
# Modes the DEPLOY normalizer (openwam.dataloader.transforms.normalize.YAML_TO_NORM_MODE) can
# invert. quantile is deliberately excluded — it's not in YAML_TO_NORM_MODE, so a quantile-trained
# checkpoint would silently lose normalization at serve time. Mirrors robotwin's mode guard.
_DEPLOY_RESOLVABLE_MODES = ("min-max", "z-score")

# observation.state slices (16-D).
_STATE_EEF_POS = slice(7, 10)
_STATE_EEF_ROT = slice(10, 14)  # quaternion (xyzw)

# LeRobot ``action`` field (12-D, "layout B" from PandaOmron_modality.json). The base COMMAND dims
# (raw, RoboCasa-native — NOT reconstructed from state like the arm):
#   [0:3] base x/y/yaw velocity, [3] torso lift (position 0-0.34 m), [4] control_mode {-1,+1}.
_ACTION_BASE = slice(0, 5)
BASE_ACTION_DIM = 5
_ACTION_GRIPPER = 11  # LeRobot action field gripper_close: recorded binary command {-1=open, +1=close}
BASE_VEL_DIM = 3  # the 3 base-velocity dims within base5 (proprio populates these; torso+mode masked)
# Gripper is rendered into the [-1, +1] COMMAND space (matching the env's action.gripper_close, +1=close):
#   ACTION gripper = the recorded command (exact timing, no actuation lag).
#   PROPRIO gripper = the ACHIEVED finger-separation width linearly mapped to [-1, +1] via _gripper_width_to_cmd
#     (open width _GRIPPER_WIDTH_OPEN → -1, closed 0 → +1), so both share the gripper stats and the deploy
#     bridge decides open/close by CONFIDENT-CLOSE (>0.5 → close; neutral/uncertain ~0 output defaults to
#     open, avoiding spurious grasps) — no width binarization / actuation-lag delay.
_GRIPPER_WIDTH_OPEN = 0.1  # finger-separation width mapped to -1 (fully open); 0 (closed) → +1
# The mobile raw vector folds the base command INTO the pre-unify vector (like BEHAVIOR's RAW-27):
#   raw25 = [arm20 (single-arm EEF, left real + right zero), base5]. unify then maps the WHOLE 25-D via
# one map (["0-9","34-43","68-72"]); base is NOT a bypass channel. Deploy gathers 80->25 and
# un-normalizes with ONE stats block — no base special-casing (see openwam/deploy/model_loader.py).
RAW_MOBILE_DIM = EEF_DIM + BASE_ACTION_DIM  # 25
# Deploy stats key for the combined 25-D mobile vector. NOT 'eef' (that is the repo-level 20-D bimanual
# schema shared by robotwin/OXE via materialize_eef_stats); the wider mobile vector gets its own key
# (mirrors behavior.py's per-mode DEPLOY_ACTION_MODE keys). Non-mobile stays 20-D under 'eef'.
_MOBILE_STATS_KEY = "eef_base"
DATASET_FPS = 20  # v3 atomic + composite are both fps=20 (asserted == meta/info.json on load for mobile)
# A′ base-velocity rescale: per-axis physical base speed at command saturation (measured p99.9 over
# NavigateKitchen). The proprio finite-diff base velocity (m/frame) is rescaled into the action's
# [-1, 1] command space via ``× fps / PHYS_MAX`` so the ACHIEVED proprio velocity and the COMMANDED
# action base velocity share ONE stats block (the dual of BEHAVIOR's _BASE_VEL_OUTPUT_SCALE). Without
# this the two differ ~30× in scale and cannot share stats. [vx m/s, vy m/s, vyaw rad/s].
_BASE_VEL_PHYS_MAX = np.array([0.75, 0.88, 1.33], np.float32)
# 25-D raw dim masks (arm left-valid / right-masked, then base). ACTION supervises the 5 base command
# dims; PROPRIO only the 3 base-velocity dims are observable (torso + control_mode have no achieved
# value → masked, zero-filled). Scattered to 80-D through the unify map when unify_action.
#   torso (base idx 3, raw idx 23) is CONSTANT 0 across the whole dataset (29.1M frames) yet a LIVE sim
#   actuator (base_motion[3] → robot0_torso JOINT_POSITION delta). Predicting it risks a nonzero output
#   driving the torso at eval, so ``mask_torso_action`` (default true) masks it out of the action loss;
#   the eval client then zeros it before the env (see the interface). control_mode stays supervised.
_ARM_MASK = np.asarray(LEFT_ARM_DIM_MASK, dtype=bool)
_RAW_ACTION_MASK = np.concatenate([_ARM_MASK, np.array([True, True, True, True, True])])  # all base dims
_RAW_ACTION_MASK_NO_TORSO = np.concatenate([_ARM_MASK, np.array([True, True, True, False, True])])  # torso[3] masked
_RAW_PROPRIO_MASK = np.concatenate([_ARM_MASK, np.array([True, True, True, False, False])])
# base_proprio="global_pose": proprio base5 = [x, y, sin(yaw), cos(yaw), 0] (world planar base pose;
# slots 0-3 observable, slot 4 spare). sin/cos instead of raw yaw: no ±π seam — the planar reduction
# of GR00T's rot6d base_rotation. Pose is meters/unit-circle, NOT command space, so the proprio can
# no longer share the action's base stats: a separate proprio stats block is required (below).
_RAW_PROPRIO_MASK_POSE = np.concatenate([_ARM_MASK, np.array([True, True, True, True, False])])
_PROPRIO_POSE_STATS_KEY = "eef_base_pose_proprio"
# The only two-point {-1, +1} command dims in raw25 (l_grip cmd, control_mode). binary_action_dims
# may list these: their targets stay the RAW ±1 (identity, immune to stats drift) and the deploy
# server snaps its decoded output back to exact ±1 (see openwam/deploy/model_loader.py).
_BINARY_ACTION_DIMS_ALLOWED = (ARM10_DIM - 1, RAW_MOBILE_DIM - 1)  # (9, 24)

# Multiview L-shape slot sizes (must match assemble_multiview_layout defaults at
# height=384/width=320: top 256x320, each bottom 128x160).
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320
_WRIST_SLOT_H, _WRIST_SLOT_W = 128, 160


def _task_dir_name(data_root: str) -> str:
    """Fallback display / stats-file name when ``task_name`` is unset: the v3 repo's directory name
    (a single-task reader normally gets an explicit ``task_name`` that filters the repo)."""
    return os.path.basename(data_root.rstrip("/")) or "robocasa365"


def _task_from_source_prefix(prefix: str) -> str:
    """Task name from a v3 ``source_prefix`` like ``pretrain/atomic/OpenDrawer/20250819`` -> ``OpenDrawer``.

    v3 aggregates all tasks into one repo, tagging each episode's origin task with ``source_prefix``
    (``<split>/<atomic|composite>/<Task>/<date>``); the task is the second-to-last path segment.
    """
    parts = str(prefix).strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else str(prefix)


@functools.lru_cache(maxsize=8)
def _episodes_with_offsets(data_root: str):
    """Load the v3 episodes table + compute file-local row/frame offsets ONCE per repo (process-global
    cache). The multi-task wrapper (``_resolve_task_roots``) AND every per-task sub-dataset need the same
    table; without this a full 300-task run re-reads + re-offsets it 300+ times at startup. Offsets are
    over the FULL table (before any single-task filter) so a task that isn't first in its shard reads at
    its true file-local offset. Callers MUST treat the returned frame as read-only (filter to a copy)."""
    eps = load_episodes_parquet(Path(data_root))
    eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
    for cam in (HEAD_CAMERA, WRIST_CAMERA):
        if f"videos/{cam}/chunk_index" in eps.columns:
            eps[f"_voff/{cam}"] = compute_file_local_offsets(
                eps, f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
            )
    return eps


@functools.lru_cache(maxsize=8)
def _read_shard_cached(path: str) -> pd.DataFrame:
    """Process-global cache of one decoded v3 data shard (state+action columns), shared across ALL
    RoboCasa365 sub-datasets in the process. v3 aggregated shards are shared by many tasks, so a
    per-instance cache would hold one copy PER task per DataLoader worker (a memory bomb at 300-task
    scale); a module-level cache keeps it to one copy per shard per worker, bounded by maxsize. Mirrors
    the process-global shard cache ``LeRobotV3Reader`` adopted (commit 9b95134)."""
    return pd.read_parquet(path, columns=["observation.state", "action"])


def _compute_shared_stats_rank0_synced(shared_path: str, roots: list, include_base: bool = False) -> None:
    """Compute + persist the shared multitask stats with rank-0 synchronization.

    ``roots`` is ``[(task_name, repo), ...]`` (one per selected task, paired with its repo). On a
    multi-GPU first run, only rank 0 computes + atomically writes; other ranks poll for the file
    (mirrors robotwin). Without this, every rank races to write the same ``{path}.tmp`` → torn writes
    + N× redundant compute over all tasks. ``include_base`` emits the combined 25-D ``eef_base`` block
    (arm20 + base5 command) instead of the arm-only 20-D ``eef`` block.
    """
    from openwam.dataloader.utils.stats_computation.robocasa365_stats_computation import (
        atomic_save_stats_npy,
        compute_multitask_stats,
    )

    try:
        import torch.distributed as dist

        dist_ready = dist.is_available() and dist.is_initialized()
    except Exception:
        dist_ready = False
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

    if not dist_ready or rank == 0:
        print(f"  [normalizer] computing SHARED multitask stats over {len(roots)} tasks -> {shared_path}")
        atomic_save_stats_npy(shared_path, compute_multitask_stats(roots, include_base=include_base))
        return
    # non-rank0: wait for rank 0 to produce the file
    import time

    deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
    while not os.path.exists(shared_path):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for rank 0 to produce shared stats: {shared_path}")
        time.sleep(float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10)))


def _gripper_width_to_cmd(width: np.ndarray) -> np.ndarray:
    """Achieved finger-separation width → [-1, +1] gripper COMMAND space (closed→+1, open→-1), matching
    the RoboCasa ``action.gripper_close`` convention (+1=close). Linear over ``[0, _GRIPPER_WIDTH_OPEN]``,
    clipped. The eval client reproduces this exactly (benchmarks/utils.robocasa_state_to_eef20d)."""
    return np.clip(1.0 - 2.0 * np.asarray(width) / _GRIPPER_WIDTH_OPEN, -1.0, 1.0).astype(np.float32)


def state_to_arm10(state: np.ndarray) -> np.ndarray:
    """``(T, 16)`` observation.state -> ``(T, 10)`` single-arm EEF (raw, unnormalized).

    arm10 = [eef_pos_rel(3), rot6d(eef_rot_rel quat xyzw, 6), gripper(1)]. The gripper is the ACHIEVED
    finger-separation width ``qpos[0] - qpos[1]`` rendered into the [-1, +1] command space
    (``_gripper_width_to_cmd``: open→-1, closed→+1). This is the PROPRIO gripper; the ACTION gripper is
    replaced with the recorded command in ``_build_sample`` (both live in the same command space)."""
    pos = state[:, _STATE_EEF_POS]
    rot6d = quat_xyzw_to_rotation_6d(state[:, _STATE_EEF_ROT])
    grip = _gripper_width_to_cmd(state[:, 14] - state[:, 15])[:, None]
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Yaw (rotation about world +z) from a quaternion ``(x, y, z, w)``."""
    x, y, z, w = (float(v) for v in np.asarray(q).reshape(-1)[:4])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _base_velocity_body(base_pose: np.ndarray) -> np.ndarray:
    """Body-frame base velocity from 2 consecutive base poses (finite difference).

    ``base_pose`` = ``(2, 7)`` rows ``[prev, cur]``, each ``base_position(3, world) +
    base_rotation(4, world quat xyzw)``. Returns ``(3,)`` = ``[vx, vy, vyaw]`` in the robot's body
    frame at ``cur`` (per-step displacement; the constant 1/dt scale is absorbed by normalization).
    SE(2): z + roll/pitch are ignored (ground base). Δyaw is wrapped to (-pi, pi].
    """
    prev = np.asarray(base_pose[0], np.float64)
    cur = np.asarray(base_pose[1], np.float64)
    d_world = cur[0:2] - prev[0:2]  # world planar displacement
    yaw_cur, yaw_prev = _yaw_from_quat_xyzw(cur[3:7]), _yaw_from_quat_xyzw(prev[3:7])
    c, s = np.cos(yaw_cur), np.sin(yaw_cur)
    vx = c * d_world[0] + s * d_world[1]  # R(-yaw_cur) @ d_world -> body frame
    vy = -s * d_world[0] + c * d_world[1]
    d_yaw = np.arctan2(np.sin(yaw_cur - yaw_prev), np.cos(yaw_cur - yaw_prev))  # wrapped Δyaw
    return np.array([vx, vy, d_yaw], np.float32)


def base_velocity_cmd(base_pose: np.ndarray, fps: int = DATASET_FPS) -> np.ndarray:
    """Body-frame base velocity finite-diff rescaled into the action's [-1, 1] command space (A′).

    ``_base_velocity_body`` returns per-FRAME body displacement (m/frame, rad/frame); ``× fps`` gives
    physical velocity (m/s, rad/s) and ``÷ _BASE_VEL_PHYS_MAX`` (per-axis base max speed at command
    saturation) lands it in the SAME space as the recorded action base-velocity command, so proprio
    velocity and action command normalize with ONE base stats block. ``base_pose`` = ``(2, 7)``
    ``[prev, cur]``. The eval client reproduces this exactly (benchmarks/utils.base_velocity_cmd)."""
    return (_base_velocity_body(base_pose) * float(fps) / _BASE_VEL_PHYS_MAX).astype(np.float32)


def _expand_stats_to_20d(s10: dict) -> dict:
    """Left-pad a 10-D arm stats dict into the 20-D bimanual schema.

    The right arm carries no data (always 0, masked), so its stats are NEUTRAL:
    mean=0/std=1/min=-1/max=1/q01=-1/q99=1 — i.e. a normalized 0 maps back to 0.
    The 20-D result is what gets persisted to the checkpoint (via
    ``normalization_stats_path``) and read back by the deploy normalizer, and is
    the same dict returned by the ``normalization_stats`` property.
    """

    def pad(arm, neutral):
        return np.concatenate([np.asarray(arm, np.float32), np.full(ARM10_DIM, neutral, np.float32)])

    return {
        "mean": pad(s10["mean"], 0.0),
        "std": pad(s10["std"], 1.0),
        "min": pad(s10["min"], -1.0),
        "max": pad(s10["max"], 1.0),
        "q01": pad(s10["q01"], -1.0),
        "q99": pad(s10["q99"], 1.0),
    }


class RoboCasa365Dataset(BaseDataset):
    """Single-task RoboCasa365 reader (raw LeRobot v3.0 aggregated repo, single-arm 20-D EEF).

    Mirrors ``RoboTwinDataset``: exhaustive ``(episode, start)`` window enumeration,
    deterministic train/val split, multiview L-shape composition, per-task action
    normalization with auto-compute on first use. RoboCasa-specific: reads the v3
    aggregated shards (``task_name`` filters the repo to one task via ``source_prefix``)
    and assembles the 20-D EEF from ``observation.state`` (see module docstring).
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 33,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        val_ratio: float = 0.0,
        repeat: int = 1,
        task_name: Optional[str] = None,
        seed: int = 42,
        normalization_stats_path: Optional[str] = None,
        normalize_mode: Optional[str] = "min-max",
        num_val_samples: int = 4,
        window_stride: int = 1,
        video_stride: int = 4,
        multiview: bool = True,
        camera_layout: Optional[list] = None,
        temporal_compression: int = 4,
        causal_temporal: bool = True,
        filter_static_segments: bool = True,
        static_segment_threshold: float = 1e-5,
        max_static_retry: int = 3,
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        mobile_base: bool = False,
        mask_torso_action: bool = True,
        base_proprio: str = "velocity",
        binary_action_dims: Optional[Any] = None,
        color_jitter: Optional[Any] = None,
        **_unused,
    ):
        super().__init__()
        self.data_root = data_root
        self.task_name = task_name or _task_dir_name(data_root)
        self.normalize_mode = normalize_mode if normalize_mode not in ("", "none", "null") else None
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        self.num_frames = int(num_frames)
        self.num_action_steps = self.num_frames - 1
        self.height = int(height)
        self.width = int(width)
        # Resolution guard (mirrors robotwin): VAE downsamples by 16, patch size 2,
        # so both dims must be divisible by 32 or latent shapes silently break.
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(
                f"Resolution {self.height}x{self.width} must be divisible by 32 (VAE downsamples by 16, patch size 2)."
            )
        self.repeat = int(repeat)
        self.split = split
        # Load-time color jitter (same random factors across the whole clip, via VideoColorJitter);
        # train-only, disabled / val keeps video byte-identical (mirrors robotwin).
        self._color_jitter = None
        if color_jitter and split == "train":
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.2)),
                contrast=float(cj_get("contrast", 0.2)),
                saturation=float(cj_get("saturation", 0.2)),
                hue=float(cj_get("hue", 0.0)),
            )
        # Static-segment filtering (mirrors robotwin): resample train windows that "haven't
        # started moving" so the model isn't taught to output ~zero motion.
        self._filter_static_segments = bool(filter_static_segments)
        self._static_segment_threshold = float(static_segment_threshold)
        self._max_static_retry = int(max_static_retry)
        # Unified 80-D action space (opt-in; mirrors the OXE/BEHAVIOR base-reader path). The raw
        # pre-unify vector is the arm-only 20-D EEF, or 25-D [arm20, base5] when mobile_base. When
        # unify_action, the WHOLE raw vector scatters into UNIFY_DIM via ONE map (base included, not a
        # bypass channel); the per-dim raw masks (arm left/right + base) are honored through the
        # scatter. mobile_base and unify_action are DECOUPLED — non-unify emits the raw 20/25-D head.
        self._unify_action = bool(unify_action)
        self._unify_action_map = unify_action_map
        # Mobile base: fold the RoboCasa-native base command (x/y/yaw vel + torso + control_mode) from
        # the LeRobot ``action`` field INTO the raw vector as base5; proprio carries the A′-rescaled
        # body-frame base velocity in those 3 velocity slots (torso + control_mode masked).
        self._mobile_base = bool(mobile_base)
        # torso is a live sim actuator but constant 0 in the data → mask it out of the ACTION loss
        # (default) so a nonzero prediction can't drive it at eval; the eval client zeros it too.
        self._mask_torso_action = bool(mask_torso_action)
        # base_proprio: what fills the 5 proprio base slots. "velocity" (historical; a ckpt config
        # without the key trained this way) = A′-rescaled finite-diff body velocity in slots 0-2;
        # "global_pose" = world planar pose [x, y, sin(yaw), cos(yaw), 0] (needs its own stats block).
        if base_proprio not in ("velocity", "global_pose"):
            raise ValueError(f"base_proprio must be 'velocity' or 'global_pose', got {base_proprio!r}")
        if base_proprio == "global_pose" and not mobile_base:
            raise ValueError("base_proprio='global_pose' requires mobile_base=True (there is no base5 proprio block)")
        self._base_proprio = base_proprio
        # binary_action_dims: raw dims whose targets are the two-point {-1, +1} command set. Identity
        # in training (immune to stats drift); the deploy server snaps decoded outputs back to ±1.
        if binary_action_dims is None:
            self._binary_action_dims: tuple = ()
        else:
            dims = tuple(int(d) for d in binary_action_dims)
            bad = [d for d in dims if d not in _BINARY_ACTION_DIMS_ALLOWED]
            if bad:
                raise ValueError(
                    f"binary_action_dims {bad} not in the two-point command dims {_BINARY_ACTION_DIMS_ALLOWED} "
                    "(l_grip cmd, control_mode); other dims are continuous and must not be snapped."
                )
            if RAW_MOBILE_DIM - 1 in dims and not mobile_base:
                raise ValueError("binary_action_dims includes control_mode (24) but mobile_base=False (raw is 20-D)")
            self._binary_action_dims = dims
        self._raw_dim = RAW_MOBILE_DIM if self._mobile_base else EEF_DIM  # 25 or 20
        if self._mobile_base:
            self._raw_action_mask = _RAW_ACTION_MASK_NO_TORSO if self._mask_torso_action else _RAW_ACTION_MASK
            self._raw_proprio_mask = _RAW_PROPRIO_MASK_POSE if self._base_proprio == "global_pose" else _RAW_PROPRIO_MASK
        else:
            self._raw_action_mask = _ARM_MASK
            self._raw_proprio_mask = _ARM_MASK
        self._unify_dst_index = None
        self._unify_dim_mask = None  # proprio dim mask (80-D, scattered from _raw_proprio_mask)
        self._unify_action_dim_mask = None  # action dim mask (80-D, scattered from _raw_action_mask)
        if self._unify_action:
            spec = self._unify_action_map if self._unify_action_map is not None else list(range(self._raw_dim))
            self._unify_dst_index = parse_unify_spec(spec, UNIFY_DIM)
            if self._unify_dst_index.shape[0] != self._raw_dim:
                raise ValueError(
                    f"robocasa365 unify_action_map maps {self._unify_dst_index.shape[0]} source dims but the raw "
                    f"action is {self._raw_dim}-D ({'arm20+base5' if self._mobile_base else 'arm20'}); they must match."
                )
            # Scatter the raw per-dim masks into the unified space (right-arm + unmapped slots stay masked;
            # proprio also masks torso + control_mode, which have no achieved value).
            self._unify_dim_mask = np.zeros(UNIFY_DIM, dtype=bool)
            self._unify_dim_mask[self._unify_dst_index] = self._raw_proprio_mask
            self._unify_action_dim_mask = np.zeros(UNIFY_DIM, dtype=bool)
            self._unify_action_dim_mask[self._unify_dst_index] = self._raw_action_mask
        self.window_stride = max(1, int(window_stride))
        self.video_stride = max(1, int(video_stride))
        if (self.num_frames - 1) % self.video_stride != 0:
            valid = [s for s in range(1, self.num_frames) if (self.num_frames - 1) % s == 0]
            raise ValueError(
                f"(num_frames - 1) must be divisible by video_stride. Got num_frames={self.num_frames}, "
                f"video_stride={self.video_stride}. Valid strides: {valid}"
            )
        self._video_sample_indices = list(range(0, self.num_frames, self.video_stride))
        self.num_video_frames = len(self._video_sample_indices)
        # Encoder temporal contract (mirrors robotwin/base reader, which no longer enforce this at
        # runtime): for clean downsampling causal encoders want (num_video_frames - 1) % tc == 0,
        # non-causal want num_video_frames % tc == 0 — documented, not enforced. temporal_compression
        # / causal_temporal are accepted for config compatibility.
        self.multiview = bool(multiview)
        # Camera layout: head (top), wrist (bot-left), missing right (bot-right=black).
        self.camera_layout = list(camera_layout) if camera_layout else [HEAD_CAMERA, WRIST_CAMERA, _MISSING_RIGHT]

        # ── info.json: v3 aggregated path templates ───────────────────────
        with open(os.path.join(data_root, "meta", "info.json")) as f:
            info = json.load(f)
        self._data_path_tmpl = info["data_path"]  # data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
        self._video_path_tmpl = info[
            "video_path"
        ]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
        # fps for the A′ base-velocity rescale (proprio finite-diff m/frame → m/s → command space). The
        # eval client hardcodes DATASET_FPS, so a mobile run on a differently-sampled repo would diverge
        # train↔eval — fail loud (no silent fallback) rather than silently mis-scale the base velocity.
        if "fps" not in info:
            raise ValueError(f"meta/info.json under {data_root} has no 'fps'.")
        self._fps = int(info["fps"])
        if self._mobile_base and self._fps != DATASET_FPS:
            raise ValueError(
                f"RoboCasa365 mobile_base pins the A′ base-velocity rescale (and the eval client) to "
                f"fps={DATASET_FPS}; this repo's meta/info.json says fps={self._fps}. A different rate "
                "would diverge train↔eval — re-verify _BASE_VEL_PHYS_MAX and the eval client fps."
            )

        # ── episodes (v3 aggregated meta) + single-task filter by source_prefix ──
        # v3 packs ALL tasks into one repo; ``source_prefix`` tags each episode's origin task, and the
        # aggregated data/video shards mix tasks. A single-task reader (task_name given) filters the
        # episode table to that task; the rest of the pipeline (window enumeration, split, reads) is
        # unchanged — it just sees a filtered episode set. Reads resolve each episode's (chunk, file) +
        # file-local row/frame offset (mirrors LeRobotV3Reader; offsets via compute_file_local_offsets).
        # Episodes table + file-local offsets, cached once per repo (see _episodes_with_offsets;
        # offsets are over the FULL table so a task not first in its shard reads at its true offset).
        # The cached frame is shared/read-only — filter to a single task on a COPY (reset_index).
        eps = _episodes_with_offsets(data_root)
        if task_name is not None:
            eps = eps[eps["source_prefix"].map(_task_from_source_prefix) == self.task_name].reset_index(drop=True)
        if len(eps) == 0:
            raise FileNotFoundError(f"No episodes for task {self.task_name!r} under {data_root}")
        # Per-episode dicts for window enumeration/split (episode_index/length/tasks) + a read-metadata
        # map keyed by episode_index (aggregated (chunk,file) + file-local row/frame offsets).
        episodes, self._ep_meta = [], {}
        for _, r in eps.iterrows():
            epi = int(r["episode_index"])
            episodes.append({"episode_index": epi, "length": int(r["length"]), "tasks": list(r["tasks"])})
            self._ep_meta[epi] = {
                "chunk": int(r["data/chunk_index"]),
                "file": int(r["data/file_index"]),
                "row_offset": int(r["_data_row_offset"]),
                "voff": {c: int(r[f"_voff/{c}"]) for c in (HEAD_CAMERA, WRIST_CAMERA) if f"_voff/{c}" in eps.columns},
                "vcf": {
                    c: (int(r[f"videos/{c}/chunk_index"]), int(r[f"videos/{c}/file_index"]))
                    for c in (HEAD_CAMERA, WRIST_CAMERA)
                    if f"videos/{c}/chunk_index" in eps.columns
                },
            }
        self._episodes = episodes

        # ── deterministic train/val split ─────────────────────────────────
        rng = random.Random(seed)
        order = list(range(len(episodes)))
        rng.shuffle(order)
        if val_ratio <= 0.0:
            n_val = 0
        elif val_ratio >= 1.0:
            n_val = len(episodes)
        else:
            n_val = max(1, int(len(episodes) * val_ratio))
        selected = sorted(order[:n_val]) if split == "val" else sorted(order[n_val:])
        self._ep_pos = selected  # positions into self._episodes for this split
        if not self._ep_pos:
            raise ValueError(f"No episodes for split='{split}' (val_ratio={val_ratio}, {len(episodes)} total)")
        self._ep_lengths = [int(episodes[p]["length"]) for p in self._ep_pos]
        print(f"RoboCasa365Dataset[{self.task_name}]: {len(self._ep_pos)} episodes ({split})")

        # ── exhaustive window enumeration (FastWAM tail semantics) ────────
        self._window_index = []
        for local_idx, ep_len in enumerate(self._ep_lengths):
            if ep_len < 2:
                continue
            max_start = max(0, ep_len - self.num_frames) if split == "val" else max(0, ep_len - 2)
            for start in range(0, max_start + 1, self.window_stride):
                self._window_index.append((local_idx, start))
        if self.repeat > 1:
            self._window_index = self._window_index * self.repeat
        if not self._window_index:
            raise ValueError(f"No valid windows for split='{split}' in {data_root}")

        # ── fixed val samples ─────────────────────────────────────────────
        self._val_samples = None
        if split == "val" and num_val_samples > 0:
            vr = random.Random(seed + 1)
            eligible = [i for i, ep_len in enumerate(self._ep_lengths) if ep_len >= 2]
            self._val_samples = []
            for _ in range(num_val_samples):
                li = eligible[vr.randint(0, len(eligible) - 1)]
                self._val_samples.append((li, vr.randint(0, max(0, self._ep_lengths[li] - self.num_frames))))

        # ── action normalization (ONE combined stats block; auto-compute if missing) ─
        # Mobile → 25-D 'eef_base' ([arm20, base5]); non-mobile → 20-D 'eef' (repo-level bimanual
        # schema). The whole raw vector normalizes with this single block, so at deploy the server
        # gathers 80->raw and un-normalizes with it — no base special-casing (wayrise convergence).
        # base_proprio="global_pose" is the one exception: pose (meters) cannot share the command-space
        # base stats, so the PROPRIO normalizes with its own 'eef_base_pose_proprio' block instead
        # (arm20 dims identical to 'eef_base'; base dims are pose stats) — dual of robocasa-gr1's
        # separate hand command/state stats.
        self._stats: Optional[dict] = None
        self._proprio_stats: Optional[dict] = None
        self.normalization_stats_path: Optional[str] = None
        if self.normalize_mode is not None:
            if self.normalize_mode not in _DEPLOY_RESOLVABLE_MODES:
                raise ValueError(
                    f"normalize_mode={self.normalize_mode!r} is not deploy-resolvable. "
                    f"Use one of {sorted(_DEPLOY_RESOLVABLE_MODES)} or null — a mode the deploy "
                    "normalizer (YAML_TO_NORM_MODE) can invert, else the checkpoint silently "
                    "loses normalization at serve time."
                )
            stats_path = self._resolve_stats_path(normalization_stats_path)
            if stats_path.endswith(".json"):
                with open(stats_path) as f:
                    full = json.load(f)
            else:
                full = np.load(stats_path, allow_pickle=True).item()
            if self._mobile_base:
                raw = full.get(_MOBILE_STATS_KEY) if isinstance(full, dict) else None
                if raw is None:
                    raise ValueError(
                        f"mobile_base=True but stats file {stats_path} has no {_MOBILE_STATS_KEY!r} block "
                        f"(the combined 25-D [arm20, base5]). Recompute stats "
                        "(robocasa365_stats_computation --mobile-base emits it)."
                    )
                self._stats = {k: np.asarray(raw[k], np.float32).reshape(-1) for k in ("mean", "std", "min", "max")}
                if self._stats["mean"].shape[0] != RAW_MOBILE_DIM:
                    raise ValueError(
                        f"{_MOBILE_STATS_KEY} stats dim {self._stats['mean'].shape[0]} != {RAW_MOBILE_DIM}; "
                        "recompute stats."
                    )
                self._proprio_stats = self._stats
                if self._base_proprio == "global_pose":
                    raw_p = full.get(_PROPRIO_POSE_STATS_KEY) if isinstance(full, dict) else None
                    if raw_p is None:
                        raise ValueError(
                            f"base_proprio='global_pose' but stats file {stats_path} has no "
                            f"{_PROPRIO_POSE_STATS_KEY!r} block (arm20 + planar base pose "
                            "[x, y, sin_yaw, cos_yaw, 0]). Recompute stats "
                            "(robocasa365_stats_computation --mobile-base emits it)."
                        )
                    self._proprio_stats = {
                        k: np.asarray(raw_p[k], np.float32).reshape(-1) for k in ("mean", "std", "min", "max")
                    }
                    if self._proprio_stats["mean"].shape[0] != RAW_MOBILE_DIM:
                        raise ValueError(
                            f"{_PROPRIO_POSE_STATS_KEY} stats dim {self._proprio_stats['mean'].shape[0]} != "
                            f"{RAW_MOBILE_DIM}; recompute stats."
                        )
            else:
                eef_raw = full.get("eef", full) if isinstance(full, dict) else full  # flat or {"eef": {...}}
                self._stats = materialize_eef_stats(
                    eef_raw, self.normalize_mode, dim=EEF_DIM, strict_minmax=True, source_hint=stats_path
                )
                self._proprio_stats = self._stats
            self.normalization_stats_path = stats_path
            print(
                f"  [normalizer] {self.normalize_mode}, dim={self._raw_dim}"
                f"{' (arm20+base5)' if self._mobile_base else ''}, stats={stats_path}"
            )
        else:
            print("  [normalizer] DISABLED (normalize_mode=None)")

    def _resolve_stats_path(self, explicit: Optional[str]) -> str:
        """Explicit path wins; else ``{data_root}/{task}_{eef|eefbase}_stats.npy`` (auto-compute).

        Mobile runs use a distinct ``_eefbase_`` suffix so they never load a stale arm-only
        ``_eef_`` file (which lacks the required 25-D ``eef_base`` block).
        """
        if explicit and os.path.exists(explicit):
            return explicit
        suffix = "eefbase" if self._mobile_base else "eef"
        stats_path = os.path.join(self.data_root, f"{self.task_name}_{suffix}_stats.npy")
        if not os.path.exists(stats_path):
            from openwam.dataloader.utils.stats_computation.robocasa365_stats_computation import (
                atomic_save_stats_npy,
                compute_normalization_stats,
            )

            print(
                f"  [normalizer] computing arm-10{' + base5 (combined eef_base)' if self._mobile_base else ''} "
                f"stats from {self.data_root} -> {stats_path}"
            )
            atomic_save_stats_npy(
                stats_path,
                compute_normalization_stats(self.data_root, include_base=self._mobile_base, task_name=self.task_name),
            )
        return stats_path

    # ----- BaseDataset interface -----

    @property
    def action_dim(self) -> int:
        # unify → UNIFY_DIM (80); else the raw head width (25 mobile / 20 arm-only).
        return UNIFY_DIM if self._unify_action else self._raw_dim

    @property
    def normalization_stats(self) -> Optional[dict]:
        """The combined raw-width stats dict (25-D ``[arm20, base5]`` mobile / 20-D arm-only) — the
        SAME dict persisted to the checkpoint (via ``normalization_stats_path``) and read by the deploy
        normalizer. Also surfaced through the multi-task wrapper's ``normalization_stats`` (mirrors
        robotwin). None when normalization is disabled."""
        return dict(self._stats) if self._stats is not None else None

    def _unnormalize(self, arr: np.ndarray, stats: dict) -> np.ndarray:
        """Invert min-max / z-score with the given stats (no-op for other modes)."""
        arr = np.asarray(arr, dtype=np.float32)
        if stats is None or self.normalize_mode is None:
            return arr.copy()
        if self.normalize_mode == "z-score":
            return (arr * stats["std"] + stats["mean"]).astype(np.float32)
        if self.normalize_mode == "min-max":
            return ((arr + 1.0) * 0.5 * (stats["max"] - stats["min"]) + stats["min"]).astype(np.float32)
        return arr.copy()

    def denormalize_action(self, action) -> np.ndarray:
        """Invert normalization + un-unify for the active mode (no-op when disabled).

        Generic: gather the unified 80-D back to the raw width via the ONE map, then un-normalize with
        the single combined stats block (NO base special-casing — the same contract as the deploy
        ``_UnifyAwareNormalizer``). Returns the raw RoboCasa action the client bridges:
          * unify off → the raw head (20-D arm EEF / 25-D ``[arm20, base5]`` mobile).
          * unify on  → un-unified raw (80-D → 20-D / 25-D).
        base5 = raw ``[x/y/yaw vel, torso, control_mode]``; its stats live inside the combined block.
        """
        arr = np.asarray(action, dtype=np.float32)
        if self._unify_dst_index is not None:
            arr = unmap_from_unify(arr, self._unify_dst_index).astype(np.float32)  # (..., 80) -> (..., raw_dim)
        return self._unnormalize(arr, self._stats)

    def __len__(self) -> int:
        return len(self._val_samples) if self._val_samples is not None else len(self._window_index)

    # ----- IO helpers -----

    def _data_file_path(self, chunk: int, file: int) -> str:
        return os.path.join(self.data_root, self._data_path_tmpl.format(chunk_index=chunk, file_index=file))

    def _video_path(self, camera: str, chunk: int, file: int) -> str:
        return os.path.join(
            self.data_root, self._video_path_tmpl.format(video_key=camera, chunk_index=chunk, file_index=file)
        )

    def _read_state(self, ep_global_idx: int, start: int, end: int) -> np.ndarray:
        m = self._ep_meta[ep_global_idx]
        df = _read_shard_cached(self._data_file_path(m["chunk"], m["file"]))  # process-global shard cache
        o = m["row_offset"]
        return np.stack(df["observation.state"].values[o + start : o + end]).astype(np.float32)  # (n, 16)

    def _read_base_action(self, ep_global_idx: int, start: int, end: int) -> np.ndarray:
        """Base command window [start, end) from the LeRobot ``action`` field: ``(n, 5)`` =
        [x_vel, y_vel, yaw_vel, torso, control_mode] (RoboCasa-native, raw, mobile only)."""
        m = self._ep_meta[ep_global_idx]
        df = _read_shard_cached(self._data_file_path(m["chunk"], m["file"]))
        o = m["row_offset"]
        return np.stack(df["action"].values[o + start : o + end]).astype(np.float32)[:, _ACTION_BASE]  # (n, 5)

    def _read_gripper_command(self, ep_global_idx: int, start: int, end: int) -> np.ndarray:
        """Recorded gripper command window [start, end) from the LeRobot ``action`` field: ``(n, 1)`` =
        gripper_close (binary {-1=open, +1=close}). The ACTION gripper target — exact timing, no
        actuation lag (unlike the achieved width). Aligned to action steps like the base command."""
        m = self._ep_meta[ep_global_idx]
        df = _read_shard_cached(self._data_file_path(m["chunk"], m["file"]))
        o = m["row_offset"]
        col = np.stack(df["action"].values[o + start : o + end]).astype(np.float32)[:, _ACTION_GRIPPER]
        return col[:, None]  # (n, 1)

    def _read_video(self, ep_global_idx: int, start: int, actual_end: int):
        """Decode the window's sampled frames into a list of L-shape canvases (or single-view PIL
        frames). v3 stores aggregated mp4s, so an absolute frame index = the episode's file-local
        frame offset within its (chunk,file) video shard + the episode-local index."""
        m = self._ep_meta[ep_global_idx]
        local = [start + i for i in self._video_sample_indices if start + i < actual_end]
        if self.multiview:
            hc, hf = m["vcf"][HEAD_CAMERA]
            wc, wf = m["vcf"][WRIST_CAMERA]
            head = decode_video_frames(
                self._video_path(HEAD_CAMERA, hc, hf),
                [m["voff"][HEAD_CAMERA] + a for a in local],
                _HEAD_SLOT_H,
                _HEAD_SLOT_W,
            )
            wrist = decode_video_frames(
                self._video_path(WRIST_CAMERA, wc, wf),
                [m["voff"][WRIST_CAMERA] + a for a in local],
                _WRIST_SLOT_H,
                _WRIST_SLOT_W,
            )
            frames = [
                assemble_multiview_layout(
                    {HEAD_CAMERA: head[fi], WRIST_CAMERA: wrist[fi]}, self.camera_layout, self.height, self.width
                )
                for fi in range(len(local))
            ]
        else:
            hc, hf = m["vcf"][HEAD_CAMERA]
            head = decode_video_frames(
                self._video_path(HEAD_CAMERA, hc, hf),
                [m["voff"][HEAD_CAMERA] + a for a in local],
                self.height,
                self.width,
            )
            frames = [crop_and_resize(f, self.height, self.width) for f in head]
        # Pad to num_video_frames with the last real frame.
        if frames and len(frames) < self.num_video_frames:
            frames = frames + [frames[-1]] * (self.num_video_frames - len(frames))
        return frames

    def _get_prompt(self, local_idx: int) -> str:
        ep = self._episodes[self._ep_pos[local_idx]]
        tasks = ep.get("tasks") or []
        base = tasks[0] if tasks else self.task_name
        return format_prompt_for_inference(base)

    # ----- sample assembly -----

    def _build_sample(self, local_idx: int, start: int) -> dict:
        ep_global = int(self._episodes[self._ep_pos[local_idx]]["episode_index"])
        ep_len = self._ep_lengths[local_idx]
        actual_end = min(start + self.num_frames, ep_len)
        actual_len = max(0, actual_end - start)
        if actual_len < 2:
            raise IndexError(f"window [{start},{start + self.num_frames}) has no action label (ep_len={ep_len})")

        state = self._read_state(ep_global, start, actual_end)  # (actual_len, 16)
        arm10 = state_to_arm10(state)  # (actual_len, 10), raw
        frames = self._read_video(ep_global, start, actual_end)

        # pad arm10 to the full window with the last real row
        if actual_len < self.num_frames:
            pad = self.num_frames - actual_len
            arm10 = np.concatenate([arm10, np.repeat(arm10[-1:], pad, axis=0)], axis=0)
        arm20 = assemble_single_arm_left(arm10)  # (num_frames, 20) raw single-arm EEF (right 10 = 0)

        n_valid_action = max(0, min(actual_len - 1, self.num_action_steps))
        # ── raw pre-normalization vectors: PROPRIO = current pose [0:1], ACTION = next-frame poses ──
        proprio_raw = arm20[0:1]  # (1, 20)  gripper = rendered achieved width
        action_raw = arm20[1 : self.num_frames].copy()  # (T, 20)  pos/rot = next-frame achieved pose
        # ACTION gripper (dim 9) = the recorded command (exact timing, no actuation lag), replacing the
        # achieved width; aligned to the action steps + padded like the base command (padded rows are
        # dropped by the time mask). Both proprio and action gripper live in the same [-1, +1] space.
        grip_cmd = self._read_gripper_command(ep_global, start, start + n_valid_action)  # (n_valid, 1)
        if grip_cmd.shape[0] < self.num_action_steps:
            pad_row = grip_cmd[-1:] if grip_cmd.shape[0] else np.zeros((1, 1), np.float32)
            grip_cmd = np.concatenate(
                [grip_cmd, np.repeat(pad_row, self.num_action_steps - grip_cmd.shape[0], axis=0)], axis=0
            )
        action_raw[:, ARM10_DIM - 1] = grip_cmd[:, 0]  # dim 9 = left-arm gripper
        if self._mobile_base:
            # ACTION base5 = RoboCasa-native command (raw) aligned to the action steps: command at
            # frame start+i drives the transition to step i. Padded rows land past n_valid_action so
            # the time mask drops them (value irrelevant).
            base_act = self._read_base_action(ep_global, start, start + n_valid_action)  # (n_valid, 5)
            if base_act.shape[0] < self.num_action_steps:
                pad_row = base_act[-1:] if base_act.shape[0] else np.zeros((1, BASE_ACTION_DIM), np.float32)
                base_act = np.concatenate(
                    [base_act, np.repeat(pad_row, self.num_action_steps - base_act.shape[0], axis=0)], axis=0
                )
            # PROPRIO base5, by self._base_proprio (the eval client reproduces the chosen mode
            # exactly, so train/eval match — no exposure bias):
            #   "global_pose": [x, y, sin(yaw), cos(yaw), 0] — world planar base pose at the window's
            #     frame 0 (ground robot: z / roll / pitch constant; sin/cos = no ±π seam).
            #   "velocity" (historical): [vx, vy, vyaw (A′ command-space), 0, 0] — finite-diff of the
            #     base pose (start-1 → start); start=0 has no previous frame → 0.
            base_pro = np.zeros((1, BASE_ACTION_DIM), np.float32)
            if self._base_proprio == "global_pose":
                cur = self._read_state(ep_global, start, start + 1)[0, 0:7]  # base_position(3) + base_rotation(4)
                yaw = _yaw_from_quat_xyzw(cur[3:7])
                base_pro[0, 0:4] = (cur[0], cur[1], np.sin(yaw), np.cos(yaw))
            elif start > 0:
                base_pose = self._read_state(ep_global, start - 1, start + 1)[:, 0:7]  # (2, 7) prev+cur
                base_pro[0, 0:BASE_VEL_DIM] = base_velocity_cmd(base_pose, self._fps)
            action_raw = np.concatenate([action_raw, base_act.astype(np.float32)], axis=-1)  # (T, 25)
            proprio_raw = np.concatenate([proprio_raw, base_pro], axis=-1)  # (1, 25)

        # Normalize the whole raw vectors. Action always uses the ONE combined command-space block;
        # proprio uses the same block ("velocity": the A′ rescale put its velocity in command space)
        # or its own pose block ("global_pose": meters/unit-circle can't share command stats).
        action = apply_normalization(action_raw, self._stats, self.normalize_mode).astype(np.float32)
        proprio = apply_normalization(proprio_raw, self._proprio_stats, self.normalize_mode).astype(np.float32)
        # Two-point command dims (binary_action_dims): targets stay the RAW ±1 — identity regardless
        # of what the stats say (min-max over ±1 data is identity today; this makes it structural).
        # The deploy server snaps its decoded outputs back to exact ±1 (model_loader).
        for d in self._binary_action_dims:
            vals = action_raw[:n_valid_action, d]
            if vals.size and np.any(np.abs(np.abs(vals) - 1.0) > 1e-4):
                raise ValueError(
                    f"binary action dim {d} has non-±1 values in {self.task_name}: {np.unique(vals)[:8]}"
                )
            action[:, d] = action_raw[:, d]
        # Static-window flag (robotwin parity) on the arm POSE dims [0:9] only (pos3 + rot6d6): base
        # velocity is a separate channel, and the gripper dim is a COMMAND on the action side vs an
        # achieved width on the proprio side (they legitimately differ), so both are excluded. First
        # action step vs proprio in the normalized representation the model sees.
        _pose = ARM10_DIM - 1  # 9: left-arm pos+rot6d (exclude gripper)
        is_static = bool(np.max(np.abs(action[0, :_pose] - proprio[0, :_pose])) < self._static_segment_threshold)

        video_mask = torch.tensor([start + i < actual_end for i in self._video_sample_indices], dtype=torch.bool)
        # Unified 80-D scatter of the whole raw vector (arm + base) through the ONE map. Use the raw
        # per-dim masks scattered to 80-D (built in __init__) — right-arm + torso + control_mode slots
        # stay masked (map_to_unify's own mask would mark every mapped slot valid).
        if self._unify_dst_index is not None:
            proprio, _ = map_to_unify(proprio, self._unify_dst_index, UNIFY_DIM)  # (1, 80)
            action, _ = map_to_unify(action, self._unify_dst_index, UNIFY_DIM)  # (T, 80)
            mask_dim = UNIFY_DIM
            action_dim_mask, proprio_dim_mask = self._unify_action_dim_mask, self._unify_dim_mask
        else:
            mask_dim = self._raw_dim
            action_dim_mask, proprio_dim_mask = self._raw_action_mask, self._raw_proprio_mask
        action_mask = torch.from_numpy(
            build_action_mask_2d(self.num_action_steps, mask_dim, n_valid_action, dim_mask=action_dim_mask)
        )
        proprio_mask = torch.from_numpy(
            build_proprio_mask_2d(mask_dim, enabled=actual_len > 0, dim_mask=proprio_dim_mask)
        )

        return {
            "video": frames,
            "vace_video": None,
            "first_frame_image": [frames[0]] if frames else [],
            "action": torch.from_numpy(action),
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": torch.from_numpy(proprio),
            "proprio_mask": proprio_mask,
            "prompt": self._get_prompt(local_idx),
            "episode_index": ep_global,
            "start_frame": start,
            "episode_length": ep_len,
            "task_name": self.task_name,
            "_is_static": is_static,
        }

    def __getitem__(self, idx):
        if self._val_samples is not None:
            local_idx, start = self._val_samples[idx]
            return self._build_sample(local_idx, start)
        local_idx, start = self._window_index[idx]
        sample = self._build_sample(local_idx, start)
        # Training-only: resample away from a static window (keep val deterministic). Mirrors robotwin.
        if (
            self._filter_static_segments
            and self.split == "train"
            and sample.get("_is_static")
            and len(self._window_index) > 1
        ):
            for _ in range(self._max_static_retry):
                li, st = self._window_index[random.randint(0, len(self._window_index) - 1)]
                sample = self._build_sample(li, st)
                if not sample.get("_is_static"):
                    break
        if self._color_jitter is not None:
            # Same jitter factors across the whole clip (temporal consistency).
            sample["video"] = self._color_jitter.apply({"video": sample["video"]})["video"]
            # Keep the first-frame conditioning image in sync with the jittered clip.
            sample["first_frame_image"] = [sample["video"][0]]
        return sample


class MultiTaskRoboCasa365Dataset(BaseDataset):
    """Multi-task wrapper over per-task ``RoboCasa365Dataset`` (mirrors
    ``MultiTaskRoboTwinDataset``).

    Concatenates one ``RoboCasa365Dataset`` per task so one epoch covers all tasks.
    ``dataset_dir`` is a single v3 aggregated repo OR a LIST of repos (the full 300-task
    atomic+composite case = two separate HF repos); tasks are discovered across all of them by
    ``source_prefix``. ``task_name`` selects one task, ``task_roots`` a subset (task names), else
    EVERY task across the repo(s) is discovered (full RoboCasa365, mobile + fixed — the base command
    is trained via ``mobile_base``, not dropped). All sub-datasets share one pooled stats file
    (multi-repo requires an explicit ``normalization_stats_path`` — no single root to auto-place it).
    """

    @classmethod
    def from_config(cls, config, split: str = "train"):
        norm = get_cfg(config, "normalize_mode", "min-max")
        if isinstance(norm, str) and norm.lower() in ("none", "null", ""):
            norm = None
        cam_layout = get_cfg(config, "camera_layout", None)
        return cls(
            dataset_dir=get_cfg(config, "dataset_dir"),
            task_name=get_cfg(config, "task_name", None),
            task_roots=get_cfg(config, "task_roots", None),
            normalize_mode=norm,
            normalization_stats_path=get_cfg(config, "normalization_stats_path", None),
            num_frames=int(get_cfg(config, "num_frames", 33)),
            height=int(get_cfg(config, "height", 384)),
            width=int(get_cfg(config, "width", 320)),
            split=split,
            val_ratio=float(get_cfg(config, "val_ratio", 0.0)),
            repeat=int(get_cfg(config, "repeat", 1)),
            window_stride=int(get_cfg(config, "window_stride", 1)),
            video_stride=int(get_cfg(config, "video_stride", 4)),
            multiview=bool(get_cfg(config, "multiview", True)),
            camera_layout=list(cam_layout) if cam_layout is not None else None,
            temporal_compression=int(get_cfg(config, "temporal_compression", 4)),
            causal_temporal=bool(get_cfg(config, "causal_temporal", True)),
            filter_static_segments=bool(get_cfg(config, "filter_static_segments", True)),
            static_segment_threshold=float(get_cfg(config, "static_segment_threshold", 1e-5)),
            max_static_retry=int(get_cfg(config, "max_static_retry", 3)),
            unify_action=bool(get_cfg(config, "unify_action", False)),
            unify_action_map=get_cfg(config, "unify_action_map", None),
            mobile_base=bool(get_cfg(config, "mobile_base", False)),
            mask_torso_action=bool(get_cfg(config, "mask_torso_action", True)),
            # Absent keys = historical behavior (what every pre-existing ckpt trained with).
            base_proprio=str(get_cfg(config, "base_proprio", "velocity")),
            binary_action_dims=get_cfg(config, "binary_action_dims", None),
            color_jitter=get_cfg(config, "color_jitter", None),
            seed=int(get_cfg(config, "seed", 42)),
        )

    def __init__(
        self,
        dataset_dir: str,
        task_name: Optional[str] = None,
        task_roots: Optional[list] = None,
        normalize_mode: Optional[str] = "min-max",
        normalization_stats_path: Optional[str] = None,
        mobile_base: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        self._mobile_base = bool(mobile_base)
        roots = self._resolve_task_roots(dataset_dir, task_name, task_roots)
        if not roots:
            raise FileNotFoundError(f"No RoboCasa365 task buckets found under {dataset_dir}")
        print(f"MultiTaskRoboCasa365Dataset: {len(roots)} task bucket(s)")
        norm = normalize_mode if normalize_mode not in ("", "none", "null") else None
        # Resolve ONE shared stats path across all buckets so every task trains in
        # the SAME normalized space (mirrors robotwin's multi-task shared-stats
        # contract). Single-bucket → None lets the sub-dataset auto-resolve its own
        # per-task stats (the verified single-task path, unchanged).
        shared_stats = self._resolve_shared_stats(
            dataset_dir, task_name, roots, normalization_stats_path, norm, self._mobile_base
        )
        self._datasets = [
            RoboCasa365Dataset(
                data_root=dr,
                task_name=tn,
                normalize_mode=norm,
                normalization_stats_path=shared_stats,
                mobile_base=self._mobile_base,
                **kwargs,
            )
            for tn, dr in roots
        ]
        self._cum = np.cumsum([0] + [len(d) for d in self._datasets]).astype(np.int64)
        # Surface the resolved stats path so the trainer's save_normalization_stats() copies it
        # into the checkpoint dir — deploy's _build_normalizer REQUIRES normalization_stats.npy
        # there (raises FileNotFoundError if missing). Mirrors MultiTaskRoboTwinDataset; delegates
        # to the sub-dataset (single bucket auto-resolves its own path; multi shares the pooled one)
        # exactly like the normalization_stats property below.
        self.normalization_stats_path = self._datasets[0].normalization_stats_path if self._datasets else None

    @staticmethod
    def _resolve_shared_stats(dataset_dir, task_name, roots, explicit, norm, mobile_base=False):
        """Resolve a single stats path shared by every sub-dataset (or None).

        An explicit ``normalization_stats_path`` is ALWAYS honored — read if it exists, else used as
        the compute target (no silent fallback to a different location). Without an explicit path: a
        single bucket returns None (the sub-dataset auto-resolves its own per-task stats); a multi-task
        single repo pools stats to a ``dataset_dir``-level file; a multi-repo list requires an explicit
        path (raises otherwise). The suffix encodes the layout — ``_eefbase`` (25-D [arm20, base5])
        when mobile, else ``_eef`` (20-D arm) — matching ``_resolve_stats_path``.
        """
        if norm is None:
            return None
        if explicit:
            # Honor the user's path: read it if present, else it is the compute target below. Never
            # quietly ignore an explicit path and compute stats somewhere else (no silent fallback).
            if os.path.exists(explicit):
                return explicit
            shared = explicit
        elif len(roots) <= 1:
            return None  # single bucket, no explicit path: sub-dataset auto-resolves its own per-task stats
        elif isinstance(dataset_dir, str):
            tag = "eefbase" if mobile_base else "eef"
            name = f"{task_name}_{tag}_stats.npy" if task_name else f"robocasa365_multitask_{tag}_stats.npy"
            shared = os.path.join(dataset_dir, name)
        else:
            # Multi-repo (dataset_dir is a list of repos) has no single root to auto-place the pooled
            # stats; require an explicit target (no silent fallback — surface the missing config).
            raise ValueError(
                "multi-repo robocasa365 (dataset_dir is a list of repos) has no single root to auto-place "
                "the shared stats; set dataloader.normalization_stats_path to the target .npy "
                "(it is computed there on first use)."
            )
        if not os.path.exists(shared):
            _compute_shared_stats_rank0_synced(shared, roots, include_base=mobile_base)
        return shared

    @staticmethod
    def _resolve_task_roots(dataset_dir, task_name: Optional[str], task_roots: Optional[list]):
        """Return ``[(task_name, repo), ...]`` — one entry per selected task, paired with its repo.

        ``dataset_dir`` is a single v3 aggregated repo (str) OR a list of repos (the full 300-task
        atomic+composite case = two separate HF repos). v3 packs many tasks per repo, distinguished by
        ``source_prefix``; each task is discovered in the repo it lives in (atomic/composite task sets
        are disjoint). ``task_name`` selects one; ``task_roots`` (task names) a subset; else EVERY task
        across all repos. The sub-dataset later filters its own repo to its task via ``source_prefix``.
        """
        repos = [dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir)
        pairs = []  # [(task, repo), ...] across all repos, in repo order
        for repo in repos:
            eps = _episodes_with_offsets(repo)  # cached: reused by each sub-dataset's __init__
            for t in sorted({_task_from_source_prefix(p) for p in eps["source_prefix"]}):
                pairs.append((t, repo))
        all_tasks = {t for t, _ in pairs}
        if task_name is not None:
            sel = {task_name} if task_name in all_tasks else set()
        elif task_roots:
            sel = {t for t in task_roots if t in all_tasks}
        else:
            sel = all_tasks
        return [(t, repo) for t, repo in pairs if t in sel]

    @property
    def action_dim(self) -> int:
        return self._datasets[0].action_dim if self._datasets else EEF_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        return self._datasets[0].normalization_stats if self._datasets else None

    def denormalize_action(self, action) -> np.ndarray:
        return self._datasets[0].denormalize_action(action)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx):
        d = int(np.searchsorted(self._cum, idx, side="right") - 1)
        return self._datasets[d][idx - int(self._cum[d])]


__all__ = ["RoboCasa365Dataset", "MultiTaskRoboCasa365Dataset", "state_to_arm10"]
