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
robotwin's endpose): both are the ABSOLUTE single-arm end-effector pose taken from
``observation.state`` (NOT the OSC-delta ``action`` field)::

    arm10 = [eef_pos_rel(3) + eef_rot_rel(quat->rot6d, 6) + gripper(1)]
          -> single-arm LEFT 10 of the canonical 20-D EEF.
    proprio = eef20d[0:1]      # current pose
    action  = eef20d[1:T]      # future-pose trajectory (model predicts poses)

Mobile base (``mobile_base=True``, requires ``unify_action``): the RoboCasa-native base command
is read RAW from the LeRobot ``action`` field ([x/y/yaw velocity, torso position, control_mode])
and scattered into the 80-D reserved slots ``[68:73)`` — direct-to-env at eval, no bridge. The
arm stays absolute-EEF (bridged). Base is action-only; world-frame base pose is scene-arbitrary so
it is NOT added to proprio (the policy perceives base state from the robot-mounted head video).
See ``docs/plans/robocasa365-full-mobile.md``.

``observation.state`` layout (16-D, from meta/modality.json)::

    base_position(0:3) + base_rotation(3:7) + eef_pos_rel(7:10)
    + eef_rot_rel(10:14, quat xyzw) + gripper_qpos(14:16)

Cameras (Phase-1 mapping): head=``robot0_agentview_left``,
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
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    assemble_single_arm_left,
    build_action_mask_2d,
    build_proprio_mask_2d,
)
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec, unmap_from_unify
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.video_io import decode_video_frames

# Phase-1 2-view mapping (also what the deploy server composes).
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

# LeRobot ``action`` field (12-D, "layout B" from PandaOmron_modality.json). Mobile support reads
# the base COMMAND dims (raw, RoboCasa-native — NOT reconstructed from state like the arm):
#   [0:3] base x/y/yaw velocity, [3] torso lift (position 0-0.34 m), [4] control_mode {-1,+1}.
_ACTION_BASE = slice(0, 5)
BASE_ACTION_DIM = 5
# 80-D reserved-slot span for the base command (ACTION side only, [68:73)). Proprio stays 20-D EEF:
# absolute world-frame base pose is scene-arbitrary (differs per kitchen), a poor generalizable
# proprio signal, so it is deliberately NOT added — the policy perceives base state from the
# robot-mounted head video. See docs/plans/robocasa365-full-mobile.md.
_UNIFY_BASE = slice(68, 68 + BASE_ACTION_DIM)

# Multiview L-shape slot sizes (must match assemble_multiview_layout defaults at
# height=384/width=320: top 256x320, each bottom 128x160).
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320
_WRIST_SLOT_H, _WRIST_SLOT_W = 128, 160


def _task_dir_name(lerobot_dir: str) -> str:
    """Task name from a ``.../<Task>/<date>/lerobot`` bucket path."""
    return os.path.basename(os.path.dirname(os.path.dirname(lerobot_dir.rstrip("/")))) or "task"


def _task_from_source_prefix(prefix: str) -> str:
    """Task name from a v3 ``source_prefix`` like ``pretrain/atomic/OpenDrawer/20250819`` -> ``OpenDrawer``.

    v3 aggregates all tasks into one repo, tagging each episode's origin task with ``source_prefix``
    (``<split>/<atomic|composite>/<Task>/<date>``); the task is the second-to-last path segment.
    """
    parts = str(prefix).strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else str(prefix)


def _compute_shared_stats_rank0_synced(shared_path: str, roots: list, include_base: bool = False) -> None:
    """Compute + persist the shared multitask stats with rank-0 synchronization.

    ``roots`` is ``[(task_name, repo), ...]`` (v3: every entry shares the same aggregated repo, one
    per selected task). On a multi-GPU first run, only rank 0 computes + atomically writes; other
    ranks poll for the file (mirrors robotwin). Without this, every rank races to write the same
    ``{path}.tmp`` → torn writes + N× redundant compute over all tasks. ``include_base`` adds the
    mobile ``base`` block (5-D base command stats) to the pooled file.
    """
    from openwam.dataloader.robocasa365_stats_computation import atomic_save_stats_npy, compute_multitask_stats

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


def state_to_arm10(state: np.ndarray) -> np.ndarray:
    """``(T, 16)`` observation.state -> ``(T, 10)`` single-arm EEF (raw, unnormalized).

    arm10 = [eef_pos_rel(3), rot6d(eef_rot_rel quat xyzw, 6), gripper_opening(1)].
    Gripper 1-D = finger separation ``qpos[0] - qpos[1]``.
    """
    pos = state[:, _STATE_EEF_POS]
    rot6d = quat_xyzw_to_rotation_6d(state[:, _STATE_EEF_ROT])
    grip = (state[:, 14] - state[:, 15])[:, None]
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


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
                f"Resolution {self.height}x{self.width} must be divisible by 32 "
                "(VAE downsamples by 16, patch size 2)."
            )
        self.repeat = int(repeat)
        self.split = split
        # Static-segment filtering (mirrors robotwin): resample train windows that "haven't
        # started moving" so the model isn't taught to output ~zero motion.
        self._filter_static_segments = bool(filter_static_segments)
        self._static_segment_threshold = float(static_segment_threshold)
        self._max_static_retry = int(max_static_retry)
        # Unified 80-D action space (opt-in; mirrors the OXE single-arm base-reader path). When on,
        # the normalized 20-D EEF is scattered into UNIFY_DIM and LEFT_ARM_DIM_MASK is honored through
        # the scatter (right-arm slots stay masked out of the loss). Off (default) → 20-D as before.
        self._unify_action = bool(unify_action)
        self._unify_action_map = unify_action_map
        # Mobile base: read the RoboCasa-native base command (x/y/yaw vel + torso + control_mode)
        # from the LeRobot ``action`` field and scatter it into the 80-D reserved slots [68:73).
        # Requires unify_action (the base lives in the unified reserved region).
        self._mobile_base = bool(mobile_base)
        if self._mobile_base and not self._unify_action:
            raise ValueError("mobile_base=True requires unify_action=True (base occupies the 80-D reserved slots)")
        self._unify_dst_index = None
        self._unify_dim_mask = None  # proprio dim mask (arm only)
        self._unify_action_dim_mask = None  # action dim mask (arm + base when mobile)
        if self._unify_action:
            spec = self._unify_action_map if self._unify_action_map is not None else list(range(EEF_DIM))
            self._unify_dst_index = parse_unify_spec(spec, UNIFY_DIM)
            if self._unify_dst_index.shape[0] != EEF_DIM:
                raise ValueError(
                    f"robocasa365 unify_action_map maps {self._unify_dst_index.shape[0]} source dims "
                    f"but the EEF action is {EEF_DIM}-D; they must match."
                )
            # Proprio: only the left-arm slots are valid (right arm zero-padded + masked).
            self._unify_dim_mask = np.zeros(UNIFY_DIM, dtype=bool)
            self._unify_dim_mask[self._unify_dst_index] = np.asarray(LEFT_ARM_DIM_MASK, dtype=bool)
            # Action: same arm mask, plus the base command slots when mobile (proprio has no base —
            # world-frame base pose is scene-arbitrary; see _UNIFY_BASE).
            self._unify_action_dim_mask = self._unify_dim_mask.copy()
            if self._mobile_base:
                self._unify_action_dim_mask[_UNIFY_BASE] = True
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
        self._video_path_tmpl = info["video_path"]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4

        # ── episodes (v3 aggregated meta) + single-task filter by source_prefix ──
        # v3 packs ALL tasks into one repo; ``source_prefix`` tags each episode's origin task, and the
        # aggregated data/video shards mix tasks. A single-task reader (task_name given) filters the
        # episode table to that task; the rest of the pipeline (window enumeration, split, reads) is
        # unchanged — it just sees a filtered episode set. Reads resolve each episode's (chunk, file) +
        # file-local row/frame offset (mirrors LeRobotV3Reader; offsets via compute_file_local_offsets).
        eps = load_episodes_parquet(Path(data_root))
        # File-local offsets (row/frame position WITHIN the aggregated shard) MUST be computed over the
        # FULL episode table, BEFORE the single-task filter — a task that isn't first in its shard would
        # otherwise get an offset counting only its own episodes, not the other tasks' rows physically
        # preceding it in the same file.
        eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
        for cam in (HEAD_CAMERA, WRIST_CAMERA):
            if f"videos/{cam}/chunk_index" in eps.columns:
                eps[f"_voff/{cam}"] = compute_file_local_offsets(eps, f"videos/{cam}/chunk_index", f"videos/{cam}/file_index")
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

        # ── action normalization (20-D EEF stats; auto-compute if missing) ─
        # Stats are persisted + loaded at the full 20-D action dim (left=arm, right=neutral),
        # so the deploy-time un-normalization of the model's 20-D output round-trips cleanly
        # (the 10-D forward path was the deploy-break; see _DEPLOY_RESOLVABLE_MODES).
        self._stats: Optional[dict] = None  # 20-D arm stats (forward + persist + deploy)
        self._base_stats: Optional[dict] = None  # 5-D base command stats (mobile only)
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
            eef_raw = full.get("eef", full) if isinstance(full, dict) else full  # flat or {"eef": {...}}
            self._stats = materialize_eef_stats(
                eef_raw, self.normalize_mode, dim=EEF_DIM, strict_minmax=True, source_hint=stats_path
            )
            if self._mobile_base:
                base_raw = full.get("base") if isinstance(full, dict) else None
                if base_raw is None:
                    raise ValueError(
                        f"mobile_base=True but stats file {stats_path} has no 'base' block. Recompute stats "
                        "(robocasa365_stats_computation emits a 'base' block when the action field is read)."
                    )
                self._base_stats = {
                    k: np.asarray(base_raw[k], np.float32).reshape(-1) for k in ("mean", "std", "min", "max")
                }
                if self._base_stats["mean"].shape[0] != BASE_ACTION_DIM:
                    raise ValueError(
                        f"base stats dim {self._base_stats['mean'].shape[0]} != {BASE_ACTION_DIM}; recompute stats."
                    )
            self.normalization_stats_path = stats_path
            print(
                f"  [normalizer] {self.normalize_mode}, dim={EEF_DIM}"
                f"{' +base5' if self._mobile_base else ''}, stats={stats_path}"
            )
        else:
            print("  [normalizer] DISABLED (normalize_mode=None)")

    def _resolve_stats_path(self, explicit: Optional[str]) -> str:
        """Explicit path wins; else ``{data_root}/{task}_{eef|eefbase}_stats.npy`` (auto-compute).

        Mobile runs use a distinct ``_eefbase_`` suffix so they never load a stale arm-only
        ``_eef_`` file (which lacks the required ``base`` block).
        """
        if explicit and os.path.exists(explicit):
            return explicit
        suffix = "eefbase" if self._mobile_base else "eef"
        stats_path = os.path.join(self.data_root, f"{self.task_name}_{suffix}_stats.npy")
        if not os.path.exists(stats_path):
            from openwam.dataloader.robocasa365_stats_computation import (
                atomic_save_stats_npy,
                compute_normalization_stats,
            )

            print(f"  [normalizer] computing {'arm-10 + base' if self._mobile_base else 'arm-10'} stats "
                  f"from {self.data_root} -> {stats_path}")
            atomic_save_stats_npy(
                stats_path,
                compute_normalization_stats(self.data_root, include_base=self._mobile_base, task_name=self.task_name),
            )
        return stats_path

    # ----- BaseDataset interface -----

    @property
    def action_dim(self) -> int:
        return UNIFY_DIM if self._unify_action else EEF_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        """20-D stats (arm10 left, neutral right) — the SAME dict persisted to the checkpoint
        (via ``normalization_stats_path``) and read by the deploy normalizer. Also surfaced
        through the multi-task wrapper's ``normalization_stats`` (mirrors robotwin). None when
        normalization is disabled."""
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

        Returns the raw RoboCasa action the client bridges:
          * unify off → 20-D arm EEF.
          * unify on, no base → 20-D arm EEF (un-unified from 80-D).
          * unify on + mobile_base → 25-D ``[arm20, base5]`` (base5 = raw x/y/yaw vel, torso, mode).

        Arm right-10 dims carry neutral stats so they pass through ~unchanged; left-10 use the real
        arm stats; base5 uses the separate base stats. Same contract as the deploy normalizer.
        """
        arr = np.asarray(action, dtype=np.float32)
        base_phys = None
        if self._unify_dst_index is not None:
            if self._mobile_base:
                # Gather base BEFORE un-unifying the arm (unmap collapses to 20-D and drops [68:73)).
                base_phys = self._unnormalize(arr[..., _UNIFY_BASE], self._base_stats)
            arr = unmap_from_unify(arr, self._unify_dst_index).astype(np.float32)  # (..., 80) -> (..., 20)
        arm = self._unnormalize(arr, self._stats)
        if base_phys is not None:
            return np.concatenate([arm, base_phys], axis=-1)  # (..., 25)
        return arm

    def __len__(self) -> int:
        return len(self._val_samples) if self._val_samples is not None else len(self._window_index)

    # ----- IO helpers -----

    def _data_file_path(self, chunk: int, file: int) -> str:
        return os.path.join(self.data_root, self._data_path_tmpl.format(chunk_index=chunk, file_index=file))

    def _video_path(self, camera: str, chunk: int, file: int) -> str:
        return os.path.join(
            self.data_root, self._video_path_tmpl.format(video_key=camera, chunk_index=chunk, file_index=file)
        )

    def _read_data_file_uncached(self, chunk: int, file: int) -> pd.DataFrame:
        return pd.read_parquet(self._data_file_path(chunk, file), columns=["observation.state", "action"])

    def _file_table(self, chunk: int, file: int) -> pd.DataFrame:
        # Cache the aggregated-shard decode: with window_stride=1 all windows of an episode hit the
        # same (chunk,file) shard — without the cache each re-decodes the whole shard (mirrors
        # LeRobotV3Reader's lru_cache). Lazy (re)build so it survives DataLoader-worker pickling.
        cache = getattr(self, "_file_cache", None)
        if cache is None:
            cache = self._file_cache = functools.lru_cache(maxsize=8)(self._read_data_file_uncached)
        return cache(chunk, file)

    def _read_state(self, ep_global_idx: int, start: int, end: int) -> np.ndarray:
        m = self._ep_meta[ep_global_idx]
        df = self._file_table(m["chunk"], m["file"])
        o = m["row_offset"]
        return np.stack(df["observation.state"].values[o + start : o + end]).astype(np.float32)  # (n, 16)

    def _read_base_action(self, ep_global_idx: int, start: int, end: int) -> np.ndarray:
        """Base command window [start, end) from the LeRobot ``action`` field: ``(n, 5)`` =
        [x_vel, y_vel, yaw_vel, torso, control_mode] (RoboCasa-native, raw, mobile only)."""
        m = self._ep_meta[ep_global_idx]
        df = self._file_table(m["chunk"], m["file"])
        o = m["row_offset"]
        return np.stack(df["action"].values[o + start : o + end]).astype(np.float32)[:, _ACTION_BASE]  # (n, 5)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_file_cache", None)  # lru_cache over a bound method isn't picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_cache = None

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
                self._video_path(HEAD_CAMERA, hc, hf), [m["voff"][HEAD_CAMERA] + a for a in local], self.height, self.width
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

        # Assemble the single-arm 20-D (right 10 = 0) then normalize at the full 20-D with
        # the neutral-right stats (right-10 stay 0; identical left-10 result to the old
        # normalize-10-then-pad, but now the stats dim matches what's persisted for deploy).
        eef20d = apply_normalization(assemble_single_arm_left(arm10), self._stats, self.normalize_mode)
        proprio = eef20d[0:1].astype(np.float32)
        action = eef20d[1 : self.num_frames].astype(np.float32)
        # Static-window flag (faithful dual of robotwin): first action step vs proprio, in the SAME
        # representation the model sees (normalized when enabled), so a single threshold is
        # dimensionally consistent. actual_len>=2 is guaranteed above; __getitem__ resamples at train.
        is_static = bool(np.max(np.abs(action[0] - proprio[0])) < self._static_segment_threshold)

        video_mask = torch.tensor([start + i < actual_end for i in self._video_sample_indices], dtype=torch.bool)
        n_valid_action = max(0, min(actual_len - 1, self.num_action_steps))
        # Unified 80-D scatter (after normalization; is_static above used the 20-D). map_to_unify's
        # own mask would mark every mapped slot valid, so use the LEFT_ARM_DIM_MASK-honoring
        # _unify_dim_mask instead — otherwise the zero-padded right-arm slots leak into the loss.
        if self._unify_dst_index is not None:
            proprio, _ = map_to_unify(proprio, self._unify_dst_index, UNIFY_DIM)  # (1, 80)
            action, _ = map_to_unify(action, self._unify_dst_index, UNIFY_DIM)  # (T, 80)
            if self._mobile_base:
                # RoboCasa-native base command (raw, read from the action field), aligned with the
                # arm's action steps: command at frame start+i drives the transition to action step i.
                # Padded rows land beyond n_valid_action so the time mask drops them (value irrelevant).
                base_raw = self._read_base_action(ep_global, start, start + n_valid_action)  # (n_valid, 5)
                if self._base_stats is not None:
                    base_raw = apply_normalization(base_raw, self._base_stats, self.normalize_mode)
                if base_raw.shape[0] < self.num_action_steps:
                    pad_row = base_raw[-1:] if base_raw.shape[0] else np.zeros((1, BASE_ACTION_DIM), np.float32)
                    base_raw = np.concatenate(
                        [base_raw, np.repeat(pad_row, self.num_action_steps - base_raw.shape[0], axis=0)], axis=0
                    )
                action[:, _UNIFY_BASE] = base_raw.astype(action.dtype)
            mask_dim = UNIFY_DIM
            action_dim_mask, proprio_dim_mask = self._unify_action_dim_mask, self._unify_dim_mask
        else:
            mask_dim = EEF_DIM
            action_dim_mask = proprio_dim_mask = LEFT_ARM_DIM_MASK
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
        return sample


class MultiTaskRoboCasa365Dataset(BaseDataset):
    """Multi-task wrapper over per-task ``RoboCasa365Dataset`` (mirrors
    ``MultiTaskRoboTwinDataset``).

    Concatenates one ``RoboCasa365Dataset`` per task so one epoch covers all tasks.
    ``dataset_dir`` points at the v3 aggregated repo; tasks are distinguished by
    ``source_prefix``. ``task_name`` selects one task, ``task_roots`` a subset (task
    names), else EVERY task in the repo is discovered (full RoboCasa365, mobile + fixed —
    the base command is trained via ``mobile_base``, not dropped). All sub-datasets share
    one pooled stats file.
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

        explicit (if it exists) wins; single bucket → None (sub-dataset
        auto-resolves per-task); multi-bucket → pool stats over ALL buckets to a
        dataset_dir-level file, computing once if absent. Mobile runs use a distinct
        ``_eefbase_`` file (with a ``base`` block) so they never load a stale arm-only file.
        """
        if norm is None:
            return None
        if explicit and os.path.exists(explicit):
            return explicit
        if len(roots) <= 1:
            return None  # single bucket: keep the per-task auto-resolve path
        tag = "eefbase" if mobile_base else "eef"
        name = f"{task_name}_{tag}_stats.npy" if task_name else f"robocasa365_multitask_{tag}_stats.npy"
        shared = os.path.join(dataset_dir, name)
        if not os.path.exists(shared):
            _compute_shared_stats_rank0_synced(shared, roots, include_base=mobile_base)
        return shared

    @staticmethod
    def _resolve_task_roots(dataset_dir: str, task_name: Optional[str], task_roots: Optional[list]):
        """Return ``[(task_name, repo), ...]`` — the v3 aggregated repo, one entry per selected task.

        v3 packs ALL tasks into one repo (``dataset_dir``); tasks are distinguished by
        ``source_prefix`` (not per-task dirs). ``task_name`` selects one; ``task_roots`` (a list of
        task names) selects a subset; else discover EVERY distinct task in the repo (full RoboCasa365
        — mobile + fixed, base trained via mobile_base). Every entry shares the same ``dataset_dir``
        repo; the sub-dataset filters it to its task.
        """
        eps = load_episodes_parquet(Path(dataset_dir))
        tasks_in_repo = sorted({_task_from_source_prefix(p) for p in eps["source_prefix"]})
        if task_name is not None:
            sel = [task_name] if task_name in tasks_in_repo else []
        elif task_roots:
            sel = [t for t in task_roots if t in tasks_in_repo]
        else:
            sel = tasks_in_repo
        return [(t, dataset_dir) for t in sel]

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
