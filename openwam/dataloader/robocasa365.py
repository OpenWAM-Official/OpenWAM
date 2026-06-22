"""RoboCasa365 dataloader — fixed-base subset, single-arm 20-D EEF (LeRobot v2.1).

Bespoke ``BaseDataset`` that reads the RAW downloaded RoboCasa365 LeRobot **v2.1**
buckets directly (per-episode ``data/chunk-*/episode_*.parquet`` + per-episode
``videos/chunk-*/<cam>/episode_*.mp4`` + ``meta/episodes.jsonl``). Structurally a
faithful dual of ``robotwin_dataset.py`` (which reads raw RoboTwin HDF5): same
window enumeration, multiview L-shape, per-task stats, ``Single`` + ``Multi`` pair.
``dataset_dir`` points at the data exactly as downloaded — **no migration /
conversion / overlay**.

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

The fixed base (``base_position`` / ``base_rotation``) and the OSC-delta ``action``
field are dropped.

``observation.state`` layout (16-D, from meta/modality.json)::

    base_position(0:3) + base_rotation(3:7) + eef_pos_rel(7:10)
    + eef_rot_rel(10:14, quat xyzw) + gripper_qpos(14:16)

Cameras (Phase-1 mapping): head=``robot0_agentview_left``,
left_wrist=``robot0_eye_in_hand``, right_wrist=None -> black (single arm, no 2nd
wrist), composed into the L-shape via ``assemble_multiview_layout``.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.robotwin import _check_temporal_divisibility
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
    build_action_mask_2d,
    build_proprio_mask_2d,
    single_arm_20d,
)
from openwam.dataloader.utils.normalization import materialize_eef_stats
from openwam.dataloader.utils.video_io import decode_video_frames

# Phase-1 2-view mapping (also what the deploy server composes).
HEAD_CAMERA = "observation.images.robot0_agentview_left"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"
STATS_DIM = ARM10_DIM  # single-arm stats live on the 10-D arm
_MISSING_RIGHT = "__missing_right_wrist__"

# observation.state slices (16-D).
_STATE_EEF_POS = slice(7, 10)
_STATE_EEF_ROT = slice(10, 14)  # quaternion (xyzw)

# Multiview L-shape slot sizes (must match assemble_multiview_layout defaults at
# height=384/width=320: top 256x320, each bottom 128x160).
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320
_WRIST_SLOT_H, _WRIST_SLOT_W = 128, 160


def _task_dir_name(lerobot_dir: str) -> str:
    """Task name from a ``.../<Task>/<date>/lerobot`` bucket path."""
    return os.path.basename(os.path.dirname(os.path.dirname(lerobot_dir.rstrip("/")))) or "task"


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
    Consumed by the trainer (copies 20-D mean/std into ``action_mean``/``action_std``)
    and by eval-time denormalization.
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
    """Single-task RoboCasa365 reader (raw LeRobot v2.1, single-arm 20-D EEF).

    Mirrors ``RoboTwinDataset``: exhaustive ``(episode, start)`` window enumeration,
    deterministic train/val split, multiview L-shape composition, per-task action
    normalization with auto-compute on first use. RoboCasa-specific: reads v2.1
    parquet/mp4 (not HDF5) and assembles the 20-D EEF from ``observation.state``
    (see module docstring).
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
        # Encoder temporal-contract guard (mirrors robotwin): causal encoders need
        # (num_video_frames - 1) % tc == 0; non-causal need num_video_frames % tc == 0.
        _check_temporal_divisibility(self.num_video_frames, int(temporal_compression), bool(causal_temporal))
        self.multiview = bool(multiview)
        # Camera layout: head (top), wrist (bot-left), missing right (bot-right=black).
        self.camera_layout = list(camera_layout) if camera_layout else [HEAD_CAMERA, WRIST_CAMERA, _MISSING_RIGHT]

        # ── info.json: path templates + chunk size ────────────────────────
        with open(os.path.join(data_root, "meta", "info.json")) as f:
            info = json.load(f)
        self._data_path_tmpl = info["data_path"]
        self._video_path_tmpl = info["video_path"]
        self._chunks_size = int(info.get("chunks_size", 1000))

        # ── episodes.jsonl: per-episode index / length / task strings ─────
        episodes = []
        with open(os.path.join(data_root, "meta", "episodes.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        episodes.sort(key=lambda e: e["episode_index"])
        if not episodes:
            raise FileNotFoundError(f"No episodes in {data_root}/meta/episodes.jsonl")
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

        # ── action normalization (10-D arm stats; auto-compute if missing) ─
        self._stats: Optional[dict] = None  # 10-D arm stats (forward path)
        self._stats20: Optional[dict] = None  # 20-D expansion (trainer / eval contract)
        self.normalization_stats_path: Optional[str] = None
        if self.normalize_mode is not None:
            stats_path = self._resolve_stats_path(normalization_stats_path)
            if stats_path.endswith(".json"):
                with open(stats_path) as f:
                    raw = json.load(f)
            else:
                raw = np.load(stats_path, allow_pickle=True).item()
            raw = raw.get("eef", raw)  # accept flat or {"eef": {...}} schema
            self._stats = materialize_eef_stats(
                raw, self.normalize_mode, dim=STATS_DIM, strict_minmax=True, source_hint=stats_path
            )
            self._stats20 = _expand_stats_to_20d(self._stats)
            self.normalization_stats_path = stats_path
            print(f"  [normalizer] {self.normalize_mode}, dim={STATS_DIM}, stats={stats_path}")
        else:
            print("  [normalizer] DISABLED (normalize_mode=None)")

    def _resolve_stats_path(self, explicit: Optional[str]) -> str:
        """Explicit path wins; else ``{data_root}/{task}_eef_stats.npy`` (auto-compute)."""
        if explicit and os.path.exists(explicit):
            return explicit
        stats_path = os.path.join(self.data_root, f"{self.task_name}_eef_stats.npy")
        if not os.path.exists(stats_path):
            from openwam.dataloader.robocasa365_stats_computation import (
                atomic_save_stats_npy,
                compute_normalization_stats,
            )

            print(f"  [normalizer] computing arm-10 stats from {self.data_root} -> {stats_path}")
            atomic_save_stats_npy(stats_path, compute_normalization_stats(self.data_root))
        return stats_path

    # ----- BaseDataset interface -----

    @property
    def action_dim(self) -> int:
        return EEF_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        """20-D stats (arm10 left, neutral right) for the trainer's action buffers
        and eval denormalization. None when normalization is disabled."""
        return dict(self._stats20) if self._stats20 is not None else None

    def denormalize_action(self, action) -> np.ndarray:
        """Invert normalization on the left-arm 10 dims (right 10 stay 0).

        Inverse of ``apply_normalization`` for the active mode; no-op when
        normalization is disabled.
        """
        arr = np.asarray(action, dtype=np.float32)
        if self._stats is None or self.normalize_mode is None:
            return arr.copy()
        s, left = self._stats, arr[..., :ARM10_DIM]
        if self.normalize_mode == "z-score":
            de = left * s["std"] + s["mean"]
        elif self.normalize_mode == "min-max":
            de = (left + 1.0) * 0.5 * (s["max"] - s["min"]) + s["min"]
        elif self.normalize_mode == "quantile":
            de = (left + 1.0) * 0.5 * (s["q99"] - s["q01"]) + s["q01"]
        else:
            de = left
        out = arr.copy()
        out[..., :ARM10_DIM] = de
        return out

    def __len__(self) -> int:
        return len(self._val_samples) if self._val_samples is not None else len(self._window_index)

    # ----- IO helpers -----

    def _data_path(self, ep_global_idx: int) -> str:
        chunk = ep_global_idx // self._chunks_size
        return os.path.join(
            self.data_root, self._data_path_tmpl.format(episode_chunk=chunk, episode_index=ep_global_idx)
        )

    def _video_path(self, ep_global_idx: int, camera: str) -> str:
        chunk = ep_global_idx // self._chunks_size
        return os.path.join(
            self.data_root,
            self._video_path_tmpl.format(episode_chunk=chunk, video_key=camera, episode_index=ep_global_idx),
        )

    def _read_state(self, ep_global_idx: int, start: int, end: int) -> np.ndarray:
        df = pd.read_parquet(self._data_path(ep_global_idx), columns=["observation.state"])
        st = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 16)
        return st[start:end]

    def _decode_camera(self, ep_global_idx: int, camera: str, frame_indices, slot_h: int, slot_w: int):
        return decode_video_frames(self._video_path(ep_global_idx, camera), list(frame_indices), slot_h, slot_w)

    def _read_video(self, ep_global_idx: int, start: int, actual_end: int):
        """Decode the window's sampled frames into a list of L-shape canvases (or
        single-view PIL frames). v2.1 stores one mp4 per episode, so frame indices
        are episode-local (no concatenated-shard offset)."""
        real_abs = [start + i for i in self._video_sample_indices if start + i < actual_end]
        if self.multiview:
            head = self._decode_camera(ep_global_idx, HEAD_CAMERA, real_abs, _HEAD_SLOT_H, _HEAD_SLOT_W)
            wrist = self._decode_camera(ep_global_idx, WRIST_CAMERA, real_abs, _WRIST_SLOT_H, _WRIST_SLOT_W)
            frames = [
                assemble_multiview_layout(
                    {HEAD_CAMERA: head[fi], WRIST_CAMERA: wrist[fi]}, self.camera_layout, self.height, self.width
                )
                for fi in range(len(real_abs))
            ]
        else:
            head = self._decode_camera(ep_global_idx, HEAD_CAMERA, real_abs, self.height, self.width)
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

        # normalize the whole arm seq once, then slot into the single-arm 20-D
        # schema (right 10 zero). proprio = pose[0]; action = pose[1:].
        eef20d = single_arm_20d(arm10, self._stats, self.normalize_mode)  # (num_frames, 20)
        proprio = eef20d[0:1].astype(np.float32)
        action = eef20d[1 : self.num_frames].astype(np.float32)

        video_mask = torch.tensor([start + i < actual_end for i in self._video_sample_indices], dtype=torch.bool)
        n_valid_action = max(0, min(actual_len - 1, self.num_action_steps))
        action_mask = torch.from_numpy(
            build_action_mask_2d(self.num_action_steps, EEF_DIM, n_valid_action, dim_mask=LEFT_ARM_DIM_MASK)
        )
        proprio_mask = torch.from_numpy(
            build_proprio_mask_2d(EEF_DIM, enabled=actual_len > 0, dim_mask=LEFT_ARM_DIM_MASK)
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
        }

    def __getitem__(self, idx):
        if self._val_samples is not None:
            local_idx, start = self._val_samples[idx]
        else:
            local_idx, start = self._window_index[idx]
        return self._build_sample(local_idx, start)


class MultiTaskRoboCasa365Dataset(BaseDataset):
    """Multi-task wrapper over per-task ``RoboCasa365Dataset`` (mirrors
    ``MultiTaskRoboTwinDataset``).

    Concatenates one ``RoboCasa365Dataset`` per task so one epoch covers all
    tasks. ``dataset_dir`` may point at a single task's ``lerobot`` bucket, or at
    a root holding many ``.../<Task>/<date>/lerobot`` buckets (the RoboCasa
    download layout); the fixed-base task subset is listed in
    ``benchmarks/robocasa365/fixed_base_tasks.json``.
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
            seed=int(get_cfg(config, "seed", 42)),
        )

    def __init__(
        self,
        dataset_dir: str,
        task_name: Optional[str] = None,
        task_roots: Optional[list] = None,
        normalize_mode: Optional[str] = "min-max",
        normalization_stats_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        roots = self._resolve_task_roots(dataset_dir, task_name, task_roots)
        if not roots:
            raise FileNotFoundError(f"No RoboCasa365 task buckets found under {dataset_dir}")
        print(f"MultiTaskRoboCasa365Dataset: {len(roots)} task bucket(s)")
        norm = normalize_mode if normalize_mode not in ("", "none", "null") else None
        # Resolve ONE shared stats path across all buckets so every task trains in
        # the SAME normalized space (mirrors robotwin's multi-task shared-stats
        # contract). Single-bucket → None lets the sub-dataset auto-resolve its own
        # per-task stats (the verified single-task path, unchanged).
        shared_stats = self._resolve_shared_stats(dataset_dir, task_name, roots, normalization_stats_path, norm)
        self._datasets = [
            RoboCasa365Dataset(
                data_root=dr,
                task_name=tn,
                normalize_mode=norm,
                normalization_stats_path=shared_stats,
                **kwargs,
            )
            for tn, dr in roots
        ]
        self._cum = np.cumsum([0] + [len(d) for d in self._datasets]).astype(np.int64)

    @staticmethod
    def _resolve_shared_stats(dataset_dir, task_name, roots, explicit, norm):
        """Resolve a single stats path shared by every sub-dataset (or None).

        explicit (if it exists) wins; single bucket → None (sub-dataset
        auto-resolves per-task); multi-bucket → pool stats over ALL buckets to a
        dataset_dir-level file, computing once if absent.
        """
        if norm is None:
            return None
        if explicit and os.path.exists(explicit):
            return explicit
        if len(roots) <= 1:
            return None  # single bucket: keep the per-task auto-resolve path
        name = f"{task_name}_eef_stats.npy" if task_name else "robocasa365_multitask_eef_stats.npy"
        shared = os.path.join(dataset_dir, name)
        if not os.path.exists(shared):
            from openwam.dataloader.robocasa365_stats_computation import (
                atomic_save_stats_npy,
                compute_multitask_stats,
            )

            print(f"  [normalizer] computing SHARED multitask stats over {len(roots)} buckets -> {shared}")
            atomic_save_stats_npy(shared, compute_multitask_stats([dr for _, dr in roots]))
        return shared

    @staticmethod
    def _resolve_task_roots(dataset_dir: str, task_name: Optional[str], task_roots: Optional[list]):
        """Return ``[(task_name, lerobot_dir), ...]`` for buckets present on disk.

        ``task_roots`` (explicit relative paths) wins; else if ``dataset_dir`` is
        itself a bucket (has ``meta/info.json``) use it directly; else discover every
        ``*/lerobot/meta/info.json`` under ``dataset_dir``. ``task_name`` filters by
        the task directory name.
        """
        if task_roots:
            out = []
            for rp in task_roots:
                dr = os.path.join(dataset_dir, rp)
                if os.path.isfile(os.path.join(dr, "meta", "info.json")):
                    out.append((os.path.basename(rp.rstrip("/")) or rp, dr))
            return out
        # Single bucket: dataset_dir points straight at a lerobot/ dir.
        if os.path.isfile(os.path.join(dataset_dir, "meta", "info.json")):
            return [(task_name or _task_dir_name(dataset_dir), dataset_dir)]
        # Root mode: discover every */lerobot bucket below dataset_dir.
        out = []
        for info_path in sorted(
            glob.glob(os.path.join(dataset_dir, "**", "lerobot", "meta", "info.json"), recursive=True)
        ):
            dr = os.path.dirname(os.path.dirname(info_path))  # .../lerobot
            tn = os.path.basename(os.path.dirname(os.path.dirname(dr)))  # .../<Task>/<date>/lerobot
            if task_name and tn != task_name:
                continue
            out.append((tn, dr))
        return out

    @property
    def action_dim(self) -> int:
        return EEF_DIM

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
