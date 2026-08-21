"""RoboCasa365 compact LeRobot v3 reader.

The converted dataset stores row-aligned physical semantics directly:

* ``observation.state`` (19-D) = achieved EEF ``xyz3 + rot6d6 + gripper1``
  followed by world base ``xyz3 + rot6d6``.
* ``action`` (15-D) = current-state-anchored absolute EEF target
  ``xyz3 + rot6d6 + gripper1`` followed by the native base command
  ``vx + vy + vyaw + torso + control_mode``.

There is no 20-D or 25-D intermediate. With unified training enabled, state
and action use independent maps because their compact widths and base semantics
are intentionally different:

``state[0:10] -> unified[0:10]`` and ``state[10:19] -> unified[68:77]``;
``action[0:10] -> unified[0:10]`` and ``action[10:15] -> unified[68:73]``.
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
)
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.eef import build_action_mask_2d, build_proprio_mask_2d
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec, unmap_from_unify
from openwam.dataloader.utils.video_io import decode_video_frames

HEAD_CAMERA = "observation.images.robot0_agentview_left"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"
RIGHT_CAMERA = "observation.images.robot0_agentview_right"
VIDEO_CAMERAS = (HEAD_CAMERA, WRIST_CAMERA, RIGHT_CAMERA)
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320
_WRIST_SLOT_H, _WRIST_SLOT_W = 128, 160

STATE_DIM = 19
ACTION_DIM = 15
ACTION_STATS_KEY = "robocasa365"
STATE_STATS_KEY = "robocasa365_state"
DEFAULT_ACTION_UNIFY_MAP = ["0-9", "68-72"]
DEFAULT_STATE_UNIFY_MAP = ["0-9", "68-76"]
ACTION_DIM_MASK = np.ones(ACTION_DIM, dtype=bool)
STATE_DIM_MASK = np.ones(STATE_DIM, dtype=bool)
_DEPLOY_RESOLVABLE_MODES = ("min-max", "z-score")


def _task_dir_name(data_root: str) -> str:
    return os.path.basename(data_root.rstrip("/")) or "robocasa365"


def _task_from_source_prefix(prefix: str) -> str:
    parts = str(prefix).strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else str(prefix)


@functools.lru_cache(maxsize=8)
def _episodes_with_offsets(data_root: str) -> pd.DataFrame:
    episodes = load_episodes_parquet(Path(data_root))
    episodes["_data_row_offset"] = compute_file_local_offsets(episodes, "data/chunk_index", "data/file_index")
    for camera in VIDEO_CAMERAS:
        chunk_key = f"videos/{camera}/chunk_index"
        if chunk_key in episodes.columns:
            episodes[f"_voff/{camera}"] = compute_file_local_offsets(episodes, chunk_key, f"videos/{camera}/file_index")
    return episodes


@functools.lru_cache(maxsize=8)
def _read_shard_cached(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["observation.state", "action"])


def _load_stats(path: str) -> tuple[dict, dict]:
    raw = np.load(path, allow_pickle=True).item()
    if ACTION_STATS_KEY not in raw or STATE_STATS_KEY not in raw:
        raise KeyError(f"{path} must contain separate {ACTION_STATS_KEY!r} and {STATE_STATS_KEY!r} blocks")

    def materialize(key: str, dim: int) -> dict:
        block = raw[key]
        result = {name: np.asarray(block[name], np.float32).reshape(-1) for name in STAT_KEYS}
        bad = {name: value.shape for name, value in result.items() if value.shape != (dim,)}
        if bad:
            raise ValueError(f"{path}:{key} expected {dim}-D stats, got {bad}")
        return result

    return materialize(ACTION_STATS_KEY, ACTION_DIM), materialize(STATE_STATS_KEY, STATE_DIM)


class RoboCasa365Dataset(BaseDataset):
    """One task view into an aggregated compact RoboCasa365 LeRobot v3 repo."""

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
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        unify_state_map: Optional[Any] = None,
        color_jitter: Optional[Any] = None,
        **_unused,
    ):
        super().__init__()
        self.data_root = str(data_root)
        self.task_name = task_name or _task_dir_name(self.data_root)
        self.normalize_mode = normalize_mode if normalize_mode not in ("", "none", "null") else None
        if num_frames < 2:
            raise ValueError(f"num_frames must be >=2, got {num_frames}")
        self.num_frames = int(num_frames)
        self.num_action_steps = self.num_frames - 1
        self.height, self.width = int(height), int(width)
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(f"Resolution {self.height}x{self.width} must be divisible by 32")
        self.repeat = int(repeat)
        self.split = split
        self.window_stride = max(1, int(window_stride))
        self.video_stride = max(1, int(video_stride))
        if (self.num_frames - 1) % self.video_stride != 0:
            valid = [value for value in range(1, self.num_frames) if (self.num_frames - 1) % value == 0]
            raise ValueError(
                f"(num_frames - 1) must be divisible by video_stride; got {self.num_frames}/{self.video_stride}, valid={valid}"
            )
        self._video_sample_indices = list(range(0, self.num_frames, self.video_stride))
        self.num_video_frames = len(self._video_sample_indices)
        self.multiview = bool(multiview)
        self.camera_layout = list(camera_layout) if camera_layout else list(VIDEO_CAMERAS)
        self._color_jitter = None
        if color_jitter and split == "train":
            get = color_jitter.get if hasattr(color_jitter, "get") else lambda key, default: default
            self._color_jitter = VideoColorJitter(
                brightness=float(get("brightness", 0.2)),
                contrast=float(get("contrast", 0.2)),
                saturation=float(get("saturation", 0.2)),
                hue=float(get("hue", 0.0)),
            )
        self._unify_action = bool(unify_action)
        self._action_dst_index = None
        self._state_dst_index = None
        self._unified_action_mask = None
        self._unified_state_mask = None
        if self._unify_action:
            action_spec = unify_action_map if unify_action_map is not None else DEFAULT_ACTION_UNIFY_MAP
            state_spec = unify_state_map if unify_state_map is not None else DEFAULT_STATE_UNIFY_MAP
            self._action_dst_index = parse_unify_spec(action_spec, UNIFY_DIM)
            self._state_dst_index = parse_unify_spec(state_spec, UNIFY_DIM)
            if self._action_dst_index.shape != (ACTION_DIM,):
                raise ValueError(f"unify_action_map must map {ACTION_DIM} action dims")
            if self._state_dst_index.shape != (STATE_DIM,):
                raise ValueError(f"unify_state_map must map {STATE_DIM} state dims")
            self._unified_action_mask = np.zeros(UNIFY_DIM, dtype=bool)
            self._unified_action_mask[self._action_dst_index] = ACTION_DIM_MASK
            self._unified_state_mask = np.zeros(UNIFY_DIM, dtype=bool)
            self._unified_state_mask[self._state_dst_index] = STATE_DIM_MASK

        with open(os.path.join(self.data_root, "meta", "info.json"), encoding="utf-8") as handle:
            info = json.load(handle)
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"RoboCasa365 compact reader requires LeRobot v3.0, got {info.get('codebase_version')!r}")
        state_shape = tuple(info.get("features", {}).get("observation.state", {}).get("shape", ()))
        action_shape = tuple(info.get("features", {}).get("action", {}).get("shape", ()))
        if state_shape != (STATE_DIM,) or action_shape != (ACTION_DIM,):
            raise ValueError(
                f"RoboCasa365 compact reader requires state{STATE_DIM}/action{ACTION_DIM}; got {state_shape}/{action_shape}. "
                "Run scripts/convert_robocasa365_compact_v3.py first."
            )
        required_cameras = VIDEO_CAMERAS if self.multiview else (HEAD_CAMERA,)
        missing_camera_features = [camera for camera in required_cameras if camera not in info.get("features", {})]
        if missing_camera_features:
            raise ValueError(
                "RoboCasa365 video feature(s) required by the configured view layout are missing: "
                f"{missing_camera_features}"
            )
        self._data_path_tmpl = info["data_path"]
        self._video_path_tmpl = info["video_path"]

        episodes_frame = _episodes_with_offsets(self.data_root)
        if task_name is not None:
            episodes_frame = episodes_frame[
                episodes_frame["source_prefix"].map(_task_from_source_prefix) == self.task_name
            ].reset_index(drop=True)
        if episodes_frame.empty:
            raise FileNotFoundError(f"No episodes for task {self.task_name!r} under {self.data_root}")
        required_video_columns = {
            column
            for camera in required_cameras
            for column in (
                f"_voff/{camera}",
                f"videos/{camera}/chunk_index",
                f"videos/{camera}/file_index",
            )
        }
        missing_video_columns = sorted(required_video_columns.difference(episodes_frame.columns))
        if missing_video_columns:
            raise ValueError(f"RoboCasa365 episode metadata is missing required video columns: {missing_video_columns}")
        episodes, self._ep_meta = [], {}
        for _, row in episodes_frame.iterrows():
            episode_index = int(row["episode_index"])
            episodes.append({"episode_index": episode_index, "length": int(row["length"]), "tasks": list(row["tasks"])})
            self._ep_meta[episode_index] = {
                "chunk": int(row["data/chunk_index"]),
                "file": int(row["data/file_index"]),
                "row_offset": int(row["_data_row_offset"]),
                "voff": {
                    camera: int(row[f"_voff/{camera}"])
                    for camera in VIDEO_CAMERAS
                    if f"_voff/{camera}" in episodes_frame.columns
                },
                "vcf": {
                    camera: (
                        int(row[f"videos/{camera}/chunk_index"]),
                        int(row[f"videos/{camera}/file_index"]),
                    )
                    for camera in VIDEO_CAMERAS
                    if f"videos/{camera}/chunk_index" in episodes_frame.columns
                },
            }
        self._episodes = episodes

        rng = random.Random(seed)
        order = list(range(len(episodes)))
        rng.shuffle(order)
        if val_ratio <= 0:
            n_val = 0
        elif val_ratio >= 1:
            n_val = len(episodes)
        else:
            n_val = max(1, int(len(episodes) * val_ratio))
        selected = sorted(order[:n_val]) if split == "val" else sorted(order[n_val:])
        if not selected:
            raise ValueError(f"No episodes for split={split!r}")
        self._ep_pos = selected
        self._ep_lengths = [int(episodes[index]["length"]) for index in selected]
        self._window_index: list[tuple[int, int]] = []
        for local_index, episode_length in enumerate(self._ep_lengths):
            if episode_length < 2:
                continue
            max_start = max(0, episode_length - self.num_frames) if split == "val" else max(0, episode_length - 2)
            self._window_index.extend((local_index, start) for start in range(0, max_start + 1, self.window_stride))
        if self.repeat > 1:
            self._window_index *= self.repeat
        if not self._window_index:
            raise ValueError(f"No valid windows for split={split!r}")
        self._val_samples = None
        if split == "val" and num_val_samples > 0:
            val_rng = random.Random(seed + 1)
            eligible = [index for index, length in enumerate(self._ep_lengths) if length >= 2]
            self._val_samples = []
            for _ in range(num_val_samples):
                local_index = eligible[val_rng.randint(0, len(eligible) - 1)]
                start = val_rng.randint(0, max(0, self._ep_lengths[local_index] - self.num_frames))
                self._val_samples.append((local_index, start))
        print(f"RoboCasa365Dataset[{self.task_name}]: {len(self._ep_pos)} episodes ({split})")

        self._action_stats = None
        self._state_stats = None
        self.normalization_stats_path = None
        if self.normalize_mode is not None:
            if self.normalize_mode not in _DEPLOY_RESOLVABLE_MODES:
                raise ValueError(
                    f"normalize_mode={self.normalize_mode!r} is not deploy-resolvable; use {_DEPLOY_RESOLVABLE_MODES} or null"
                )
            stats_path = normalization_stats_path or os.path.join(self.data_root, "meta", "normalization_stats.npy")
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(
                    f"compact RoboCasa365 stats not found: {stats_path}; conversion writes meta/normalization_stats.npy"
                )
            self._action_stats, self._state_stats = _load_stats(stats_path)
            self.normalization_stats_path = stats_path
            print(f"  [normalizer] {self.normalize_mode}: action={ACTION_DIM}D state={STATE_DIM}D, stats={stats_path}")
        else:
            print("  [normalizer] DISABLED")

    @property
    def action_dim(self) -> int:
        return UNIFY_DIM if self._unify_action else ACTION_DIM

    @property
    def state_dim(self) -> int:
        return UNIFY_DIM if self._unify_action else STATE_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        return dict(self._action_stats) if self._action_stats is not None else None

    def _unnormalize_action(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, np.float32)
        if self._action_stats is None or self.normalize_mode is None:
            return values.copy()
        if self.normalize_mode == "z-score":
            return values * self._action_stats["std"] + self._action_stats["mean"]
        return (values + 1.0) * 0.5 * (self._action_stats["max"] - self._action_stats["min"]) + self._action_stats[
            "min"
        ]

    def denormalize_action(self, action) -> np.ndarray:
        values = np.asarray(action, np.float32)
        if self._action_dst_index is not None:
            values = unmap_from_unify(values, self._action_dst_index).astype(np.float32)
        return self._unnormalize_action(values).astype(np.float32)

    def __len__(self) -> int:
        return len(self._val_samples) if self._val_samples is not None else len(self._window_index)

    def _data_file_path(self, chunk: int, file_index: int) -> str:
        return os.path.join(
            self.data_root,
            self._data_path_tmpl.format(chunk_index=chunk, file_index=file_index),
        )

    def _video_path(self, camera: str, chunk: int, file_index: int) -> str:
        return os.path.join(
            self.data_root,
            self._video_path_tmpl.format(video_key=camera, chunk_index=chunk, file_index=file_index),
        )

    def _read_rows(self, episode_index: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        metadata = self._ep_meta[episode_index]
        frame = _read_shard_cached(self._data_file_path(metadata["chunk"], metadata["file"]))
        offset = metadata["row_offset"]
        state = np.stack(frame["observation.state"].values[offset + start : offset + end]).astype(np.float32)
        action = np.stack(frame["action"].values[offset + start : offset + end]).astype(np.float32)
        return state, action

    def _read_video(self, episode_index: int, start: int, actual_end: int):
        metadata = self._ep_meta[episode_index]
        local = [start + offset for offset in self._video_sample_indices if start + offset < actual_end]
        if self.multiview:
            camera_frames = {}
            for camera in VIDEO_CAMERAS:
                chunk, file_index = metadata["vcf"][camera]
                slot_h, slot_w = (
                    (_HEAD_SLOT_H, _HEAD_SLOT_W)
                    if camera == HEAD_CAMERA
                    else (_WRIST_SLOT_H, _WRIST_SLOT_W)
                )
                camera_frames[camera] = decode_video_frames(
                    self._video_path(camera, chunk, file_index),
                    [metadata["voff"][camera] + index for index in local],
                    slot_h,
                    slot_w,
                )
            frames = [
                assemble_multiview_layout(
                    {camera: camera_frames[camera][index] for camera in VIDEO_CAMERAS},
                    self.camera_layout,
                    self.height,
                    self.width,
                )
                for index in range(len(local))
            ]
        else:
            head_chunk, head_file = metadata["vcf"][HEAD_CAMERA]
            head = decode_video_frames(
                self._video_path(HEAD_CAMERA, head_chunk, head_file),
                [metadata["voff"][HEAD_CAMERA] + index for index in local],
                self.height,
                self.width,
            )
            frames = [crop_and_resize(frame, self.height, self.width) for frame in head]
        if frames and len(frames) < self.num_video_frames:
            frames += [frames[-1]] * (self.num_video_frames - len(frames))
        return frames

    def _get_prompt(self, local_index: int) -> str:
        episode = self._episodes[self._ep_pos[local_index]]
        tasks = episode.get("tasks") or []
        # RoboCasa365 uses the native LeRobot task text without a wrapper.
        return str(tasks[0] if tasks else self.task_name)

    def _build_sample(self, local_index: int, start: int) -> dict:
        episode_index = int(self._episodes[self._ep_pos[local_index]]["episode_index"])
        episode_length = self._ep_lengths[local_index]
        actual_end = min(start + self.num_frames, episode_length)
        actual_length = actual_end - start
        if actual_length < 2:
            raise IndexError(f"window [{start},{start + self.num_frames}) has no action label")
        state_rows, action_rows = self._read_rows(episode_index, start, actual_end)
        if state_rows.shape[1] != STATE_DIM or action_rows.shape[1] != ACTION_DIM:
            raise ValueError(f"compact row shape mismatch: state={state_rows.shape}, action={action_rows.shape}")
        frames = self._read_video(episode_index, start, actual_end)
        n_valid_action = min(actual_length - 1, self.num_action_steps)
        action_raw = action_rows[:n_valid_action].copy()
        if action_raw.shape[0] < self.num_action_steps:
            pad = action_raw[-1:] if action_raw.shape[0] else np.zeros((1, ACTION_DIM), np.float32)
            action_raw = np.concatenate(
                [action_raw, np.repeat(pad, self.num_action_steps - action_raw.shape[0], axis=0)], axis=0
            )
        proprio_raw = state_rows[0:1].copy()
        action = apply_normalization(action_raw, self._action_stats, self.normalize_mode).astype(np.float32)
        proprio = apply_normalization(proprio_raw, self._state_stats, self.normalize_mode).astype(np.float32)
        if self._action_dst_index is not None:
            action, _ = map_to_unify(action, self._action_dst_index, UNIFY_DIM)
            proprio, _ = map_to_unify(proprio, self._state_dst_index, UNIFY_DIM)
            action_mask_dims = self._unified_action_mask
            state_mask_dims = self._unified_state_mask
            action_width = state_width = UNIFY_DIM
        else:
            action_mask_dims, state_mask_dims = ACTION_DIM_MASK, STATE_DIM_MASK
            action_width, state_width = ACTION_DIM, STATE_DIM
        action_mask = torch.from_numpy(
            build_action_mask_2d(self.num_action_steps, action_width, n_valid_action, dim_mask=action_mask_dims)
        )
        proprio_mask = torch.from_numpy(build_proprio_mask_2d(state_width, enabled=True, dim_mask=state_mask_dims))
        video_mask = torch.tensor(
            [start + offset < actual_end for offset in self._video_sample_indices], dtype=torch.bool
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
            "prompt": self._get_prompt(local_index),
            "episode_index": episode_index,
            "start_frame": start,
            "episode_length": episode_length,
            "task_name": self.task_name,
        }

    def __getitem__(self, index):
        if self._val_samples is not None:
            local_index, start = self._val_samples[index]
        else:
            local_index, start = self._window_index[index]
        sample = self._build_sample(local_index, start)
        if self._color_jitter is not None:
            sample["video"] = self._color_jitter.apply({"video": sample["video"]})["video"]
            sample["first_frame_image"] = [sample["video"][0]]
        return sample


class MultiTaskRoboCasa365Dataset(BaseDataset):
    """Concatenate task-filtered views across one or more compact v3 repos."""

    @classmethod
    def from_config(cls, config, split: str = "train"):
        normalize_mode = get_cfg(config, "normalize_mode", "min-max")
        if isinstance(normalize_mode, str) and normalize_mode.lower() in ("", "none", "null"):
            normalize_mode = None
        camera_layout = get_cfg(config, "camera_layout", None)
        return cls(
            dataset_dir=get_cfg(config, "dataset_dir"),
            task_name=get_cfg(config, "task_name", None),
            task_roots=get_cfg(config, "task_roots", None),
            normalize_mode=normalize_mode,
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
            camera_layout=list(camera_layout) if camera_layout is not None else None,
            temporal_compression=int(get_cfg(config, "temporal_compression", 4)),
            causal_temporal=bool(get_cfg(config, "causal_temporal", True)),
            unify_action=bool(get_cfg(config, "unify_action", False)),
            unify_action_map=get_cfg(config, "unify_action_map", None),
            unify_state_map=get_cfg(config, "unify_state_map", None),
            color_jitter=get_cfg(config, "color_jitter", None),
            seed=int(get_cfg(config, "seed", 42)),
        )

    def __init__(
        self,
        dataset_dir,
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
            raise FileNotFoundError(f"No RoboCasa365 tasks found under {dataset_dir}")
        if normalize_mode is not None and normalization_stats_path is None and len({repo for _, repo in roots}) > 1:
            raise ValueError("multi-repo RoboCasa365 requires normalization_stats_path")
        print(f"MultiTaskRoboCasa365Dataset: {len(roots)} task bucket(s)")
        self._datasets = [
            RoboCasa365Dataset(
                data_root=repo,
                task_name=name,
                normalize_mode=normalize_mode,
                normalization_stats_path=normalization_stats_path,
                **kwargs,
            )
            for name, repo in roots
        ]
        self._cum = np.cumsum([0] + [len(dataset) for dataset in self._datasets]).astype(np.int64)
        self.normalization_stats_path = self._datasets[0].normalization_stats_path

    @staticmethod
    def _resolve_task_roots(dataset_dir, task_name: Optional[str], task_roots: Optional[list]):
        repos = [dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir)
        pairs = []
        for repo in repos:
            episodes = _episodes_with_offsets(str(repo))
            pairs.extend(
                (task, str(repo))
                for task in sorted({_task_from_source_prefix(prefix) for prefix in episodes["source_prefix"]})
            )
        available = {task for task, _ in pairs}
        if task_name is not None:
            selected = {task_name} if task_name in available else set()
        elif task_roots:
            selected = set(task_roots).intersection(available)
        else:
            selected = available
        return [(task, repo) for task, repo in pairs if task in selected]

    @property
    def action_dim(self) -> int:
        return self._datasets[0].action_dim if self._datasets else ACTION_DIM

    @property
    def state_dim(self) -> int:
        return self._datasets[0].state_dim if self._datasets else STATE_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        return self._datasets[0].normalization_stats if self._datasets else None

    def denormalize_action(self, action) -> np.ndarray:
        return self._datasets[0].denormalize_action(action)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, index):
        dataset_index = int(np.searchsorted(self._cum, index, side="right") - 1)
        return self._datasets[dataset_index][index - int(self._cum[dataset_index])]


__all__ = [
    "ACTION_DIM",
    "ACTION_STATS_KEY",
    "DEFAULT_ACTION_UNIFY_MAP",
    "DEFAULT_STATE_UNIFY_MAP",
    "HEAD_CAMERA",
    "MultiTaskRoboCasa365Dataset",
    "RIGHT_CAMERA",
    "RoboCasa365Dataset",
    "STATE_DIM",
    "STATE_STATS_KEY",
    "VIDEO_CAMERAS",
    "WRIST_CAMERA",
]
