"""EBench dataloader with OpenWAM 80-D action-space projection.

EBench stores bimanual control in LeRobot-style per-episode parquet files:

  * ``action.joints``: 12 arm joints, left arm first then right arm
  * ``action.gripper``: 4 finger/gripper values, two per hand
  * ``action.base`` / ``action.base_delta``: 3 mobile-base values
  * matching ``state.*`` keys for proprio

OpenWAM's 80-D pretraining head has a semantic layout shared across robots:

  * ``[0:10)`` left EEF xyz/rot6d/gripper
  * ``[10:32)`` left hand/joint slots
  * ``[32:42)`` right EEF xyz/rot6d/gripper
  * ``[42:64)`` right hand/joint slots
  * ``[64:80)`` reserved

This reader maps EBench's physically meaningful dimensions into that space:
arm joints and two-finger grippers go into hand slots, mean gripper values are
also exposed in the scalar gripper slots, and base motion goes to reserved
slots ``[64:67)``. The emitted action/proprio masks are ``(T, 80)`` and
``(1, 80)`` respectively, so only real EBench dimensions participate in loss.
"""

from __future__ import annotations

import json
import logging
import functools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout, format_prompt_for_inference
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import UNIFY_DIM
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames

logger = logging.getLogger(__name__)

EBENCH_ACTION_DIM = int(UNIFY_DIM)
EBENCH_STATE_DIM = int(UNIFY_DIM)

EBENCH_ACTION_KEYS = ("action.joints", "action.gripper", "action.base")
EBENCH_ACTION_DELTA_BASE_KEYS = ("action.joints", "action.gripper", "action.base_delta")
EBENCH_STATE_KEYS = ("state.joints", "state.gripper", "state.base")

EBENCH80_LEFT_SCALAR_GRIPPER = 9
EBENCH80_LEFT_HAND_START = 10
EBENCH80_RIGHT_SCALAR_GRIPPER = 41
EBENCH80_RIGHT_HAND_START = 42
EBENCH80_BASE_START = 64


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_plain_list(value) -> Optional[list]:
    if value is None:
        return None
    try:
        from omegaconf import ListConfig, OmegaConf

        if isinstance(value, ListConfig):
            return list(OmegaConf.to_container(value, resolve=True))
    except Exception:
        pass
    if isinstance(value, str):
        return [value]
    return list(value)


def _feature_width(info: dict, key: str) -> int:
    feat = info.get("features", {}).get(key)
    if feat is None:
        raise KeyError(f"EBench info.json missing feature {key!r}")
    shape = feat.get("shape") or []
    if not shape:
        raise ValueError(f"EBench feature {key!r} has no shape in info.json")
    return int(shape[0])


def _column_matrix(frame: pd.DataFrame, key: str, expected_width: int) -> np.ndarray:
    if key not in frame.columns:
        raise KeyError(f"EBench parquet window missing column {key!r}")
    arr = np.stack(frame[key].to_numpy()).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_width:
        raise ValueError(f"EBench column {key!r} expected shape (T, {expected_width}), got {arr.shape}")
    return arr


def _raw19_from_frame(frame: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """Concatenate EBench joint/gripper/base columns into raw 19-D vectors."""
    widths = (12, 4, 3)
    arrays = [_column_matrix(frame, key, width) for key, width in zip(keys, widths)]
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def _raw19_to_ebench80(raw: np.ndarray) -> np.ndarray:
    """Map raw EBench 19-D vectors into the OpenWAM 80-D semantic space.

    Raw layout:
      ``[0:6]`` left arm joints, ``[6:12]`` right arm joints,
      ``[12:14]`` left two-finger gripper, ``[14:16]`` right two-finger gripper,
      ``[16:19]`` base.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != 19:
        raise ValueError(f"EBench raw action/state must be 19-D, got shape {raw.shape}")

    out = np.zeros((*raw.shape[:-1], EBENCH_ACTION_DIM), dtype=np.float32)
    out[..., EBENCH80_LEFT_HAND_START : EBENCH80_LEFT_HAND_START + 6] = raw[..., 0:6]
    out[..., EBENCH80_RIGHT_HAND_START : EBENCH80_RIGHT_HAND_START + 6] = raw[..., 6:12]
    out[..., EBENCH80_LEFT_HAND_START + 6 : EBENCH80_LEFT_HAND_START + 8] = raw[..., 12:14]
    out[..., EBENCH80_RIGHT_HAND_START + 6 : EBENCH80_RIGHT_HAND_START + 8] = raw[..., 14:16]
    out[..., EBENCH80_LEFT_SCALAR_GRIPPER] = raw[..., 12:14].mean(axis=-1)
    out[..., EBENCH80_RIGHT_SCALAR_GRIPPER] = raw[..., 14:16].mean(axis=-1)
    out[..., EBENCH80_BASE_START : EBENCH80_BASE_START + 3] = raw[..., 16:19]
    return out


def ebench80_dim_mask() -> np.ndarray:
    mask = np.zeros(EBENCH_ACTION_DIM, dtype=bool)
    mask[EBENCH80_LEFT_SCALAR_GRIPPER] = True
    mask[EBENCH80_LEFT_HAND_START : EBENCH80_LEFT_HAND_START + 8] = True
    mask[EBENCH80_RIGHT_SCALAR_GRIPPER] = True
    mask[EBENCH80_RIGHT_HAND_START : EBENCH80_RIGHT_HAND_START + 8] = True
    mask[EBENCH80_BASE_START : EBENCH80_BASE_START + 3] = True
    return mask


EBENCH80_DIM_MASK = ebench80_dim_mask()


def _neutral_stats() -> Dict[str, np.ndarray]:
    zeros = np.zeros(EBENCH_ACTION_DIM, dtype=np.float32)
    ones = np.ones(EBENCH_ACTION_DIM, dtype=np.float32)
    return {
        "mean": zeros.copy(),
        "std": ones.copy(),
        "min": zeros.copy(),
        "max": ones.copy(),
        "q01": -ones.copy(),
        "q99": ones.copy(),
    }


def _json_count(stats_entry: dict) -> int:
    count = stats_entry.get("count", 0)
    if isinstance(count, list):
        return int(count[0]) if count else 0
    return int(count)


def _merge_scalar_stats(entries: list[dict]) -> dict:
    """Merge per-episode stats from EBench episodes_stats.jsonl."""
    nonempty = [(e, _json_count(e)) for e in entries if _json_count(e) > 0]
    if not nonempty:
        raise ValueError("Cannot merge empty stats entries")

    count = float(sum(c for _, c in nonempty))
    means = np.stack([np.asarray(e["mean"], dtype=np.float64) for e, _ in nonempty])
    stds = np.stack([np.asarray(e["std"], dtype=np.float64) for e, _ in nonempty])
    counts = np.asarray([c for _, c in nonempty], dtype=np.float64)

    mean = (means * counts[:, None]).sum(axis=0) / count
    second = ((stds**2 + means**2) * counts[:, None]).sum(axis=0) / count
    var = np.maximum(second - mean**2, 0.0)
    return {
        "min": np.min(np.stack([np.asarray(e["min"], dtype=np.float64) for e, _ in nonempty]), axis=0),
        "max": np.max(np.stack([np.asarray(e["max"], dtype=np.float64) for e, _ in nonempty]), axis=0),
        "mean": mean,
        "std": np.sqrt(var),
        "count": int(count),
    }


def _raw_stats_to_80(stats_by_key: dict, keys: Sequence[str]) -> dict:
    raw = {}
    raw["mean"] = np.concatenate([np.asarray(stats_by_key[k]["mean"], dtype=np.float32) for k in keys], axis=0)
    raw["std"] = np.concatenate([np.asarray(stats_by_key[k]["std"], dtype=np.float32) for k in keys], axis=0)
    raw["min"] = np.concatenate([np.asarray(stats_by_key[k]["min"], dtype=np.float32) for k in keys], axis=0)
    raw["max"] = np.concatenate([np.asarray(stats_by_key[k]["max"], dtype=np.float32) for k in keys], axis=0)

    out = _neutral_stats()
    out["mean"] = _raw19_to_ebench80(raw["mean"])
    out["std"] = np.maximum(_raw19_to_ebench80(raw["std"]), 1e-3)
    out["min"] = _raw19_to_ebench80(raw["min"])
    out["max"] = _raw19_to_ebench80(raw["max"])
    out["q01"] = out["min"].copy()
    out["q99"] = out["max"].copy()

    for name in out:
        out[name] = out[name].astype(np.float32)
    out["std"][~EBENCH80_DIM_MASK] = 1.0
    out["min"][~EBENCH80_DIM_MASK] = 0.0
    out["max"][~EBENCH80_DIM_MASK] = 1.0
    out["q01"][~EBENCH80_DIM_MASK] = -1.0
    out["q99"][~EBENCH80_DIM_MASK] = 1.0
    return out


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_stats_from_bucket(bucket_dir: Path, keys: Sequence[str]) -> Tuple[dict, int]:
    stats_path = bucket_dir / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"EBench stats file missing: {stats_path}")

    by_key: dict[str, list[dict]] = {k: [] for k in keys}
    total_count = 0
    for row in _read_jsonl(stats_path):
        episode_stats = row.get("stats", {})
        for key in keys:
            if key not in episode_stats:
                raise KeyError(f"{stats_path} missing stats for {key!r}")
            by_key[key].append(episode_stats[key])
        first_key = keys[0]
        total_count += _json_count(episode_stats[first_key])

    merged = {key: _merge_scalar_stats(entries) for key, entries in by_key.items()}
    return _raw_stats_to_80(merged, keys), total_count


def _merge_80_stats(parts: list[Tuple[dict, int]]) -> dict:
    parts = [(s, int(c)) for s, c in parts if int(c) > 0]
    if not parts:
        return _neutral_stats()
    counts = np.asarray([c for _, c in parts], dtype=np.float64)
    total = counts.sum()
    out = _neutral_stats()
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        values = np.stack([np.asarray(stats[key], dtype=np.float64) for stats, _ in parts])
        if key == "mean":
            out[key] = (values * counts[:, None]).sum(axis=0) / total
        elif key == "std":
            means = np.stack([np.asarray(stats["mean"], dtype=np.float64) for stats, _ in parts])
            second = ((values**2 + means**2) * counts[:, None]).sum(axis=0) / total
            mean = out["mean"].astype(np.float64)
            out[key] = np.sqrt(np.maximum(second - mean**2, 0.0))
        elif key in ("min", "q01"):
            out[key] = values.min(axis=0)
        else:
            out[key] = values.max(axis=0)
    for key in out:
        out[key] = np.asarray(out[key], dtype=np.float32)
    out["std"] = np.maximum(out["std"], 1e-3)
    out["std"][~EBENCH80_DIM_MASK] = 1.0
    return out


def _stats_cache_payload(action_stats: dict, state_stats: dict, num_timesteps: int) -> dict:
    return {
        "ebench80": action_stats,
        "ebench80_state": state_stats,
        "num_timesteps": int(num_timesteps),
        "action_dim_mask": EBENCH80_DIM_MASK.astype(np.bool_),
    }


def _load_or_build_stats(
    buckets: Sequence[Path],
    action_keys: Sequence[str],
    state_keys: Sequence[str],
    stats_path: Optional[str],
) -> Tuple[dict, dict, Optional[str]]:
    if stats_path:
        path = Path(stats_path)
        if path.exists():
            raw = np.load(path, allow_pickle=True).item()
            action_stats = raw["ebench80"] if "ebench80" in raw else raw
            state_stats = raw.get("ebench80_state", action_stats)
            return action_stats, state_stats, str(path)

    action_parts = []
    state_parts = []
    total = 0
    for bucket in buckets:
        action_stats, count = _build_stats_from_bucket(bucket, action_keys)
        state_stats, _ = _build_stats_from_bucket(bucket, state_keys)
        action_parts.append((action_stats, count))
        state_parts.append((state_stats, count))
        total += int(count)

    action_stats = _merge_80_stats(action_parts)
    state_stats = _merge_80_stats(state_parts)
    if stats_path:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, _stats_cache_payload(action_stats, state_stats, total))
        logger.info("Saved EBench 80-D normalization stats to %s", path)
        return action_stats, state_stats, str(path)
    return action_stats, state_stats, None


def discover_ebench_buckets(
    dataset_dir: str,
    *,
    groups: Optional[Sequence[str]] = None,
    buckets: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Discover EBench task buckets containing ``meta/info.json``."""
    root = Path(dataset_dir)
    if buckets:
        resolved = [root / b for b in buckets]
    else:
        groups = list(groups or ("simple_pnp", "teleop_tasks"))
        resolved = []
        for group in groups:
            group_dir = root / group
            if not group_dir.is_dir():
                continue
            for child in sorted(group_dir.iterdir()):
                if (child / "meta" / "info.json").is_file():
                    resolved.append(child)
    existing = [p for p in resolved if (p / "meta" / "info.json").is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No EBench buckets found under {root}. "
            "Expected paths like simple_pnp/task1/meta/info.json or teleop_tasks/peg_in_hole/meta/info.json."
        )
    return existing


class EBenchDataset(BaseDataset):
    """Single EBench bucket reader that emits OpenWAM 80-D action/proprio."""

    DATASET_NAME = "EBench"

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        multiview: bool = True,
        target_camera: str = "video.overlook_camera_view",
        camera_layout: Optional[Sequence[str]] = None,
        normalize_mode: Optional[str] = "z-score",
        normalization_stats_path: Optional[str] = None,
        action_stats: Optional[dict] = None,
        state_stats: Optional[dict] = None,
        action_key_variant: str = "absolute_base",
        dataset_id: Optional[str] = None,
        **_unused: Any,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = dataset_id or "/".join(self._dataset_dir.parts[-2:])
        self._split = split
        self._num_frames = int(num_frames)
        self._video_stride = max(1, int(video_stride))
        self._window_stride = max(1, int(window_stride))
        self._height = int(height)
        self._width = int(width)
        self._multiview = bool(multiview)
        self._target_camera = target_camera
        self._camera_layout = list(camera_layout or [target_camera, "video.left_camera_view", "video.right_camera_view"])
        self._normalize_mode = normalize_mode
        self.normalization_stats_path = normalization_stats_path
        self._action_key_variant = action_key_variant

        if action_key_variant == "absolute_base":
            self._action_keys = EBENCH_ACTION_KEYS
        elif action_key_variant == "delta_base":
            self._action_keys = EBENCH_ACTION_DELTA_BASE_KEYS
        else:
            raise ValueError("EBench action_key_variant must be 'absolute_base' or 'delta_base'")
        self._state_keys = EBENCH_STATE_KEYS
        self._data_columns = list(dict.fromkeys((*self._action_keys, *self._state_keys, "task_index")))

        info_path = self._dataset_dir / "meta" / "info.json"
        with info_path.open() as f:
            info = json.load(f)
        self._fps = float(info["fps"])
        self._chunks_size = int(info.get("chunks_size", 1000))
        self._data_path_template = info["data_path"]
        self._video_path_template = info["video_path"]
        self._validate_schema(info)
        self._load_episode_table = functools.lru_cache(maxsize=16)(self._read_episode_table_uncached)

        self._action_stats = action_stats
        self._state_stats = state_stats if state_stats is not None else action_stats

        self._episodes = list(_read_jsonl(self._dataset_dir / "meta" / "episodes.jsonl"))
        self._tasks = self._load_tasks()
        self._episode_tasks = {
            int(ep["episode_index"]): (ep.get("tasks") or []) for ep in self._episodes
        }
        self._episodes = self._select_split(self._episodes, info.get("splits", {}))
        if not self._episodes:
            raise ValueError(f"EBench({self._dataset_id}) split={split!r} has no episodes")

        self._video_sample_indices = np.arange(0, self._num_frames, self._video_stride, dtype=np.int64)
        if self._video_sample_indices[-1] != self._num_frames - 1:
            self._video_sample_indices = np.append(self._video_sample_indices, self._num_frames - 1)
        self._num_video_frames = int(len(self._video_sample_indices))

        lengths = np.asarray([int(ep["length"]) for ep in self._episodes], dtype=np.int64)
        min_window_len = self._num_frames if split == "val" else 2
        n_starts = np.where(
            lengths >= min_window_len,
            (lengths - min_window_len) // self._window_stride + 1,
            0,
        ).astype(np.int64)
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        self._n_total = int(self._cum_n_starts[-1])
        if self._n_total <= 0:
            raise ValueError(f"EBench({self._dataset_id}) split={split!r} produced 0 windows")

        logger.info(
            "EBench(%s, %s): %d episodes, %d windows, fps=%.1f, multiview=%s, normalize=%s",
            self._dataset_id,
            split,
            len(self._episodes),
            self._n_total,
            self._fps,
            self._multiview,
            self._normalize_mode,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_load_episode_table", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._load_episode_table = functools.lru_cache(maxsize=16)(self._read_episode_table_uncached)

    def _validate_schema(self, info: dict) -> None:
        for key, width in (
            ("action.joints", 12),
            ("action.gripper", 4),
            ("state.joints", 12),
            ("state.gripper", 4),
            ("state.base", 3),
        ):
            got = _feature_width(info, key)
            if got != width:
                raise ValueError(f"EBench({self._dataset_id}) {key} width {got} != expected {width}")
        base_key = self._action_keys[-1]
        got = _feature_width(info, base_key)
        if got != 3:
            raise ValueError(f"EBench({self._dataset_id}) {base_key} width {got} != expected 3")
        for cam in set(self._camera_layout + [self._target_camera]):
            if cam not in info.get("features", {}):
                raise KeyError(f"EBench({self._dataset_id}) camera {cam!r} not present in info.json")

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self._dataset_dir / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            return {}
        tasks = {}
        for row in _read_jsonl(tasks_path):
            tasks[int(row["task_index"])] = str(row["task"])
        return tasks

    def _select_split(self, episodes: list[dict], splits: dict) -> list[dict]:
        if self._split not in splits:
            if self._split == "train":
                return episodes
            return []
        spec = str(splits[self._split])
        if ":" not in spec:
            return episodes
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else len(episodes)
        return [ep for ep in episodes if start <= int(ep["episode_index"]) < end]

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._n_total:
            raise IndexError(f"EBench idx {idx} out of range [0, {self._n_total})")
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        ep = self._episodes[ep_local]
        ep_idx = int(ep["episode_index"])
        ep_len = int(ep["length"])
        actual_raw_len = min(self._num_frames, ep_len - offset)

        frame = self._read_episode_data(ep_idx, offset, actual_raw_len)
        action, action_mask = self._build_action(frame, actual_raw_len)
        proprio, proprio_mask = self._build_proprio(frame)
        video = self._decode_window_video(ep_idx, offset, actual_raw_len)
        video_mask = torch.from_numpy(self._video_sample_indices < actual_raw_len)

        return {
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]] if video else [],
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "video_mask": video_mask,
            "proprio": torch.from_numpy(proprio).float(),
            "proprio_mask": torch.from_numpy(proprio_mask),
            "prompt": self._prompt_for_episode(ep_idx, frame),
        }

    def _read_episode_data(self, episode_index: int, offset: int, length: int) -> pd.DataFrame:
        table = self._load_episode_table(episode_index)
        return table.slice(offset, length).to_pandas()

    def _read_episode_table_uncached(self, episode_index: int):
        chunk = episode_index // self._chunks_size
        path = self._dataset_dir / self._data_path_template.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            chunk_index=chunk,
        )
        return pq.read_table(path, memory_map=True, columns=self._data_columns)

    def _build_action(self, frame: pd.DataFrame, actual_raw_len: int) -> tuple[np.ndarray, np.ndarray]:
        T_action = self._num_frames - 1
        action = np.zeros((T_action, EBENCH_ACTION_DIM), dtype=np.float32)
        mask = np.zeros((T_action, EBENCH_ACTION_DIM), dtype=bool)
        n_valid = min(actual_raw_len, T_action)
        if n_valid > 0:
            raw = _raw19_from_frame(frame.iloc[:n_valid], self._action_keys)
            mapped = _raw19_to_ebench80(raw)
            mapped = self._normalize(mapped, self._action_stats)
            action[:n_valid] = mapped
            mask[:n_valid, EBENCH80_DIM_MASK] = True
        return action, mask

    def _build_proprio(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = _raw19_from_frame(frame.iloc[:1], self._state_keys)
        proprio = _raw19_to_ebench80(raw)
        proprio = self._normalize(proprio, self._state_stats)
        mask = np.zeros((1, EBENCH_STATE_DIM), dtype=bool)
        mask[:, EBENCH80_DIM_MASK] = True
        return proprio.astype(np.float32), mask

    def _normalize(self, arr: np.ndarray, stats: Optional[dict]) -> np.ndarray:
        if self._normalize_mode in (None, "none", "null"):
            return arr.astype(np.float32)
        if stats is None:
            raise ValueError(
                f"EBench({self._dataset_id}) normalize_mode={self._normalize_mode!r} but no stats were provided"
            )
        normalized = apply_normalization(arr, stats, self._normalize_mode)
        # Ensure all masked-out slots stay exactly zero after z-score/min-max.
        normalized[..., ~EBENCH80_DIM_MASK] = 0.0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.astype(np.float32)

    def _prompt_for_episode(self, episode_index: int, frame: pd.DataFrame) -> str:
        episode_tasks = self._episode_tasks.get(episode_index) or []
        if episode_tasks:
            return format_prompt_for_inference(str(episode_tasks[0]))
        task_idx = int(frame["task_index"].iloc[0]) if "task_index" in frame.columns else None
        if task_idx is not None and task_idx in self._tasks:
            return format_prompt_for_inference(self._tasks[task_idx])
        return format_prompt_for_inference("Complete the manipulation task.")

    def _decode_window_video(self, episode_index: int, offset: int, actual_raw_len: int) -> List:
        real_indices = self._video_sample_indices[self._video_sample_indices < actual_raw_len]
        real_indices = (real_indices + offset).tolist()
        if not real_indices:
            real_indices = [offset]

        frames_by_camera = {}
        cameras = self._camera_layout if self._multiview else [self._target_camera]
        for cam in cameras:
            path = self._video_path(cam, episode_index)
            h, w = self._camera_slot_size(cam)
            frames_by_camera[cam] = _decode_video_frames(str(path), real_indices, h, w)

        n_real = len(next(iter(frames_by_camera.values())))
        for cam, frames in list(frames_by_camera.items()):
            if n_real < self._num_video_frames:
                frames_by_camera[cam] = frames + [frames[-1]] * (self._num_video_frames - n_real)

        if not self._multiview:
            return frames_by_camera[self._target_camera]

        video = []
        for i in range(self._num_video_frames):
            video.append(
                assemble_multiview_layout(
                    {cam: frames_by_camera[cam][i] for cam in cameras},
                    list(cameras),
                    out_h=self._height,
                    out_w=self._width,
                )
            )
        return video

    def _video_path(self, camera: str, episode_index: int) -> Path:
        chunk = episode_index // self._chunks_size
        return self._dataset_dir / self._video_path_template.format(
            video_key=camera,
            episode_chunk=chunk,
            episode_index=episode_index,
            chunk_index=chunk,
        )

    def _camera_slot_size(self, camera: str) -> tuple[int, int]:
        if not self._multiview:
            return self._height, self._width
        top_h = int(round(self._height * 2.0 / 3.0))
        if camera == self._camera_layout[0]:
            return top_h, self._width
        bottom_h = self._height - top_h
        half_w = self._width // 2
        right_w = self._width - half_w
        if len(self._camera_layout) > 1 and camera == self._camera_layout[1]:
            return bottom_h, half_w
        return bottom_h, right_w

    @property
    def action_dim(self) -> int:
        return EBENCH_ACTION_DIM

    @property
    def state_dim(self) -> int:
        return EBENCH_STATE_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        return self._action_stats if self._normalize_mode not in (None, "none", "null") else None

    @classmethod
    def from_config(cls, config, split: str = "train"):
        dataset_dir = _cfg_get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError("EBenchDataset: missing dataloader.dataset_dir")

        groups = _as_plain_list(_cfg_get(config, "groups", ["simple_pnp", "teleop_tasks"]))
        buckets_cfg = _as_plain_list(_cfg_get(config, "buckets", None))
        buckets = discover_ebench_buckets(dataset_dir, groups=groups, buckets=buckets_cfg)

        normalize_mode = _cfg_get(config, "normalize_mode", "z-score")
        stats_path = _cfg_get(config, "normalization_stats_path", None)
        action_variant = _cfg_get(config, "action_key_variant", "absolute_base")
        if action_variant not in ("absolute_base", "delta_base"):
            raise ValueError("EBench action_key_variant must be 'absolute_base' or 'delta_base'")
        action_keys = EBENCH_ACTION_KEYS if action_variant == "absolute_base" else EBENCH_ACTION_DELTA_BASE_KEYS

        action_stats = None
        state_stats = None
        resolved_stats_path = None
        if normalize_mode not in (None, "none", "null"):
            action_stats, state_stats, resolved_stats_path = _load_or_build_stats(
                buckets,
                action_keys,
                EBENCH_STATE_KEYS,
                stats_path,
            )

        common = {
            "split": split,
            "num_frames": int(_cfg_get(config, "num_frames", 33)),
            "video_stride": int(_cfg_get(config, "video_stride", 4)),
            "window_stride": int(_cfg_get(config, "window_stride", 1)),
            "height": int(_cfg_get(config, "height", 384)),
            "width": int(_cfg_get(config, "width", 320)),
            "multiview": bool(_cfg_get(config, "multiview", True)),
            "target_camera": _cfg_get(config, "target_camera", "video.overlook_camera_view"),
            "camera_layout": _as_plain_list(
                _cfg_get(
                    config,
                    "camera_layout",
                    ["video.overlook_camera_view", "video.left_camera_view", "video.right_camera_view"],
                )
            ),
            "normalize_mode": normalize_mode,
            "normalization_stats_path": resolved_stats_path,
            "action_stats": action_stats,
            "state_stats": state_stats,
            "action_key_variant": action_variant,
        }

        readers = [
            cls(str(bucket), dataset_id=str(bucket.relative_to(Path(dataset_dir))), **common)
            for bucket in buckets
        ]
        if len(readers) == 1:
            return readers[0]
        return MultiEBenchDataset(readers, normalization_stats_path=resolved_stats_path, normalization_stats=action_stats)


class MultiEBenchDataset(BaseDataset):
    """Aggregate multiple EBench task buckets."""

    def __init__(
        self,
        buckets: Sequence[EBenchDataset],
        *,
        normalization_stats_path: Optional[str] = None,
        normalization_stats: Optional[dict] = None,
    ):
        if not buckets:
            raise ValueError("MultiEBenchDataset requires at least one bucket")
        self._buckets = list(buckets)
        lens = np.asarray([len(b) for b in self._buckets], dtype=np.int64)
        self._cum_lens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.normalization_stats_path = normalization_stats_path
        self._normalization_stats = normalization_stats
        logger.info(
            "MultiEBenchDataset: %d buckets, %d windows",
            len(self._buckets),
            len(self),
        )

    def __len__(self) -> int:
        return int(self._cum_lens[-1])

    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        if not 0 <= idx < n:
            raise IndexError(f"MultiEBenchDataset idx {idx} out of range [0, {n})")
        bi = int(np.searchsorted(self._cum_lens, idx, side="right") - 1)
        local = idx - int(self._cum_lens[bi])
        return self._buckets[bi][local]

    @property
    def action_dim(self) -> int:
        return EBENCH_ACTION_DIM

    @property
    def state_dim(self) -> int:
        return EBENCH_STATE_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        return self._normalization_stats

    @property
    def buckets(self) -> List[EBenchDataset]:
        return self._buckets


__all__ = [
    "EBENCH80_DIM_MASK",
    "EBENCH_ACTION_DIM",
    "EBENCH_STATE_DIM",
    "EBenchDataset",
    "MultiEBenchDataset",
    "discover_ebench_buckets",
    "_raw19_to_ebench80",
]
