"""LIBERO LeRobot v3 dataloader.

Targets the official ``nvidia/LIBERO_LeRobot_v3`` schema and the compatible
``HuggingFaceVLA/libero`` conversion:

* action: 7-D OSC command
* observation.state: 8-D EEF/gripper state (not consumed; see below)
* observation.images.image: agent-view RGB
* observation.images.wrist_image or image2: wrist RGB
* task_index: language lookup through meta/tasks.parquet

LIBERO action commands and proprioception are different physical spaces
(delta command versus absolute EEF/gripper state). OpenWAM's deploy normalizer
currently shares one stats block between action and proprio, so this reader
deliberately disables proprioception instead of applying action statistics to
the 8-D state. Train with ``model.architecture.use_proprioception=false``.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

_ACTION_DIM = 7
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def _as_priority(value: Optional[Sequence[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    return next((key for key in priorities if key in features), None)


@functools.lru_cache(maxsize=8)
def _load_libero_stats(path: str, action_mode: str, normalize_mode: str) -> dict:
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    if stats_path.suffix == ".json":
        with stats_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = np.load(stats_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"LIBERO stats must be a mapping, got {type(payload).__name__}")
    raw = payload.get(action_mode, payload)
    stats = materialize_eef_stats(
        raw,
        normalize_mode,
        dim=_ACTION_DIM,
        strict_minmax=False,
        source_hint=f"{stats_path}:{action_mode}",
    )
    bad = {key: value.shape for key, value in stats.items() if value.shape != (_ACTION_DIM,)}
    if bad:
        raise ValueError(f"LIBERO normalization stats must be 7-D, got {bad}")
    return stats


class LiberoDataset(LeRobotV3Reader):
    """Single-bucket LIBERO reader for LeRobot v3 parquet/video datasets."""

    DATASET_NAME = "LIBERO"
    ACTION_DIM = _ACTION_DIM
    NEEDED_COLS = ("action", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEPLOY_ACTION_MODE = "libero"

    HEAD_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.image",
        "observation.images.agentview_image",
    )
    WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.wrist_image",
        "observation.images.image2",
        "observation.images.robot0_eye_in_hand_image",
    )
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "head_camera_priority",
        "wrist_camera_priority",
        "normalization_stats_path",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = "libero",
        head_camera_priority: Optional[Sequence[str]] = None,
        wrist_camera_priority: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
        unify_action: bool = False,
        **kwargs: Any,
    ):
        if action_mode != "libero":
            raise ValueError(f"LIBERO action_mode must be 'libero', got {action_mode!r}")
        if unify_action:
            raise ValueError("LIBERO uses native 7-D actions and does not support unify_action")
        self.action_mode = action_mode
        self._head_priority = _as_priority(head_camera_priority, self.HEAD_CAMERA_PRIORITY)
        self._wrist_priority = _as_priority(wrist_camera_priority, self.WRIST_CAMERA_PRIORITY)
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        super().__init__(dataset_dir=dataset_dir, unify_action=False, **kwargs)

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        if self._target_camera is not None:
            return self._target_camera, None, None
        return (
            _pick_feature(features, self._head_priority),
            _pick_feature(features, self._wrist_priority),
            None,
        )

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        action = features.get("action", {})
        shape = tuple(action.get("shape", ()))
        if shape and shape != (_ACTION_DIM,):
            raise ValueError(f"LIBERO action feature must have shape [7], got {shape}")

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if not self._source_stats_path:
            raise FileNotFoundError(
                "LIBERO normalize_mode is enabled but normalization_stats_path is unset. "
                "Run scripts/libero_compute_stats.py or set normalize_mode=null."
            )
        stats = _load_libero_stats(
            str(Path(self._source_stats_path).expanduser().resolve()),
            self.action_mode,
            str(self._normalize_mode),
        )
        self._write_deploy_normalizer_stats(stats, _STAT_KEYS)
        return stats

    def _action_20d(self, win) -> np.ndarray:
        action = np.stack(win["action"].values).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != _ACTION_DIM:
            raise ValueError(f"LIBERO action must be (T, 7), got {action.shape}")
        return apply_normalization(action, self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win):
        return None

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiLiberoDataset


class MultiLiberoDataset(MultiLeRobotV3Reader):
    """Aggregate homogeneous LIBERO LeRobot v3 suite/task buckets."""

    def __init__(self, buckets: List[LiberoDataset]):
        super().__init__(buckets)
        source_paths = {bucket._source_stats_path for bucket in self._buckets}
        if len(source_paths) != 1:
            raise ValueError("MultiLiberoDataset requires one shared normalization_stats_path")

    @property
    def normalization_stats_path(self) -> Optional[str]:
        return self._buckets[0].normalization_stats_path

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return LiberoDataset.from_config(config, split)


__all__ = ["LiberoDataset", "MultiLiberoDataset"]
