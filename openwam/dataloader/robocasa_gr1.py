"""RoboCasa GR1 LeRobot v3 dataloader.

This reader is intentionally schema-configurable. RoboCasa GR1 data may be
exported as joint, EEF, or pre-unified vectors; the LeRobot v3 container stays
the same, but feature names can differ across conversion jobs.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.robocasa_gr1_stats import load_stats_file, neutralize_rot6d_stats
from openwam.dataloader.utils.eef import (
    EEF_DIM,
    build_action_mask_2d,
    build_proprio_mask_2d,
    eef14_to_eef20,
)
from openwam.dataloader.utils.normalization import apply_normalization

logger = logging.getLogger(__name__)

_ACTION_MODES = {"joint", "eef", "unify"}
_UNIFY_DIM = 80
_DEFAULT_PROMPT = "Perform the RoboCasa GR1 tabletop task."

# 80-D pretraining action space:
#   L xyz[0:3] rot6d[3:9] gripper[9] dexterous_hand[10:34]
#   R xyz[34:37] rot6d[37:43] gripper[43] dexterous_hand[44:68]
#   redundant/base[68:80]
_EEF20_TO_UNIFY80 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
]

_EEF_ROT6D_SLICES = ((3, 9), (13, 19))
_UNIFY_ROT6D_SLICES = ((3, 9), (37, 43))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        pass
    return [value]


def _as_bool_mask(value: Any, dim: int, *, field: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.shape != (dim,):
        raise ValueError(f"{field} must be a boolean list of length {dim}, got shape {arr.shape}")
    return arr


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    for key in priorities:
        if key and key in features:
            return key
    return None


class RoboCasaGR1Dataset(LeRobotV3Reader):
    """Single-bucket RoboCasa GR1 reader for LeRobot v3 datasets."""

    DATASET_NAME = "RoboCasaGR1"
    PROMPT_FILE_REQUIRED = False

    HEAD_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.ego_view",
        "observation.images.ego_view_rgb",
        "observation.images.ego_rgb",
        "observation.images.head_rgb",
        "observation.images.cam_head_rgb",
        "observation.images.cam_high_rgb",
        "video.ego_view_pad_res256_freq20",
    )
    LEFT_WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.left_wrist",
        "observation.images.left_wrist_rgb",
        "observation.images.cam_left_wrist_rgb",
    )
    RIGHT_WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.right_wrist",
        "observation.images.right_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
    )

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "action_dim",
        "action_column",
        "state_column",
        "joint_action_column",
        "joint_state_column",
        "eef_action_column",
        "eef_state_column",
        "eef_pose_action_column",
        "eef_gripper_action_column",
        "eef_pose_state_column",
        "eef_gripper_state_column",
        "prompt_columns",
        "head_camera_priority",
        "left_wrist_camera_priority",
        "right_wrist_camera_priority",
        "normalization_stats_path",
        "unify_action",
        "unify_action_map",
        "state_mask",
        "action_mask",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = "eef",
        action_dim: Optional[int] = None,
        action_column: Optional[str] = None,
        state_column: Optional[str] = None,
        joint_action_column: str = "action",
        joint_state_column: str = "observation.state",
        eef_action_column: Optional[str] = None,
        eef_state_column: Optional[str] = None,
        eef_pose_action_column: str = "eef_sim_pose_action",
        eef_gripper_action_column: str = "gripper_open_scale_action",
        eef_pose_state_column: str = "eef_sim_pose_state",
        eef_gripper_state_column: str = "gripper_open_scale_state",
        prompt_columns: Optional[Sequence[str]] = None,
        head_camera_priority: Optional[Sequence[str]] = None,
        left_wrist_camera_priority: Optional[Sequence[str]] = None,
        right_wrist_camera_priority: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
        unify_action: Optional[bool] = None,
        unify_action_map: Optional[Sequence[int]] = None,
        action_mask: Optional[Sequence[bool]] = None,
        state_mask: Optional[Sequence[bool]] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if unify_action is True:
            mode = "unify"
        if mode not in _ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(_ACTION_MODES)}, got {action_mode!r}")
        self.action_mode = mode
        self._normalization_stats_path = str(normalization_stats_path) if normalization_stats_path else None

        self._prompt_columns = [str(x) for x in _as_list(prompt_columns)]
        self._head_priority = tuple(str(x) for x in (head_camera_priority or self.HEAD_CAMERA_PRIORITY))
        self._left_wrist_priority = tuple(str(x) for x in (left_wrist_camera_priority or self.LEFT_WRIST_CAMERA_PRIORITY))
        self._right_wrist_priority = tuple(
            str(x) for x in (right_wrist_camera_priority or self.RIGHT_WRIST_CAMERA_PRIORITY)
        )

        self._action_column = action_column
        self._state_column = state_column
        self._pose_action_column = None
        self._gripper_action_column = None
        self._pose_state_column = None
        self._gripper_state_column = None

        if mode == "joint":
            self._action_column = action_column or joint_action_column
            self._state_column = state_column or joint_state_column
            dim = int(action_dim) if action_dim is not None else None
        else:
            self._action_column = action_column or eef_action_column
            self._state_column = state_column or eef_state_column
            self._pose_action_column = None if self._action_column else eef_pose_action_column
            self._gripper_action_column = None if self._action_column else eef_gripper_action_column
            self._pose_state_column = None if self._state_column else eef_pose_state_column
            self._gripper_state_column = None if self._state_column else eef_gripper_state_column
            dim = _UNIFY_DIM if mode == "unify" else (int(action_dim) if action_dim is not None else EEF_DIM)

        if mode == "joint" and dim is None:
            # LeRobot v3 joint conversions should set action_dim explicitly.
            # A clear constructor error is better than a late assignment failure.
            raise ValueError("RoboCasaGR1Dataset action_mode='joint' requires action_dim in the dataloader config")

        self.ACTION_DIM = int(dim)
        self._raw_action_dim = int(action_dim) if action_dim is not None and mode == "unify" else None
        self._unify_action_map = self._resolve_unify_map(unify_action_map)
        default_unify_mask = None
        if mode == "unify":
            default_unify_mask = np.zeros((self.ACTION_DIM,), dtype=bool)
            default_unify_mask[self._unify_action_map] = True
        self._action_dim_mask = (
            _as_bool_mask(action_mask, self.ACTION_DIM, field="action_mask")
            if action_mask is not None
            else default_unify_mask
        )
        self._proprio_dim_mask = (
            _as_bool_mask(state_mask, self.ACTION_DIM, field="state_mask")
            if state_mask is not None
            else default_unify_mask
        )
        self.ACTION_DIM_MASK = self._action_dim_mask

        cols: List[str] = []
        for col in (
            self._action_column,
            self._state_column,
            self._pose_action_column,
            self._gripper_action_column,
            self._pose_state_column,
            self._gripper_state_column,
            *self._prompt_columns,
        ):
            if col:
                cols.append(str(col))
        if not self._prompt_columns:
            cols.append("task_index")
        self.NEEDED_COLS = tuple(dict.fromkeys(cols))

        super().__init__(dataset_dir=dataset_dir, **kwargs)

    def _resolve_unify_map(self, raw_map: Optional[Sequence[int]]) -> Optional[np.ndarray]:
        if self.action_mode != "unify":
            return None
        mapping = _EEF20_TO_UNIFY80 if raw_map is None else [int(x) for x in raw_map]
        arr = np.asarray(mapping, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("unify_action_map must be a 1-D list of destination indices")
        if (arr < 0).any() or (arr >= _UNIFY_DIM).any():
            raise ValueError(f"unify_action_map indices must be in [0, {_UNIFY_DIM}), got {arr.tolist()}")
        if len(set(arr.tolist())) != len(arr):
            raise ValueError("unify_action_map must not contain duplicate destination indices")
        return arr

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        return (
            _pick_feature(features, self._head_priority),
            _pick_feature(features, self._left_wrist_priority),
            _pick_feature(features, self._right_wrist_priority),
        )

    def _read_data_file_uncached(self, chunk_idx: int, file_idx: int):
        path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        try:
            return pq.read_table(path, memory_map=True, columns=list(self.NEEDED_COLS))
        except pa.ArrowInvalid as exc:
            # Flat LeRobot feature names may contain dots (e.g.
            # annotation.human.coarse_action). Some pyarrow versions parse those
            # projection strings as nested field paths. Fall back to reading the
            # shard and let the normal hook-level column checks produce precise
            # missing-column errors.
            if "Dot path" not in str(exc):
                raise
            return pq.read_table(path, memory_map=True)

    def _resolve_prompt(self, row, win) -> str:
        for col in self._prompt_columns:
            if col in win:
                value = win[col].iloc[0]
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
        try:
            return super()._resolve_prompt(row, win)
        except Exception:
            return _DEFAULT_PROMPT

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in (None, "none", "null"):
            return None
        if not self._normalization_stats_path:
            raise FileNotFoundError(
                "RoboCasaGR1Dataset normalize_mode is enabled but normalization_stats_path is unset. "
                "Run scripts/robocasa_gr1_compute_stats.py or set normalize_mode=null."
            )
        stats = load_stats_file(self._normalization_stats_path, action_mode=self.action_mode)
        return neutralize_rot6d_stats(stats, self._rot6d_slices_for_stats())

    def _rot6d_slices_for_stats(self) -> Tuple[Tuple[int, int], ...]:
        if self.action_mode == "eef":
            return _EEF_ROT6D_SLICES
        if self.action_mode == "unify":
            # Stats are computed/applied before mapping raw vectors into the
            # 80-D space. The default raw vector is canonical 20-D EEF.
            if self._unify_action_map is not None and len(self._unify_action_map) == EEF_DIM:
                return _EEF_ROT6D_SLICES
        return ()

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        raw = self._raw_action(win)
        return self._map_after_normalize(raw)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._raw_state(win)
        return self._map_after_normalize(raw[:1])

    def _raw_action(self, win) -> np.ndarray:
        return self._read_vector_window(
            win,
            vector_col=self._action_column,
            pose_col=self._pose_action_column,
            grip_col=self._gripper_action_column,
            label="action",
        )

    def _raw_state(self, win) -> np.ndarray:
        return self._read_vector_window(
            win,
            vector_col=self._state_column,
            pose_col=self._pose_state_column,
            grip_col=self._gripper_state_column,
            label="state",
        )

    def _read_vector_window(
        self,
        win,
        *,
        vector_col: Optional[str],
        pose_col: Optional[str],
        grip_col: Optional[str],
        label: str,
    ) -> np.ndarray:
        if vector_col:
            if vector_col not in win:
                raise KeyError(f"RoboCasaGR1 {label} column {vector_col!r} not found in parquet window")
            return np.stack(win[vector_col].values).astype(np.float32)
        if not pose_col or not grip_col:
            raise KeyError(f"RoboCasaGR1 {label} needs either a vector column or pose+gripper columns")
        if pose_col not in win or grip_col not in win:
            raise KeyError(
                f"RoboCasaGR1 {label} columns missing: pose={pose_col!r} present={pose_col in win}, "
                f"gripper={grip_col!r} present={grip_col in win}"
            )
        pose = np.stack(win[pose_col].values).astype(np.float32)
        grip = np.stack(win[grip_col].values).astype(np.float32)
        if pose.shape[-1] != 12 or grip.shape[-1] != 2:
            raise ValueError(
                f"RoboCasaGR1 {label} pose+gripper EEF conversion expects 12D+2D, "
                f"got {pose.shape[-1]}D+{grip.shape[-1]}D"
            )
        return eef14_to_eef20(pose, grip)

    def _map_after_normalize(self, raw: np.ndarray) -> np.ndarray:
        normalized = self._normalize_array(raw)
        if self.action_mode != "unify":
            if normalized.shape[-1] != self.ACTION_DIM:
                raise ValueError(
                    f"RoboCasaGR1 action_mode={self.action_mode!r} expected {self.ACTION_DIM}D vectors, "
                    f"got {normalized.shape[-1]}D"
                )
            return normalized.astype(np.float32)
        return self.map_to_unify(normalized)

    def map_to_unify(self, arr: np.ndarray) -> np.ndarray:
        if self._unify_action_map is None:
            raise RuntimeError("map_to_unify called outside action_mode='unify'")
        if arr.shape[-1] != len(self._unify_action_map):
            raise ValueError(
                f"unify_action_map length {len(self._unify_action_map)} does not match raw vector dim {arr.shape[-1]}"
            )
        out = np.zeros(arr.shape[:-1] + (_UNIFY_DIM,), dtype=np.float32)
        out[..., self._unify_action_map] = arr
        return out

    def unmap_from_unify(self, arr: np.ndarray) -> np.ndarray:
        if self._unify_action_map is None:
            return arr
        return np.asarray(arr)[..., self._unify_action_map]

    def _finalize_action(self, action_20d: Optional[np.ndarray], actual_raw_len: int):
        T_action = self._num_frames - 1
        action = np.zeros((T_action, self.ACTION_DIM), dtype=np.float32)
        n_valid = 0
        if action_20d is not None:
            n_valid = min(actual_raw_len, T_action)
            if n_valid > 0:
                action[:n_valid] = action_20d[:n_valid]
        action_mask = build_action_mask_2d(
            T_action=T_action,
            action_dim=self.ACTION_DIM,
            n_valid_time=n_valid if self._enable_action_supervision else 0,
            dim_mask=self._action_dim_mask,
        )
        return action, action_mask

    def _finalize_proprio(self, proprio_20d: Optional[np.ndarray]):
        if proprio_20d is None:
            proprio = np.zeros((1, self.ACTION_DIM), dtype=np.float32)
            mask = build_proprio_mask_2d(action_dim=self.ACTION_DIM, enabled=False)
            return proprio, mask
        mask = build_proprio_mask_2d(
            action_dim=self.ACTION_DIM,
            enabled=self._enable_action_supervision,
            dim_mask=self._proprio_dim_mask if self._proprio_dim_mask is not None else self._action_dim_mask,
        )
        return np.asarray(proprio_20d, dtype=np.float32), mask

    @property
    def normalization_stats_path(self) -> Optional[str]:
        return self._normalization_stats_path

    @property
    def normalization_stats(self):
        # Samples leave this reader already normalized when normalize_mode is set.
        return None

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiRoboCasaGR1Dataset


class MultiRoboCasaGR1Dataset(MultiLeRobotV3Reader):
    """Aggregate multiple RoboCasa GR1 LeRobot v3 buckets."""

    def __init__(self, buckets: List[RoboCasaGR1Dataset]):
        super().__init__(buckets)
        dims = {int(b.action_dim) for b in self._buckets}
        modes = {b.action_mode for b in self._buckets}
        if len(dims) != 1:
            raise ValueError(f"MultiRoboCasaGR1Dataset requires homogeneous action_dim, got {sorted(dims)}")
        if len(modes) != 1:
            raise ValueError(f"MultiRoboCasaGR1Dataset requires homogeneous action_mode, got {sorted(modes)}")
        logger.info(
            "MultiRoboCasaGR1Dataset: %d buckets, %d windows, action_mode=%s, action_dim=%d",
            len(self._buckets),
            len(self),
            next(iter(modes)),
            next(iter(dims)),
        )

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return RoboCasaGR1Dataset.from_config(config, split)


__all__ = ["RoboCasaGR1Dataset", "MultiRoboCasaGR1Dataset"]
