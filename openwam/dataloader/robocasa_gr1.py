"""RoboCasa GR1 LeRobot v3 dataloader.

This reader is intentionally schema-configurable. RoboCasa GR1 data may be
exported as joint, EEF, or pre-unified vectors; the LeRobot v3 container stays
the same, but feature names can differ across conversion jobs.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.robocasa_gr1_stats import load_stats_file
from openwam.dataloader.utils.eef import EEF_DIM, eef14_to_eef20
from openwam.dataloader.utils.normalization import apply_normalization

logger = logging.getLogger(__name__)

_ACTION_MODES = {"joint", "eef", "unify"}
_EEF20_UNIFY_SPEC = ("0-9", "34-43")
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


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
        unify_action_map: Optional[Any] = None,
        action_mask: Optional[Sequence[bool]] = None,
        state_mask: Optional[Sequence[bool]] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode not in _ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(_ACTION_MODES)}, got {action_mode!r}")
        unify_on = bool(unify_action)
        if mode == "unify" and not unify_on:
            raise ValueError("RoboCasaGR1 action_mode='unify' requires unify_action=true")
        if mode != "unify" and unify_on:
            raise ValueError(
                f"RoboCasaGR1 action_mode={mode!r} requires unify_action=false; "
                "use action_mode='unify' for the shared 80-D action space"
            )
        self.action_mode = mode
        self.DEPLOY_ACTION_MODE = mode
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None

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
            dim = int(action_dim) if action_dim is not None else EEF_DIM

        if mode == "joint" and dim is None:
            # LeRobot v3 joint conversions should set action_dim explicitly.
            # A clear constructor error is better than a late assignment failure.
            raise ValueError("RoboCasaGR1Dataset action_mode='joint' requires action_dim in the dataloader config")
        if mode != "joint" and dim != EEF_DIM:
            raise ValueError(f"RoboCasaGR1 action_mode={mode!r} requires raw 20-D EEF vectors, got action_dim={dim}")

        self.ACTION_DIM = int(dim)
        action_dim_mask = _as_bool_mask(action_mask, self.ACTION_DIM, field="action_mask")
        state_dim_mask = _as_bool_mask(state_mask, self.ACTION_DIM, field="state_mask")
        if action_dim_mask is not None and state_dim_mask is not None and not np.array_equal(
            action_dim_mask, state_dim_mask
        ):
            raise ValueError("RoboCasaGR1 action_mask and state_mask must match; the shared reader uses one raw mask")
        self.ACTION_DIM_MASK = action_dim_mask if action_dim_mask is not None else state_dim_mask

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
        cols.append("task_index")
        self.NEEDED_COLS = tuple(dict.fromkeys(cols))

        if mode == "unify" and unify_action_map is None:
            unify_action_map = list(_EEF20_UNIFY_SPEC)
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=unify_on,
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        return (
            _pick_feature(features, self._head_priority),
            _pick_feature(features, self._left_wrist_priority),
            _pick_feature(features, self._right_wrist_priority),
        )

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        configured = set(self._prompt_columns)
        self._prompt_columns = [col for col in self._prompt_columns if col in features]
        missing = configured.difference(self._prompt_columns)
        if missing:
            logger.info(
                "RoboCasaGR1(%s): ignoring prompt columns absent from info.features: %s",
                self._dataset_id,
                sorted(missing),
            )
        self.NEEDED_COLS = tuple(
            col for col in self.NEEDED_COLS if col not in configured or col in self._prompt_columns
        )

    def _resolve_prompt(self, row, win) -> str:
        for col in self._prompt_columns:
            if col in win:
                value = win[col].iloc[0]
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
        return super()._resolve_prompt(row, win)

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in (None, "none", "null"):
            return None
        if not self._source_stats_path:
            raise FileNotFoundError(
                "RoboCasaGR1Dataset normalize_mode is enabled but normalization_stats_path is unset. "
                "Run scripts/robocasa_gr1_compute_stats.py or set normalize_mode=null."
            )
        stats = load_stats_file(
            self._source_stats_path,
            action_mode=self.action_mode,
            normalize_mode=self._normalize_mode,
            dim=self._raw_action_dim,
        )
        self._write_deploy_normalizer_stats(stats, _STAT_KEYS)
        return stats

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        raw = self._raw_action(win)
        return self._normalize_array(raw)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._read_vector_window(
            win,
            vector_col=self._state_column,
            pose_col=self._pose_state_column,
            grip_col=self._gripper_state_column,
            label="state",
            first_only=True,
        )
        return self._normalize_array(raw)

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
        first_only: bool = False,
    ) -> np.ndarray:
        values = slice(0, 1) if first_only else slice(None)
        if vector_col:
            if vector_col not in win:
                raise KeyError(f"RoboCasaGR1 {label} column {vector_col!r} not found in parquet window")
            return np.stack(win[vector_col].values[values]).astype(np.float32)
        if not pose_col or not grip_col:
            raise KeyError(f"RoboCasaGR1 {label} needs either a vector column or pose+gripper columns")
        if pose_col not in win or grip_col not in win:
            raise KeyError(
                f"RoboCasaGR1 {label} columns missing: pose={pose_col!r} present={pose_col in win}, "
                f"gripper={grip_col!r} present={grip_col in win}"
            )
        pose = np.stack(win[pose_col].values[values]).astype(np.float32)
        grip = np.stack(win[grip_col].values[values]).astype(np.float32)
        if pose.shape[-1] != 12 or grip.shape[-1] != 2:
            raise ValueError(
                f"RoboCasaGR1 {label} pose+gripper EEF conversion expects 12D+2D, "
                f"got {pose.shape[-1]}D+{grip.shape[-1]}D"
            )
        return eef14_to_eef20(pose, grip)

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
        stats_paths = {b._source_stats_path for b in self._buckets}
        if len(stats_paths) != 1:
            raise ValueError("MultiRoboCasaGR1Dataset requires one shared normalization_stats_path")
        logger.info(
            "MultiRoboCasaGR1Dataset: %d buckets, %d windows, action_mode=%s, action_dim=%d",
            len(self._buckets),
            len(self),
            next(iter(modes)),
            next(iter(dims)),
        )

    @property
    def normalization_stats_path(self) -> Optional[str]:
        """Deploy artifact generated by the first homogeneous bucket."""
        return self._buckets[0].normalization_stats_path

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return RoboCasaGR1Dataset.from_config(config, split)


__all__ = ["RoboCasaGR1Dataset", "MultiRoboCasaGR1Dataset"]
