"""LIBERO LeRobot v3 dataloader (single-arm EEF10 -> unified 80-D).

Targets the official ``nvidia/LIBERO_LeRobot_v3`` schema and the compatible
``HuggingFaceVLA/libero`` conversion:

* action: 7-D OSC delta command ``[dpos3, drot_axis_angle3, gripper1]``
* observation.state: 8-D ``[eef_pos3, eef_axis_angle3, gripper_qpos2]`` (world frame)
* observation.images.image: agent-view RGB
* observation.images.wrist_image or image2: wrist RGB
* task_index: language lookup through meta/tasks.parquet

The reader trains on the repo-standard single-arm **EEF10** representation::

    eef10 = [xyz(3), rot6d(6), gripper_cmd(1)]        (world frame, full pose)

* PROPRIO at window frame 0: the achieved ``observation.state`` rendered to
  EEF10 (axis-angle -> rot6d; finger separation width -> [-1, +1] command
  space, +1 = close — the same convention as the recorded ``action[6]``).
* ACTION target at step ``t``: the **next frame's achieved pose**
  (``state[t+1]`` -> xyz + rot6d) plus the **recorded gripper command**
  ``action[t][6]`` — a full absolute pose target, NOT the env's per-step OSC
  delta. The final window step has no ``t+1`` target and is masked out of the
  loss (``_n_supervised_action_steps``). The eval bridge
  (``benchmarks/utils/action_conversion.eef10_to_libero7d``) inverts the full
  pose back to the env's 7-D OSC delta using live controller scales.

``unify_action: true`` scatters the 10 physical dims into the unified 80-D
space via ``unify_action_map: ["0-9"]`` (left-arm slots; everything else stays
masked). Action commands and achieved proprio use SEPARATE normalization
statistics (``eef`` / ``eef_state`` blocks, rot6d dims pinned to identity) —
the deploy ``_AsymmetricNormalizer`` unnormalizes actions with the former and
normalizes incoming proprio with the latter.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_ARM10,
    STAT_KEYS,
    apply_normalization,
    load_stats_file,
)

logger = logging.getLogger(__name__)

_ACTION_MODE = "eef"
EEF10_DIM = 10
STATE8_DIM = 8
ACTION7_DIM = 7
# Panda finger separation at fully open (m): gripper_qpos ~= [0.04, -0.04] open,
# [~0, ~0] closed -> width = qpos[0] - qpos[1] in [0, 0.08]. Rendered into the
# recorded command space (+1 = close, -1 = open) so proprio and action gripper
# share one convention. Kept in lockstep with
# benchmarks/utils/action_conversion.LIBERO_GRIPPER_WIDTH_OPEN.
LIBERO_GRIPPER_WIDTH_OPEN = 0.08


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: ``(..., 3)`` rotation vector -> ``(..., 3, 3)`` rotation matrix.

    Matches robosuite/LeRobot ``quat2axisangle`` semantics (rotvec = axis * angle),
    so the stored ``observation.state[3:6]`` converts back to the same rotation.
    """
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    small = angle[..., 0] < 1e-8
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-8), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(angle[..., 0])
    s = np.sin(angle[..., 0])
    C = 1.0 - c
    R = np.empty(aa.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    R[small] = np.eye(3)
    return R


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 6)`` rot6d (first two columns)."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1).astype(np.float32)


def gripper_qpos_to_cmd(width: np.ndarray) -> np.ndarray:
    """Achieved finger separation width -> [-1, +1] command space (+1 = close).

    Linear over ``[0, LIBERO_GRIPPER_WIDTH_OPEN]``, clipped. The eval client
    reproduces this exactly (``benchmarks.utils.libero_gripper_qpos_to_cmd``).
    """
    return np.clip(1.0 - 2.0 * np.asarray(width, np.float64) / LIBERO_GRIPPER_WIDTH_OPEN, -1.0, 1.0).astype(
        np.float32
    )


def state8_to_eef10(state: np.ndarray) -> np.ndarray:
    """``(T, 8)`` observation.state -> ``(T, 10)`` raw EEF10 (achieved, unnormalized).

    ``[pos3, rot6d(axis-angle), gripper_cmd]`` — the gripper is the ACHIEVED
    finger-separation width rendered into command space. This is the PROPRIO
    gripper; the ACTION gripper is replaced with the recorded command in
    ``_action_20d`` (both live in the same [-1, +1] space).
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != STATE8_DIM:
        raise ValueError(f"LIBERO observation.state must be (T, {STATE8_DIM}), got {state.shape}")
    pos = state[:, 0:3].astype(np.float32)
    rot6d = matrix_to_rot6d(axis_angle_to_matrix(state[:, 3:6]))
    grip = gripper_qpos_to_cmd(state[:, 6] - state[:, 7])[:, None]
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


def _as_priority(value: Optional[Sequence[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    return next((key for key in priorities if key in features), None)


class LiberoDataset(LeRobotV3Reader):
    """Single-bucket LIBERO reader for LeRobot v3 parquet/video datasets."""

    DATASET_NAME = "LIBERO"
    ACTION_DIM = EEF10_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEPLOY_ACTION_MODE = _ACTION_MODE

    HEAD_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.image",
        "observation.images.agentview_image",
        "observation.images.front",
    )
    WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.wrist_image",
        "observation.images.wrist",
        "observation.images.image2",
        "observation.images.robot0_eye_in_hand_image",
    )
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "head_camera_priority",
        "wrist_camera_priority",
        "prompt_columns",
        "normalization_stats_path",
        "state_stats_mode",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = _ACTION_MODE,
        head_camera_priority: Optional[Sequence[str]] = None,
        wrist_camera_priority: Optional[Sequence[str]] = None,
        prompt_columns: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
        state_stats_mode: str = "eef_state",
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != _ACTION_MODE:
            raise ValueError(
                f"LIBERO currently supports only action_mode='eef', got {action_mode!r}. "
                "EEF is raw 10-D: [xyz3, rot6d6, gripper1] (world frame, full pose)."
            )
        unify_on = bool(unify_action)
        if unify_on and unify_action_map is None:
            raise ValueError(
                "LIBERO unify_action=true requires an explicit unify_action_map; "
                'set ["0-9"] for the canonical single-arm left-slot mapping'
            )
        self.action_mode = mode
        self._state_stats_mode = str(state_stats_mode)
        self._state_normalization_stats: Optional[dict] = None
        self._head_priority = _as_priority(head_camera_priority, self.HEAD_CAMERA_PRIORITY)
        self._wrist_priority = _as_priority(wrist_camera_priority, self.WRIST_CAMERA_PRIORITY)
        self._prompt_columns = _as_priority(
            prompt_columns,
            ("language_instruction", "task", "prompt"),
        )
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=unify_on,
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        if self._target_camera is not None:
            return self._target_camera, None, None
        return (
            _pick_feature(features, self._head_priority),
            _pick_feature(features, self._wrist_priority),
            None,
        )

    def _train_min_window_len(self) -> int:
        # The action target at step t is the ACHIEVED pose at t+1, so a window
        # needs at least 2 frames to carry one real supervised step.
        return 2

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        # The final window row has no t+1 achieved-pose target (clamped copy) —
        # drop it from the loss. Full windows keep all T_action steps because
        # the caller min-caps to num_frames - 1.
        return max(0, actual_raw_len - 1)

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"LIBERO {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        action = features.get("action", {})
        shape = tuple(action.get("shape", ()))
        if shape and shape != (ACTION7_DIM,):
            raise ValueError(f"LIBERO action feature must have shape [{ACTION7_DIM}], got {shape}")
        state = features.get("observation.state", {})
        state_shape = tuple(state.get("shape", ()))
        if state_shape and state_shape != (STATE8_DIM,):
            raise ValueError(f"LIBERO observation.state feature must have shape [{STATE8_DIM}], got {state_shape}")
        if "observation.state" not in features:
            raise KeyError(
                "LIBERO EEF10 requires the 8-D observation.state column (achieved EEF pose + "
                "gripper qpos); this conversion does not carry it."
            )
        self._prompt_columns = tuple(col for col in self._prompt_columns if col in features)
        self.NEEDED_COLS = self.NEEDED_COLS + self._prompt_columns

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if not self._source_stats_path:
            raise FileNotFoundError(
                "LIBERO normalize_mode is enabled but normalization_stats_path is unset. "
                "Run python -m openwam.dataloader.utils.stats_computation.libero_stats_computation "
                "or set normalize_mode=null."
            )
        action_stats = load_stats_file(
            self._source_stats_path,
            action_mode=self.action_mode,
            normalize_mode=str(self._normalize_mode),
            dim=self._raw_action_dim,
        )
        self._state_normalization_stats = load_stats_file(
            self._source_stats_path,
            action_mode=self._state_stats_mode,
            normalize_mode=str(self._normalize_mode),
            dim=self._raw_action_dim,
        )
        self._write_deploy_normalizer_stats(
            action_stats,
            STAT_KEYS,
            additional_entries={self._state_stats_mode: self._state_normalization_stats},
        )
        return action_stats

    def _read_state8(self, win) -> np.ndarray:
        state = np.stack(win["observation.state"].values).astype(np.float32)
        if state.ndim != 2 or state.shape[1] != STATE8_DIM:
            raise ValueError(f"LIBERO observation.state must be (T, {STATE8_DIM}), got {state.shape}")
        return state

    def _read_action7(self, win) -> np.ndarray:
        action = np.stack(win["action"].values).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != ACTION7_DIM:
            raise ValueError(f"LIBERO action must be (T, {ACTION7_DIM}), got {action.shape}")
        return action

    def _raw_action_eef10(self, win) -> np.ndarray:
        """``(L, 10)`` raw absolute EEF10 targets: next-frame achieved pose +
        recorded gripper command. The final row has no ``t+1`` and is a clamped
        copy — it carries no supervision (see ``_n_supervised_action_steps``)."""
        state = self._read_state8(win)
        action7 = self._read_action7(win)
        eef10 = state8_to_eef10(state)  # (L, 10) achieved
        target = np.empty_like(eef10)
        if eef10.shape[0] > 1:
            target[:-1, 0:9] = eef10[1:, 0:9]  # next-frame achieved pose
            target[:-1, 9] = action7[:-1, 6]  # recorded gripper COMMAND (exact timing)
        target[-1] = target[-2] if eef10.shape[0] > 1 else eef10[0]
        return target

    def _raw_state_eef10(self, win) -> np.ndarray:
        """``(L, 10)`` raw achieved EEF10 proprio (used by the stats script)."""
        return state8_to_eef10(self._read_state8(win))

    def _action_20d(self, win) -> np.ndarray:
        return apply_normalization(self._raw_action_eef10(win), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = state8_to_eef10(self._read_state8(win)[0:1])
        return apply_normalization(raw, self._state_normalization_stats, self._normalize_mode)

    def _resolve_prompt(self, row, win) -> str:
        for column in self._prompt_columns:
            value = win[column].iloc[0]
            if value is not None and not pd.isna(value):
                text = str(value).strip()
                if text:
                    return text
        return super()._resolve_prompt(row, win)

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


ROT6D_DIMS_EEF10 = ROT6D_DIMS_ARM10

__all__ = [
    "ACTION7_DIM",
    "EEF10_DIM",
    "LIBERO_GRIPPER_WIDTH_OPEN",
    "ROT6D_DIMS_EEF10",
    "STATE8_DIM",
    "LiberoDataset",
    "MultiLiberoDataset",
    "axis_angle_to_matrix",
    "gripper_qpos_to_cmd",
    "matrix_to_rot6d",
    "state8_to_eef10",
]
