"""MUKA Franka single-arm LeRobot v3 dataloader.

The downloaded dataset stores row-aligned 7-D values as::

    observation.state = achieved [xyz3, Euler-XYZ3, gripper_closedness1]
    action            = next achieved [xyz3, Euler-XYZ3, gripper_closedness1]

Euler angles are extrinsic XYZ radians (``R = Rz @ Ry @ Rx``).  Raw gripper
closedness is ``0 = open, 1 = closed``.  This reader converts both streams to
the OpenWAM single-arm EEF10 contract::

    [xyz3, rot6d6, gripper_open_scale1],  -1 = closed, +1 = open

The conversion is ``open_scale = 1 - 2 * closedness``.  With
``unify_action: true``, EEF10 is normalized first and then scattered into the
shared 80-D left-arm slots 0..9.

The source action at row ``t`` equals the state at row ``t+1``.  The final
episode row repeats the terminal state, so it is excluded from action
supervision and from action normalization statistics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import euler_xyz_to_rot6d
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_ARM10,
    apply_normalization,
    load_stats_file,
    load_stats_metadata,
)

logger = logging.getLogger(__name__)

_ACTION_MODE = "eef"
RAW_EEF7_DIM = 7
EEF10_DIM = 10
RAW_GRIPPER_CONVENTION = "zero_open_one_closed"
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
GRIPPER_TRANSFORM = "1_minus_2_times_closedness"
ROTATION_CONVENTION = "extrinsic_xyz_radians_r_equals_rz_ry_rx"
NORMALIZATION_STATS_FILENAME = "normalization_stats.npy"
STATS_POPULATION = "all_episodes"

_EXPECTED_FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper_closedness",
)
_EXPECTED_SOURCE_SCHEMA = {
    "position_frame": "robot_base",
    "rotation_storage": "Euler XYZ",
    "rotation_convention": "extrinsic; R = Rz @ Ry @ Rx",
    "rotation_unit": "radians",
    "action_type": "absolute achieved EEF state",
    "action_alignment": "action[t] = observation.state[t+1]; terminal row repeated and masked by reader",
    "gripper": "continuous closedness; 0=open, 1=closed",
}


def _as_priority(value: Optional[Sequence[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    return next((key for key in priorities if key in features), None)


def closedness_to_open_scale(closedness: np.ndarray) -> np.ndarray:
    """Map source gripper closedness to OpenWAM's ``[-1 closed, +1 open]``."""
    values = np.asarray(closedness, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("MUKA Franka gripper closedness contains NaN or infinity")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        observed_min = float(values.min())
        observed_max = float(values.max())
        raise ValueError(
            f"MUKA Franka gripper closedness must be in [0, 1], got min={observed_min:.8g}, max={observed_max:.8g}"
        )
    return (1.0 - 2.0 * values).astype(np.float32)


def euler7_to_eef10(values: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, extrinsic XYZ Euler, closedness]`` to canonical EEF10."""
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_EEF7_DIM:
        raise ValueError(f"MUKA Franka EEF input must be (T, {RAW_EEF7_DIM}), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("MUKA Franka EEF input contains NaN or infinity")
    return np.concatenate(
        (
            raw[:, :3],
            euler_xyz_to_rot6d(raw[:, 3:6]),
            closedness_to_open_scale(raw[:, 6:7]),
        ),
        axis=-1,
    ).astype(np.float32)


class MukaFrankaDataset(LeRobotV3Reader):
    """Single-bucket reader for ``ewykric/muka_franka_lerobot_v3``."""

    DATASET_NAME = "MUKAFranka"
    ACTION_DIM = EEF10_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEPLOY_ACTION_MODE = _ACTION_MODE

    HEAD_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = ("observation.images.left",)
    WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = ("observation.images.left_wrist",)
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "gripper_convention",
        "head_camera_priority",
        "wrist_camera_priority",
        "normalization_stats_path",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = _ACTION_MODE,
        gripper_convention: str = GRIPPER_CONVENTION,
        head_camera_priority: Optional[Sequence[str]] = None,
        wrist_camera_priority: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != _ACTION_MODE:
            raise ValueError(f"MUKA Franka supports only action_mode='eef', got {action_mode!r}")
        convention = str(gripper_convention).strip()
        if convention != GRIPPER_CONVENTION:
            raise ValueError(
                f"MUKA Franka requires gripper_convention={GRIPPER_CONVENTION!r}, got {gripper_convention!r}"
            )
        unify_on = bool(unify_action)
        if unify_on and unify_action_map is None:
            raise ValueError(
                "MUKA Franka unify_action=true requires an explicit unify_action_map; "
                'set ["0-9"] for the canonical single-arm left-slot mapping'
            )
        self.action_mode = mode
        self._head_priority = _as_priority(head_camera_priority, self.HEAD_CAMERA_PRIORITY)
        self._wrist_priority = _as_priority(wrist_camera_priority, self.WRIST_CAMERA_PRIORITY)
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._resolved_stats_path: Optional[str] = None
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

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        # action[t] targets state[t+1]; a window ending at the episode boundary
        # includes one repeated terminal target that must stay out of the loss.
        return max(0, actual_raw_len - 1)

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"MUKA Franka {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        if info.get("robot_type") != "muka_franka_single_arm":
            raise ValueError(
                "MUKA Franka info.json must declare robot_type='muka_franka_single_arm', "
                f"got {info.get('robot_type')!r}"
            )
        for column in ("action", "observation.state"):
            if column not in features:
                raise KeyError(f"MUKA Franka requires {column!r}")
            feature = features[column]
            shape = tuple(feature.get("shape", ()))
            if shape != (RAW_EEF7_DIM,):
                raise ValueError(f"MUKA Franka {column} feature must have shape [{RAW_EEF7_DIM}], got {shape}")
            names = tuple(feature.get("names") or ())
            if names != _EXPECTED_FEATURE_NAMES:
                raise ValueError(f"MUKA Franka {column} names must be {_EXPECTED_FEATURE_NAMES}, got {names}")
        self._validate_source_schema()

    def _validate_source_schema(self) -> None:
        path = self._dataset_dir / "meta" / "muka_schema.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"MUKA Franka source contract is missing: {path}; cannot safely infer rotation/gripper semantics"
            )
        with path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        mismatches = {
            key: (schema.get(key), expected)
            for key, expected in _EXPECTED_SOURCE_SCHEMA.items()
            if schema.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"MUKA Franka source contract mismatch in {path}: {mismatches}")

    @staticmethod
    def _stats_contract_matches(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            raw = np.load(path, allow_pickle=True).item()
            block = raw.get(_ACTION_MODE, raw) if isinstance(raw, dict) else {}
            return (
                block.get("gripper_convention") == GRIPPER_CONVENTION
                and block.get("raw_gripper_convention") == RAW_GRIPPER_CONVENTION
                and block.get("gripper_transform") == GRIPPER_TRANSFORM
                and block.get("rotation_convention") == ROTATION_CONVENTION
                and block.get("stats_population") == STATS_POPULATION
            )
        except (OSError, ValueError, EOFError, AttributeError):
            return False

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if self._source_stats_path:
            stats_path = Path(self._source_stats_path)
            if not self._stats_contract_matches(stats_path):
                raise ValueError(f"MUKA Franka stats {stats_path} do not match the required rot6d/gripper contract")
        else:
            stats_path = self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
            if not self._stats_contract_matches(stats_path):
                self._build_default_stats(stats_path)
        self._resolved_stats_path = str(stats_path)
        global_stats = load_stats_file(
            stats_path,
            action_mode=self.action_mode,
            normalize_mode=str(self._normalize_mode),
            dim=self._raw_action_dim,
        )
        self._check_stats_contract(stats_path)
        self.normalization_stats_path = str(stats_path)
        return global_stats

    def _check_stats_contract(self, stats_path: Path) -> None:
        metadata = load_stats_metadata(stats_path, action_mode=self.action_mode)
        expected = {
            "gripper_convention": GRIPPER_CONVENTION,
            "raw_gripper_convention": RAW_GRIPPER_CONVENTION,
            "gripper_transform": GRIPPER_TRANSFORM,
            "rotation_convention": ROTATION_CONVENTION,
            "stats_population": STATS_POPULATION,
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"MUKA Franka stats contract mismatch in {stats_path}: {mismatches}")

    def _build_default_stats(self, path: Path) -> None:
        from openwam.dataloader.utils.stats_computation.muka_franka_stats_computation import (
            build_and_save_muka_franka_stats,
        )

        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        rank = dist.get_rank() if dist_ready else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            logger.info(
                "MUKAFranka(%s): building OpenWAM normalization stats at %s",
                self._dataset_id,
                path,
            )
            build_and_save_muka_franka_stats(self, path)
            return

        deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
        poll_interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
        while not self._stats_contract_matches(path):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for rank 0 to build MUKA Franka stats: {path}")
            time.sleep(poll_interval)

    @staticmethod
    def _read_euler7_column(win, column: str) -> np.ndarray:
        values = np.stack(win[column].values).astype(np.float32)
        return euler7_to_eef10(values)

    def _raw_action_eef10(self, win) -> np.ndarray:
        return self._read_euler7_column(win, "action")

    def _raw_state_eef10(self, win) -> np.ndarray:
        return self._read_euler7_column(win, "observation.state")

    def _action_20d(self, win) -> np.ndarray:
        return apply_normalization(self._raw_action_eef10(win), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._raw_state_eef10(win)[0:1]
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)


ROT6D_DIMS_EEF10 = ROT6D_DIMS_ARM10

__all__ = [
    "EEF10_DIM",
    "GRIPPER_CONVENTION",
    "GRIPPER_TRANSFORM",
    "MukaFrankaDataset",
    "RAW_EEF7_DIM",
    "RAW_GRIPPER_CONVENTION",
    "ROT6D_DIMS_EEF10",
    "ROTATION_CONVENTION",
    "STATS_POPULATION",
    "closedness_to_open_scale",
    "euler7_to_eef10",
]
