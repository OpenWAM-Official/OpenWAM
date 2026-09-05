"""Canonical LeRobot v3 reader for LIBERO native-action EEF10.

The canonical on-disk contract is achieved EEF10 state plus native normalized
LIBERO action encoded as
``[delta_xyz3, rot6d(Exp(delta_axis_angle3)), gripper_open_command]``.

The reader accepts only datasets whose conversion metadata declares the exact
``native_delta_eef10`` representation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_ARM10,
    apply_normalization,
    materialize_eef_stats,
)

logger = logging.getLogger(__name__)

ACTION_MODE = "eef"
ACTION_STATS_KEY = ACTION_MODE
STATE_STATS_KEY = f"{ACTION_MODE}_state"
OUTPUT_REPRESENTATION = "native_delta_eef10"
EEF10_DIM = 10
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
NORMALIZATION_STATS_FILENAME = "libero_normalization_stats.npy"
EEF10_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "rot6d_col0_x",
    "rot6d_col0_y",
    "rot6d_col0_z",
    "rot6d_col1_x",
    "rot6d_col1_y",
    "rot6d_col1_z",
    "gripper_open_scale",
]
NATIVE_ACTION_NAMES = [
    "eef_native_delta_x",
    "eef_native_delta_y",
    "eef_native_delta_z",
    "eef_native_delta_rot6d_col0_x",
    "eef_native_delta_rot6d_col0_y",
    "eef_native_delta_rot6d_col0_z",
    "eef_native_delta_rot6d_col1_x",
    "eef_native_delta_rot6d_col1_y",
    "eef_native_delta_rot6d_col1_z",
    "gripper_open_command",
]


def _as_priority(value: Optional[Sequence[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    return next((key for key in priorities if key in features), None)


class LiberoDataset(LeRobotV3Reader):
    """Single-bucket canonical native-action LIBERO reader.

    State and action are both 10-D on disk, but are normalized from separate
    stats blocks because state xyz is a metric achieved pose whereas action xyz
    is a normalized controller command.  ``unify_action`` remains available for
    the normal 80-D model interface and maps both streams to slots 0..9.
    """

    DATASET_NAME = "LIBERO"
    ACTION_DIM = EEF10_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEPLOY_ACTION_MODE = ACTION_MODE

    # Fixed camera layout (head, wrist, unused); None slots are rendered black
    # in multiview mode. Override via ``camera_layout``.
    DEFAULT_CAMERA_LAYOUT: ClassVar[Tuple[Optional[str], ...]] = (
        "observation.images.image",
        "observation.images.image2",
        None,
    )
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "normalization_stats_path",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = ACTION_MODE,
        normalization_stats_path: Optional[str] = None,
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != ACTION_MODE:
            raise ValueError(f"LIBERO supports only action_mode={ACTION_MODE!r}, got {action_mode!r}.")
        if bool(unify_action) and unify_action_map is None:
            raise ValueError(
                "LIBERO unify_action=true requires an explicit unify_action_map; "
                'set ["0-9"] for the canonical single-arm left-slot mapping'
            )

        self.action_mode = mode
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._state_normalization_stats: Optional[dict] = None
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=bool(unify_action),
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        if self._target_camera is not None:
            return self._target_camera, None, None
        layout = list(self._camera_layout_param or self.DEFAULT_CAMERA_LAYOUT)
        layout += [None] * (3 - len(layout))
        return tuple(str(cam) if cam else None for cam in layout[:3])

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"LIBERO {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        for column in ("observation.state", "action"):
            shape = tuple(features.get(column, {}).get("shape", ()))
            if shape != (EEF10_DIM,):
                raise ValueError(f"LIBERO {column} feature must have shape [{EEF10_DIM}], got {shape}")

        conversion_path = self._dataset_dir / "meta" / "conversion.json"
        if not conversion_path.is_file():
            raise FileNotFoundError(f"LIBERO requires {conversion_path} (native-action conversion metadata)")
        with conversion_path.open(encoding="utf-8") as handle:
            conversion = json.load(handle)
        if conversion.get("output_representation") != OUTPUT_REPRESENTATION:
            raise ValueError(
                f"{conversion_path} declares output_representation={conversion.get('output_representation')!r}, "
                f"expected {OUTPUT_REPRESENTATION!r}; this is not a native-delta LIBERO dataset"
            )

    def _build_stats_rank0(self, path: Path) -> None:
        """Auto-build the pooled stats file: rank 0 scans, other ranks wait."""
        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        if dist_ready:
            rank = dist.get_rank()
        else:
            # torchrun sets RANK before init_process_group; honor it so
            # pre-init constructions still elect a single builder.
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            from openwam.dataloader.utils.stats_computation.libero_stats_computation import (
                build_and_save_libero_stats,
            )

            logger.info("No LIBERO stats at %s — scanning the dataset (rank 0; other ranks wait)", path)
            build_and_save_libero_stats(self._dataset_dir, output=path)
        else:
            deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
            poll_interval_s = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
            while not path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank 0 to build LIBERO stats: {path}")
                time.sleep(poll_interval_s)

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            self._state_normalization_stats = None
            return None
        stats_path = (
            Path(self._source_stats_path)
            if self._source_stats_path
            else self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
        )
        if not stats_path.is_file():
            self._build_stats_rank0(stats_path)

        raw = np.load(stats_path, allow_pickle=True).item()
        if not isinstance(raw, dict):
            raise ValueError(f"{stats_path} must contain a dictionary payload")
        if ACTION_STATS_KEY not in raw or STATE_STATS_KEY not in raw:
            raise KeyError(f"{stats_path} must contain {ACTION_STATS_KEY!r} and {STATE_STATS_KEY!r} blocks")
        action_raw = raw[ACTION_STATS_KEY]
        state_raw = raw[STATE_STATS_KEY]

        action_stats = materialize_eef_stats(
            dict(action_raw),
            self._normalize_mode,
            dim=self._raw_action_dim,
            strict_minmax=False,
            source_hint=f"{stats_path}:{ACTION_STATS_KEY}",
            force_rot6d_identity=True,
        )
        self._state_normalization_stats = materialize_eef_stats(
            dict(state_raw),
            self._normalize_mode,
            dim=self._raw_action_dim,
            strict_minmax=False,
            source_hint=f"{stats_path}:{STATE_STATS_KEY}",
            force_rot6d_identity=True,
        )
        for key, block in ((ACTION_STATS_KEY, action_raw), (STATE_STATS_KEY, state_raw)):
            recorded = block.get("gripper_convention")
            if recorded != GRIPPER_CONVENTION:
                raise ValueError(
                    f"{stats_path}:{key} declares gripper_convention={recorded!r}, expected {GRIPPER_CONVENTION!r}"
                )
            representation = block.get("representation")
            if representation is not None and representation != OUTPUT_REPRESENTATION:
                raise ValueError(
                    f"{stats_path}:{key} declares representation={representation!r}, expected {OUTPUT_REPRESENTATION!r}"
                )
        self.normalization_stats_path = str(stats_path)
        return action_stats

    @staticmethod
    def _read_eef10_column(win, column: str) -> np.ndarray:
        if column not in win:
            raise KeyError(f"LIBERO parquet window is missing {column!r}")
        values = np.stack(win[column].values).astype(np.float32)
        if values.ndim != 2 or values.shape[1] != EEF10_DIM:
            raise ValueError(f"LIBERO {column} must be (T, {EEF10_DIM}), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"LIBERO {column} contains NaN or infinity")
        return values

    def _normalize_array(self, arr: np.ndarray, stats: Optional[dict] = None) -> np.ndarray:
        """Normalize non-rotation dimensions while preserving rot6d exactly.

        Rot6d stores the first two columns of an SO(3) matrix. It is already
        bounded and geometrically coupled, so an affine per-dimension transform
        would change the represented rotation. The stats loader pins these
        dimensions to identity values as a first line of defense; this explicit
        overwrite is the hard runtime guarantee, including custom statistics.
        """
        raw = np.asarray(arr, dtype=np.float32)
        normalized = apply_normalization(
            arr,
            self._normalization_stats if stats is None else stats,
            self._normalize_mode,
        )
        output = np.array(normalized, dtype=np.float32, copy=True)
        output[..., ROT6D_DIMS_ARM10] = raw[..., ROT6D_DIMS_ARM10]
        return output

    def _raw_action_eef10(self, win) -> np.ndarray:
        """Return row-aligned native delta-action EEF10 values."""
        return self._read_eef10_column(win, "action")

    def _raw_state_eef10(self, win) -> np.ndarray:
        """Return row-aligned achieved-state EEF10 values."""
        return self._read_eef10_column(win, "observation.state")

    def _action_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        return self._normalize_array(self._raw_action_eef10(win))

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        return self._normalize_array(self._raw_state_eef10(win)[0:1], self._state_normalization_stats)


ROT6D_DIMS_EEF10 = ROT6D_DIMS_ARM10

__all__ = [
    "ACTION_MODE",
    "ACTION_STATS_KEY",
    "EEF10_DIM",
    "GRIPPER_CONVENTION",
    "LiberoDataset",
    "NATIVE_ACTION_NAMES",
    "OUTPUT_REPRESENTATION",
    "ROT6D_DIMS_EEF10",
    "STATE_STATS_KEY",
]
