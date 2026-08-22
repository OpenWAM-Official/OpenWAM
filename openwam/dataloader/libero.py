"""LIBERO LeRobot v3 dataloader for the canonical row-aligned EEF10 dataset.

The only accepted on-disk contract is::

    observation.state = achieved [xyz3, rot6d6, gripper_open_scale1]
    action            = absolute OSC goal [xyz3, rot6d6, gripper_open_scale1]

Both columns are world-frame, full-pose EEF10 values and are consumed directly
from the same row. The reader performs no state/action representation conversion
and no temporal shifting. The gripper convention is ``-1 = closed, +1 = open``.

With ``unify_action: true``, the 10 raw dimensions are normalized by one shared
``eef`` statistics block and scattered into unified slots 0..9.

Training and deployment share the single authoritative artifact
``meta/normalization_stats.npy``.  Its ``eef`` block is a superset of the six
vectors deployment consumes and also carries the LIBERO representation metadata.
"""

from __future__ import annotations

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
    load_stats_file,
    load_stats_metadata,
)

logger = logging.getLogger(__name__)

_ACTION_MODE = "eef"
EEF10_DIM = 10
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
NORMALIZATION_STATS_FILENAME = "normalization_stats.npy"


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
    )
    WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.wrist_image",
        "observation.images.image2",
        "observation.images.robot0_eye_in_hand_image",
    )
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "gripper_convention",
        "head_camera_priority",
        "wrist_camera_priority",
        "prompt_columns",
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
        prompt_columns: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
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
        convention = str(gripper_convention).strip()
        if convention != GRIPPER_CONVENTION:
            raise ValueError(
                f"LIBERO requires gripper_convention={GRIPPER_CONVENTION!r}, got {gripper_convention!r}"
            )
        unify_on = bool(unify_action)
        if unify_on and unify_action_map is None:
            raise ValueError(
                "LIBERO unify_action=true requires an explicit unify_action_map; "
                'set ["0-9"] for the canonical single-arm left-slot mapping'
            )
        self.action_mode = mode
        self._head_priority = _as_priority(head_camera_priority, self.HEAD_CAMERA_PRIORITY)
        self._wrist_priority = _as_priority(wrist_camera_priority, self.WRIST_CAMERA_PRIORITY)
        self._prompt_columns = _as_priority(
            prompt_columns,
            ("language_instruction", "task", "prompt"),
        )
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._resolved_stats_path: Optional[str] = None  # set by _load_stats when normalization is on
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
        return 1

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        return actual_raw_len

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"LIBERO {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        if "action" not in features:
            raise KeyError("LIBERO requires a row-aligned 10-D action column")
        action = features["action"]
        shape = tuple(action.get("shape", ()))
        if shape != (EEF10_DIM,):
            raise ValueError(f"LIBERO action feature must have shape [{EEF10_DIM}], got {shape}")
        if "observation.state" not in features:
            raise KeyError("LIBERO requires a row-aligned 10-D observation.state column")
        state = features["observation.state"]
        state_shape = tuple(state.get("shape", ()))
        if state_shape != (EEF10_DIM,):
            raise ValueError(f"LIBERO observation.state feature must have shape [{EEF10_DIM}], got {state_shape}")
        self._prompt_columns = tuple(col for col in self._prompt_columns if col in features)
        self.NEEDED_COLS = self.NEEDED_COLS + self._prompt_columns

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if self._source_stats_path:
            # An explicit path is authoritative and is never silently rebuilt.
            stats_path = Path(self._source_stats_path)
        else:
            stats_path = self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
            if not stats_path.is_file():
                self._build_default_stats(stats_path)
        self._resolved_stats_path = str(stats_path)
        global_stats = load_stats_file(
            stats_path,
            action_mode=self.action_mode,
            normalize_mode=str(self._normalize_mode),
            dim=self._raw_action_dim,
        )
        self._check_gripper_convention(stats_path)
        # The complete training payload is already deployment-compatible: the
        # deploy reader selects the same ``eef`` block and ignores its extra
        # provenance/quantile fields.  Surface this exact file to checkpointing
        # instead of materializing a second, reduced artifact.
        self.normalization_stats_path = str(stats_path)
        return global_stats

    def _check_gripper_convention(self, stats_path: Path) -> None:
        """Require stats generated for the canonical EEF10 gripper convention."""
        recorded = load_stats_metadata(stats_path, action_mode=self.action_mode).get("gripper_convention")
        if recorded == GRIPPER_CONVENTION:
            return
        raise ValueError(
            f"LIBERO stats {stats_path} declare gripper_convention={recorded!r}, expected "
            f"{GRIPPER_CONVENTION!r}. Regenerate the stats from the canonical EEF10 dataset."
        )

    def _build_default_stats(self, path: Path) -> None:
        """Build the default per-bucket stats file, coordinated across ranks.

        Rank 0 owns the full parquet scan (minutes over the complete dataset)
        and writes atomically; other ranks poll for the file. ``dist.barrier()``
        is deliberately avoided — a minutes-long scan would trip NCCL's
        collective timeout (the ebench precedent).
        """
        # Lazy import: the stats module imports this reader at module level.
        from openwam.dataloader.utils.stats_computation.libero_stats_computation import (
            build_and_save_libero_stats,
        )

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
            logger.info(
                "LIBERO(%s): no normalization stats at %s — computing from the dataset (rank 0; other ranks wait)",
                self._dataset_id,
                path,
            )
            build_and_save_libero_stats(self, path)
            return

        deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
        poll_interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for rank 0 to build LIBERO stats: {path}")
            time.sleep(poll_interval)

    @staticmethod
    def _read_eef10_column(win, column: str) -> np.ndarray:
        values = np.stack(win[column].values).astype(np.float32)
        if values.ndim != 2 or values.shape[1] != EEF10_DIM:
            raise ValueError(f"LIBERO {column} must be (T, {EEF10_DIM}), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"LIBERO {column} contains NaN or infinity")
        return values

    def _raw_action_eef10(self, win) -> np.ndarray:
        """Return the row-aligned absolute EEF10 action without conversion."""
        return self._read_eef10_column(win, "action")

    def _raw_state_eef10(self, win) -> np.ndarray:
        """Return the row-aligned achieved EEF10 state without conversion."""
        return self._read_eef10_column(win, "observation.state")

    def _action_20d(self, win) -> np.ndarray:
        return apply_normalization(self._raw_action_eef10(win), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._raw_state_eef10(win)[0:1]
        return apply_normalization(raw, self._normalization_stats, self._normalize_mode)

    def _resolve_prompt(self, row, win) -> str:
        for column in self._prompt_columns:
            value = win[column].iloc[0]
            if value is not None and not pd.isna(value):
                text = str(value).strip()
                if text:
                    return text
        return super()._resolve_prompt(row, win)

ROT6D_DIMS_EEF10 = ROT6D_DIMS_ARM10

__all__ = [
    "EEF10_DIM",
    "GRIPPER_CONVENTION",
    "ROT6D_DIMS_EEF10",
    "LiberoDataset",
]
