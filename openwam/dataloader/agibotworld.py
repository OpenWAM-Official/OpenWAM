"""Public implementation. Dataset-specific audit notes were removed."""


































































from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.robocoin import GRIP_EXCLUDED_DIM_MASK, _build_dex_unify_map
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    apply_normalization,
    materialize_eef_stats,
    pin_rot6d_identity,
)

logger = logging.getLogger(__name__)


HEAD_CAMERA = "observation.images.head"
LEFT_WRIST_CAMERA = "observation.images.hand_left"
RIGHT_WRIST_CAMERA = "observation.images.hand_right"


_DEX_PER_HAND = 6

_EEF_RAW_DIM = 20
_DEX_RAW_DIM = 18 + 2 * _DEX_PER_HAND










_MOVE_SRC_DIMS = (0, 2)
_MOVE_DIM = len(_MOVE_SRC_DIMS)
_MOVE_SLOTS = (68, 70)
_MOVE_EPS = 1e-6





_DEX_BUCKET_IDS = frozenset(
    {
        "475", "536", "549", "554", "577", "578", "595", "608", "620", "622",
        "660", "679", "705", "710", "727", "730", "731", "749", "753",
    }
)


_GRIP_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
    "action.gripper",
    "observation.state.gripper",
)
_DEX_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
    "action.dex",
    "observation.state.dex",
)

_POSE_ONLY_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
)

_MOVE_COLS = ("action.robot_velocity", "observation.state.robot_velocity")




_ROT6D_DIMS_DEX30 = (3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23)

_STAT_FIELDS = ("min", "max", "mean", "std", "q01", "q99")




_UNIFIED_STATS_FILENAME = "stats_g2a.json"




_BUCKET_STATS_FILENAME = "stats.json"

_SEGMENT_FLAG_COL = "segment_flag"
_SEGMENT_DELTA_COL = "segment_delta"


def _validate_trim_ratio(value) -> float | None:
    """Public implementation. Dataset-specific audit notes were removed."""





    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"segment_max_trim_ratio must be a float in (0, 1] or null; got {value!r}") from e
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"segment_max_trim_ratio must be a fraction in (0, 1] or null; got {ratio!r}. "
            "Use 0.7 for '70% or more of the episode trimmed', not 70."
        )
    return ratio


def _bucket_has_base_motion(dataset_dir) -> bool:
    """Public implementation. Dataset-specific audit notes were removed."""










    p = Path(dataset_dir) / "meta" / _BUCKET_STATS_FILENAME
    if not p.exists():
        logger.warning(
            "AgiBotWorld %s: no %s — treating base as stationary (move slots unmapped).",
            dataset_dir, _BUCKET_STATS_FILENAME,
        )
        return False
    try:
        with open(p) as f:
            st = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(
            "AgiBotWorld %s: could not read %s (%s) — treating base as stationary (move slots unmapped).",
            dataset_dir, p.name, e,
        )
        return False
    moved = False
    for key in ("action.robot_velocity", "observation.state.robot_velocity"):
        blk = st.get(key)
        if not blk:
            continue
        lo = np.abs(np.asarray(blk.get("min", [0.0]), dtype=np.float64)).max()
        hi = np.abs(np.asarray(blk.get("max", [0.0]), dtype=np.float64)).max()
        moved = moved or max(lo, hi) > _MOVE_EPS
    return moved


def _eef18_to_eef20(ee18: np.ndarray, grip2: np.ndarray) -> np.ndarray:
    """Public implementation. Dataset-specific audit notes were removed."""













    l_pose9 = ee18[:, 0:9]
    r_pose9 = ee18[:, 9:18]
    l_grip = grip2[:, 0:1]
    r_grip = grip2[:, 1:2]
    return np.concatenate([l_pose9, l_grip, r_pose9, r_grip], axis=-1).astype(np.float32)







class AgiBotWorldDataset(LeRobotV3Reader):
    """Public implementation. Dataset-specific audit notes were removed."""






    DATASET_NAME = "AgiBotWorld"
    HEAD_CAMERA = HEAD_CAMERA
    LEFT_WRIST_CAMERA = LEFT_WRIST_CAMERA
    RIGHT_WRIST_CAMERA = RIGHT_WRIST_CAMERA

    PROMPT_FILE_REQUIRED = True

    DEFAULT_NORMALIZE_MODE = None

    WRIST_DECODE_TOLERATED = (Exception,)
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("use_segment_annotations", "segment_max_trim_ratio")



    def __init__(
        self,
        dataset_dir,
        *,
        unify_action: bool = False,
        unify_action_map=None,
        use_segment_annotations: bool = True,
        segment_max_trim_ratio: float | None = None,
        **kwargs,
    ):
        """Public implementation. Dataset-specific audit notes were removed."""


























        self._is_dex = Path(dataset_dir).name in _DEX_BUCKET_IDS
        self._has_move = unify_action and _bucket_has_base_motion(dataset_dir)
        self._use_segment_annotations = bool(use_segment_annotations)
        self._segment_max_trim_ratio = _validate_trim_ratio(segment_max_trim_ratio)
        if self._segment_max_trim_ratio is not None and not self._use_segment_annotations:
            logger.warning(
                "AgiBotWorld %s: segment_max_trim_ratio=%s is ignored because "
                "use_segment_annotations=False (there is no trim to measure).",
                dataset_dir,
                self._segment_max_trim_ratio,
            )
        if unify_action:


            move_dim = _MOVE_DIM if self._has_move else 0
            move_slots = list(_MOVE_SLOTS) if self._has_move else []
            if self._is_dex:
                self.ACTION_DIM = _DEX_RAW_DIM + move_dim
                unify_action_map = _build_dex_unify_map(_DEX_PER_HAND, _DEX_PER_HAND) + move_slots
            else:
                self.ACTION_DIM = _EEF_RAW_DIM + move_dim
                unify_action_map = ["0-9", "34-43"] + move_slots
        super().__init__(dataset_dir, unify_action=unify_action, unify_action_map=unify_action_map, **kwargs)



    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """Public implementation. Dataset-specific audit notes were removed."""
























        if not self._use_segment_annotations:
            return eps_df.reset_index(drop=True)
        if _SEGMENT_FLAG_COL not in eps_df.columns or _SEGMENT_DELTA_COL not in eps_df.columns:
            logger.warning(
                "AgiBotWorld(%s): segment annotation columns %s/%s missing; using full segments.",
                self._dataset_id,
                _SEGMENT_FLAG_COL,
                _SEGMENT_DELTA_COL,
            )
            return eps_df.reset_index(drop=True)

        out = eps_df.copy()
        flag = out[_SEGMENT_FLAG_COL].fillna(0).astype(np.int64).to_numpy()
        delta = np.maximum(0, out[_SEGMENT_DELTA_COL].fillna(0).astype(np.int64).to_numpy())
        length = out["length"].astype(np.int64).to_numpy()

        valid_start = np.zeros(len(out), dtype=np.int64)
        valid_end = length.copy()
        start_mask = flag == 1
        end_mask = flag == 2
        valid_start[start_mask] = np.minimum(delta[start_mask], length[start_mask])
        valid_end[end_mask] = np.maximum(0, length[end_mask] - delta[end_mask])

        keep = (flag != 3) & (valid_end > valid_start)
        dropped_static = int(np.sum(flag == 3))
        dropped_empty = int(np.sum((flag != 3) & (valid_end <= valid_start)))






        dropped_over_trim = 0
        if self._segment_max_trim_ratio is not None:
            trimmed = np.maximum(0, length - np.maximum(0, valid_end - valid_start))
            with np.errstate(invalid="ignore", divide="ignore"):
                trim_ratio = np.where(length > 0, trimmed / np.maximum(length, 1), 0.0)
            over = (flag != 3) & keep & (trim_ratio >= self._segment_max_trim_ratio)
            dropped_over_trim = int(np.sum(over))
            keep = keep & ~over

        out["_valid_start"] = valid_start
        out["_valid_end"] = valid_end
        out = out.loc[keep].reset_index(drop=True)
        if dropped_static or dropped_empty or dropped_over_trim:
            logger.info(
                "AgiBotWorld(%s): segment annotations dropped %d static, %d empty-after-trim and "
                "%d over-trimmed (>=%s of the episode) episodes; kept %d episodes.",
                self._dataset_id,
                dropped_static,
                dropped_empty,
                dropped_over_trim,
                "n/a" if self._segment_max_trim_ratio is None else f"{self._segment_max_trim_ratio:.0%}",
                len(out),
            )
        return out

    def _resolve_cameras(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""







        self._robot_type = info.get("robot_type", "unknown")
        move_cols = _MOVE_COLS if self._has_move else ()
        if self._is_dex:
            if self._unify:




                self.NEEDED_COLS = _DEX_COLS + move_cols
            else:

                self.NEEDED_COLS = _POSE_ONLY_COLS
                self.ACTION_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
        else:
            self.NEEDED_COLS = (_GRIP_COLS + move_cols) if self._unify else _GRIP_COLS
        return self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA

    def _load_stats(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""























        self._action_norm_stats = None
        self._proprio_norm_stats = None
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir.parent / "meta" / _UNIFIED_STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but the unified stats file {stats_path} is missing. "
                f"Run `python -m openwam.dataloader.utils.stats_computation.agibotworld_stats_computation "
                f"--dataset_dir {self._dataset_dir.parent}` to generate it, or set normalize_mode=null to disable."
            )
        with open(stats_path) as f:
            raw = json.load(f)

        def _mat(key: str, dim: int):
            if key not in raw:
                raise KeyError(f"AgiBotWorld bucket {self._dataset_id}: '{key}' missing from {stats_path}.")
            return materialize_eef_stats(
                raw[key], self._normalize_mode, dim=dim, strict_minmax=False, source_hint=f"{stats_path}: {key}"
            )

        def _build(prefix: str):
            """Public implementation. Dataset-specific audit notes were removed."""

            ee = _mat(f"{prefix}.ee_base", 18)
            combined = {}
            if self._is_dex and self._unify:
                fing = _mat(f"{prefix}.dex", 2 * _DEX_PER_HAND)
                for k in _STAT_FIELDS:
                    e, h = ee[k], fing[k]
                    combined[k] = np.concatenate(
                        [e[0:9], h[0:_DEX_PER_HAND], e[9:18], h[_DEX_PER_HAND : 2 * _DEX_PER_HAND]]
                    ).astype(np.float32)
                pin_rot6d_identity(combined, _ROT6D_DIMS_DEX30)
            else:
                g = _mat(f"{prefix}.gripper", 2)
                for k in _STAT_FIELDS:
                    e, gg = ee[k], g[k]
                    combined[k] = np.concatenate([e[0:9], gg[0:1], e[9:18], gg[1:2]]).astype(np.float32)
                pin_rot6d_identity(combined, ROT6D_DIMS_EEF20)
            if self._has_move:



                vel = _mat(f"{prefix}.robot_velocity", 3)
                src = list(_MOVE_SRC_DIMS)
                for k in _STAT_FIELDS:
                    combined[k] = np.concatenate([combined[k], vel[k][src]]).astype(np.float32)
            return combined

        self._action_norm_stats = _build("action")
        self._proprio_norm_stats = _build("observation.state")


        return self._action_norm_stats

    def _normalize_array(self, arr: np.ndarray, stats) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""
        return apply_normalization(arr, stats, self._normalize_mode)

    def _grip_or_zeros(self, win, col: str, n: int) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""





        if not self._is_dex:
            return np.stack(win[col].values[:n]).astype(np.float32)
        return np.zeros((n, 2), dtype=np.float32)

    def _dex_pose_fingers(self, ee18: np.ndarray, dex12: np.ndarray) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""







        l_pose9 = ee18[:, 0:9]
        r_pose9 = ee18[:, 9:18]
        l_fing = dex12[:, 0:_DEX_PER_HAND]
        r_fing = dex12[:, _DEX_PER_HAND : 2 * _DEX_PER_HAND]
        return np.concatenate([l_pose9, l_fing, r_pose9, r_fing], axis=-1).astype(np.float32)

    def _append_move(self, raw: np.ndarray, win, col: str, n: int) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""

        if not self._has_move:
            return raw
        vel = np.stack(win[col].values[:n]).astype(np.float32)[:, _MOVE_SRC_DIMS]
        return np.concatenate([raw, vel], axis=-1)

    def _action_20d(self, win) -> np.ndarray:
        ee = np.stack(win["action.ee_base"].values).astype(np.float32)
        n = len(ee)
        if self._is_dex and self._unify:
            dex = np.stack(win["action.dex"].values).astype(np.float32)
            raw = self._dex_pose_fingers(ee, dex)
        else:
            grip = self._grip_or_zeros(win, "action.gripper", n)
            raw = _eef18_to_eef20(ee, grip)
        raw = self._append_move(raw, win, "action.robot_velocity", n)
        return self._normalize_array(raw, self._action_norm_stats)

    def _proprio_20d(self, win) -> np.ndarray:
        ee = np.stack(win["observation.state.ee_base"].values[:1]).astype(np.float32)
        if self._is_dex and self._unify:
            dex = np.stack(win["observation.state.dex"].values[:1]).astype(np.float32)
            raw = self._dex_pose_fingers(ee, dex)
        else:
            grip = self._grip_or_zeros(win, "observation.state.gripper", 1)
            raw = _eef18_to_eef20(ee, grip)
        raw = self._append_move(raw, win, "observation.state.robot_velocity", 1)
        return self._normalize_array(raw, self._proprio_norm_stats)

    @property
    def robot_type(self):
        return self._robot_type

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiAgiBotWorldDataset







class MultiAgiBotWorldDataset(MultiLeRobotV3Reader):
    """Public implementation. Dataset-specific audit notes were removed."""

    def __init__(self, buckets: List[AgiBotWorldDataset]):
        super().__init__(buckets)
        n_dex = sum(1 for b in self._buckets if getattr(b, "_is_dex", False))
        logger.info(
            "MultiAgiBotWorldDataset: %d datasets (%d dex-hand, %d grippered), %d windows",
            len(self._buckets),
            n_dex,
            len(self._buckets) - n_dex,
            len(self),
        )




    @classmethod
    def from_config(cls, config, split: str = "train"):
        return AgiBotWorldDataset.from_config(config, split)


__all__ = ["AgiBotWorldDataset", "MultiAgiBotWorldDataset"]
