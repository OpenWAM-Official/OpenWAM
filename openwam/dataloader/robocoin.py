"""Public implementation. Dataset-specific audit notes were removed."""







































from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import EEF_DIM as _ACTION_DIM
from openwam.dataloader.utils.eef import eef14_to_eef20
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

_STATE_DIM = _ACTION_DIM


_NEEDED_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "gripper_open_scale_action",
    "eef_sim_pose_state",
    "gripper_open_scale_state",
)



_eef14_to_eef20 = eef14_to_eef20






HEAD_CAMERA_PRIORITY = [
    "observation.images.cam_high_rgb",
    "observation.images.cam_head_rgb",
    "observation.images.cam_head_right_rgb",
    "observation.images.cam_head_left_rgb",
    "observation.images.cam_high_right_rgb",
    "observation.images.cam_high_left_rgb",
    "observation.images.cam_high_realsense_rgb",
    "observation.images.cam_front_rgb",
    "observation.images.cam_front_chest_rgb",
    "observation.images.cam_chest_rgb",
]

WRIST_LEFT_CANDIDATES = [
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_left_wrist_rgb_rgb",
]

WRIST_RIGHT_CANDIDATES = [
    "observation.images.cam_right_wrist_rgb",
    "observation.images.cam_right_wrist_rgb_rgb",
]


def _resolve_robocoin_cameras(features: dict) -> tuple:
    """Public implementation. Dataset-specific audit notes were removed."""




    feat_keys = set(features.keys())
    head = None
    for c in HEAD_CAMERA_PRIORITY:
        if c in feat_keys:
            head = c
            break
    left_wrist = None
    for c in WRIST_LEFT_CANDIDATES:
        if c in feat_keys:
            left_wrist = c
            break
    right_wrist = None
    for c in WRIST_RIGHT_CANDIDATES:
        if c in feat_keys:
            right_wrist = c
            break
    return head, left_wrist, right_wrist







class RoboCOINDataset(LeRobotV3Reader):
    """Public implementation. Dataset-specific audit notes were removed."""







    DATASET_NAME = "RoboCOIN"
    NEEDED_COLS = _NEEDED_COLS


    PROMPT_FILE_REQUIRED = False


    WRIST_DECODE_TOLERATED = (Exception,)



    def _resolve_cameras(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""
        features = info.get("features", {})
        head, left_wrist, right_wrist = _resolve_robocoin_cameras(features)
        if head is None:
            raise ValueError(f"No head camera found in {self._dataset_id}")
        self._robot_type = info.get("robot_type", "unknown")
        return head, left_wrist, right_wrist

    def _add_data_offsets(self, eps) -> None:


        self._add_data_offsets_from_files(eps)

    def _add_data_offsets_from_files(self, eps):
        """Public implementation. Dataset-specific audit notes were removed."""









        paths = sorted((self._dataset_dir / "data").glob("chunk-*/file-*.parquet"))

        def _read_meta(path):
            chunk_m = re.search(r"chunk-(\d+)$", path.parent.name)
            file_m = re.search(r"file-(\d+)$", path.stem)
            if chunk_m is None or file_m is None:
                return None
            return (int(chunk_m.group(1)), int(file_m.group(1)), pq.ParquetFile(path).metadata.num_rows)

        if not paths:
            raise FileNotFoundError(f"No data parquet files under {self._dataset_dir}/data")


        n_workers = min(len(paths), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_read_meta, paths))
        data_files = [r for r in results if r is not None]
        if not data_files:
            raise FileNotFoundError(f"No data parquet files under {self._dataset_dir}/data")

        starts = np.concatenate([[0], np.cumsum([n for _, _, n in data_files])]).astype(np.int64)
        global_starts = eps["dataset_from_index"].to_numpy().astype(np.int64)
        file_pos = np.searchsorted(starts, global_starts, side="right") - 1
        if (file_pos < 0).any() or (file_pos >= len(data_files)).any():
            raise ValueError(f"{self._dataset_id}: dataset_from_index outside data parquet row range")

        chunks = np.array([data_files[i][0] for i in file_pos], dtype=np.int64)
        files = np.array([data_files[i][1] for i in file_pos], dtype=np.int64)
        eps["data/chunk_index"] = chunks
        eps["data/file_index"] = files
        eps["_data_row_offset"] = global_starts - starts[file_pos]

    def _load_stats(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir.parent / "meta" / f"stats_{self._robot_type}.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but stats file is missing: {stats_path}. "
                f"Run python -m openwam.dataloader.utils.stats_computation.robocoin_stats_computation to generate it, or set normalize_mode=null."
            )
        with open(stats_path) as f:
            raw = json.load(f)
        eef = raw.get("eef", {})
        return materialize_eef_stats(
            eef,
            self._normalize_mode,
            dim=_ACTION_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: eef.* — re-run python -m openwam.dataloader.utils.stats_computation.robocoin_stats_computation",
        )

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""






        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        eef_action = np.stack(win["eef_sim_pose_action"].values).astype(np.float32)
        grip_action = np.stack(win["gripper_open_scale_action"].values).astype(np.float32)
        return self._normalize_array(eef14_to_eef20(eef_action, grip_action))

    def _proprio_20d(self, win) -> np.ndarray:
        eef_state = np.stack(win["eef_sim_pose_state"].values[:1]).astype(np.float32)
        grip_state = np.stack(win["gripper_open_scale_state"].values[:1]).astype(np.float32)
        return self._normalize_array(eef14_to_eef20(eef_state, grip_state))

    @property
    def robot_type(self):
        return self._robot_type

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiRobotCOINDataset







class MultiRobotCOINDataset(MultiLeRobotV3Reader):
    """Public implementation. Dataset-specific audit notes were removed."""

    def __init__(self, buckets: List[RoboCOINDataset]):
        super().__init__(buckets)
        robot_types = set(b.robot_type for b in self._buckets)
        logger.info(
            "MultiRobotCOINDataset: %d datasets, %d windows, %d robot types: %s",
            len(self._buckets),
            len(self),
            len(robot_types),
            sorted(robot_types),
        )

    @property
    def action_dim(self):



        return self._buckets[0].action_dim if self._buckets else _ACTION_DIM

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return RoboCOINDataset.from_config(config, split)


__all__ = ["RoboCOINDataset", "MultiRobotCOINDataset"]
