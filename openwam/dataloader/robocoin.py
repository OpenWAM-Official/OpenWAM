"""Public implementation. Dataset-specific audit notes were removed."""




























































from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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



_BASE_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "eef_sim_pose_state",
)
_GRIP_COLS = ("gripper_open_scale_action", "gripper_open_scale_state")





_DEX_UNIFY_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "eef_sim_pose_state",
    "action",
    "observation.state",
)




GRIP_EXCLUDED_DIM_MASK = np.ones(_ACTION_DIM, dtype=bool)
GRIP_EXCLUDED_DIM_MASK[9] = False
GRIP_EXCLUDED_DIM_MASK[19] = False


def _finger_indices(feature: dict):
    """Public implementation. Dataset-specific audit notes were removed."""




    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    names = names or []
    left = [i for i, x in enumerate(names) if str(x).startswith("left_hand_joint")]
    right = [i for i, x in enumerate(names) if str(x).startswith("right_hand_joint")]
    return left, right






MAX_HAND_DOF = 24


def dex_finger_layout(features: dict):
    """Public implementation. Dataset-specific audit notes were removed."""











    has_grip = all(c in features for c in _GRIP_COLS)
    if has_grip or "eef_sim_pose_action" not in features:
        return None
    aL, aR = _finger_indices(features.get("action", {}))
    sL, sR = _finger_indices(features.get("observation.state", {}))
    kL, kR = len(aL), len(aR)
    if not (0 < kL <= MAX_HAND_DOF and 0 < kR <= MAX_HAND_DOF and len(sL) == kL and len(sR) == kR):
        return None
    return aL, aR, sL, sR


def _build_dex_unify_map(k_left: int, k_right: int):
    """Public implementation. Dataset-specific audit notes were removed."""








    l_hand = 10
    r_pos = l_hand + MAX_HAND_DOF
    r_hand = r_pos + 10
    return (
        list(range(0, 3))
        + list(range(3, 9))
        + list(range(l_hand, l_hand + k_left))
        + list(range(r_pos, r_pos + 3))
        + list(range(r_pos + 3, r_pos + 9))
        + list(range(r_hand, r_hand + k_right))
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



    "observation.images.cam_front_head_rgb",
    "observation.images.cam_front_chest_rgb",
    "observation.images.cam_chest_rgb",


    "observation.images.camera_head_rgb",
    "observation.images.cam_left_high",
    "observation.images.ego_view",
]

WRIST_LEFT_CANDIDATES = [
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_left_wrist_rgb_rgb",
    "observation.images.camera_left_wrist_rgb",
    "observation.images.cam_left_wrist",
]

WRIST_RIGHT_CANDIDATES = [
    "observation.images.cam_right_wrist_rgb",
    "observation.images.cam_right_wrist_rgb_rgb",
    "observation.images.camera_right_wrist_rgb",
    "observation.images.cam_right_wrist",
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



    def __init__(self, dataset_dir, *, unify_action: bool = False, unify_action_map=None, **kwargs):
        """Public implementation. Dataset-specific audit notes were removed."""









        self._dex_unify = False
        self._k_left = 0
        self._k_right = 0
        if unify_action:
            try:
                with open(Path(dataset_dir) / "meta" / "info.json") as f:
                    features = json.load(f).get("features", {})
            except (OSError, ValueError):
                features = {}
            layout = dex_finger_layout(features)
            if layout is not None:
                aL, aR, sL, sR = layout
                kL, kR = len(aL), len(aR)
                self._dex_unify = True
                self._k_left, self._k_right = kL, kR
                self._fidx_act = (np.asarray(aL, dtype=np.int64), np.asarray(aR, dtype=np.int64))
                self._fidx_state = (np.asarray(sL, dtype=np.int64), np.asarray(sR, dtype=np.int64))


                self.ACTION_DIM = 18 + kL + kR
                unify_action_map = _build_dex_unify_map(kL, kR)
            elif "eef_sim_pose_action" in features and not all(c in features for c in _GRIP_COLS):




                aL, aR = _finger_indices(features.get("action", {}))
                sL, sR = _finger_indices(features.get("observation.state", {}))
                logger.warning(
                    "RoboCOIN %s: dexterous-hand bucket (pose, no gripper) but finger layout "
                    "failed the gate (action L/R=%d/%d, state L/R=%d/%d, max=%d); falling back "
                    "to pose-only under unify_action (fingers dropped).",
                    dataset_dir, len(aL), len(aR), len(sL), len(sR), MAX_HAND_DOF,
                )
        super().__init__(dataset_dir, unify_action=unify_action, unify_action_map=unify_action_map, **kwargs)



    def _resolve_cameras(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""









        features = info.get("features", {})
        head, left_wrist, right_wrist = _resolve_robocoin_cameras(features)
        if head is None:
            raise ValueError(f"No head camera found in {self._dataset_id}")
        self._robot_type = info.get("robot_type", "unknown")
        self._has_grip = all(c in features for c in _GRIP_COLS)
        if self._has_grip:
            self.NEEDED_COLS = _NEEDED_COLS
        elif self._dex_unify:



            self.NEEDED_COLS = _DEX_UNIFY_COLS
        else:

            self.NEEDED_COLS = _BASE_COLS
            self.ACTION_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
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
        eef_stats = materialize_eef_stats(
            raw.get("eef", {}),
            self._normalize_mode,
            dim=_ACTION_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: eef.* — re-run python -m openwam.dataloader.utils.stats_computation.robocoin_stats_computation",
        )





        for k in ("mean", "std", "min", "max", "q01", "q99"):
            if eef_stats[k].shape[0] != _ACTION_DIM:
                raise ValueError(
                    f"RoboCOIN bucket {self._dataset_id}: 'eef' stats '{k}' width "
                    f"{eef_stats[k].shape[0]} in {stats_path} != expected {_ACTION_DIM}. "
                    f"Re-run robocoin_stats_computation."
                )
        if not self._dex_unify:
            return eef_stats


        hand_raw = raw.get("hand")
        if not hand_raw:
            raise FileNotFoundError(
                f"unify_action + dexterous-hand bucket {self._dataset_id} needs a 'hand' stats "
                f"block in {stats_path}; re-run robocoin_stats_computation (it now emits hand stats)."
            )
        kL, kR = self._k_left, self._k_right














        dof_l, dof_r = hand_raw.get("dof_left"), hand_raw.get("dof_right")
        if dof_l is None or dof_r is None:
            raise ValueError(
                f"unify_action + dexterous-hand bucket {self._dataset_id}: 'hand' stats block in "
                f"{stats_path} is missing dof_left/dof_right (got left={dof_l}, right={dof_r}); "
                f"re-run robocoin_stats_computation."
            )
        if dof_l != kL or dof_r != kR:
            raise ValueError(
                f"unify_action + dexterous-hand bucket {self._dataset_id}: 'hand' stats DOF "
                f"(left={dof_l}, right={dof_r}) in {stats_path} != this bucket's finger DOF "
                f"(left={kL}, right={kR}). The per-robot-type 'hand' block is locked to the first "
                f"dataset's DOF; re-run robocoin_stats_computation (it now hard-fails on mixed DOF), "
                f"or split mismatched datasets into distinct robot_types."
            )
        hand_stats = materialize_eef_stats(
            hand_raw, self._normalize_mode, dim=kL + kR, strict_minmax=False, source_hint=f"{stats_path}: hand.*"
        )



        for k in ("mean", "std", "min", "max", "q01", "q99"):
            if hand_stats[k].shape[0] != kL + kR:
                raise ValueError(
                    f"unify_action + dexterous-hand bucket {self._dataset_id}: 'hand' stats '{k}' "
                    f"width {hand_stats[k].shape[0]} in {stats_path} != expected kL+kR={kL + kR}. "
                    f"Re-run robocoin_stats_computation."
                )
        combined = {}
        for k in ("mean", "std", "min", "max", "q01", "q99"):
            e, h = eef_stats[k], hand_stats[k]
            combined[k] = np.concatenate([e[0:9], h[0:kL], e[10:19], h[kL : kL + kR]]).astype(np.float32)
        return combined

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""






        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _grip_or_zeros(self, win, col: str, n: int) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""





        if self._has_grip:
            return np.stack(win[col].values[:n]).astype(np.float32)
        return np.zeros((n, 2), dtype=np.float32)

    def _dex_raw(self, eef12: np.ndarray, raw_arr: np.ndarray, fidx) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""






        pose20 = eef14_to_eef20(eef12, np.zeros((len(eef12), 2), dtype=np.float32))
        l_pose, r_pose = pose20[:, 0:9], pose20[:, 10:19]
        fL, fR = fidx
        l_fing = raw_arr[:, fL].astype(np.float32)
        r_fing = raw_arr[:, fR].astype(np.float32)
        raw = np.concatenate([l_pose, l_fing, r_pose, r_fing], axis=-1)
        return self._normalize_array(raw)

    def _action_20d(self, win) -> np.ndarray:
        eef_action = np.stack(win["eef_sim_pose_action"].values).astype(np.float32)
        if self._dex_unify:
            raw_arr = np.stack(win["action"].values).astype(np.float32)
            return self._dex_raw(eef_action, raw_arr, self._fidx_act)
        grip_action = self._grip_or_zeros(win, "gripper_open_scale_action", len(eef_action))
        return self._normalize_array(eef14_to_eef20(eef_action, grip_action))

    def _proprio_20d(self, win) -> np.ndarray:
        eef_state = np.stack(win["eef_sim_pose_state"].values[:1]).astype(np.float32)
        if self._dex_unify:
            raw_arr = np.stack(win["observation.state"].values[:1]).astype(np.float32)
            return self._dex_raw(eef_state, raw_arr, self._fidx_state)
        grip_state = self._grip_or_zeros(win, "gripper_open_scale_state", len(eef_state))
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
