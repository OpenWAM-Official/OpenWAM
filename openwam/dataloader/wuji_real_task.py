"""Wuji/Astribot Grasp Anything single-bucket reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.lerobotv3 import apply_info_splits
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.rot6d import convert_wuji_58

_CAMERAS = (
    "observation.images.head_view",
    "observation.images.left_wrist_view",
    "observation.images.right_wrist_view",
)
_STATS_KEYS = ("mean", "std", "min", "max", "q01", "q99")
_ROT6D_DIMS = (3, 4, 5, 6, 7, 8, 32, 33, 34, 35, 36, 37)


class WujiRealTaskDataset(LeRobotV3Reader):
    DATASET_NAME = "WujiRealTask"
    ACTION_DIM = 58
    NEEDED_COLS = ("action", "observation.state", "task_index")
    DEPLOY_ACTION_MODE = "eef"
    PROMPT_FILE_REQUIRED = True
    HEAD_CAMERA = _CAMERAS[0]
    LEFT_WRIST_CAMERA = _CAMERAS[1]
    RIGHT_WRIST_CAMERA = _CAMERAS[2]
    CONFIG_KEYS: ClassVar[tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "rot6d_convention",
        "enable_action_supervision",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = "eef",
        rot6d_convention: str = "column",
        enable_action_supervision: bool = True,
        **kwargs: Any,
    ):
        if str(action_mode).strip().lower() != "eef":
            raise ValueError(f"WujiRealTask supports only action_mode='eef', got {action_mode!r}")
        convention = str(rot6d_convention).strip().lower()
        if convention not in {"row", "column"}:
            raise ValueError(f"rot6d_convention must be 'row' or 'column', got {rot6d_convention!r}")
        self.action_mode = "eef"
        self.rot6d_convention = convention
        self._enable_action_supervision = bool(enable_action_supervision)
        self._source_stats_path: Optional[Path] = None
        super().__init__(dataset_dir=dataset_dir, **kwargs)

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        path = self._dataset_dir / "meta" / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{self.DATASET_NAME}: missing {path}")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError(f"{self.DATASET_NAME}: episodes.jsonl is empty")
        # Normalize the v2 episode-addressed templates to the chunk/file names
        # consumed by the shared base reader.
        self._data_path_template = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
        self._video_path_template = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
        rows = []
        data_start = 0
        for record in sorted(records, key=lambda item: int(item["episode_index"])):
            ep = int(record["episode_index"])
            length = int(record["length"])
            if length <= 0:
                raise ValueError(f"{self.DATASET_NAME}: episode {ep} has invalid length={length}")
            row = dict(record)
            row.update(
                {
                    "episode_index": ep,
                    "length": length,
                    "dataset_from_index": data_start,
                    "data/chunk_index": 0,
                    "data/file_index": ep,
                }
            )
            for camera in _CAMERAS:
                row[f"videos/{camera}/chunk_index"] = 0
                row[f"videos/{camera}/file_index"] = ep
            rows.append(row)
            data_start += length
        eps = pd.DataFrame(rows)
        eps["_data_row_offset"] = 0
        for camera in _CAMERAS:
            eps[f"_video_frame_offset/{camera}"] = 0
        return apply_info_splits(
            eps,
            self._split,
            info.get("splits", {}) or {},
            source_name=f"{self.DATASET_NAME}({self._dataset_id})",
        )

    def _load_prompts(self) -> None:
        path = self._dataset_dir / "meta" / "tasks.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{self.DATASET_NAME}: missing {path}")
        self._task_idx_to_text = {}
        for record in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            idx = int(record["task_index"])
            text = str(record.get("task", "")).strip()
            if not text:
                raise ValueError(f"{self.DATASET_NAME}: task_index={idx} has an empty prompt")
            self._task_idx_to_text[idx] = text

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        for camera in _CAMERAS:
            if camera not in features or features[camera].get("dtype") != "video":
                raise ValueError(f"{self.DATASET_NAME}: required camera feature {camera!r} is missing")
        return _CAMERAS

    def _load_stats(self, info: dict):
        if not self._normalize_mode or str(self._normalize_mode).lower() in {"none", "null"}:
            return None
        path = self._dataset_dir / "meta" / "normalization_stats.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{self.DATASET_NAME}: normalization stats missing: {path}")
        payload = np.load(path, allow_pickle=True).item()
        stats = payload.get("eef") if isinstance(payload, dict) else None
        if not isinstance(stats, dict):
            raise ValueError(f"{path} must contain an 'eef' stats mapping")
        materialized = {}
        for key in _STATS_KEYS:
            if key not in stats or np.asarray(stats[key]).shape != (58,):
                raise ValueError(f"{path}: eef.{key} must be a 58-D vector")
            materialized[key] = np.asarray(stats[key], dtype=np.float32)
        identity_values = (
            ("mean", 0.0),
            ("std", 1.0),
            ("min", -1.0),
            ("max", 1.0),
            ("q01", -1.0),
            ("q99", 1.0),
        )
        for key, expected in identity_values:
            if not np.allclose(materialized[key][list(_ROT6D_DIMS)], expected, atol=0.0):
                raise ValueError(f"{path}: eef.{key} has non-identity rot6d statistics")
        self.normalization_stats_path = str(path)
        self._source_stats_path = path
        return materialized

    def _post_init(self, info: dict) -> None:
        robot_type = str(info.get("robot_type", ""))
        expected_type = (
            "WUJI_ASTRIBOT_EEF_ABSOLUTE_HAND_ABSOLUTE_ROT6D"
            if self.rot6d_convention == "row"
            else "WUJI_ASTRIBOT_EEF_ABSOLUTE_HAND_ABSOLUTE_ROT6D_COLUMN"
        )
        if robot_type != expected_type:
            raise ValueError(f"{self.DATASET_NAME}: unexpected robot_type={robot_type!r}")
        for name in ("action", "observation.state"):
            shape = tuple((info.get("features", {}).get(name, {}) or {}).get("shape", ()))
            if shape != (58,):
                raise ValueError(f"{self.DATASET_NAME}: feature {name!r} must have shape [58], got {shape}")

    def _canonical(self, values: np.ndarray) -> np.ndarray:
        if self.rot6d_convention == "row":
            return convert_wuji_58(values, row_rot6d=True)
        return np.asarray(values, dtype=np.float32)

    def _read_vector(self, win: pd.DataFrame, column: str, *, first_only: bool = False) -> np.ndarray:
        values = np.asarray(np.stack(win[column].values), dtype=np.float32)
        if first_only:
            values = values[:1]
        if values.ndim != 2 or values.shape[-1] != 58:
            raise ValueError(f"{self.DATASET_NAME} {column} must be (T, 58), got {values.shape}")
        return self._canonical(values)

    def _normalize_array(self, values: np.ndarray) -> np.ndarray:
        return apply_normalization(values, self._normalization_stats, self._normalize_mode).astype(np.float32)

    def _action_20d(self, win: pd.DataFrame):
        if not self._enable_action_supervision:
            return None
        return self._normalize_array(self._read_vector(win, "action"))

    def _proprio_20d(self, win: pd.DataFrame):
        return self._normalize_array(self._read_vector(win, "observation.state", first_only=True))


__all__ = ["WujiRealTaskDataset"]
