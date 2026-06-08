"""OXE BC-Z reader.

Single-arm Euler-XYZ EEF source from the BC-Z dataset (Google Robot, 10 fps,
1 camera, 171×213). Loads from the LeRobot v3 conversion at
``/path/to/OXE/BC-Z-Dataset``.

Schema:
  * observation.state[8] = [x, y, z, roll, pitch, yaw, **pad**, gripper]
    The pad slot is silently dropped.
  * action[7] = [x, y, z, roll, pitch, yaw, gripper]
  * Single head camera ``observation.images.image`` — no wrist cameras.
    Multiview mode stretches the head into the top slot of the L-shape canvas
    and leaves both wrist slots black.

Emits the canonical 20-D EEF sample dict (front 10 dims = real left arm,
back 10 dims = zero-padded). See plans/oxe_dataloaders.md for the broader
contract. All LeRobot v3 reading machinery lives in
:class:`~openwam.dataloader.bases.lerobot_v3_reader.LeRobotV3Reader`; this
reader only declares its schema + the per-dataset 10-D EEF conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.oxe_schema import bcz_state_to_arm10, euler7_action_to_arm10


class OxeBczDataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-BC-Z"
    HEAD_CAMERA = "observation.images.image"
    # Prompts come from tasks_annotated.parquet (per-episode), so task_index
    # is no longer consumed from the data parquet.
    NEEDED_COLS = ("observation.state", "action")
    ACTION_DIM_MASK = LEFT_ARM_DIM_MASK
    PROMPT_SOURCE = "episode_annotated"
    DEFAULT_NORMALIZE_MODE = "quantile"  # OXE readers historically default to quantile
    STATS_FILENAME = "eef_stats.json"
    STATS_DIM = 10
    STATS_STRICT_MINMAX = True

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        action = np.stack(win["action"].values).astype(np.float32)  # (T_actual, 7)
        return single_arm_20d(euler7_action_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        state = np.stack(win["observation.state"].values[:1]).astype(np.float32)  # (1, 8)
        return single_arm_20d(bcz_state_to_arm10(state), self._normalization_stats, self._normalize_mode)


__all__ = ["OxeBczDataset"]
