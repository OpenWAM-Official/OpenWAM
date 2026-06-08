"""OXE Bridge reader.

Bridge data v2 (~105h, 53,192 episodes, WidowX). Same Euler-XYZ schema as
BC-Z, but uses ``observation.images.image_0`` as the head camera (image_1/2
have only ~40% coverage and image_3 is mostly empty — see
plans/oxe_dataloaders.md §1.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.oxe_schema import bcz_state_to_arm10, euler7_action_to_arm10


class OxeBridgeDataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-Bridge"
    HEAD_CAMERA = "observation.images.image_0"
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


__all__ = ["OxeBridgeDataset"]
