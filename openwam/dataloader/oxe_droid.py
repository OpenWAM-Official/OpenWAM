"""Public implementation. Dataset-specific audit notes were removed."""













from __future__ import annotations

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.oxe_schema import droid_state_to_arm10, euler7_action_to_arm10


class OxeDroidDataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-DROID"
    HEAD_CAMERA = "observation.images.exterior_1_left"
    LEFT_WRIST_CAMERA = "observation.images.wrist_left"
    RIGHT_WRIST_CAMERA = None


    NEEDED_COLS = (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.original",
    )
    ACTION_DIM_MASK = LEFT_ARM_DIM_MASK
    PROMPT_SOURCE = "episode_annotated"
    DEFAULT_NORMALIZE_MODE = "quantile"
    STATS_FILENAME = "eef_stats.json"
    STATS_DIM = 10
    STATS_STRICT_MINMAX = True

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        action = np.stack(win["action.original"].values).astype(np.float32)
        return single_arm_20d(euler7_action_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        cart = np.stack(win["observation.state.cartesian_position"].values[:1]).astype(np.float32)

        grip_raw = win["observation.state.gripper_position"].values[:1]
        grip = np.asarray([list(g) if isinstance(g, (list, np.ndarray)) else [g] for g in grip_raw], dtype=np.float32)
        return single_arm_20d(droid_state_to_arm10(cart, grip), self._normalization_stats, self._normalize_mode)


__all__ = ["OxeDroidDataset"]
