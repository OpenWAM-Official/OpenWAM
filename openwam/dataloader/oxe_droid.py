"""Public implementation. Dataset-specific audit notes were removed."""

























































from __future__ import annotations

import re
from typing import ClassVar, Optional, Tuple

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10






_PLACEHOLDER_RE = re.compile(
    r"^(?:no[\s_-]*action\.?|not[\s_-]*action|no[\s_-]*instruction|n/?a|null|none|nothing|test"
    r"|[.\-_/]+|pree|pm|op)$"
)


def _clean_text(value) -> str:
    """Public implementation. Dataset-specific audit notes were removed."""
    if value is None or not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or _PLACEHOLDER_RE.match(text.lower()):
        return ""
    return text


class OxeDroidDataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-DROID"
    HEAD_CAMERA = "observation.images.primary"
    LEFT_WRIST_CAMERA = "observation.images.wrist"
    RIGHT_WRIST_CAMERA = None












    PROMPT_FALLBACK_COLS: ClassVar[Tuple[str, ...]] = (
        "other_information.language_instruction_2",
        "other_information.language_instruction_3",
        "annotation.substask",
        "annotation.instruction_add",
    )
    NEEDED_COLS = (
        "state",
        "other_information.action_tcp_pose",
        "task_index",
    ) + PROMPT_FALLBACK_COLS
    ACTION_DIM_MASK = LEFT_ARM_DIM_MASK

    PROMPT_SOURCE = "task_index"
    DEFAULT_NORMALIZE_MODE = "quantile"
    STATS_FILENAME = "eef_stats.json"
    STATS_DIM = 10
    STATS_STRICT_MINMAX = True

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        """Public implementation. Dataset-specific audit notes were removed."""









        task_idx = int(win["task_index"].iloc[0])
        if task_idx not in self._task_idx_to_text:
            raise KeyError(
                f"{self.DATASET_NAME} prompt lookup failed: task_index={task_idx} not present "
                "in this bucket's tasks.parquet."
            )
        text = _clean_text(self._task_idx_to_text[task_idx])
        if text:
            return text
        for col in self.PROMPT_FALLBACK_COLS:
            text = _clean_text(win[col].iloc[0])
            if text:
                return text
        raise ValueError(
            f"{self.DATASET_NAME} task_index={task_idx} has a blank prompt and every fallback "
            f"column {list(self.PROMPT_FALLBACK_COLS)} is blank too (episode_index="
            f"{int(row['episode_index'])}). Generate meta/excluded_episodes.json with scripts/write_droid_prompt_exclusions.py before loading the dataset."
        )

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:

        action = np.stack(win["other_information.action_tcp_pose"].values).astype(np.float32)
        return single_arm_20d(euler7_action_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:


        state = np.stack(win["state"].values[:1]).astype(np.float32)
        return single_arm_20d(euler7_action_to_arm10(state), self._normalization_stats, self._normalize_mode)


__all__ = ["OxeDroidDataset"]
