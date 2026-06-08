"""OXE RT-1 reader.

Google Robotics Transformer 1 (~351h, 87,212 episodes, Google Robot).
State uses **quaternion (xyzw)** for orientation; action uses Euler XYZ
(same 7-D layout as BC-Z/Bridge/DROID).

Quat convention is validated at __init__ by sampling state[3:7] and
asserting unit norm — catches a wrongly-routed wxyz before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import LEFT_ARM_DIM_MASK, assert_unit_quaternion, single_arm_20d
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10, rt1_state_to_arm10


class OxeRt1Dataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-RT-1"
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

    def _post_init(self, info: dict) -> None:
        """Sanity-check the RT-1 quaternion convention.

        Reads up to 64 rows from the first parquet shard and asserts that
        ``state[:, 3:7]`` is unit-norm (xyzw). Catches data that is not
        normalized at all, which would yield garbage rot6d.
        """
        first_path = self._dataset_dir / self._data_path_template.format(chunk_index=0, file_index=0)
        if not first_path.exists():
            return
        table = pq.read_table(first_path, memory_map=True, columns=["observation.state"])
        sample = np.stack(table.column("observation.state").to_pylist()[:64]).astype(np.float32)
        if sample.shape[1] >= 7:
            assert_unit_quaternion(sample[:, 3:7], tol=0.05, sample_n=len(sample))

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        action = np.stack(win["action"].values).astype(np.float32)  # (T_actual, 7)
        return single_arm_20d(euler7_action_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        state = np.stack(win["observation.state"].values[:1]).astype(np.float32)  # (1, 8)
        return single_arm_20d(rt1_state_to_arm10(state), self._normalization_stats, self._normalize_mode)


__all__ = ["OxeRt1Dataset"]
