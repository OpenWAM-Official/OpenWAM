"""Public implementation. Dataset-specific audit notes were removed."""




























































from __future__ import annotations

import json
import re
from pathlib import Path
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



DROID_PROMPT_EXCLUSION_SCHEMA_VERSION = 1
DROID_PROMPT_EXCLUSION_KEY = "droid_prompt_exclusions"
DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY = "independently_owned_episode_indices"
DROID_PROMPT_FALLBACK_COLS = (
    "other_information.language_instruction_2",
    "other_information.language_instruction_3",
    "annotation.substask",
    "annotation.instruction_add",
)


def _clean_text(value) -> str:
    """Public implementation. Dataset-specific audit notes were removed."""
    if value is None or not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or _PLACEHOLDER_RE.match(text.lower()):
        return ""
    return text


def _parse_episode_indices(value, field: str) -> set[int]:
    if not isinstance(value, list) or any(type(i) is not int or i < 0 for i in value):
        raise ValueError(f"{field} must be a list of non-negative integers")
    return set(value)


def _parse_nonnegative_int(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_droid_prompt_exclusions(dataset_dir: str | Path) -> tuple[dict, set[int]]:
    """Public implementation. Dataset-specific audit notes were removed."""
    path = Path(dataset_dir) / "meta" / "excluded_episodes.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/write_droid_prompt_exclusions.py before constructing OXE-DROID."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical = _parse_episode_indices(payload["episode_indices"], "episode_indices")
        prompt = payload[DROID_PROMPT_EXCLUSION_KEY]
        if not isinstance(prompt, dict):
            raise ValueError(f"{DROID_PROMPT_EXCLUSION_KEY} must be an object")
        if type(prompt["schema_version"]) is not int or (
            prompt["schema_version"] != DROID_PROMPT_EXCLUSION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"{DROID_PROMPT_EXCLUSION_KEY}.schema_version={prompt['schema_version']!r}, "
                f"expected {DROID_PROMPT_EXCLUSION_SCHEMA_VERSION}"
            )
        if prompt["fallback_chain"] != list(DROID_PROMPT_FALLBACK_COLS):
            raise ValueError(
                f"{DROID_PROMPT_EXCLUSION_KEY}.fallback_chain={prompt['fallback_chain']!r}, "
                f"expected {list(DROID_PROMPT_FALLBACK_COLS)!r}"
            )
        prompt_owned = _parse_episode_indices(
            prompt["episode_indices"],
            f"{DROID_PROMPT_EXCLUSION_KEY}.episode_indices",
        )
        independently_owned = _parse_episode_indices(
            prompt[DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY],
            f"{DROID_PROMPT_EXCLUSION_KEY}.{DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY}",
        )
        if canonical != prompt_owned | independently_owned:
            raise ValueError("episode_indices must equal the union of prompt-owned and independently-owned exclusions")

        latest_scan = prompt["latest_scan"]
        if not isinstance(latest_scan, dict):
            raise ValueError(f"{DROID_PROMPT_EXCLUSION_KEY}.latest_scan must be an object")
        latest_scan_owned = _parse_episode_indices(
            latest_scan["episode_indices"],
            f"{DROID_PROMPT_EXCLUSION_KEY}.latest_scan.episode_indices",
        )
        if latest_scan_owned != prompt_owned:
            raise ValueError("latest_scan.episode_indices do not match the prompt-owned exclusions")
        scan_stats = latest_scan["stats"]
        if not isinstance(scan_stats, dict):
            raise ValueError(f"{DROID_PROMPT_EXCLUSION_KEY}.latest_scan.stats must be an object")
        if scan_stats["fallback_chain"] != list(DROID_PROMPT_FALLBACK_COLS):
            raise ValueError("latest_scan.stats.fallback_chain does not match the reader fallback chain")
        rows_scanned = _parse_nonnegative_int(scan_stats["rows_scanned"], "latest_scan.stats.rows_scanned")
        unresolved_rows = _parse_nonnegative_int(
            scan_stats["unresolved_rows"],
            "latest_scan.stats.unresolved_rows",
        )
        if rows_scanned == 0 or unresolved_rows > rows_scanned:
            raise ValueError("latest_scan.stats row counts are inconsistent")
        if _parse_nonnegative_int(
            scan_stats["episodes_all_unresolved"],
            "latest_scan.stats.episodes_all_unresolved",
        ) != len(prompt_owned):
            raise ValueError("latest_scan.stats.episodes_all_unresolved does not match prompt-owned exclusions")
        for field in ("episodes_partially_unresolved", "task_index_missing_from_tasks_parquet"):
            if _parse_nonnegative_int(scan_stats[field], f"latest_scan.stats.{field}") != 0:
                raise ValueError(f"latest_scan.stats.{field} must be zero for a completed scan")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} is stale or malformed ({exc}). Re-run scripts/write_droid_prompt_exclusions.py."
        ) from exc
    return payload, canonical


class OxeDroidDataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-DROID"
    HEAD_CAMERA = "observation.images.primary"
    LEFT_WRIST_CAMERA = "observation.images.wrist"
    RIGHT_WRIST_CAMERA = None












    PROMPT_FALLBACK_COLS: ClassVar[Tuple[str, ...]] = DROID_PROMPT_FALLBACK_COLS
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

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        payload, self._droid_excluded_episode_indices = load_droid_prompt_exclusions(self._dataset_dir)
        expected_rows = info.get("total_frames")
        scanned_rows = payload[DROID_PROMPT_EXCLUSION_KEY]["latest_scan"]["stats"]["rows_scanned"]
        if expected_rows is not None and (
            type(expected_rows) is not int or expected_rows < 0 or scanned_rows != expected_rows
        ):
            raise ValueError(
                f"{self._dataset_dir}/meta/excluded_episodes.json is stale: its prompt scan covered "
                f"{scanned_rows} rows but meta/info.json declares {expected_rows}. "
                "Re-run scripts/write_droid_prompt_exclusions.py."
            )
        return super()._build_episode_index(info)

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return super()._load_stats(info)
        stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
        if not stats_path.exists():
            return super()._load_stats(info)
        try:
            raw = json.loads(stats_path.read_text(encoding="utf-8"))
            stats_excluded = _parse_episode_indices(
                raw["excluded_episode_indices"],
                "excluded_episode_indices",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{stats_path} has missing or invalid excluded_episode_indices provenance ({exc}). "
                "Re-run oxe_stats_computation for DROID."
            ) from exc
        if stats_excluded != self._droid_excluded_episode_indices:
            raise ValueError(
                f"{stats_path} excluded_episode_indices do not match meta/excluded_episodes.json. "
                "Re-run oxe_stats_computation for DROID after updating exclusions."
            )
        return super()._load_stats(info)

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


__all__ = [
    "DROID_PROMPT_EXCLUSION_KEY",
    "DROID_PROMPT_EXCLUSION_SCHEMA_VERSION",
    "DROID_PROMPT_FALLBACK_COLS",
    "DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY",
    "OxeDroidDataset",
    "load_droid_prompt_exclusions",
]
