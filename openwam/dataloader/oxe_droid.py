"""Public implementation. Dataset-specific audit notes were removed."""





























































from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10






_PLACEHOLDER_RE = re.compile(
    r"^(?:no[\s_-]*action\.?|not[\s_-]*action|no[\s_-]*instruction|n/?a|null|none|nothing|test"
    r"|[.\-_/]+|pree|pm|op)$"
)



DROID_PROMPT_EXCLUSION_SCHEMA_VERSION = 2
DROID_PROMPT_EXCLUSION_KEY = "droid_prompt_exclusions"
DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY = "independently_owned_episode_indices"
DROID_PROMPT_INPUTS_DIGEST_KEY = "prompt_inputs_digest"
DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION = 1
DROID_PROMPT_FALLBACK_COLS = (
    "other_information.language_instruction_2",
    "other_information.language_instruction_3",
    "annotation.substask",
    "annotation.instruction_add",
)
DROID_PROMPT_SOURCE_COLUMNS = ("episode_index", "task_index", *DROID_PROMPT_FALLBACK_COLS)


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


def _digest_framed_bytes(hasher, value: bytes | bytearray | memoryview) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def _digest_framed_text(hasher, value: str) -> None:
    _digest_framed_bytes(hasher, value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_droid_prompt_source_table(path: Path) -> pa.Table:
    try:
        return pq.read_table(path, columns=list(DROID_PROMPT_SOURCE_COLUMNS), memory_map=True)
    except pa.ArrowInvalid as exc:
        if "Dot path" not in str(exc):
            raise
        return pq.read_table(path, memory_map=True).select(list(DROID_PROMPT_SOURCE_COLUMNS))


def digest_droid_prompt_shard(table: pa.Table) -> str:
    """Public implementation. Dataset-specific audit notes were removed."""







    missing = [column for column in DROID_PROMPT_SOURCE_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"prompt source table is missing columns {missing}")
    table = table.select(list(DROID_PROMPT_SOURCE_COLUMNS))
    hasher = hashlib.sha256(b"openwam:droid-prompt-shard:v1\0")
    hasher.update(table.num_rows.to_bytes(8, "little"))
    for column_name in DROID_PROMPT_SOURCE_COLUMNS:
        _digest_framed_text(hasher, column_name)
        column = table.column(column_name)
        if column_name in ("episode_index", "task_index"):
            array = column.combine_chunks()
            if array.null_count:
                raise ValueError(f"{column_name} contains null values")
            try:
                array = pc.cast(array, pa.int64(), safe=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise ValueError(f"{column_name} must contain integer values") from exc
            values = np.asarray(array.to_numpy(zero_copy_only=False), dtype="<i8")
            _digest_framed_text(hasher, "int64")
            _digest_framed_bytes(hasher, values.tobytes(order="C"))
            continue

        normalized_chunks = []
        for chunk in column.chunks:
            if pa.types.is_dictionary(chunk.type):
                chunk = pc.dictionary_decode(chunk)
            if pa.types.is_null(chunk.type):
                chunk = pa.nulls(len(chunk), type=pa.large_string())
            elif pa.types.is_string(chunk.type) or pa.types.is_large_string(chunk.type):
                chunk = pc.cast(chunk, pa.large_string())
            else:
                raise ValueError(f"{column_name} must contain string or null values, got {chunk.type}")
            normalized_chunks.append(chunk)
        array = pa.chunked_array(normalized_chunks, type=pa.large_string()).combine_chunks()
        valid = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=np.uint8)
        normalized = pc.fill_null(array, "")
        _, offsets_buffer, data_buffer = normalized.buffers()
        if offsets_buffer is None:
            raise ValueError(f"{column_name} has no string offsets")
        buffer_offsets = np.frombuffer(memoryview(offsets_buffer), dtype=np.int64)
        start = normalized.offset
        offsets = np.asarray(buffer_offsets[start : start + len(normalized) + 1], dtype="<i8").copy()
        data_start = int(offsets[0])
        data_end = int(offsets[-1])
        offsets -= data_start
        if data_buffer is None:
            data = b""
        else:
            data = memoryview(data_buffer)[data_start:data_end]
        _digest_framed_text(hasher, "large_string")
        _digest_framed_bytes(hasher, valid.tobytes(order="C"))
        _digest_framed_bytes(hasher, offsets.tobytes(order="C"))
        _digest_framed_bytes(hasher, data)
    return hasher.hexdigest()


def compute_droid_prompt_inputs_digest(
    dataset_dir: str | Path,
    *,
    shard_digests: Mapping[str, str] | None = None,
    tasks_sha256: str | None = None,
) -> dict:
    """Public implementation. Dataset-specific audit notes were removed."""
    root = Path(dataset_dir)
    tasks_digest = tasks_sha256 or _sha256_file(root / "meta" / "tasks.parquet")
    shard_paths = sorted((root / "data").rglob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no data parquet under {root}/data")
    relative_paths = [path.relative_to(root).as_posix() for path in shard_paths]
    if shard_digests is None:
        resolved_shard_digests = {
            relative_path: digest_droid_prompt_shard(_read_droid_prompt_source_table(path))
            for relative_path, path in zip(relative_paths, shard_paths)
        }
    else:
        resolved_shard_digests = dict(shard_digests)
        if set(resolved_shard_digests) != set(relative_paths):
            raise ValueError("prompt shard digest paths do not match the current data shard set")

    hasher = hashlib.sha256(b"openwam:droid-prompt-inputs:v1\0")
    _digest_framed_text(hasher, "meta/tasks.parquet")
    try:
        tasks_digest_bytes = bytes.fromhex(tasks_digest)
    except ValueError as exc:
        raise ValueError("tasks_sha256 must be a 64-character hexadecimal SHA-256 digest") from exc
    if len(tasks_digest_bytes) != hashlib.sha256().digest_size:
        raise ValueError("tasks_sha256 must be a 64-character hexadecimal SHA-256 digest")
    _digest_framed_bytes(hasher, tasks_digest_bytes)
    for relative_path in relative_paths:
        _digest_framed_text(hasher, relative_path)
        digest = resolved_shard_digests[relative_path]
        try:
            digest_bytes = bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError(f"invalid SHA-256 digest for {relative_path}") from exc
        if len(digest_bytes) != hashlib.sha256().digest_size:
            raise ValueError(f"invalid SHA-256 digest for {relative_path}")
        _digest_framed_bytes(hasher, digest_bytes)
    return {
        "algorithm": "sha256",
        "format_version": DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION,
        "value": hasher.hexdigest(),
    }


def _parse_prompt_inputs_digest(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{DROID_PROMPT_INPUTS_DIGEST_KEY} must be an object")
    if value.get("algorithm") != "sha256":
        raise ValueError(f"{DROID_PROMPT_INPUTS_DIGEST_KEY}.algorithm must be 'sha256'")
    format_version = value.get("format_version")
    if type(format_version) is not int or format_version != DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION:
        raise ValueError(
            f"{DROID_PROMPT_INPUTS_DIGEST_KEY}.format_version must be {DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION}"
        )
    digest = value.get("value")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{DROID_PROMPT_INPUTS_DIGEST_KEY}.value must be a lowercase SHA-256 digest")
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
        recorded_inputs_digest = _parse_prompt_inputs_digest(latest_scan[DROID_PROMPT_INPUTS_DIGEST_KEY])
        current_inputs_digest = compute_droid_prompt_inputs_digest(dataset_dir)
        if current_inputs_digest != recorded_inputs_digest:
            raise ValueError(f"{DROID_PROMPT_INPUTS_DIGEST_KEY} does not match tasks.parquet and prompt source columns")
    except (KeyError, OSError, TypeError, ValueError) as exc:
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
    "DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION",
    "DROID_PROMPT_INPUTS_DIGEST_KEY",
    "DROID_PROMPT_SOURCE_COLUMNS",
    "OxeDroidDataset",
    "compute_droid_prompt_inputs_digest",
    "digest_droid_prompt_shard",
    "load_droid_prompt_exclusions",
]
