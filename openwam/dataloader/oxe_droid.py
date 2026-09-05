"""LeRobot v3 reader for the DROID portion of OXE.

The reader uses the rigid arm-side wrist/gripper-mount frame: achieved state is
read from ``observation_gripper_pose6d`` and commanded action from
``action_wrist_pose``.  The physically distinct task-TCP and clipped-delta
streams are intentionally not mixed into the same action slots.  Euler XYZ
poses are converted to a 10-D single-arm EEF vector and placed in the left arm
of the shared schema.  Raw gripper closedness is inverted to the project-wide
``0=closed, 1=open`` convention.

Prompt text resolves from ``tasks.parquet`` and a deterministic per-row fallback
chain.  Episodes without any resolvable prompt must be represented by the
canonical exclusion artifact.  The reader validates prompt inputs, manifest
ranges, exclusions, and normalization provenance so stale artifacts fail
before training.
"""

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

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import EEF_POSE_FRAME_CONTRACT, LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.lerobotv3 import (
    LeRobotV3DataPopulation,
    LeRobotV3DataShard,
    apply_info_splits,
    digest_lerobot_v3_data_population,
    read_lerobot_v3_population_shard,
    resolve_lerobot_v3_data_population,
)
from openwam.dataloader.utils.normalization import materialize_eef_stats
from openwam.dataloader.utils.oxe_schema import droid_euler7_to_arm10, droid_pose6_closedness_to_arm10

# Present-but-empty instructions fall through to the next prompt source.
_PLACEHOLDER_RE = re.compile(
    r"^(?:no[\s_-]*action\.?|not[\s_-]*action|no[\s_-]*instruction|n/?a|null|none|nothing|test"
    r"|[.\-_/]+|pree|pm|op)$"
)


# Versioned contracts keep prompt exclusions and EEF stats tied to source data.
DROID_PROMPT_EXCLUSION_SCHEMA_VERSION = 3
DROID_PROMPT_EXCLUSION_KEY = "droid_prompt_exclusions"
DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY = "independently_owned_episode_indices"
DROID_PROMPT_INPUTS_DIGEST_KEY = "prompt_inputs_digest"
DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION = 2
DROID_DATA_POPULATION_DIGEST_KEY = "data_population_digest"
DROID_STATS_POPULATION_KEY = "stats_population"
DROID_STATS_POPULATION_SCHEMA_VERSION = 1
DROID_STATS_POPULATION_POLICY = "info_split_then_prompt_exclusion"
DROID_EEF_STATS_CONTRACT_KEY = "droid_eef_stats_contract"
DROID_EEF_STATS_CONTRACT_VERSION = 2
DROID_EEF_STATS_CONTRACT = {
    "schema_version": DROID_EEF_STATS_CONTRACT_VERSION,
    "pose_frame_semantics": EEF_POSE_FRAME_CONTRACT,
    "action_pose_source": "other_information.action_wrist_pose",
    "state_pose_source": "other_information.observation_gripper_pose6d",
    "state_gripper_source": "state[6]",
    "raw_gripper_semantics": "closedness:0=open,1=closed",
    "output_gripper_semantics": "openness:0=closed,1=open",
    "gripper_transform": "1-raw",
}
DROID_PROMPT_FALLBACK_COLS = (
    "other_information.language_instruction_2",
    "other_information.language_instruction_3",
    "annotation.substask",
    "annotation.instruction_add",
)
DROID_PROMPT_SOURCE_COLUMNS = ("episode_index", "task_index", *DROID_PROMPT_FALLBACK_COLS)


def _validate_droid_eef_stats_contract(value) -> None:
    """Reject stats from a different pose frame or gripper conversion."""
    if not isinstance(value, dict) or set(value) != set(DROID_EEF_STATS_CONTRACT):
        raise ValueError(f"expected exactly {DROID_EEF_STATS_CONTRACT!r}")
    if type(value["schema_version"]) is not int or value["schema_version"] != DROID_EEF_STATS_CONTRACT_VERSION:
        raise ValueError(f"schema_version must be integer {DROID_EEF_STATS_CONTRACT_VERSION}")
    for key, expected in DROID_EEF_STATS_CONTRACT.items():
        if key == "schema_version":
            continue
        if not isinstance(value[key], str) or value[key] != expected:
            raise ValueError(f"{key} must be {expected!r}")


def _clean_text(value) -> str:
    """Strip a raw prompt candidate, or return ``""`` when it carries no instruction."""
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


def resolve_droid_stats_population(
    population: LeRobotV3DataPopulation,
    info: dict,
    excluded_episode_indices,
    *,
    split: str = "train",
) -> tuple[pd.DataFrame, dict]:
    """Resolve and certify the exact DROID population used for normalization.

    The raw data-population digest deliberately describes the pre-split corpus.
    This second certificate binds the stats to the selected info.json split and
    prompt-exclusion set, including each episode's physical manifest range.
    """

    selected = apply_info_splits(
        population.episodes,
        split,
        info.get("splits", {}) or {},
        source_name="OXE-DROID normalization stats",
    )
    excluded = set(int(value) for value in excluded_episode_indices)
    if excluded:
        selected = selected[~selected["episode_index"].isin(excluded)].reset_index(drop=True)

    required = (
        "episode_index",
        "dataset_from_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "_data_row_offset",
    )
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise ValueError(f"DROID stats population is missing manifest columns {missing}")

    hasher = hashlib.sha256(b"openwam:droid-stats-population:v1\0")
    ordered = selected.sort_values("episode_index", kind="stable")
    for row in ordered.loc[:, required].itertuples(index=False, name=None):
        for value in row:
            integer = int(value)
            if integer < 0:
                raise ValueError("DROID stats population manifest values must be non-negative")
            hasher.update(integer.to_bytes(8, "little", signed=False))
    provenance = {
        "schema_version": DROID_STATS_POPULATION_SCHEMA_VERSION,
        "split": split,
        "policy": DROID_STATS_POPULATION_POLICY,
        "effective_population_digest": hasher.hexdigest(),
        "num_episodes": int(len(ordered)),
        "num_rows": int(ordered["length"].sum()),
    }
    return selected, provenance


def read_droid_prompt_population_shard(
    dataset_dir: str | Path,
    shard: LeRobotV3DataShard,
) -> pa.Table:
    """Read and validate the prompt rows addressed by one manifest shard."""
    try:
        return read_lerobot_v3_population_shard(
            Path(dataset_dir),
            shard,
            DROID_PROMPT_SOURCE_COLUMNS,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"prompt source population is invalid ({exc})") from exc


def digest_droid_prompt_shard(table: pa.Table) -> str:
    """Hash the logical prompt-bearing values of one data shard.

    Numeric IDs are normalized to little-endian int64. Fallback text is
    normalized to one combined large-string array with an explicit validity
    vector and nulls filled to empty before hashing offsets/data. This keeps the
    digest independent of parquet row groups, compression, and dictionary
    encoding while preserving row order and null-vs-empty distinctions.
    """

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
    population: LeRobotV3DataPopulation | None = None,
) -> dict:
    """Hash tasks and the exact prompt population addressed by the reader."""
    root = Path(dataset_dir)
    resolved_population = population or resolve_lerobot_v3_data_population(root)
    tasks_digest = tasks_sha256 if tasks_sha256 is not None else _sha256_file(root / "meta" / "tasks.parquet")
    relative_paths = [shard.relative_path for shard in resolved_population.shards]
    if shard_digests is None:
        resolved_shard_digests = {
            shard.relative_path: digest_droid_prompt_shard(read_droid_prompt_population_shard(root, shard))
            for shard in resolved_population.shards
        }
    else:
        resolved_shard_digests = dict(shard_digests)
        if set(resolved_shard_digests) != set(relative_paths):
            raise ValueError("prompt shard digest paths do not match the manifest-addressed shard set")

    hasher = hashlib.sha256(b"openwam:droid-prompt-inputs:v2\0")
    _digest_framed_text(hasher, "meta/tasks.parquet")
    try:
        tasks_digest_bytes = bytes.fromhex(tasks_digest)
    except ValueError as exc:
        raise ValueError("tasks_sha256 must be a 64-character hexadecimal SHA-256 digest") from exc
    if len(tasks_digest_bytes) != hashlib.sha256().digest_size:
        raise ValueError("tasks_sha256 must be a 64-character hexadecimal SHA-256 digest")
    _digest_framed_bytes(hasher, tasks_digest_bytes)
    population_digest = digest_lerobot_v3_data_population(resolved_population)
    _digest_framed_text(hasher, DROID_DATA_POPULATION_DIGEST_KEY)
    _digest_framed_bytes(hasher, bytes.fromhex(population_digest))
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


def load_droid_prompt_exclusions(
    dataset_dir: str | Path,
    *,
    population: LeRobotV3DataPopulation | None = None,
) -> tuple[dict, set[int]]:
    """Load and validate the prompt-generator provenance required by this reader."""
    root = Path(dataset_dir)
    path = root / "meta" / "excluded_episodes.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Provide the prompt-exclusion manifest before constructing OXE-DROID."
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
        resolved_population = population or resolve_lerobot_v3_data_population(root)
        if rows_scanned != resolved_population.total_rows:
            raise ValueError(
                f"latest_scan.stats.rows_scanned={rows_scanned} but the data manifest addresses "
                f"{resolved_population.total_rows} rows"
            )
        recorded_inputs_digest = _parse_prompt_inputs_digest(latest_scan[DROID_PROMPT_INPUTS_DIGEST_KEY])
        current_inputs_digest = compute_droid_prompt_inputs_digest(
            root,
            population=resolved_population,
        )
        if current_inputs_digest != recorded_inputs_digest:
            raise ValueError(f"{DROID_PROMPT_INPUTS_DIGEST_KEY} does not match tasks.parquet and prompt source columns")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} is stale or malformed ({exc}). Regenerate the prompt-exclusion manifest.") from exc
    return payload, canonical


class OxeDroidDataset(LeRobotV3Reader):
    DATASET_NAME = "OXE-DROID"
    HEAD_CAMERA = "observation.images.primary"
    LEFT_WRIST_CAMERA = "observation.images.wrist"
    RIGHT_WRIST_CAMERA = None

    PROMPT_FALLBACK_COLS: ClassVar[Tuple[str, ...]] = DROID_PROMPT_FALLBACK_COLS
    NEEDED_COLS = (
        "state",
        "other_information.observation_gripper_pose6d",
        "other_information.action_wrist_pose",
        "task_index",
    ) + PROMPT_FALLBACK_COLS
    ACTION_DIM_MASK = LEFT_ARM_DIM_MASK

    PROMPT_SOURCE = "task_index"
    DEFAULT_NORMALIZE_MODE = "quantile"
    STATS_FILENAME = "eef_stats.json"
    STATS_DIM = 10
    STATS_STRICT_MINMAX = True

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        population = resolve_lerobot_v3_data_population(self._dataset_dir, info=info)
        _, self._droid_excluded_episode_indices = load_droid_prompt_exclusions(
            self._dataset_dir,
            population=population,
        )
        self._droid_data_population_digest = digest_lerobot_v3_data_population(population)
        _, self._droid_stats_population = resolve_droid_stats_population(
            population,
            info,
            self._droid_excluded_episode_indices,
            split="train",
        )
        eps = population.episodes.copy()
        self._add_episode_offsets(eps)
        info_splits = info.get("splits", {}) or {}
        return apply_info_splits(
            eps,
            self._split,
            info_splits,
            source_name=f"{self.DATASET_NAME}({self._dataset_id})",
        )

    def _load_excluded_episode_indices(self) -> set[int]:
        """Reuse the canonical set validated with the prompt certificate."""
        return set(self._droid_excluded_episode_indices)

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
            stats_population_digest = raw[DROID_DATA_POPULATION_DIGEST_KEY]
            stats_population = raw[DROID_STATS_POPULATION_KEY]
            stats_contract = raw[DROID_EEF_STATS_CONTRACT_KEY]
            if (
                not isinstance(stats_population_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", stats_population_digest) is None
            ):
                raise ValueError(f"{DROID_DATA_POPULATION_DIGEST_KEY} must be a lowercase SHA-256 digest")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{stats_path} has missing or invalid DROID normalization provenance ({exc}). "
                "Re-run oxe_stats_computation for DROID."
            ) from exc
        if stats_excluded != self._droid_excluded_episode_indices:
            raise ValueError(
                f"{stats_path} excluded_episode_indices do not match meta/excluded_episodes.json. "
                "Re-run oxe_stats_computation for DROID after updating exclusions."
            )
        if stats_population_digest != self._droid_data_population_digest:
            raise ValueError(
                f"{stats_path} {DROID_DATA_POPULATION_DIGEST_KEY} does not match the current data manifest. "
                "Re-run oxe_stats_computation for DROID."
            )
        if stats_population != self._droid_stats_population:
            raise ValueError(
                f"{stats_path} {DROID_STATS_POPULATION_KEY} does not match the current train "
                "split/exclusion population. Re-run oxe_stats_computation for DROID."
            )
        try:
            _validate_droid_eef_stats_contract(stats_contract)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{stats_path} {DROID_EEF_STATS_CONTRACT_KEY} is stale or invalid; expected "
                "the rigid terminal-arm pose frame and 0=closed, 1=open gripper transform. "
                f"Re-run oxe_stats_computation for DROID ({exc})."
            ) from exc
        stats = materialize_eef_stats(
            raw,
            self._normalize_mode,
            dim=self.STATS_DIM,
            strict_minmax=self.STATS_STRICT_MINMAX,
            source_hint=str(stats_path),
            force_rot6d_identity=True,
        )
        return stats

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        """tasks.parquet text, else the first non-placeholder fallback column.

        Raises when every source is blank. This is a **backstop, not a runtime
        degradation path**: those episodes must already be gone from the index
        via ``meta/excluded_episodes.json`` (see the module docstring). Relying
        on ``_safe_get`` to retry past them does NOT work — its ``idx + 1`` walk
        stays inside the same episode, and the condition is all-or-nothing per
        episode, so any episode longer than ``_GETITEM_MAX_RETRIES`` frames
        exhausts the retries and kills the DataLoader worker.
        """

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
            f"{int(row['episode_index'])}). Generate meta/excluded_episodes.json "
            "before loading the dataset."
        )

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        action = np.stack(win["other_information.action_wrist_pose"].values).astype(np.float32)
        return single_arm_20d(droid_euler7_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        pose = np.stack(win["other_information.observation_gripper_pose6d"].values[:1]).astype(np.float32)
        state = np.stack(win["state"].values[:1]).astype(np.float32)
        arm10 = droid_pose6_closedness_to_arm10(pose, state[:, 6:7])
        return single_arm_20d(arm10, self._normalization_stats, self._normalize_mode)


__all__ = [
    "DROID_DATA_POPULATION_DIGEST_KEY",
    "DROID_EEF_STATS_CONTRACT",
    "DROID_EEF_STATS_CONTRACT_KEY",
    "DROID_EEF_STATS_CONTRACT_VERSION",
    "DROID_PROMPT_EXCLUSION_KEY",
    "DROID_PROMPT_EXCLUSION_SCHEMA_VERSION",
    "DROID_PROMPT_FALLBACK_COLS",
    "DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY",
    "DROID_PROMPT_INPUTS_DIGEST_FORMAT_VERSION",
    "DROID_PROMPT_INPUTS_DIGEST_KEY",
    "DROID_PROMPT_SOURCE_COLUMNS",
    "DROID_STATS_POPULATION_KEY",
    "DROID_STATS_POPULATION_POLICY",
    "DROID_STATS_POPULATION_SCHEMA_VERSION",
    "OxeDroidDataset",
    "compute_droid_prompt_inputs_digest",
    "digest_droid_prompt_shard",
    "load_droid_prompt_exclusions",
    "read_droid_prompt_population_shard",
    "resolve_droid_stats_population",
]
