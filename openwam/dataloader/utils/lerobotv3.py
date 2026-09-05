"""Shared helpers for LeRobot v3 single-bucket readers.

Both ``RoboCOINDataset`` and ``EgoDexDataset`` follow the same setup
recipe at init: read ``meta/info.json``, concatenate the per-shard
``meta/episodes/*.parquet``, sort by ``episode_index``, compute file-local
row + frame offsets via lexsort + groupby cumsum, then maybe trim by an
info-declared train/val split. This module hosts the identical pieces so
each reader's ``__init__`` only has to wire them together + do its
data-specific work (action conversion, multiview canvas, normalization).

These helpers are intentionally stateless module-level functions; they
take whatever they need as arguments and return plain values. The readers
remain ``BaseDataset`` subclasses (no shared base class wrapped
around them) so the inheritance graph stays flat and easy to follow.

Functions
---------
- parse_info_json(dataset_dir)
    Read ``<dataset_dir>/meta/info.json`` and return the dict along with
    a couple of normalised defaults (``data_path`` and ``video_path``
    templates filled in when missing).

- load_episodes_parquet(dataset_dir)
    Concatenate every ``meta/episodes/*.parquet`` shard into a single
    sorted pandas DataFrame, dropping any ``stats/*`` columns.

- compute_file_local_offsets(eps, chunk_col, file_col)
    Return an int64 numpy array of per-episode file-local cumulative
    offsets, suitable as a column appended to ``eps`` (e.g.
    ``_data_row_offset`` or ``_video_frame_offset``).

- resolve_lerobot_v3_data_population(dataset_dir, ...)
    Resolve the complete pre-split manifest into the exact data paths and
    file-local episode ranges addressed by the reader.

- digest_lerobot_v3_data_population(population)
    Hash the effective data-path template and resolved episode-range mapping.

- read_lerobot_v3_population_shard(dataset_dir, shard, columns)
    Read only a manifest-addressed shard population and verify every physical
    ``episode_index`` row against its declared range.

- apply_info_splits(eps_df, split, info_splits, *, source_name=...)
    Honor ``info.json["splits"][split]`` when present; otherwise default
    train to the full eps_df and val to empty. Raises with a clear
    message on malformed split specs.

- water_fill_hours(bucket_hours, total_budget)
    Allocate ``total_budget`` hours across N buckets via water-filling.
    Each bucket gets at most its own ``bucket_hours[i]``; surplus from
    under-budgetable buckets is redistributed among the rest.

- effective_episode_frames(eps_df, length_col="length")
    Validate and return each row's post-trim sampleable span.

- subsample_episodes_by_hours(eps_df, target_hours, fps, seed, ...)
    Random subset of ``eps_df`` rows totalling >= ``target_hours`` of effective
    post-trim footage. ``_valid_start/_valid_end`` take precedence when present;
    otherwise the (possibly trim-adjusted) ``length`` column is used. Seeded
    permutation + greedy prefix; preserves original row order so downstream
    offset columns stay valid.

Exceptions
----------
- DataContractError
    Raised by a reader that has PROVEN its data is broken (as opposed to an
    IO / environment fault). ``build_multibucket`` tolerates every other
    per-bucket exception by dropping the bucket with a warning; this one it
    re-raises, so broken data cannot silently shrink the training set.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


class DataContractError(RuntimeError):
    """A per-bucket failure that proves the DATA is broken, not the environment.

    :func:`build_multibucket` deliberately tolerates a bucket that fails to
    construct — a truncated shard, an NFS blip, a stale stats file — by logging
    a warning and dropping it, so one bad corner of a large multi-bucket root cannot
    abort a training launch. That tolerance is wrong for a failure that means
    the dataset itself violates the reader's contract (a prompt table that has
    diverged from the episodes table, a mis-typed text column): dropping such a
    bucket silently removes a slice of the training set behind a single WARNING,
    which is exactly the silent data loss the warning was meant to surface.

    A reader raises this (instead of a plain exception) when it has PROVEN a
    data-contract violation. ``_build_one`` re-raises it so the launch fails
    loudly with the reader's own diagnostic, in root mode as well as in
    single-bucket mode. Environmental failures must keep using ordinary
    exceptions so they stay tolerated.
    """


@dataclass(frozen=True)
class ExcludedEpisodesSnapshot:
    """Immutable effective population from one excluded-episodes file read."""

    path: str
    episode_indices: tuple[int, ...]


def load_excluded_episodes_snapshot(dataset_dir: Path) -> ExcludedEpisodesSnapshot:
    """Load the canonical per-bucket episode blacklist as sorted unique IDs."""
    path = Path(dataset_dir) / "meta" / "excluded_episodes.json"
    if not path.exists():
        return ExcludedEpisodesSnapshot(path=str(path), episode_indices=())
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as e:
        raise DataContractError(f"Malformed LeRobot episode exclusions {path}: {e}") from e
    if not isinstance(payload, dict) or "episode_indices" not in payload:
        raise DataContractError(
            f"Malformed LeRobot episode exclusions {path}: expected an object with 'episode_indices'"
        )
    values = payload["episode_indices"]
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise DataContractError(
            f"Malformed LeRobot episode exclusions {path}: 'episode_indices' must be a list of non-negative integers"
        )
    return ExcludedEpisodesSnapshot(path=str(path), episode_indices=tuple(sorted(set(values))))


def assert_excluded_episodes_snapshot_current(
    snapshot: ExcludedEpisodesSnapshot,
    *,
    context: str,
) -> None:
    """Fail if a bucket's effective exclusion population changed after loading."""
    actual = load_excluded_episodes_snapshot(Path(snapshot.path).parent.parent)
    if actual.episode_indices != snapshot.episode_indices:
        raise DataContractError(
            f"RoboCOIN episode exclusions {snapshot.path} changed while {context}; "
            f"expected {list(snapshot.episode_indices)}, got {list(actual.episode_indices)}"
        )


@dataclass(frozen=True)
class LeRobotV3DataRange:
    """One episode range addressed inside a LeRobot v3 data shard."""

    episode_index: int
    dataset_from_index: int
    row_offset: int
    length: int


@dataclass(frozen=True)
class LeRobotV3DataShard:
    """One logical data shard and the exact ranges addressed by its manifest."""

    chunk_index: int
    file_index: int
    relative_path: str
    ranges: tuple[LeRobotV3DataRange, ...]


@dataclass(frozen=True)
class LeRobotV3DataPopulation:
    """The complete pre-split/pre-exclusion data population a reader addresses."""

    data_path_template: str
    total_rows: int
    shards: tuple[LeRobotV3DataShard, ...]
    episodes: pd.DataFrame = field(compare=False, repr=False)


def parse_info_json(dataset_dir: Path) -> dict:
    """Read ``<dataset_dir>/meta/info.json`` and fill in default templates.

    Returns the raw dict augmented with default ``data_path`` /
    ``video_path`` entries when the file omits them.
    """
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.setdefault("data_path", _DEFAULT_DATA_PATH)
    info.setdefault("video_path", _DEFAULT_VIDEO_PATH)
    return info


def load_episodes_parquet(dataset_dir: Path) -> pd.DataFrame:
    """Concatenate per-shard episodes parquet into one sorted DataFrame.

    Drops any ``stats/*`` column at read time (they aren't consumed by the
    readers and skipping them halves the read cost on shards that ship
    pre-computed stats columns).

    Raises FileNotFoundError when no shard exists.
    """
    eps_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not eps_paths:
        raise FileNotFoundError(f"No episode parquet under {dataset_dir}/meta/episodes")
    schema_cols = pq.read_schema(eps_paths[0]).names
    keep_cols = [c for c in schema_cols if not c.startswith("stats/")]
    eps_tables = [pq.read_table(p, columns=keep_cols) for p in eps_paths]
    eps = pa.concat_tables(eps_tables).to_pandas()
    return eps.sort_values("episode_index").reset_index(drop=True)


def compute_file_local_offsets(eps: pd.DataFrame, chunk_col: str, file_col: str) -> np.ndarray:
    """Return per-episode file-local cumulative offsets.

    For a contiguous-on-disk format like LeRobot v3, episode N within a
    given (chunk, file) shard starts at offset = Σ length of episodes
    earlier than N in the same shard. The lexsort key
    ``(dataset_from_index, file_col, chunk_col)`` puts episodes in shard
    order; the ``groupby([chunk_col, file_col]).cumsum() - length`` trick
    converts inclusive cumulative length to exclusive offset; the inverse
    permutation puts results back into eps's original row order.
    """
    order = np.lexsort(
        (
            eps["dataset_from_index"].to_numpy(),
            eps[file_col].to_numpy(),
            eps[chunk_col].to_numpy(),
        )
    )
    sorted_eps = eps.iloc[order]
    cum_inclusive = sorted_eps.groupby([chunk_col, file_col], sort=False)["length"].cumsum()
    cum_exclusive = (cum_inclusive - sorted_eps["length"]).to_numpy().astype(np.int64)
    out = np.empty(len(eps), dtype=np.int64)
    out[order] = cum_exclusive
    return out


def _manifest_int_column(eps: pd.DataFrame, column: str, *, minimum: int) -> np.ndarray:
    if column not in eps.columns:
        raise ValueError(f"episodes manifest is missing {column!r}")
    values = []
    for value in eps[column].tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"episodes manifest {column!r} must contain integers")
        integer = int(value)
        if integer < minimum:
            raise ValueError(f"episodes manifest {column!r} values must be >= {minimum}")
        values.append(integer)
    try:
        return np.asarray(values, dtype=np.int64)
    except OverflowError as exc:
        raise ValueError(f"episodes manifest {column!r} values must fit int64") from exc


def resolve_lerobot_v3_data_population(
    dataset_dir: Path,
    *,
    info: dict | None = None,
    episodes: pd.DataFrame | None = None,
) -> LeRobotV3DataPopulation:
    """Resolve the exact pre-split data ranges addressed by a v3 reader."""
    root = Path(dataset_dir)
    resolved_info = parse_info_json(root) if info is None else info
    data_path_template = resolved_info.get("data_path")
    if not isinstance(data_path_template, str) or not data_path_template:
        raise ValueError("info.json data_path must be a non-empty string")

    eps = load_episodes_parquet(root) if episodes is None else episodes.copy()
    if eps.empty:
        raise ValueError("episodes manifest must contain at least one episode")
    int_columns = {
        "episode_index": _manifest_int_column(eps, "episode_index", minimum=0),
        "length": _manifest_int_column(eps, "length", minimum=1),
        "dataset_from_index": _manifest_int_column(eps, "dataset_from_index", minimum=0),
        "data/chunk_index": _manifest_int_column(eps, "data/chunk_index", minimum=0),
        "data/file_index": _manifest_int_column(eps, "data/file_index", minimum=0),
    }
    for column, values in int_columns.items():
        eps[column] = values
    if eps["episode_index"].duplicated().any():
        raise ValueError("episodes manifest episode_index values must be unique")
    if eps.duplicated(["data/chunk_index", "data/file_index", "dataset_from_index"]).any():
        raise ValueError("episodes manifest has duplicate per-shard dataset_from_index values")

    eps["_data_row_offset"] = compute_file_local_offsets(
        eps,
        "data/chunk_index",
        "data/file_index",
    )
    total_rows = int(eps["length"].sum())
    declared_total = resolved_info.get("total_frames")
    if declared_total is not None:
        if type(declared_total) is not int or declared_total < 0:
            raise ValueError("info.json total_frames must be a non-negative integer")
        if declared_total != total_rows:
            raise ValueError(
                f"episodes manifest addresses {total_rows} rows but info.json declares total_frames={declared_total}"
            )

    grouped: dict[tuple[int, int, str], list[LeRobotV3DataRange]] = {}
    path_owners: dict[str, tuple[int, int]] = {}
    for _, row in eps.iterrows():
        chunk_index = int(row["data/chunk_index"])
        file_index = int(row["data/file_index"])
        try:
            rendered_path = data_path_template.format(
                chunk_index=chunk_index,
                file_index=file_index,
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"info.json data_path cannot resolve chunk/file indices ({exc})") from exc
        relative = Path(rendered_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"info.json data_path resolved outside the dataset root: {rendered_path!r}")
        relative_path = relative.as_posix()
        owner = (chunk_index, file_index)
        if relative_path in path_owners and path_owners[relative_path] != owner:
            raise ValueError(f"info.json data_path maps multiple chunk/file pairs to {relative_path!r}")
        path_owners[relative_path] = owner
        grouped.setdefault((chunk_index, file_index, relative_path), []).append(
            LeRobotV3DataRange(
                episode_index=int(row["episode_index"]),
                dataset_from_index=int(row["dataset_from_index"]),
                row_offset=int(row["_data_row_offset"]),
                length=int(row["length"]),
            )
        )

    shards = []
    for (chunk_index, file_index, relative_path), ranges in grouped.items():
        ordered_ranges = tuple(sorted(ranges, key=lambda item: item.row_offset))
        expected_offset = 0
        for episode_range in ordered_ranges:
            if episode_range.row_offset != expected_offset:
                raise ValueError(f"episodes manifest ranges for {relative_path} overlap or contain a gap")
            expected_offset += episode_range.length
        shards.append(
            LeRobotV3DataShard(
                chunk_index=chunk_index,
                file_index=file_index,
                relative_path=relative_path,
                ranges=ordered_ranges,
            )
        )
    shards.sort(key=lambda shard: shard.relative_path)
    return LeRobotV3DataPopulation(
        data_path_template=data_path_template,
        total_rows=total_rows,
        shards=tuple(shards),
        episodes=eps,
    )


def _digest_length_prefixed(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def digest_lerobot_v3_data_population(population: LeRobotV3DataPopulation) -> str:
    """Hash the effective data template and every addressed episode range."""
    hasher = hashlib.sha256(b"openwam:lerobot-v3-data-population:v1\0")
    _digest_length_prefixed(hasher, population.data_path_template.encode("utf-8"))
    hasher.update(population.total_rows.to_bytes(8, "little"))
    hasher.update(len(population.shards).to_bytes(8, "little"))
    for shard in population.shards:
        _digest_length_prefixed(hasher, shard.relative_path.encode("utf-8"))
        hasher.update(shard.chunk_index.to_bytes(8, "little"))
        hasher.update(shard.file_index.to_bytes(8, "little"))
        hasher.update(len(shard.ranges).to_bytes(8, "little"))
        for episode_range in shard.ranges:
            for value in (
                episode_range.episode_index,
                episode_range.dataset_from_index,
                episode_range.row_offset,
                episode_range.length,
            ):
                hasher.update(value.to_bytes(8, "little"))
    return hasher.hexdigest()


def read_lerobot_v3_population_shard(
    dataset_dir: Path,
    shard: LeRobotV3DataShard,
    columns: List[str] | tuple[str, ...],
) -> pa.Table:
    """Read and validate only the rows addressed by one logical shard."""
    requested = list(dict.fromkeys(columns))
    read_columns = list(dict.fromkeys(["episode_index", *requested]))
    path = Path(dataset_dir) / shard.relative_path
    try:
        table = pq.read_table(path, memory_map=True, columns=read_columns)
    except pa.ArrowInvalid as exc:
        if "Dot path" not in str(exc):
            raise
        table = pq.read_table(path, memory_map=True).select(read_columns)

    slices = []
    for episode_range in shard.ranges:
        stop = episode_range.row_offset + episode_range.length
        if episode_range.row_offset < 0 or stop > table.num_rows:
            raise ValueError(
                f"manifest range for episode_index={episode_range.episode_index} exceeds {shard.relative_path} "
                f"({episode_range.row_offset}:{stop} of {table.num_rows} rows)"
            )
        episode_table = table.slice(episode_range.row_offset, episode_range.length)
        if episode_table.num_rows != episode_range.length:
            raise ValueError(
                f"manifest range for episode_index={episode_range.episode_index} is truncated in {shard.relative_path}"
            )
        episode_column = episode_table.column("episode_index").combine_chunks()
        if episode_column.null_count:
            raise ValueError(f"episode_index contains nulls in {shard.relative_path}")
        physical_ids = episode_column.to_numpy(zero_copy_only=False)
        if not np.all(physical_ids == episode_range.episode_index):
            raise ValueError(
                f"manifest episode_index={episode_range.episode_index} does not match every physical row in "
                f"{shard.relative_path}[{episode_range.row_offset}:{stop}]"
            )
        slices.append(episode_table)
    population_table = pa.concat_tables(slices) if len(slices) > 1 else slices[0]
    return population_table.select(requested)


def apply_info_splits(
    eps_df: pd.DataFrame,
    split: str,
    info_splits: dict,
    *,
    source_name: str = "dataset",
) -> pd.DataFrame:
    """Honor ``info.json[splits][split]`` when declared; default otherwise.

    - If ``split`` appears in ``info_splits``: expect ``"start:end"`` (int
      bounds, half-open), filter ``eps_df`` to ``episode_index ∈ [start, end)``.
    - Otherwise: ``train`` returns the full eps_df, anything else returns
      empty. Pretraining-only projects depend on the empty-val path; if
      future consumers need a different fallback they can add it here.

    Raises ValueError on a malformed split spec.
    """
    if split in info_splits:
        spec = info_splits[split]
        try:
            start_str, end_str = spec.split(":")
            start, end = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            raise ValueError(f"info.json splits.{split} must be 'start:end', got {spec!r}")
        sel = eps_df[(eps_df["episode_index"] >= start) & (eps_df["episode_index"] < end)].reset_index(drop=True)
        logger.info(
            "%s: split=%s from info.json (episode_index %d:%d, %d eps)",
            source_name,
            split,
            start,
            end,
            len(sel),
        )
        return sel
    return eps_df.reset_index(drop=True) if split == "train" else eps_df.iloc[0:0].reset_index(drop=True)


def water_fill_hours(bucket_hours: List[float], total_budget: float) -> List[float]:
    """Allocate ``total_budget`` hours across N buckets via water-filling.

    Each bucket gets at most its own ``bucket_hours[i]``. Surplus from
    under-budgetable buckets (too small to absorb fair share) is
    redistributed equally among the rest. The result is parallel to
    ``bucket_hours``; the sum is min(total_budget, sum(bucket_hours)).

    Args:
        bucket_hours: List of total available hours per bucket.
        total_budget: Target total hours across all buckets. A non-positive
            value returns all-zeros; a value >= sum(bucket_hours) returns
            ``bucket_hours`` unchanged (everyone gets full).

    Returns:
        List of allocations parallel to ``bucket_hours``. Each element is
        in [0, bucket_hours[i]].

    Examples:
        >>> water_fill_hours([10.0, 10.0, 10.0], 15.0)
        [5.0, 5.0, 5.0]
        >>> water_fill_hours([0.5, 10.0, 10.0], 5.0)
        [0.5, 2.25, 2.25]
        >>> water_fill_hours([1.0, 1.0, 1.0], 100.0)   # over-budget
        [1.0, 1.0, 1.0]
    """
    n = len(bucket_hours)
    if n == 0:
        return []
    if total_budget <= 0:
        return [0.0] * n
    if total_budget >= sum(bucket_hours):
        return [float(h) for h in bucket_hours]

    alloc = [0.0] * n
    active = list(range(n))
    remaining = float(total_budget)

    while active and remaining > 1e-9:
        fair_share = remaining / len(active)
        # Buckets too small to absorb fair_share — cap them at their max.
        small = [i for i in active if bucket_hours[i] <= fair_share]
        if not small:
            # Every remaining bucket can absorb fair_share; done.
            for i in active:
                alloc[i] = fair_share
            break
        for i in small:
            alloc[i] = float(bucket_hours[i])
            remaining -= bucket_hours[i]
            active.remove(i)

    return alloc


def effective_episode_frames(eps_df: pd.DataFrame, length_col: str = "length") -> np.ndarray:
    """Return validated post-trim frame counts for each episode row.

    Readers that physically shift offsets during trimming also replace
    ``length`` and therefore need no special handling. Segment-based readers
    retain nominal ``length`` and provide the sampleable half-open range through
    the paired ``_valid_start`` / ``_valid_end`` columns.
    """
    nominal = eps_df[length_col].to_numpy(dtype=np.int64)
    if np.any(nominal < 0):
        raise ValueError(f"episode {length_col!r} values must be non-negative")
    if length_col != "length":
        return nominal

    has_start = "_valid_start" in eps_df.columns
    has_end = "_valid_end" in eps_df.columns
    if has_start != has_end:
        missing = "_valid_end" if has_start else "_valid_start"
        raise ValueError(f"episode table has only one valid-range column; missing {missing!r}")
    if not has_start:
        return nominal

    valid_start = eps_df["_valid_start"].to_numpy(dtype=np.int64)
    valid_end = eps_df["_valid_end"].to_numpy(dtype=np.int64)
    invalid = (valid_start < 0) | (valid_end < valid_start) | (valid_end > nominal)
    if np.any(invalid):
        bad_pos = np.flatnonzero(invalid)[:5]
        preview = ", ".join(
            f"row {int(i)}: [{int(valid_start[i])}, {int(valid_end[i])}) / length {int(nominal[i])}" for i in bad_pos
        )
        raise ValueError(f"episode valid ranges must satisfy 0 <= start <= end <= length ({preview})")
    return valid_end - valid_start


def subsample_episodes_by_hours(
    eps_df: pd.DataFrame,
    target_hours: float,
    fps: float,
    seed: int,
    length_col: str = "length",
    episode_frames: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Random subset totalling >= ``target_hours`` of effective footage.

    Selection: seeded permutation over rows, then greedy prefix until
    cumulative effective length >= target_hours * fps * 3600.  When the
    standard ``length`` column is requested, optional ``_valid_start`` and
    ``_valid_end`` columns define the count; this is how segment-trimmed
    datasets budget only frames that can actually be sampled. Readers that
    implement trimming by mutating ``length`` need no special case. Returns
    rows in their *original* eps_df order — important because downstream offset
    columns (``_data_row_offset`` / ``_video_frame_offset``) are computed per-row
    and must stay aligned.

    If the full eps_df already fits in target_hours, returns it unchanged
    (identity short-circuit, no permutation overhead).

    Args:
        eps_df: Episodes DataFrame with at minimum a ``length_col`` column.
        target_hours: Target total duration in hours. Must be > 0.
        fps: Frame rate (from info.json).
        seed: Random seed for the permutation.
        length_col: Name of the nominal episode-length column. For the default
            ``"length"``, ``_valid_start/_valid_end`` override its sampled span
            when both are present.
        episode_frames: Optional explicit non-negative frame count per row.
            Reader-level callers use this to zero episodes that cannot produce a
            window. When omitted, :func:`effective_episode_frames` is used.

    Returns:
        Filtered DataFrame (same columns, subset of rows, in original order,
        with index reset).

    Raises:
        ValueError if target_hours <= 0.
    """
    if not np.isfinite(target_hours) or target_hours <= 0:
        raise ValueError(f"target_hours must be > 0, got {target_hours}")
    target_float = float(target_hours) * 3600.0 * float(fps)
    # ``250 / fps / 3600`` can round-trip as 250.00000000000003. Nudge one
    # representable float toward -inf before ceil so an exactly frame-aligned
    # request does not spuriously demand a sixth 50-frame episode.
    target_frames = max(1, int(np.ceil(np.nextafter(target_float, -np.inf))))
    if episode_frames is None:
        lengths = effective_episode_frames(eps_df, length_col=length_col)
    else:
        raw = np.asarray(episode_frames)
        if raw.shape != (len(eps_df),):
            raise ValueError(f"episode_frames shape {raw.shape} != ({len(eps_df)},)")
        lengths = raw.astype(np.int64)
        if np.any(raw != lengths) or np.any(lengths < 0):
            raise ValueError("episode_frames must contain non-negative integer frame counts")
    available_frames = int(lengths.sum())
    if available_frames <= 0:
        raise ValueError("episode population contains no positive sampleable frame span")
    if available_frames <= target_frames:
        return eps_df.reset_index(drop=True)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(eps_df))
    cum = 0
    selected = []
    for idx in perm:
        if lengths[idx] <= 0:
            continue
        cum += int(lengths[idx])
        selected.append(int(idx))
        if cum >= target_frames:
            break
    return eps_df.iloc[sorted(selected)].reset_index(drop=True)


def load_tasks_annotated(dataset_dir: Path, *, source_name: str = "dataset") -> Dict[int, str]:
    """Load ``meta/tasks_annotated.parquet`` → ``{episode_index: text}``.

    OXE prompt source: per-episode LLM-rewritten descriptions indexed by
    ``episode_index``. Raises FileNotFoundError when absent and ValueError on a
    wrong index name. Warns (does not raise) if any entry is empty.
    """
    ann_path = Path(dataset_dir) / "meta" / "tasks_annotated.parquet"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"{ann_path} missing — OXE readers require tasks_annotated.parquet for prompts. "
            "Re-run the upstream OXE conversion if this file is absent."
        )
    ann_df = pd.read_parquet(ann_path)
    if ann_df.index.name != "episode_index":
        raise ValueError(f"{ann_path}: expected index 'episode_index', got {ann_df.index.name!r}")
    txt_col = "task" if "task" in ann_df.columns else ann_df.columns[0]
    mapping: Dict[int, str] = {int(ep_idx): str(text) for ep_idx, text in ann_df[txt_col].items()}
    n_empty = sum(1 for t in mapping.values() if not t.strip())
    if n_empty:
        logger.warning(
            "%s: %d/%d annotated prompts are empty (unexpected — coverage was 100%% at conversion).",
            source_name,
            n_empty,
            len(mapping),
        )
    return mapping


def resolve_prompt_by_episode(episode_idx_to_text: Dict[int, str], episode_index: int, ds_name: str) -> str:
    """OXE prompt resolver: look up per-episode text by ``episode_index``.

    Raises KeyError on a missing episode_index — a real data inconsistency
    that should not be silently swallowed.
    """
    if int(episode_index) not in episode_idx_to_text:
        raise KeyError(
            f"{ds_name} prompt lookup failed: episode_index={episode_index} not present in tasks_annotated.parquet."
        )
    return episode_idx_to_text[int(episode_index)]


def build_multibucket(
    reader_cls: type,
    sub_dirs: List[Path],
    common: Dict[str, Any],
    *,
    base_seed: int,
    total_hours: Optional[float],
    wrapper_cls: type,
    source_name: str = "dataset",
    per_bucket_kwargs: Optional[Callable[[Path], Dict[str, Any]]] = None,
) -> Any:
    """Construct N per-bucket readers from ``sub_dirs`` and wrap them.

    Shared multi-bucket root-mode builder for RoboCOIN / EgoDex (and any future
    LeRobot v3 reader family). Honors an optional ``total_hours`` water-fill
    budget over the readers' effective post-split/exclusion/trim population.
    Leaves are initialized uncapped once, their exact effective capacities are
    water-filled, and the resulting whole-episode subsets are applied in place.
    This avoids estimating from the raw manifest, which overstates capacity for
    trimmed datasets. Construction is **robust**: a bucket whose construction
    raises is skipped (logged), empty buckets are filtered, and only an
    all-failed result raises.

    Args:
        reader_cls: the per-bucket reader class to instantiate.
        sub_dirs: discovered bucket directories (each with ``meta/info.json``).
        common: kwargs shared by every bucket (split + window/video knobs).
        base_seed: seed base for per-bucket subsample (``base_seed + i*7919``).
        total_hours: optional global hour budget; None → load full dataset.
        wrapper_cls: the ``MultiLeRobotV3Reader`` subclass to wrap buckets in.
        source_name: label for log lines.
        per_bucket_kwargs: optional callable ``(sub_dir) -> dict`` for extra
            per-bucket kwargs (unused by current readers; kept for extension).
    """
    if total_hours is not None:
        total_hours = float(total_hours)
        if not np.isfinite(total_hours) or total_hours <= 0:
            raise ValueError(f"{source_name} total_hours must be > 0 or null/unset; got {total_hours}.")
    else:
        logger.info("%s: total_hours unset, loading full dataset", source_name)

    def _build_one(args):
        _, sub = args
        kwargs = dict(common)
        kwargs["dataset_dir"] = str(sub)
        if per_bucket_kwargs is not None:
            kwargs.update(per_bucket_kwargs(sub))
        try:
            return reader_cls(**kwargs)
        except DataContractError:
            # Proven broken data (see DataContractError): dropping the bucket
            # would hide a slice of the training set behind a warning, so let
            # the reader's own diagnostic abort the launch.
            raise
        except Exception as e:
            logger.warning("%s: skipping %s: %s", source_name, sub.name, e)
            return None

    build_args = list(enumerate(sub_dirs))
    with ThreadPoolExecutor(max_workers=min(len(sub_dirs), 16)) as pool:
        results = list(pool.map(_build_one, build_args))
    loaded = [(i, sub, r) for i, (sub, r) in enumerate(zip(sub_dirs, results)) if r is not None and len(r) > 0]
    if not loaded:
        raise RuntimeError(f"All {source_name} buckets failed to load")

    if total_hours is not None:
        bucket_hours = [float(reader.effective_hours) for _, _, reader in loaded]
        total_avail = sum(bucket_hours)
        if total_hours > total_avail - 1e-3:
            # Over budget: every bucket already represents the complete effective
            # population, so applying a near-equal cap only risks a float-rounding
            # loss. Keep each initialized leaf unchanged.
            logger.warning(
                "%s total_hours=%.3f exceeds available effective footage %.3fh; "
                "loading the full post-split/exclusion/trim population.",
                source_name,
                total_hours,
                total_avail,
            )
        else:
            allocations = water_fill_hours(bucket_hours, total_hours)
            for (original_i, _, reader), allocation in zip(loaded, allocations):
                reader._apply_effective_hour_budget(allocation, base_seed + original_i * 7919)
            realized = sum(float(reader.effective_hours) for _, _, reader in loaded if len(reader) > 0)
            logger.info(
                "%s total_hours=%.3f → allocated %.3f effective h across %d buckets; whole-episode realization %.3fh.",
                source_name,
                total_hours,
                sum(allocations),
                len(loaded),
                realized,
            )

    buckets = [reader for _, _, reader in loaded if len(reader) > 0]
    if not buckets:
        raise RuntimeError(f"All {source_name} buckets became empty after effective-hour subsampling")
    # _build_one swallows per-bucket construction errors (missing/corrupt data,
    # and — for RoboCOIN — stats-integrity validation raised in _load_stats) into
    # a warning + None, so a misconfigured bucket is dropped rather than aborting
    # the run. Surface the dropped set explicitly so that silent data loss (e.g. a
    # whole robot_type lost to a stale stats file) is visible, not buried.
    dropped = [sub.name for sub, r in zip(sub_dirs, results) if r is None or len(r) == 0]
    if dropped:
        logger.warning(
            "%s: dropped %d / %d bucket(s) during load (construction failed or empty): %s",
            source_name,
            len(dropped),
            len(sub_dirs),
            ", ".join(sorted(dropped)),
        )
    logger.info("%s: loaded %d / %d buckets", source_name, len(buckets), len(sub_dirs))
    return wrapper_cls(buckets)


__all__ = [
    "DataContractError",
    "parse_info_json",
    "load_episodes_parquet",
    "compute_file_local_offsets",
    "apply_info_splits",
    "water_fill_hours",
    "effective_episode_frames",
    "subsample_episodes_by_hours",
    "load_tasks_annotated",
    "resolve_prompt_by_episode",
    "build_multibucket",
]
