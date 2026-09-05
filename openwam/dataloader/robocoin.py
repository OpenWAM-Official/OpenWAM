"""RoboCOIN LeRobot v3 reader with real action/state supervision.

Grippered buckets consume the published ``eef_sim_pose_*`` Euler-XYZ poses and
``gripper_open_scale_*`` values, producing the canonical 20-D bimanual EEF
layout.  The pose follows a rigid terminal-arm attachment in the robot base
frame and the gripper convention is ``0=closed, 1=open``.

Dexterous-hand buckets have no scalar gripper columns.  Without unification
they emit pose with masked grip slots; with unification they append left/right
finger joints and scatter pose/fingers into the corresponding 80-D slots while
leaving scalar-grip and unused hand slots masked.

Rows are temporally aligned: action row ``t`` is supervised at row ``t``.
Normalization is pooled per robot type over action and state, with rot6d
dimensions pinned to identity.  Optional episode trimming and exclusions are
validated against the exact manifest population before stats are accepted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import List

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import EEF_DIM as _ACTION_DIM
from openwam.dataloader.utils.eef import eef14_to_eef20
from openwam.dataloader.utils.lerobotv3 import (
    DataContractError,
    ExcludedEpisodesSnapshot,
    apply_info_splits,
    assert_excluded_episodes_snapshot_current,
    build_multibucket,
    load_episodes_parquet,
    load_excluded_episodes_snapshot,
    parse_info_json,
)
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

_STATE_DIM = _ACTION_DIM


# One canonical denylist is shared by reader and stats discovery.
ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS = MappingProxyType(
    {
        "Airbot_MMK2_storage_peach_pear": MappingProxyType(
            {
                "robot_type": "airbot_mmk2",
                "reason": "fails the public action-unit contract and must not enter normalization statistics",
            }
        ),
    }
)
_WHOLE_BUCKET_EXCLUSIONS_SCHEMA_VERSION = 1
_WHOLE_BUCKET_EXCLUSIONS_POLICY = "drop_whole_bucket_before_reader_and_stats_discovery"


def is_robocoin_bucket_excluded(dataset_dir) -> bool:
    """Whether ``dataset_dir`` is excluded as one indivisible task bucket."""
    return Path(dataset_dir).name in ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS


def robocoin_bucket_exclusions_provenance(robot_type: str) -> dict:
    """Canonical whole-bucket exclusion contract for one robot-type stats file."""
    datasets = {
        dataset_id: dict(entry)
        for dataset_id, entry in ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS.items()
        if entry["robot_type"] == str(robot_type)
    }
    return {
        "schema_version": _WHOLE_BUCKET_EXCLUSIONS_SCHEMA_VERSION,
        "policy": _WHOLE_BUCKET_EXCLUSIONS_POLICY,
        "datasets": dict(sorted(datasets.items())),
    }


def _validate_whole_bucket_exclusions_provenance(actual, *, robot_type, stats_path) -> None:
    """Fail closed when stats for an affected robot type predate its denylist."""
    expected = robocoin_bucket_exclusions_provenance(robot_type)
    if expected["datasets"]:
        if actual != expected:
            raise DataContractError(
                f"RoboCOIN stats {stats_path} whole_bucket_exclusions_provenance does not "
                f"match the repository exclusion contract for robot_type={robot_type!r}; "
                f"expected {expected}, got {actual}. Regenerate stats."
            )
    elif actual is not None and actual != expected:
        raise DataContractError(
            f"RoboCOIN stats {stats_path} has incompatible "
            f"whole_bucket_exclusions_provenance for robot_type={robot_type!r}; "
            f"expected {expected}, got {actual}. Regenerate stats."
        )


def _discover_data_parquets(dataset_dir: Path, data_path_template: str):
    """Return canonical numeric RoboCOIN data shards in reader order."""
    if not isinstance(data_path_template, str) or not data_path_template:
        raise DataContractError(
            f"RoboCOIN({Path(dataset_dir).name}): info.json data_path must be a non-empty string, "
            f"got {data_path_template!r}"
        )
    data_files = {}
    for path in sorted((Path(dataset_dir) / "data").glob("chunk-*/file-*.parquet")):
        chunk_m = re.search(r"chunk-(\d+)$", path.parent.name)
        file_m = re.search(r"file-(\d+)$", path.stem)
        if chunk_m is not None and file_m is not None:
            coordinates = (int(chunk_m.group(1)), int(file_m.group(1)))
            previous = data_files.get(coordinates)
            if previous is not None:
                raise DataContractError(
                    f"RoboCOIN({Path(dataset_dir).name}): duplicate numeric data shard "
                    f"coordinates {coordinates}: {previous} and {path}"
                )
            data_files[coordinates] = path

    result = []
    for (chunk_index, file_index), path in sorted(data_files.items()):
        try:
            expected = Path(dataset_dir) / data_path_template.format(
                chunk_index=chunk_index,
                file_index=file_index,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            raise DataContractError(
                f"RoboCOIN({Path(dataset_dir).name}): invalid info.json data_path template {data_path_template!r}: {e}"
            ) from e
        if path != expected:
            raise DataContractError(
                f"RoboCOIN({Path(dataset_dir).name}): non-canonical numeric data shard {path}; "
                f"info.json data_path resolves coordinates {(chunk_index, file_index)} to {expected}"
            )
        result.append((path, chunk_index, file_index))
    return result


_NEEDED_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "gripper_open_scale_action",
    "eef_sim_pose_state",
    "gripper_open_scale_state",
)


_BASE_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "eef_sim_pose_state",
)
_GRIP_COLS = ("gripper_open_scale_action", "gripper_open_scale_state")


_DEX_UNIFY_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "eef_sim_pose_state",
    "action",
    "observation.state",
)


GRIP_EXCLUDED_DIM_MASK = np.ones(_ACTION_DIM, dtype=bool)
GRIP_EXCLUDED_DIM_MASK[9] = False
GRIP_EXCLUDED_DIM_MASK[19] = False


def _finger_indices(feature: dict):
    """Indices of left/right dexterous-hand joints within an action/state array.

    Reads the ``names`` (or ``names.motors``) list from one info.features entry
    and returns ``(left_idx, right_idx)`` for ``*_hand_joint_*`` motors.
    """

    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    names = names or []
    left = [i for i, x in enumerate(names) if str(x).startswith("left_hand_joint")]
    right = [i for i, x in enumerate(names) if str(x).startswith("right_hand_joint")]
    return left, right


MAX_HAND_DOF = 24


def dex_finger_layout(features: dict):
    """Validated dexterous-hand finger layout for one ``info.features`` dict.

    A bucket qualifies as dexterous-hand ONLY when it has the unified pose column
    but NO ``gripper_open_scale_*`` columns, AND both hands carry
    ``1..MAX_HAND_DOF`` ``*_hand_joint_*`` motors with the action and state finger
    counts agreeing. Returns ``(aL, aR, sL, sR)`` — the left/right finger indices
    within the raw ``action`` / ``observation.state`` arrays — when all hold, else
    ``None``.

    Single source of truth for the dex-bucket gate, shared by the reader
    (:meth:`RoboCOINDataset.__init__`) and the stats script so they cannot drift.
    """

    has_grip = all(c in features for c in _GRIP_COLS)
    if has_grip or "eef_sim_pose_action" not in features:
        return None
    aL, aR = _finger_indices(features.get("action", {}))
    sL, sR = _finger_indices(features.get("observation.state", {}))
    kL, kR = len(aL), len(aR)
    if not (0 < kL <= MAX_HAND_DOF and 0 < kR <= MAX_HAND_DOF and len(sL) == kL and len(sR) == kR):
        return None
    return aL, aR, sL, sR


def _build_dex_unify_map(k_left: int, k_right: int):
    """``dst_index`` (single-list spec) for a dex-hand raw action laid out as
    ``[L_pos(3), L_rot6d(6), L_fingers(kL), R_pos(3), R_rot6d(6), R_fingers(kR)]``.

    Maps pose → shared 80-D pose slots [0:9)/[R_POS:R_POS+9) and fingers → the
    dedicated hand slots [10:10+kL)/[R_HAND:R_HAND+kR) (front-aligned; the unused
    tail of each MAX_HAND_DOF-wide hand region and the gripper slots stay unmapped
    → masked). All right-side offsets are derived from MAX_HAND_DOF so the layout
    follows the hand width automatically (currently: R pose 34:43, R hand 44:68).
    """

    l_hand = 10
    r_pos = l_hand + MAX_HAND_DOF
    r_hand = r_pos + 10
    return (
        list(range(0, 3))
        + list(range(3, 9))
        + list(range(l_hand, l_hand + k_left))
        + list(range(r_pos, r_pos + 3))
        + list(range(r_pos + 3, r_pos + 9))
        + list(range(r_hand, r_hand + k_right))
    )


_eef14_to_eef20 = eef14_to_eef20


HEAD_CAMERA_PRIORITY = [
    "observation.images.cam_high_rgb",
    "observation.images.cam_head_rgb",
    "observation.images.cam_head_right_rgb",
    "observation.images.cam_head_left_rgb",
    "observation.images.cam_high_right_rgb",
    "observation.images.cam_high_left_rgb",
    "observation.images.cam_high_realsense_rgb",
    "observation.images.cam_front_rgb",
    "observation.images.cam_front_head_rgb",
    "observation.images.cam_front_chest_rgb",
    "observation.images.cam_chest_rgb",
    "observation.images.camera_head_rgb",
    "observation.images.cam_left_high",
    "observation.images.ego_view",
]

WRIST_LEFT_CANDIDATES = [
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_left_wrist_rgb_rgb",
    "observation.images.camera_left_wrist_rgb",
    "observation.images.cam_left_wrist",
]

WRIST_RIGHT_CANDIDATES = [
    "observation.images.cam_right_wrist_rgb",
    "observation.images.cam_right_wrist_rgb_rgb",
    "observation.images.camera_right_wrist_rgb",
    "observation.images.cam_right_wrist",
]


_TRIM_SNAPSHOT_CACHE: dict = {}
_TRIM_SCHEMA_VERSION = 1
_TRIM_MIN_LEN = 1
_TRIM_ZERO_SPAN_POLICY = "drop"
_EXCLUDED_EPISODES_SCHEMA_VERSION = 1
_EXCLUDED_EPISODES_POLICY = "drop_matching_episode_index_before_trim"
_STATS_POPULATION_SCHEMA_VERSION = 2
_STATS_POPULATION_POLICY = "info_split_then_exclusion_then_trim"
_TRIM_CSV_COLUMNS = (
    "dataset",
    "episode_index",
    "total_frames",
    "trim_head_to",
    "trim_tail_from",
)


def _excluded_episodes_provenance(
    snapshots: dict[str, ExcludedEpisodesSnapshot],
) -> dict:
    """Serialize the exact effective blacklist for every stats contributor."""
    return {
        "schema_version": _EXCLUDED_EPISODES_SCHEMA_VERSION,
        "policy": _EXCLUDED_EPISODES_POLICY,
        "datasets": {
            dataset_id: {"episode_indices": list(snapshot.episode_indices)}
            for dataset_id, snapshot in sorted(snapshots.items())
        },
    }


def _validate_excluded_episodes_provenance(
    actual,
    *,
    dataset_dir: Path,
    dataset_id: str,
    current_snapshot: ExcludedEpisodesSnapshot,
    num_datasets,
    stats_path: Path,
) -> dict[str, ExcludedEpisodesSnapshot]:
    """Validate every bucket population pooled into one robot-type stats file."""
    expected_keys = {"schema_version", "policy", "datasets"}
    if (
        not isinstance(actual, dict)
        or set(actual) != expected_keys
        or type(actual.get("schema_version")) is not int
        or actual.get("schema_version") != _EXCLUDED_EPISODES_SCHEMA_VERSION
        or actual.get("policy") != _EXCLUDED_EPISODES_POLICY
        or not isinstance(actual.get("datasets"), dict)
    ):
        raise DataContractError(
            f"RoboCOIN bucket {dataset_id}: stats {stats_path} excluded_episodes_provenance "
            "has a missing or incompatible schema/policy. Regenerate stats with the configured trim CSV."
        )

    datasets = actual["datasets"]
    if type(num_datasets) is not int or num_datasets != len(datasets):
        raise DataContractError(
            f"RoboCOIN bucket {dataset_id}: stats {stats_path} declares num_datasets="
            f"{num_datasets!r} but excluded_episodes_provenance has {len(datasets)} dataset entries. "
            "Regenerate stats with the configured trim CSV."
        )
    if dataset_id not in datasets:
        raise DataContractError(
            f"RoboCOIN bucket {dataset_id}: stats {stats_path} excluded_episodes_provenance "
            "does not include this bucket. Regenerate stats with the configured trim CSV."
        )

    root = Path(dataset_dir).parent
    snapshots = {}
    for contributor, entry in datasets.items():
        if not isinstance(contributor, str) or not contributor:
            raise DataContractError(
                f"RoboCOIN bucket {dataset_id}: stats {stats_path} has invalid contributor "
                f"name {contributor!r} in excluded_episodes_provenance"
            )
        contributor_path = Path(contributor)
        if contributor in (".", "..") or contributor_path.is_absolute() or len(contributor_path.parts) != 1:
            raise DataContractError(
                f"RoboCOIN bucket {dataset_id}: stats {stats_path} has invalid contributor "
                f"name {contributor!r} in excluded_episodes_provenance"
            )
        bucket_dir = root / contributor
        if not bucket_dir.is_dir():
            raise DataContractError(
                f"RoboCOIN bucket {dataset_id}: stats {stats_path} references missing contributor "
                f"directory {bucket_dir} in excluded_episodes_provenance"
            )
        snapshot = current_snapshot if contributor == dataset_id else load_excluded_episodes_snapshot(bucket_dir)
        snapshots[contributor] = snapshot
        expected_entry = {"episode_indices": list(snapshot.episode_indices)}
        if (
            not isinstance(entry, dict)
            or set(entry) != {"episode_indices"}
            or not isinstance(entry.get("episode_indices"), list)
            or any(type(value) is not int for value in entry["episode_indices"])
            or entry != expected_entry
        ):
            raise DataContractError(
                f"RoboCOIN bucket {dataset_id}: stats {stats_path} excluded_episodes_provenance "
                f"for contributor {contributor!r} does not match the current exclusion population; "
                f"expected {expected_entry}, got {entry}. Regenerate stats with the configured trim CSV."
            )
    return snapshots


def _validate_stats_population_provenance(
    actual,
    *,
    dataset_dir: Path,
    current_dataset_id: str,
    current_info: dict,
    trim_snapshot: _TrimSnapshot,
    exclusion_snapshots: dict[str, ExcludedEpisodesSnapshot],
    num_datasets,
    stats_path: Path,
) -> None:
    """Validate every contributor to one pooled robot-type stats file.

    Checking only the leaf currently being constructed is insufficient: a
    sibling bucket contributes to the same pooled numbers, so changing that
    sibling's train split would stale normalization for this leaf too.  Rebuild
    the exact physical train spans for every contributor using the immutable
    trim/exclusion snapshots already validated above.
    """

    expected_header = {
        "schema_version": _STATS_POPULATION_SCHEMA_VERSION,
        "split": "train",
        "policy": _STATS_POPULATION_POLICY,
    }
    if (
        not isinstance(actual, dict)
        or {key: actual.get(key) for key in expected_header} != expected_header
        or set(actual) != {*expected_header, "datasets"}
        or not isinstance(actual.get("datasets"), dict)
    ):
        raise DataContractError(
            f"RoboCOIN bucket {current_dataset_id}: stats {stats_path} population must be "
            "train-derived with the reader's split/exclusion/trim order; "
            f"expected schema/header {expected_header}, got {actual}. Regenerate stats "
            "with --split train and the configured trim CSV."
        )
    datasets = actual["datasets"]
    if type(num_datasets) is not int or len(datasets) != num_datasets or set(datasets) != set(exclusion_snapshots):
        raise DataContractError(
            f"RoboCOIN bucket {current_dataset_id}: stats {stats_path} population datasets "
            "do not exactly match the pooled exclusion contributors; regenerate stats."
        )

    root = Path(dataset_dir).parent

    def _expected(contributor: str, snapshot: ExcludedEpisodesSnapshot) -> dict:
        bucket_dir = root / contributor
        info = current_info if contributor == current_dataset_id else parse_info_json(bucket_dir)
        _spans, _manifest_end, expected_entry = _stats_population_spans(
            bucket_dir,
            dataset_id=contributor,
            split="train",
            dataset_spec=trim_snapshot.spec.get(contributor, {}),
            excluded_episode_indices=snapshot.episode_indices,
            info=info,
        )
        return expected_entry

    current_expected = _expected(current_dataset_id, exclusion_snapshots[current_dataset_id])
    current_actual = datasets.get(current_dataset_id)
    if current_actual != current_expected:
        raise DataContractError(
            f"RoboCOIN bucket {current_dataset_id}: stats {stats_path} effective train "
            "population no longer matches its manifest/split/exclusion/trim spans; "
            f"expected {current_expected}, got {current_actual}. Regenerate stats."
        )

    population_digest = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_key = (str(stats_path.resolve()), population_digest)
    if cache_key in trim_snapshot.population_validation_cache:
        return

    for contributor, snapshot in exclusion_snapshots.items():
        if contributor == current_dataset_id:
            continue
        expected_entry = _expected(contributor, snapshot)
        actual_entry = datasets.get(contributor)
        if actual_entry != expected_entry:
            raise DataContractError(
                f"RoboCOIN bucket {current_dataset_id}: stats {stats_path} effective train "
                f"population for contributor {contributor!r} no longer matches its "
                "manifest/split/exclusion/trim spans; "
                f"expected {expected_entry}, got {actual_entry}. Regenerate stats."
            )
    trim_snapshot.population_validation_cache.add(cache_key)


@dataclass(frozen=True)
class _TrimSnapshot:
    """Parsed trim rules and provenance derived from one immutable byte snapshot."""

    path: str
    spec: dict
    sha256: str

    population_validation_cache: set[tuple[str, str]] = field(
        default_factory=set,
        compare=False,
        repr=False,
    )

    @property
    def provenance(self) -> dict:
        return {
            "schema_version": _TRIM_SCHEMA_VERSION,
            "sha256": self.sha256,
            "min_len": _TRIM_MIN_LEN,
            "zero_span_policy": _TRIM_ZERO_SPAN_POLICY,
        }


class _TrimSnapshotConfig:
    """Overlay one pinned snapshot onto dict, DictConfig, or namespace configs."""

    def __init__(self, base, snapshot: _TrimSnapshot):
        self._base = base
        self._trim_snapshot = snapshot

    def __getattr__(self, key):
        return getattr(self._base, key)

    def get(self, key, default=None):
        if key == "_trim_snapshot":
            return self._trim_snapshot
        if hasattr(self._base, "get"):
            return self._base.get(key, default)
        return getattr(self._base, key, default)


def _trim_csv_int(row, name: str, *, path: str, line_number: int, required: bool = False):
    """Parse one optional integer field, with actionable CSV error context."""
    raw = row.get(name)
    value = raw.strip() if isinstance(raw, str) else ""
    if not value:
        if required:
            raise ValueError(f"RoboCOIN trim_csv {path}, line {line_number}: '{name}' is required")
        return None
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(
            f"RoboCOIN trim_csv {path}, line {line_number}: '{name}' must be an integer, got {value!r}"
        ) from e


def _load_trim_snapshot(path) -> _TrimSnapshot:
    """Read, parse, and hash one immutable trim CSV byte snapshot.

    Cached per unchanged path: MultiRobotCOINDataset builds one reader per
    bucket, and each would otherwise re-read the same large CSV.

    Expected columns — ``dataset``, ``episode_index``, ``total_frames``,
    ``trim_head_to``, ``trim_tail_from``. Empty trim fields mean "no trim on that
    end". ``total_frames`` is required and carried through as a staleness check
    (see :meth:`RoboCOINDataset._filter_episodes`). Invalid files fail fast; only
    a completely parsed snapshot is cached. The parsed spec and SHA-256 always
    come from the same bytes, so a path replacement cannot pair old spans with
    new provenance inside one reader construction or stats scan.
    """

    key = str(path)
    try:
        stat = Path(key).stat()
    except OSError as e:
        raise OSError(
            e.errno,
            f"RoboCOIN trim_csv {key} could not be read: {e.strerror or e}",
            key,
        ) from e
    signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    cached = _TRIM_SNAPSHOT_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        raw = Path(key).read_bytes()
    except OSError as e:
        raise OSError(
            e.errno,
            f"RoboCOIN trim_csv {key} could not be read: {e.strerror or e}",
            key,
        ) from e
    try:
        text = raw.decode("utf-8")
    except UnicodeError as e:
        raise ValueError(f"RoboCOIN trim_csv {key}, line unknown: invalid text encoding: {e}") from e

    spec: dict = {}
    seen_entries = set()
    n = 0
    try:
        with io.StringIO(text, newline="") as fh:
            reader = csv.DictReader(fh, strict=True)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"RoboCOIN trim_csv {key}, line 1: missing CSV header")
            missing = [name for name in _TRIM_CSV_COLUMNS if name not in fieldnames]
            if missing:
                raise ValueError(f"RoboCOIN trim_csv {key}, line 1: missing required column(s): {', '.join(missing)}")
            duplicate = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
            if duplicate:
                raise ValueError(f"RoboCOIN trim_csv {key}, line 1: duplicate column(s): {', '.join(duplicate)}")

            for row in reader:
                line_number = reader.line_num
                if None in row:
                    raise ValueError(
                        f"RoboCOIN trim_csv {key}, line {line_number}: row has more values than the CSV header"
                    )
                dataset_raw = row.get("dataset")
                dataset = dataset_raw.strip() if isinstance(dataset_raw, str) else ""
                if not dataset:
                    raise ValueError(f"RoboCOIN trim_csv {key}, line {line_number}: 'dataset' is required")
                episode_index = _trim_csv_int(row, "episode_index", path=key, line_number=line_number, required=True)
                total = _trim_csv_int(row, "total_frames", path=key, line_number=line_number, required=True)
                head = _trim_csv_int(row, "trim_head_to", path=key, line_number=line_number)
                tail = _trim_csv_int(row, "trim_tail_from", path=key, line_number=line_number)
                if episode_index < 0:
                    raise ValueError(
                        f"RoboCOIN trim_csv {key}, line {line_number}: "
                        f"'episode_index' must be >= 0, got {episode_index}"
                    )
                if total <= 0:
                    raise ValueError(
                        f"RoboCOIN trim_csv {key}, line {line_number}: 'total_frames' must be > 0, got {total}"
                    )
                for name, value in (("trim_head_to", head), ("trim_tail_from", tail)):
                    if value is not None and not 0 <= value <= total:
                        raise ValueError(
                            f"RoboCOIN trim_csv {key}, line {line_number}: "
                            f"'{name}' must be in [0, total_frames={total}], got {value}"
                        )
                head = head or 0
                effective_tail = total if tail is None else tail
                if head > effective_tail:
                    raise ValueError(
                        f"RoboCOIN trim_csv {key}, line {line_number}: trim_head_to={head} "
                        f"must not exceed trim_tail_from={effective_tail}"
                    )
                entry_key = (dataset, episode_index)
                if entry_key in seen_entries:
                    raise ValueError(
                        f"RoboCOIN trim_csv {key}, line {line_number}: duplicate entry for "
                        f"dataset={dataset!r}, episode_index={episode_index}"
                    )
                seen_entries.add(entry_key)
                if not head and tail is None:
                    continue
                dataset_spec = spec.setdefault(dataset, {})
                dataset_spec[episode_index] = (head, tail, total)
                n += 1
    except csv.Error as e:
        line_number = getattr(locals().get("reader"), "line_num", "unknown")
        raise ValueError(f"RoboCOIN trim_csv {key}, line {line_number}: malformed CSV: {e}") from e

    logger.info("RoboCOIN: loaded %d trim entries across %d datasets from %s", n, len(spec), key)
    immutable_spec = MappingProxyType({dataset: MappingProxyType(entries) for dataset, entries in spec.items()})
    snapshot = _TrimSnapshot(path=key, spec=immutable_spec, sha256=hashlib.sha256(raw).hexdigest())
    _TRIM_SNAPSHOT_CACHE[key] = (signature, snapshot)
    return snapshot


def _load_trim_spec(path) -> dict:
    """Return the parsed spec from one cached immutable trim snapshot."""
    return {dataset: dict(entries) for dataset, entries in _load_trim_snapshot(path).spec.items()}


def _assert_trim_snapshot_current(snapshot: _TrimSnapshot, *, context: str) -> None:
    """Fail if the configured path changed after ``snapshot`` was pinned."""
    try:
        actual_sha256 = hashlib.sha256(Path(snapshot.path).read_bytes()).hexdigest()
    except OSError as e:
        raise OSError(
            e.errno,
            f"RoboCOIN trim_csv {snapshot.path} could not be re-read after {context}: {e.strerror or e}",
            snapshot.path,
        ) from e
    if actual_sha256 != snapshot.sha256:
        raise DataContractError(
            f"RoboCOIN trim_csv {snapshot.path} changed while {context}; "
            f"started with sha256={snapshot.sha256}, now sha256={actual_sha256}. "
            "Refusing mixed trim spans/provenance; retry with an immutable CSV."
        )


def _validate_trim_manifest(dataset_id: str, manifest, spec: dict) -> dict:
    """Validate one bucket's trim spec against its full pre-split manifest.

    Returns ``{episode_index: (head, tail)}`` relative spans. Unknown IDs and
    every ``total_frames`` mismatch are data-contract failures, including rows
    outside the selected split or later excluded by the reader.
    """

    if not spec:
        return {}
    try:
        episode_ids = [int(ep) for ep in manifest["episode_index"].to_numpy()]
        lengths = [int(length) for length in manifest["length"].to_numpy()]
    except (KeyError, TypeError, ValueError) as e:
        raise DataContractError(
            f"RoboCOIN({dataset_id}): full manifest must carry integer episode_index and length columns"
        ) from e
    manifest_lengths = dict(zip(episode_ids, lengths))
    if len(manifest_lengths) != len(episode_ids):
        raise DataContractError(f"RoboCOIN({dataset_id}): full manifest has duplicate episode_index values")

    unknown = sorted(set(spec).difference(manifest_lengths))
    if unknown:
        unknown_ids = ",".join(map(str, unknown[:10])) + ("..." if len(unknown) > 10 else "")
        raise DataContractError(
            f"RoboCOIN({dataset_id}): trim list references {len(unknown)} unknown episode_index "
            f"value(s) absent from the full pre-split manifest (episode_index={unknown_ids}). "
            "Refusing dataset construction; re-run "
            "Regenerate the trim manifest against the current corpus before retrying."
        )

    stale = sorted(ep for ep, (_, _, total) in spec.items() if int(total) != manifest_lengths[ep])
    if stale:
        stale_ids = ",".join(map(str, stale[:10])) + ("..." if len(stale) > 10 else "")
        raise DataContractError(
            f"RoboCOIN({dataset_id}): trim list is STALE for {len(stale)} matching episode(s) "
            f"(episode_index={stale_ids}): recorded total_frames disagrees with the full manifest "
            "length. Refusing dataset construction because episode indices shift after physical "
            "deletion and same-length collisions cannot be detected individually. Re-run "
            "Regenerate the trim manifest against the current corpus before retrying."
        )

    return {
        ep: (int(head), manifest_lengths[ep] if tail is None else int(tail)) for ep, (head, tail, _) in spec.items()
    }


def _stats_population_spans(
    dataset_dir,
    *,
    dataset_id: str,
    split: str,
    dataset_spec: dict,
    excluded_episode_indices,
    info: dict | None = None,
):
    """Return physical kept spans plus a digest of the exact stats population.

    The order deliberately mirrors the reader: validate trims against the full
    pre-split manifest, apply ``info.json`` split membership, drop exclusions,
    then apply the head/tail span.  The digest binds each surviving episode to
    its physical global ``[start, end)`` rows, so changing a split range cannot
    be hidden by merely leaving a top-level ``split: train`` label in an old
    stats file.
    """

    root = Path(dataset_dir)
    manifest = load_episodes_parquet(root)
    trim_spans = _validate_trim_manifest(dataset_id, manifest, dataset_spec)
    if info is None:
        info = parse_info_json(root)
    selected = apply_info_splits(
        manifest,
        split,
        info.get("splits", {}) or {},
        source_name=f"RoboCOIN stats population({dataset_id})",
    )
    selected_ids = set(int(ep) for ep in selected["episode_index"].to_numpy())
    try:
        episode_ids = [int(ep) for ep in manifest["episode_index"].to_numpy()]
        starts = [int(start) for start in manifest["dataset_from_index"].to_numpy()]
        lengths = [int(length) for length in manifest["length"].to_numpy()]
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(
            f"RoboCOIN({dataset_id}): full manifest must carry integer "
            "episode_index, dataset_from_index, and length columns"
        ) from exc

    excluded = set(int(ep) for ep in excluded_episode_indices)
    raw_spans = []
    kept = []
    for episode_id, start, length in zip(episode_ids, starts, lengths):
        if start < 0 or length < 0:
            raise DataContractError(
                f"RoboCOIN({dataset_id}): invalid manifest span for episode_index={episode_id}: "
                f"dataset_from_index={start}, length={length}"
            )
        raw_spans.append((start, start + length, episode_id))
        if episode_id not in selected_ids or episode_id in excluded:
            continue
        head, tail = trim_spans.get(episode_id, (0, length))
        if tail - head >= _TRIM_MIN_LEN:
            kept.append((start + head, start + tail, episode_id))

    raw_spans.sort()
    for previous, current in zip(raw_spans, raw_spans[1:]):
        if current[0] < previous[1]:
            raise DataContractError(
                f"RoboCOIN({dataset_id}): overlapping global manifest spans for "
                f"episode_index={previous[2]} and {current[2]}"
            )
    kept.sort()
    hasher = hashlib.sha256(b"openwam:robocoin-stats-population:v1\0")
    for start, end, episode_id in kept:
        for value in (episode_id, start, end):
            hasher.update(int(value).to_bytes(8, "little", signed=False))
    entry = {
        "effective_population_digest": hasher.hexdigest(),
        "num_episodes": len(kept),
        "num_rows": sum(end - start for start, end, _episode_id in kept),
    }
    manifest_end = max((end for _start, end, _episode_id in raw_spans), default=0)
    return [(start, end) for start, end, _episode_id in kept], manifest_end, entry


def _resolve_robocoin_cameras(features: dict) -> tuple:
    """Resolve head / left-wrist / right-wrist camera keys from info.json features.

    Returns:
        (head_cam, left_wrist_cam, right_wrist_cam) — each is a string or None.
    """

    feat_keys = set(features.keys())
    head = None
    for c in HEAD_CAMERA_PRIORITY:
        if c in feat_keys:
            head = c
            break
    left_wrist = None
    for c in WRIST_LEFT_CANDIDATES:
        if c in feat_keys:
            left_wrist = c
            break
    right_wrist = None
    for c in WRIST_RIGHT_CANDIDATES:
        if c in feat_keys:
            right_wrist = c
            break
    return head, left_wrist, right_wrist


class RoboCOINDataset(LeRobotV3Reader):
    """Single-dataset reader for one RoboCOIN task bucket (LeRobot v3).

    Bimanual 20-D EEF with real action/state supervision and per-robot-type
    in-reader normalization. All window / offset / video / pickle machinery is
    inherited from :class:`LeRobotV3Reader`; only RoboCOIN-specific bits are
    overridden below.
    """

    DATASET_NAME = "RoboCOIN"
    NEEDED_COLS = _NEEDED_COLS

    PROMPT_FILE_REQUIRED = False

    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("trim_csv", "_trim_snapshot")

    WRIST_DECODE_TOLERATED = (Exception,)

    def __init__(
        self,
        dataset_dir,
        *,
        unify_action: bool = False,
        unify_action_map=None,
        trim_csv=None,
        _trim_snapshot=None,
        **kwargs,
    ):
        """Detect dexterous-hand + unify buckets and wire finger encoding.

        Default (unify off, or grippered bucket): byte-identical to the inherited
        behavior. When ``unify_action=True`` AND this bucket is dexterous-hand
        (``eef_sim_pose_*`` present but no ``gripper_open_scale_*``), the reader
        emits a wider raw action ``[pose(18) + fingers(kL+kR)]`` and a per-bucket
        ``unify_action_map`` that scatters fingers into the dedicated 80-D hand
        slots [10:34)/[44:68) (overriding any config map). Requires unify so the
        variable-width raw collapses to the uniform 80-D head.

        ``trim_csv``: optional path to a quality-audit trim list; see
        :meth:`_filter_episodes`. ``None`` (default) disables trimming.
        """

        if is_robocoin_bucket_excluded(dataset_dir):
            exclusion = ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS[Path(dataset_dir).name]
            raise DataContractError(
                f"RoboCOIN bucket {Path(dataset_dir).name!r} is excluded as a whole: {exclusion['reason']}"
            )
        self._trim_csv = trim_csv
        trim_snapshot_was_provided = _trim_snapshot is not None
        if _trim_snapshot is not None:
            if trim_csv is None or not isinstance(_trim_snapshot, _TrimSnapshot):
                raise ValueError("_trim_snapshot requires a matching non-null trim_csv")
            if _trim_snapshot.path != str(trim_csv):
                raise ValueError(
                    f"_trim_snapshot path {_trim_snapshot.path!r} does not match trim_csv {str(trim_csv)!r}"
                )
        if _trim_snapshot is None and trim_csv is not None:
            _trim_snapshot = _load_trim_snapshot(trim_csv)

            _trim_snapshot.population_validation_cache.clear()
        self._trim_snapshot = _trim_snapshot
        self._dex_unify = False
        self._k_left = 0
        self._k_right = 0
        if unify_action:
            try:
                with open(Path(dataset_dir) / "meta" / "info.json") as f:
                    features = json.load(f).get("features", {})
            except (OSError, ValueError):
                features = {}
            layout = dex_finger_layout(features)
            if layout is not None:
                aL, aR, sL, sR = layout
                kL, kR = len(aL), len(aR)
                self._dex_unify = True
                self._k_left, self._k_right = kL, kR
                self._fidx_act = (np.asarray(aL, dtype=np.int64), np.asarray(aR, dtype=np.int64))
                self._fidx_state = (np.asarray(sL, dtype=np.int64), np.asarray(sR, dtype=np.int64))

                self.ACTION_DIM = 18 + kL + kR
                unify_action_map = _build_dex_unify_map(kL, kR)
            elif "eef_sim_pose_action" in features and not all(c in features for c in _GRIP_COLS):
                aL, aR = _finger_indices(features.get("action", {}))
                sL, sR = _finger_indices(features.get("observation.state", {}))
                logger.warning(
                    "RoboCOIN %s: dexterous-hand bucket (pose, no gripper) but finger layout "
                    "failed the gate (action L/R=%d/%d, state L/R=%d/%d, max=%d); falling back "
                    "to pose-only under unify_action (fingers dropped).",
                    dataset_dir,
                    len(aL),
                    len(aR),
                    len(sL),
                    len(sR),
                    MAX_HAND_DOF,
                )
        super().__init__(dataset_dir, unify_action=unify_action, unify_action_map=unify_action_map, **kwargs)
        if self._trim_snapshot is not None and not trim_snapshot_was_provided:
            _assert_trim_snapshot_current(self._trim_snapshot, context="reader construction")
        if self._trim_csv is not None:
            exclusion_snapshots = getattr(
                self,
                "_stats_exclusion_snapshots",
                (self._excluded_episodes_snapshot,),
            )
            for exclusion_snapshot in exclusion_snapshots:
                assert_excluded_episodes_snapshot_current(
                    exclusion_snapshot,
                    context="reader construction",
                )
            if hasattr(self, "_stats_exclusion_snapshots"):
                del self._stats_exclusion_snapshots

        self._trim_snapshot = None

    def _get_trim_snapshot(self):
        """Pin one snapshot for every stage of this reader construction."""
        if self._trim_csv is None:
            return None
        snapshot = getattr(self, "_trim_snapshot", None)
        if snapshot is None:
            snapshot = _load_trim_snapshot(self._trim_csv)
            self._trim_snapshot = snapshot
        return snapshot

    def _load_excluded_episode_indices(self) -> set[int]:
        """Pin the exact exclusion population consumed by filtering and stats."""
        self._excluded_episodes_snapshot = load_excluded_episodes_snapshot(self._dataset_dir)
        return set(self._excluded_episodes_snapshot.episode_indices)

    def _resolve_cameras(self, info: dict):
        """Resolve cameras by info.features priority, record robot_type, and
        detect whether this bucket carries unified gripper columns.

        Dexterous-hand buckets have no ``gripper_open_scale_*`` columns; for
        those we shadow ``NEEDED_COLS`` (drop the grip columns so the parquet
        read doesn't fail) and set an instance ``ACTION_DIM_MASK`` that masks
        the two grip slots out of supervision. The base ``__init__`` reads
        ``self.ACTION_DIM_MASK`` after this hook (incl. the unify-mask build),
        so setting it here is the supported extension point.
        """

        features = info.get("features", {})
        head, left_wrist, right_wrist = _resolve_robocoin_cameras(features)
        if head is None:
            raise ValueError(f"No head camera found in {self._dataset_id}")
        self._robot_type = info.get("robot_type", "unknown")
        self._has_grip = all(c in features for c in _GRIP_COLS)
        if self._has_grip:
            self.NEEDED_COLS = _NEEDED_COLS
        elif self._dex_unify:
            self.NEEDED_COLS = _DEX_UNIFY_COLS
        else:
            self.NEEDED_COLS = _BASE_COLS
            self.ACTION_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
        return head, left_wrist, right_wrist

    def _filter_episodes(self, eps_df):
        """Apply optional head/tail trim rules without breaking alignment.

        Data-row and every camera-frame offset move by the same integer head
        count, and length becomes the retained ``[head, tail)`` span.  The
        method never removes mid-episode pauses.  A trim entry is accepted only
        when its recorded manifest identity and frame count still match; the
        same retained-population contract is required by stats generation.
        """

        eps_df = super()._filter_episodes(eps_df)
        if self._trim_csv is None:
            return eps_df
        spec = self._get_trim_snapshot().spec.get(self._dataset_id)
        if not spec:
            if hasattr(self, "_trim_spans"):
                del self._trim_spans
            return eps_df

        trim_spans = getattr(self, "_trim_spans", None)
        if trim_spans is None:
            trim_spans = _validate_trim_manifest(self._dataset_id, eps_df, spec)
        else:
            del self._trim_spans

        episode_indices = eps_df["episode_index"].to_numpy()
        lengths = eps_df["length"].to_numpy().copy()

        cam_cols = [c for c in eps_df.columns if c.startswith("_video_frame_offset/")]

        row_off = eps_df["_data_row_offset"].to_numpy().copy()
        cam_off = {c: eps_df[c].to_numpy().copy() for c in cam_cols}

        min_len = _TRIM_MIN_LEN

        keep = np.ones(len(eps_df), dtype=bool)
        n_trim = n_degenerate = 0
        frames_before = int(lengths.sum())
        for pos, ep in enumerate(episode_indices):
            span = trim_spans.get(int(ep))
            if span is None:
                continue
            head, tail = span
            length = int(lengths[pos])
            if head == 0 and tail == length:
                continue
            if tail - head < min_len:
                keep[pos] = False
                n_degenerate += 1
                continue
            lengths[pos] = tail - head
            if head:
                row_off[pos] += head
                for c in cam_cols:
                    cam_off[c][pos] += head
            n_trim += 1

        if n_degenerate:
            logger.warning(
                "RoboCOIN(%s): dropped %d episode(s) whose trim leaves < %d frames.",
                self._dataset_id,
                n_degenerate,
                min_len,
            )
        if not n_trim and not n_degenerate:
            return eps_df

        eps_df = eps_df.copy()
        eps_df["length"] = lengths
        eps_df["_data_row_offset"] = row_off
        for c in cam_cols:
            eps_df[c] = cam_off[c]
        eps_df = eps_df[keep].reset_index(drop=True)
        after = int(lengths[keep].sum())
        logger.info(
            "RoboCOIN(%s): trimmed %d and dropped %d/%d episodes, %d -> %d frames (-%.1f%%, %.2f h removed)",
            self._dataset_id,
            n_trim,
            n_degenerate,
            len(keep),
            frames_before,
            after,
            100.0 * (frames_before - after) / max(frames_before, 1),
            (frames_before - after) / self._fps / 3600.0,
        )
        return eps_df

    def _add_data_offsets(self, eps) -> None:
        if self._trim_csv is not None:
            spec = self._get_trim_snapshot().spec.get(self._dataset_id, {})
            self._trim_spans = _validate_trim_manifest(self._dataset_id, eps, spec)

        self._add_data_offsets_from_files(eps)

    def _add_data_offsets_from_files(self, eps):
        """Resolve data parquet file + row offset from actual parquet row counts.

        Some RoboCOIN episode metadata has stale ``data/file_index`` values at
        file boundaries. The parquet files themselves are globally contiguous
        and ``dataset_from_index`` is correct, so use real file row counts as
        the source of truth for data lookup.

        Reads parquet metadata in parallel to avoid a serial startup bottleneck
        when a bucket has many parquet shards.
        """

        paths = _discover_data_parquets(self._dataset_dir, self._data_path_template)

        def _read_meta(entry):
            path, chunk_index, file_index = entry
            return (chunk_index, file_index, pq.ParquetFile(path).metadata.num_rows)

        if not paths:
            raise FileNotFoundError(f"No data parquet files under {self._dataset_dir}/data")

        n_workers = min(len(paths), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            data_files = list(pool.map(_read_meta, paths))

        starts = np.concatenate([[0], np.cumsum([n for _, _, n in data_files])]).astype(np.int64)
        global_starts = eps["dataset_from_index"].to_numpy().astype(np.int64)
        file_pos = np.searchsorted(starts, global_starts, side="right") - 1
        if (file_pos < 0).any() or (file_pos >= len(data_files)).any():
            raise ValueError(f"{self._dataset_id}: dataset_from_index outside data parquet row range")

        chunks = np.array([data_files[i][0] for i in file_pos], dtype=np.int64)
        files = np.array([data_files[i][1] for i in file_pos], dtype=np.int64)
        eps["data/chunk_index"] = chunks
        eps["data/file_index"] = files
        eps["_data_row_offset"] = global_starts - starts[file_pos]

    def _load_stats(self, info: dict):
        """Load per-robot-type stats (``../meta/stats_<robot_type>.json`` 'eef')."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir.parent / "meta" / f"stats_{self._robot_type}.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but stats file is missing: {stats_path}. "
                f"Run python -m openwam.dataloader.utils.stats_computation.robocoin_stats_computation to generate it, or set normalize_mode=null."
            )
        with open(stats_path) as f:
            raw = json.load(f)
        _validate_whole_bucket_exclusions_provenance(
            raw.get("whole_bucket_exclusions_provenance"),
            robot_type=self._robot_type,
            stats_path=stats_path,
        )
        has_trim_provenance = "trim_provenance" in raw
        has_excluded_provenance = "excluded_episodes_provenance" in raw
        if self._trim_csv is None:
            if has_trim_provenance or has_excluded_provenance:
                raise DataContractError(
                    f"RoboCOIN bucket {self._dataset_id}: stats {stats_path} were computed for a "
                    "trimmed/filtered population but trim_csv is disabled. Regenerate legacy "
                    "untrimmed stats or configure the matching trim_csv."
                )
        else:
            expected_provenance = self._get_trim_snapshot().provenance
            actual_provenance = raw.get("trim_provenance")
            if not has_trim_provenance or actual_provenance != expected_provenance:
                raise DataContractError(
                    f"RoboCOIN bucket {self._dataset_id}: stats {stats_path} trim_provenance "
                    f"does not exactly match trim_csv {self._trim_csv}; expected "
                    f"{expected_provenance}, got {actual_provenance}. Regenerate stats with the "
                    "configured trim CSV."
                )
            exclusion_snapshots = _validate_excluded_episodes_provenance(
                raw.get("excluded_episodes_provenance"),
                dataset_dir=self._dataset_dir,
                dataset_id=self._dataset_id,
                current_snapshot=self._excluded_episodes_snapshot,
                num_datasets=raw.get("eef", {}).get("num_datasets"),
                stats_path=stats_path,
            )
            _validate_stats_population_provenance(
                raw.get("population"),
                dataset_dir=self._dataset_dir,
                current_dataset_id=self._dataset_id,
                current_info=info,
                trim_snapshot=self._get_trim_snapshot(),
                exclusion_snapshots=exclusion_snapshots,
                num_datasets=raw.get("eef", {}).get("num_datasets"),
                stats_path=stats_path,
            )
            self._stats_exclusion_snapshots = tuple(exclusion_snapshots.values())
        eef_stats = materialize_eef_stats(
            raw.get("eef", {}),
            self._normalize_mode,
            dim=_ACTION_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: eef.* — re-run python -m openwam.dataloader.utils.stats_computation.robocoin_stats_computation",
            force_rot6d_identity=True,
        )

        for k in ("mean", "std", "min", "max", "q01", "q99"):
            if eef_stats[k].shape[0] != _ACTION_DIM:
                raise ValueError(
                    f"RoboCOIN bucket {self._dataset_id}: 'eef' stats '{k}' width "
                    f"{eef_stats[k].shape[0]} in {stats_path} != expected {_ACTION_DIM}. "
                    f"Re-run robocoin_stats_computation."
                )
        if not self._dex_unify:
            return eef_stats

        hand_raw = raw.get("hand")
        if not hand_raw:
            raise FileNotFoundError(
                f"unify_action + dexterous-hand bucket {self._dataset_id} needs a 'hand' stats "
                f"block in {stats_path}; re-run robocoin_stats_computation (it now emits hand stats)."
            )
        kL, kR = self._k_left, self._k_right

        dof_l, dof_r = hand_raw.get("dof_left"), hand_raw.get("dof_right")
        if dof_l is None or dof_r is None:
            raise ValueError(
                f"unify_action + dexterous-hand bucket {self._dataset_id}: 'hand' stats block in "
                f"{stats_path} is missing dof_left/dof_right (got left={dof_l}, right={dof_r}); "
                f"re-run robocoin_stats_computation."
            )
        if dof_l != kL or dof_r != kR:
            raise ValueError(
                f"unify_action + dexterous-hand bucket {self._dataset_id}: 'hand' stats DOF "
                f"(left={dof_l}, right={dof_r}) in {stats_path} != this bucket's finger DOF "
                f"(left={kL}, right={kR}). The per-robot-type 'hand' block is locked to the first "
                f"dataset's DOF; re-run robocoin_stats_computation (it now hard-fails on mixed DOF), "
                f"or split mismatched datasets into distinct robot_types."
            )
        hand_stats = materialize_eef_stats(
            hand_raw, self._normalize_mode, dim=kL + kR, strict_minmax=False, source_hint=f"{stats_path}: hand.*"
        )

        for k in ("mean", "std", "min", "max", "q01", "q99"):
            if hand_stats[k].shape[0] != kL + kR:
                raise ValueError(
                    f"unify_action + dexterous-hand bucket {self._dataset_id}: 'hand' stats '{k}' "
                    f"width {hand_stats[k].shape[0]} in {stats_path} != expected kL+kR={kL + kR}. "
                    f"Re-run robocoin_stats_computation."
                )
        combined = {}
        for k in ("mean", "std", "min", "max", "q01", "q99"):
            e, h = eef_stats[k], hand_stats[k]
            combined[k] = np.concatenate([e[0:9], h[0:kL], e[10:19], h[kL : kL + kR]]).astype(np.float32)
        return combined

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply per-bucket normalization to a (..., 20) array.

        Thin wrapper around :func:`apply_normalization` — kept on the class to
        preserve the call sites in the action/proprio hooks and the test suite.
        Returns the input untouched when ``normalize_mode`` is null / stats
        were not loaded.
        """

        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _grip_or_zeros(self, win, col: str, n: int) -> np.ndarray:
        """Stack ``n`` rows of a gripper column, or zeros for dex-hand buckets.

        Dex-hand buckets lack ``gripper_open_scale_*``; the (n, 2) zero fill
        flows through ``eef14_to_eef20`` into the two grip slots, which are
        masked out of supervision by ``GRIP_EXCLUDED_DIM_MASK``.
        """

        if self._has_grip:
            return np.stack(win[col].values[:n]).astype(np.float32)
        return np.zeros((n, 2), dtype=np.float32)

    def _dex_raw(self, eef12: np.ndarray, raw_arr: np.ndarray, fidx) -> np.ndarray:
        """Assemble + normalize a dex-hand raw action/state vector.

        ``[L_pos(3), L_rot6d(6), L_fingers(kL), R_pos(3), R_rot6d(6), R_fingers(kR)]``
        — pose from ``eef_sim_pose`` (grip dropped), fingers sliced from the raw
        ``action`` / ``observation.state`` array. Width = 18 + kL + kR; the base
        unify scatter places these into the 80-D pose + hand slots.
        """

        pose20 = eef14_to_eef20(eef12, np.zeros((len(eef12), 2), dtype=np.float32))
        l_pose, r_pose = pose20[:, 0:9], pose20[:, 10:19]
        fL, fR = fidx
        l_fing = raw_arr[:, fL].astype(np.float32)
        r_fing = raw_arr[:, fR].astype(np.float32)
        raw = np.concatenate([l_pose, l_fing, r_pose, r_fing], axis=-1)
        return self._normalize_array(raw)

    def _action_20d(self, win) -> np.ndarray:
        eef_action = np.stack(win["eef_sim_pose_action"].values).astype(np.float32)
        if self._dex_unify:
            raw_arr = np.stack(win["action"].values).astype(np.float32)
            return self._dex_raw(eef_action, raw_arr, self._fidx_act)
        grip_action = self._grip_or_zeros(win, "gripper_open_scale_action", len(eef_action))
        return self._normalize_array(eef14_to_eef20(eef_action, grip_action))

    def _proprio_20d(self, win) -> np.ndarray:
        eef_state = np.stack(win["eef_sim_pose_state"].values[:1]).astype(np.float32)
        if self._dex_unify:
            raw_arr = np.stack(win["observation.state"].values[:1]).astype(np.float32)
            return self._dex_raw(eef_state, raw_arr, self._fidx_state)
        grip_state = self._grip_or_zeros(win, "gripper_open_scale_state", len(eef_state))
        return self._normalize_array(eef14_to_eef20(eef_state, grip_state))

    @property
    def robot_type(self):
        return self._robot_type

    @classmethod
    def from_config(cls, config, split: str = "train"):
        """Validate global QC inputs and exclude whole buckets before fan-out."""
        from openwam.dataloader.utils import get_cfg

        dataset_dir = get_cfg(config, "dataset_dir")
        trim_csv = get_cfg(config, "trim_csv")
        snapshot = None
        if trim_csv is not None and dataset_dir is not None:
            root = Path(dataset_dir)
            if root.is_dir():
                snapshot = _load_trim_snapshot(trim_csv)
                snapshot.population_validation_cache.clear()
                spec = snapshot.spec
                if not (root / "meta" / "info.json").is_file():
                    bucket_names = {
                        path.name
                        for path in root.iterdir()
                        if (
                            path.is_dir()
                            and (path / "meta" / "info.json").is_file()
                            and not is_robocoin_bucket_excluded(path)
                        )
                    }
                    if bucket_names and bucket_names.isdisjoint(spec):
                        raise ValueError(
                            f"RoboCOIN trim_csv {trim_csv}: none of its {len(spec)} dataset key(s) "
                            f"match any bucket directory under {root}; trimming would silently no-op."
                        )
        if snapshot is not None:
            config = _TrimSnapshotConfig(config, snapshot)

        root = Path(dataset_dir) if dataset_dir is not None else None

        if root is None or (root / "meta" / "info.json").is_file():
            dataset = super().from_config(config, split=split)
        else:
            if not root.is_dir():
                raise FileNotFoundError(f"{cls.__name__}: {root} does not exist")
            discovered = sorted(
                path for path in root.iterdir() if path.is_dir() and (path / "meta" / "info.json").is_file()
            )
            excluded = [path for path in discovered if is_robocoin_bucket_excluded(path)]
            sub_dirs = [path for path in discovered if not is_robocoin_bucket_excluded(path)]
            if excluded:
                logger.info(
                    "%s.from_config: excluding %d whole bucket(s): %s",
                    cls.__name__,
                    len(excluded),
                    ", ".join(path.name for path in excluded),
                )
            if not sub_dirs:
                raise FileNotFoundError(f"{cls.__name__}: no non-excluded sub-buckets with meta/info.json under {root}")

            missing = object()
            common = {"split": split}
            for key in cls.CONFIG_KEYS:
                value = get_cfg(config, key, missing)
                if value is missing:
                    continue
                if value is None and key != "normalize_mode":
                    continue
                common[key] = value
            base_seed = int(get_cfg(config, "seed", 42))
            total_hours = get_cfg(config, "total_hours")
            dataset = build_multibucket(
                cls,
                sub_dirs,
                common,
                base_seed=base_seed,
                total_hours=total_hours,
                wrapper_cls=MultiRobotCOINDataset,
                source_name=cls.__name__,
            )
        if snapshot is not None:
            _assert_trim_snapshot_current(snapshot, context="from_config reader construction")
        return dataset

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiRobotCOINDataset


class MultiRobotCOINDataset(MultiLeRobotV3Reader):
    """Aggregate of N RoboCOIN per-task buckets across multiple robot types."""

    def __init__(self, buckets: List[RoboCOINDataset]):
        super().__init__(buckets)
        robot_types = set(b.robot_type for b in self._buckets)
        logger.info(
            "MultiRobotCOINDataset: %d datasets, %d windows, %d robot types: %s",
            len(self._buckets),
            len(self),
            len(robot_types),
            sorted(robot_types),
        )

    @property
    def action_dim(self):
        return self._buckets[0].action_dim if self._buckets else _ACTION_DIM

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return RoboCOINDataset.from_config(config, split)


__all__ = [
    "RoboCOINDataset",
    "MultiRobotCOINDataset",
    "ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS",
    "is_robocoin_bucket_excluded",
    "robocoin_bucket_exclusions_provenance",
]
