"""InternData-A1 LeRobot v3 reader with real action/state supervision.

The reader auto-detects single-arm and bimanual schemas from ``info.features``.
Pose fields are ``xyz + quaternion(wxyz)`` in the shared robot frame and refer
to the rigid arm-side controller attachment, before any task-specific TCP.
Quaternions are explicitly reordered to xyzw before conversion to rot6d.

The canonical EEF layout is ``[L_xyz(3), L_rot6d(6), L_grip,
R_xyz(3), R_rot6d(6), R_grip]``.  Single-arm Franka buckets occupy the left
half and mask the zero-padded right half.  Gripper positions are divided by
the declared full-open stroke so the emitted value is an aperture fraction in
``[0, 1]``; supported hardware variants are detected from bucket statistics.

Actions are next-state relabels and are read row-aligned.  Episode-terminal
clamped actions are excluded from supervision.  Normalization is pooled per
embodiment, with rot6d and padded dimensions pinned to identity.  Bucket
discovery is recursive because published task buckets have variable depth.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    assert_unit_quaternion,
    quat_wxyz_to_rot6d,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError, build_multibucket
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

_ACTION_DIM = EEF_DIM


# Distinguish a missing config key from one explicitly set to null.
_CONFIG_SENTINEL = object()

# Public info.json robot_type values mapped to stats-file suffixes.
ROBOT_TYPE_TO_EMBODIMENT: Dict[str, str] = {
    "Franka": "franka",
    "ARX Lift-2": "lift2",
    "Genie-1": "genie1",
    "AgileX Split Aloha": "split_aloha",
}


# Convert native gripper positions to aperture fractions (0 closed, 1 open).
GRIPPER_FULL_OPEN = {
    "franka": 0.08,
    "lift2": 0.088,
    "genie1": 5.74,
    "split_aloha": 0.1,
}

# Some embodiments publish more than one supported gripper stroke.
GRIPPER_ALT_FULL_OPEN = {
    "franka": 1.0,
}


# Values far beyond the selected stroke are surfaced but remain non-fatal.
_GRIPPER_SANE_MAX = 1.25


def resolve_gripper_scale(bucket_dir: Path, embodiment: str, grip_col: str) -> float:
    """Return the divisor mapping ``grip_col`` in this bucket onto [0, 1].

    Reads the bucket's own ``meta/stats.json`` (published with every LeRobot v3
    bucket, so this costs one small JSON read at construction). For an embodiment
    with a single gripper this is just the declared stroke. For one shipping two
    variants (franka: panda 0.08 m vs Robotiq 1.0) the observed max is matched to
    whichever declared stroke it is closer to **in log space** — the two differ by
    12.5x, so the assignment is unambiguous even for a bucket that only ever
    half-opens (0.04 is 2x from 0.08 but 25x from 1.0).

    Falls back to the primary stroke when stats are missing or the column is
    absent, which is the majority regime for every embodiment.

    Selecting the alt stroke is never silent — it rescales the whole bucket's
    gripper dim by 12.5x off a single order statistic, so it is logged, and
    logged at WARNING when the bucket's ``mean`` does not corroborate the pick.
    """

    primary = GRIPPER_FULL_OPEN.get(embodiment, 1.0)
    alt = GRIPPER_ALT_FULL_OPEN.get(embodiment)
    stats_path = bucket_dir / "meta" / "stats.json"
    try:
        with open(stats_path) as f:
            blk = json.load(f)[grip_col]
        observed_max = float(np.ravel(blk["max"])[0])

        try:
            observed_mean = float(np.ravel(blk["mean"])[0])
        except (KeyError, TypeError, IndexError, ValueError):
            observed_mean = float("nan")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        logger.debug(
            "InternData-A1: no usable %s max in %s; assuming the %s stroke %.4f",
            grip_col,
            stats_path,
            embodiment,
            primary,
        )
        return primary

    scale = primary
    if alt is not None and observed_max > 0:
        if abs(np.log(observed_max / alt)) < abs(np.log(observed_max / primary)):
            scale = alt

            if observed_mean > primary:
                logger.info(
                    "InternData-A1 %s: %s max %.4f -> alt (second-variant) stroke %.4f for %r "
                    "(mean %.4f corroborates).",
                    bucket_dir,
                    grip_col,
                    observed_max,
                    alt,
                    embodiment,
                    observed_mean,
                )
            else:
                logger.warning(
                    "InternData-A1 %s: %s max %.4f selected the alt stroke %.4f for %r, but mean "
                    "%.4f does not clear the primary stroke %.4f — if that max is a single glitch "
                    "row this bucket's gripper is being squashed by %.1fx. Verify the bucket.",
                    bucket_dir,
                    grip_col,
                    observed_max,
                    alt,
                    embodiment,
                    observed_mean,
                    primary,
                    alt / primary,
                )

    if observed_max / scale > _GRIPPER_SANE_MAX:
        logger.warning(
            "InternData-A1 %s: raw source stats %s max %.4f is %.2fx the assumed "
            "full-open stroke %.4f for %r — the source stats may include excluded/trimmed "
            "rows; otherwise this is an out-of-range outlier or GRIPPER_FULL_OPEN needs updating.",
            bucket_dir,
            grip_col,
            observed_max,
            observed_max / scale,
            scale,
            embodiment,
        )
    return scale


_BIMANUAL_COLS: Tuple[str, ...] = (
    "states.left_ee_to_robot_pose",
    "states.left_gripper.position",
    "states.right_ee_to_robot_pose",
    "states.right_gripper.position",
    "actions.left_ee_to_robot_pose",
    "actions.left_gripper.position",
    "actions.right_ee_to_robot_pose",
    "actions.right_gripper.position",
    "task_index",
)

_SINGLE_ARM_COLS: Tuple[str, ...] = (
    "states.ee_to_robot_pose",
    "states.gripper.position",
    "actions.ee_to_robot_pose",
    "actions.gripper.position",
    "task_index",
)


_BIMANUAL_SIDES = {
    "state": (
        ("states.left_ee_to_robot_pose", "states.left_gripper.position"),
        ("states.right_ee_to_robot_pose", "states.right_gripper.position"),
    ),
    "action": (
        ("actions.left_ee_to_robot_pose", "actions.left_gripper.position"),
        ("actions.right_ee_to_robot_pose", "actions.right_gripper.position"),
    ),
}
_SINGLE_ARM_SIDES = {
    "state": (("states.ee_to_robot_pose", "states.gripper.position"), None),
    "action": (("actions.ee_to_robot_pose", "actions.gripper.position"), None),
}


_HEAD_CAMERA = "images.rgb.head"

_LEFT_WRIST_BIMANUAL = "images.rgb.hand_left"
_RIGHT_WRIST_BIMANUAL = "images.rgb.hand_right"
_WRIST_SINGLE_ARM = "images.rgb.hand"


def detect_arm_layout(features: Dict[str, Any]) -> str:
    """Return ``"bimanual"`` or ``"single_arm"`` from an ``info.features`` dict.

    Detection keys off the pose feature this reader actually consumes rather
    than ``robot_type``, so a bucket with an unlisted robot_type still loads as
    long as its schema is one of the two known shapes.
    """

    if "states.left_ee_to_robot_pose" in features and "states.right_ee_to_robot_pose" in features:
        return "bimanual"
    if "states.ee_to_robot_pose" in features:
        return "single_arm"
    raise ValueError(
        "InternData-A1: info.features exposes neither the bimanual "
        "(states.left_ee_to_robot_pose + states.right_ee_to_robot_pose) nor the "
        "single-arm (states.ee_to_robot_pose) EEF schema; got keys "
        f"{sorted(k for k in features if k.startswith('states.'))}"
    )


def embodiment_key(robot_type: str, arm_layout: str) -> str:
    """Map ``info.robot_type`` to the short key used in the stats filename.

    Unknown robot types fall back to a slug of the raw string so a newly added
    embodiment gets its own stats bucket instead of silently borrowing another
    one's scale. The fallback is logged because it means the stats-computation
    script must be re-run to produce the matching file.
    """

    if robot_type in ROBOT_TYPE_TO_EMBODIMENT:
        return ROBOT_TYPE_TO_EMBODIMENT[robot_type]
    slug = "".join(c if c.isalnum() else "_" for c in str(robot_type).lower()).strip("_") or "unknown"
    logger.warning(
        "InternData-A1: unrecognized robot_type %r (%s layout) -> stats key %r. "
        "Add it to ROBOT_TYPE_TO_EMBODIMENT and re-run interndata_a1_stats_computation.",
        robot_type,
        arm_layout,
        slug,
    )
    return slug


def discover_a1_buckets(root: Path) -> List[Path]:
    """Recursively find every LeRobot bucket (a dir holding ``meta/info.json``).

    The A1 tree nests buckets at two different depths (see module docstring), so
    a fixed-depth glob misses half the dataset. Each branch is pruned as soon as
    a bucket is found — buckets never nest — which keeps the walk off the
    ``data/`` and ``videos/`` subtrees where essentially all the inodes live.

    Symlinks ARE followed. Carving a subset out of a large tree by symlinking
    buckets (rather than copying them) is the realistic usage, and the base
    reader's root mode follows symlinks too — ``os.walk``'s ``followlinks=False``
    default would silently skip every such bucket and report the tree as empty.

    A visited ``(st_dev, st_ino)`` set then makes each directory yield at most
    one bucket path. Unguarded, a symlink cycle does NOT hang: each lap adds one
    symlink component until resolution hits ``MAXSYMLINKS`` (40 on Linux) and the
    walk stops on its own — but it emits many aliases of every bucket reachable
    through the cycle, i.e. silently duplicated training data. Which alias
    survives is decided by traversal order, so ``dirnames`` is sorted below: raw
    ``readdir`` order is a filesystem-instance property (ext4 htree hashing is
    seeded per mkfs), and without the sort the same tree on two machines can keep
    different aliases — shifting every later bucket's index, hence the per-bucket
    subsample seeds in ``build_multibucket`` and the stats merge order. With it,
    the lexicographically-first path always wins.

    Dot-prefixed directories are skipped: archive extraction stages
    every archive through ``<cat>/<emb>/.partial_<name>/`` and logs into
    ``.extract_logs/``. A SIGKILL / OOM / node preemption bypasses the extractor's
    cleanup entirely, so a half-extracted staging tree whose ``meta/`` was already
    written would otherwise be discovered as a complete bucket and then fail at
    ``__getitem__`` mid-training.
    """

    out: List[Path] = []
    seen: set = set()
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
        except OSError:
            dirnames[:] = []
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            dirnames[:] = []
            continue
        seen.add(key)
        if os.path.isfile(os.path.join(dirpath, "meta", "info.json")):
            out.append(Path(dirpath))
            dirnames[:] = []
            continue

        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("data", "videos"))
    return sorted(out)


_CHUNK_DIR_RE = re.compile(r"chunk-(\d+)")
_SHARD_FILE_RE = re.compile(r"file-(\d+)\.parquet")


def parse_shard_path(path) -> Optional[Tuple[int, int]]:
    """``(chunk, file)`` for a CANONICAL LeRobot shard path, else ``None``.

    ``fullmatch`` on both components: ``file-000.backup.parquet`` matches the
    ``file-*.parquet`` glob but is not a shard. A copy left beside the original
    doubled the stats row count while the reader ignored it.

    Canonical means the name is exactly what ``info.json``'s ``data_path``
    template emits — ``chunk-{chunk_index:03d}/file-{file_index:03d}.parquet``.
    Sampling rebuilds the path from that template, never from the name found on
    disk, so accepting a name the template cannot produce enumerates a file that
    can never be read: an earlier revision took ``file-0.parquet``, scanned its
    rows into the statistics and let the reader construct windows over them,
    and only the first real data load failed — on a *different*, non-existent
    padded path. The round-trip also collapses aliases: ``file-00.parquet`` and
    ``file-000.parquet`` would otherwise both claim shard 0.
    """

    p = Path(path)
    chunk = _CHUNK_DIR_RE.fullmatch(p.parent.name)
    shard = _SHARD_FILE_RE.fullmatch(p.name)
    if chunk is None or shard is None:
        return None
    ci, fi = int(chunk.group(1)), int(shard.group(1))
    if p.parent.name != f"chunk-{ci:03d}" or p.name != f"file-{fi:03d}.parquet":
        return None
    return ci, fi


def iter_data_shards(bucket) -> List[Tuple[int, int, Path]]:
    """Every data shard of ``bucket`` as ``(chunk, file, path)``, in index order.

    Ordered by the PARSED indices, not by path: row offsets accumulate in this
    order, so under a lexical sort an unpadded ``file-10`` would follow
    ``file-1`` and shift every subsequent shard's boundary. A1 ships zero-padded
    names where the two agree, which is exactly why sorting on the parsed value
    costs nothing and removes the dependency on that padding.
    """

    out: List[Tuple[int, int, Path]] = []
    for pth in (Path(bucket) / "data").glob("chunk-*/file-*.parquet"):
        ids = parse_shard_path(pth)
        if ids is not None:
            out.append((ids[0], ids[1], pth))
    return sorted(out, key=lambda t: (t[0], t[1]))


def load_excluded_episodes(bucket) -> set:
    """``meta/excluded_episodes.json`` as a set of ints; empty set when absent.

    Strict by design. The base reader keeps the JSON values verbatim and tests
    them with ``isin`` against an integer ``episode_index`` column, so a file
    holding ``["0"]`` excludes NOTHING there — while an ``int()`` cast in the
    stats generator excluded episode 0. Same file, two populations, no error.

    Rejecting non-integers is the one resolution that cannot drift: it gives
    both sides the same answer by construction rather than by keeping two casts
    in agreement. Every file this repo emits contains plain ints, so nothing
    valid is refused. ``bool`` is excluded explicitly because it passes
    ``isinstance(x, int)`` and would silently become episode 0/1.
    """

    path = Path(bucket) / "meta" / "excluded_episodes.json"
    if not path.is_file():
        return set()
    with open(path) as fh:
        raw = json.load(fh)["episode_indices"]
    out = set()
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(
                f"{path}: episode_indices must be JSON integers, got {x!r} "
                f"({type(x).__name__}). The reader matches these against an integer "
                "episode_index column verbatim, so a quoted or float value excludes "
                "nothing there while excluding here — the two sides would then "
                "normalise over different populations with no error anywhere."
            )
        out.add(int(x))
    return out


def validate_manifest_ranges(from_idx, to_idx, lengths, episode_idx, who: str) -> None:
    """The manifest's ``[from, to)`` ranges must tile ``[0, total)`` exactly.

    A per-episode capacity check and an equal grand total are BOTH satisfied by
    ranges that overlap: ``[0,4) [2,6) [8,12)`` ends at 12 and every range fits
    inside a 12-row shard, yet episode 1 reads two of episode 0's rows and two
    of its own. The ranges are the only statement of which rows belong to whom,
    so they have to be checked as a partition — individually valid ranges say
    nothing about whether they carve the file up consistently.

    ``length`` is checked against the range width for the same reason: the
    reader uses ``to - from`` to size the window and ``length`` to bound it, so
    the two disagreeing means one of them is describing a different episode.
    """

    from_idx = np.asarray(from_idx, dtype=np.int64)
    to_idx = np.asarray(to_idx, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    episode_idx = np.asarray(episode_idx, dtype=np.int64)

    if (from_idx < 0).any() or (to_idx < from_idx).any():
        i = int(np.flatnonzero((from_idx < 0) | (to_idx < from_idx))[0])
        raise ValueError(
            f"{who}: episode {int(episode_idx[i])} has a malformed manifest range "
            f"[{int(from_idx[i])}, {int(to_idx[i])})."
        )
    bad = np.flatnonzero(to_idx - from_idx != lengths)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{who}: episode {int(episode_idx[i])} spans "
            f"[{int(from_idx[i])}, {int(to_idx[i])}) = {int(to_idx[i] - from_idx[i])} rows but "
            f"declares length={int(lengths[i])}. The reader sizes windows from one and "
            "bounds them with the other, so they must describe the same episode."
        )

    order = np.argsort(from_idx, kind="stable")
    fs, ts = from_idx[order], to_idx[order]
    if fs.size > 1:
        overlap = np.flatnonzero(fs[1:] < ts[:-1])
        if overlap.size:
            k = int(overlap[0])
            raise ValueError(
                f"{who}: episode {int(episode_idx[order][k + 1])} starts at row "
                f"{int(fs[k + 1])} but episode {int(episode_idx[order][k])} runs to "
                f"{int(ts[k])}. Overlapping manifest ranges make two episodes read the "
                "same rows — the totals and the per-episode capacity check both pass "
                "while the data is paired with the wrong episode."
            )


def _shard_episode_bounds(pf) -> Optional[Tuple[int, int]]:
    """Return the min/max ``episode_index`` advertised by an open shard.

    Row-group metadata is preferred.  If statistics are absent, the function
    reads the run-length-encoded ``episode_index`` column instead of treating
    missing evidence as success.  ``None`` means the column itself is absent.
    The result is an envelope check, not proof of row order within the shard.
    """

    try:
        col = pf.schema_arrow.names.index("episode_index")
    except ValueError:
        return None
    lo = hi = None
    meta = pf.metadata
    for g in range(meta.num_row_groups):
        st = meta.row_group(g).column(col).statistics
        if st is None or not st.has_min_max:
            vals = pf.read(columns=["episode_index"]).column("episode_index")
            if len(vals) == 0:
                return None
            return int(min(vals.to_pylist())), int(max(vals.to_pylist()))
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    if lo is None or hi is None:
        return None
    return int(lo), int(hi)


@dataclass(frozen=True)
class _A1TrimSnapshot:
    """One immutable parse of the exact trim CSV bytes used for construction."""

    path: str
    sha256: str
    spec: Dict[str, Dict[int, Tuple[int, Optional[int], Optional[int]]]]

    @property
    def provenance(self) -> dict:
        return {"schema_version": 1, "sha256": self.sha256}


_TRIM_SPEC_CACHE: Dict[str, Tuple[Tuple[int, int, int, int], _A1TrimSnapshot]] = {}


def _trim_file_signature(path: Path) -> Tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _load_trim_snapshot(path) -> _A1TrimSnapshot:
    """Read, hash and parse one trim artifact from the same byte snapshot.

    Root mode constructs many readers, so a successful parse is cached while the
    file's stat signature is unchanged.  The SHA-256 is persisted in normalization
    stats; the reader compares it before accepting those stats.
    """

    key = str(path)
    trim_path = Path(key)
    try:
        signature_before = _trim_file_signature(trim_path)
        cached = _TRIM_SPEC_CACHE.get(key)
        if cached is not None and cached[0] == signature_before:
            return cached[1]
        raw = trim_path.read_bytes()
        signature_after = _trim_file_signature(trim_path)
        if signature_before != signature_after:
            raise DataContractError(
                f"InternDataA1 trim_csv {key} changed while it was being read; retry with an immutable file."
            )

        spec: Dict[str, Dict[int, Tuple[int, Optional[int], Optional[int]]]] = {}
        seen = set()
        n = 0
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        required = {"dataset", "episode_index", "total_frames", "trim_head_to", "trim_tail_from"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"missing required column(s): {missing}")
        for row in reader:

            def _int(name):
                value = (row.get(name) or "").strip()
                return int(value) if value else None

            dataset = (row.get("dataset") or "").strip()
            if not dataset:
                raise ValueError("empty dataset key")
            episode_index = _int("episode_index")
            if episode_index is None or episode_index < 0:
                raise ValueError(f"invalid episode_index={episode_index!r}")
            entry_key = (dataset, episode_index)
            if entry_key in seen:
                raise ValueError(f"duplicate entry for dataset={dataset!r}, episode_index={episode_index}")
            seen.add(entry_key)

            head = _int("trim_head_to") or 0
            tail = _int("trim_tail_from")
            total = _int("total_frames")
            if head < 0 or (tail is not None and tail < 0) or (total is not None and total <= 0):
                raise ValueError(
                    f"invalid bounds for dataset={dataset!r}, episode_index={episode_index}: "
                    f"head={head}, tail={tail}, total={total}"
                )
            if not head and tail is None:
                continue
            spec.setdefault(dataset, {})[episode_index] = (head, tail, total)
            n += 1
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as e:
        raise ValueError(
            f"InternDataA1: trim_csv {key} could not be read ({e}). Fix the path or set "
            "trim_csv=null to run untrimmed — it will not be skipped silently."
        ) from e

    snapshot = _A1TrimSnapshot(
        path=key,
        sha256=hashlib.sha256(raw).hexdigest(),
        spec=spec,
    )
    _TRIM_SPEC_CACHE[key] = (signature_after, snapshot)
    logger.info("InternDataA1: loaded %d trim entries across %d buckets from %s", n, len(spec), key)
    return snapshot


def _load_trim_spec(path) -> Dict[str, Dict[int, Tuple[int, Optional[int], Optional[int]]]]:
    """Compatibility wrapper returning the parsed mapping from a pinned snapshot."""
    return _load_trim_snapshot(path).spec


def _assert_trim_snapshot_current(snapshot: _A1TrimSnapshot, *, context: str) -> None:
    """Fail if the configured trim bytes changed after the snapshot was pinned."""
    try:
        actual_sha256 = hashlib.sha256(Path(snapshot.path).read_bytes()).hexdigest()
    except OSError as e:
        raise DataContractError(f"InternDataA1 trim_csv {snapshot.path} disappeared during {context}: {e}") from e
    if actual_sha256 != snapshot.sha256:
        raise DataContractError(
            f"InternDataA1 trim_csv {snapshot.path} changed during {context}; "
            f"expected sha256={snapshot.sha256}, got {actual_sha256}."
        )


class AmbiguousBucketKey(LookupError):
    """A bare bucket name matches several keys, so the bucket cannot be identified."""


def resolve_bucket_key(keys, dataset_id: str, bucket_dir, *, what: str, source: str) -> Optional[str]:
    """Resolve a bucket onto a key in a mapping keyed by bucket path.

    **Shared by the reader and the stats generator on purpose** — they used to
    resolve differently, so the same trim CSV could be applied by one and skipped
    by the other while digest provenance saw nothing wrong.

    Matching is by EXACT key only, tried against progressively longer tails of
    this bucket's real path. A bare-suffix match is deliberately not used: these
    maps are frequently sparse (a trim CSV lists only trimmed buckets; a coverage
    map only successfully scanned ones), and in a sparse map a unique suffix
    proves nothing about identity. For example, with `good/…/apple`
    scanned and `bad/…/apple` skipped, a reader for `bad` matched the sole
    `apple` suffix and accepted statistics computed entirely from `good`.

    So an exact miss for an already-qualified id stays a miss, and an unqualified
    id is resolved from the directory tree rather than from the map's contents.

    Raises :class:`AmbiguousBucketKey` only when the path itself cannot
    disambiguate — the caller must decide, because silently picking one would
    apply another bucket's numbers and regenerating produces the same names
    (``<cat>/<emb>/<task>/<object>`` is a documented depth, so repeating leaf
    names is expected).
    """

    if dataset_id in keys:
        return dataset_id

    if "/" in dataset_id:
        return None

    parts = Path(bucket_dir).resolve().parts
    for n in range(min(len(parts), 6), 0, -1):
        cand = "/".join(parts[-n:])
        if cand in keys:
            return cand

    leaf = Path(bucket_dir).name
    cands = sorted(k for k in keys if k == leaf or k.endswith("/" + leaf))
    if len(cands) > 1:
        raise AmbiguousBucketKey(
            f"{source}: {what} has {len(cands)} buckets whose path ends in {leaf!r} "
            f"({', '.join(cands[:4])}{' ...' if len(cands) > 4 else ''}), and none of them "
            "matches this bucket's own path. Bucket leaf names repeat across tasks, so this "
            "one cannot be identified from its directory name. Pass dataset_id (or point "
            "--dataset_dir at the corpus root) so buckets are keyed by their path relative "
            "to the root, which is what these files key on."
        )
    return None


def resolve_trim_bounds(entry, length: int, min_len: int) -> Optional[Tuple[int, int]]:
    """Decide one episode's kept span, or ``None`` to leave it whole.

    Shared by the reader and the stats script **on purpose**. They must make the
    identical keep/skip call: if the stats path trimmed an episode the reader
    leaves whole, the normalizer would describe a distribution the reader never
    emits. Two copies of these four conditions would drift silently — nothing
    downstream compares them.

    Returns ``None`` when the entry should not be applied at all:
      * stale — recorded ``total_frames`` disagrees with the manifest length
        (the signature of a CSV built against a different corpus version);
      * a no-op (head 0, tail == length);
      * the trim would leave fewer than ``min_len`` frames, i.e. not even one
        usable window — better a whole episode than a degenerate one.
    """

    head, tail_from, total = entry
    if total is not None and int(total) != int(length):
        return None
    tail = int(length) if tail_from is None else min(int(tail_from), int(length))
    head = max(0, min(int(head), tail))
    if head == 0 and tail == int(length):
        return None
    if tail - head < min_len:
        return None
    return head, tail


def effective_a1_population_provenance(eps_df, trim_spec: Optional[Dict[int, Tuple]], min_len: int) -> dict:
    """Digest the exact post-split/post-exclusion episode spans used by A1.

    ``eps_df`` is the reader population immediately before trim application.
    Hashing the global manifest start, raw length and effective relative
    ``[head, tail)`` span detects changes to the manifest mapping, exclusion set
    or trim bounds even when the bucket name and aggregate row count stay the
    same.  The stats generator calls this same function on its selected manifest.
    """

    required = ("episode_index", "dataset_from_index", "length")
    missing = [column for column in required if column not in eps_df.columns]
    if missing:
        raise DataContractError(f"InternDataA1 effective population is missing manifest columns {missing}")

    trim_spec = trim_spec or {}
    records = []
    for _, row in eps_df.iterrows():
        episode_index = int(row["episode_index"])
        global_start = int(row["dataset_from_index"])
        length = int(row["length"])
        if episode_index < 0 or global_start < 0 or length <= 0:
            raise DataContractError(
                "InternDataA1 effective population contains an invalid manifest record: "
                f"episode_index={episode_index}, dataset_from_index={global_start}, length={length}"
            )
        entry = trim_spec.get(episode_index)
        bounds = resolve_trim_bounds(entry, length, min_len) if entry is not None else None
        head, tail = bounds if bounds is not None else (0, length)
        records.append((episode_index, global_start, length, int(head), int(tail)))

    records.sort(key=lambda record: record[0])
    if len({record[0] for record in records}) != len(records):
        raise DataContractError("InternDataA1 effective population has duplicate episode_index values")

    hasher = hashlib.sha256(b"openwam:interndata-a1-effective-population:v1\0")
    num_rows = 0
    for record in records:
        for value in record:
            hasher.update(value.to_bytes(8, "little", signed=False))
        num_rows += record[4] - record[3]
    return {
        "schema_version": 1,
        "sha256": hasher.hexdigest(),
        "num_episodes": len(records),
        "num_rows": int(num_rows),
    }


def _validate_a1_root_stats_contributors(
    root: Path,
    stats_root: Path,
    sub_dirs: List[Path],
) -> None:
    """Bind every pooled per-embodiment stats file to the current root set.

    A leaf-level membership check cannot detect a contributor directory removed
    after stats generation: all surviving leaves still find themselves in the
    file even though its pooled values include the removed bucket. Root mode has
    the authoritative discovered set, so compare it before bucket fan-out.
    """

    current: Dict[str, set[str]] = {}
    for sub in sub_dirs:
        try:
            info = json.loads((sub / "meta" / "info.json").read_text())
            layout = detect_arm_layout(info.get("features", {}) or {})
            embodiment = embodiment_key(info.get("robot_type", "unknown"), layout)
            dataset_id = str(sub.relative_to(root))
        except Exception as exc:
            raise DataContractError(f"InternData-A1 cannot classify pooled stats contributor {sub}: {exc}") from exc
        current.setdefault(embodiment, set()).add(dataset_id)

    for embodiment, current_buckets in sorted(current.items()):
        stats_path = stats_root / "meta" / f"stats_{embodiment}.json"
        try:
            raw = json.loads(stats_path.read_text())
            population = raw["population"]
            contributed = population["buckets"]
            empty = population.get("empty_buckets", [])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise DataContractError(
                f"InternData-A1 pooled stats {stats_path} has no readable contributor population; regenerate stats."
            ) from exc
        if (
            not isinstance(contributed, list)
            or not isinstance(empty, list)
            or not all(isinstance(name, str) and name for name in [*contributed, *empty])
            or len(set(contributed)) != len(contributed)
            or len(set(empty)) != len(empty)
            or set(contributed).intersection(empty)
        ):
            raise DataContractError(
                f"InternData-A1 pooled stats {stats_path} has a malformed contributor set; regenerate stats."
            )
        recorded_buckets = set(contributed) | set(empty)
        if recorded_buckets != current_buckets:
            missing = sorted(current_buckets - recorded_buckets)
            extra = sorted(recorded_buckets - current_buckets)
            raise DataContractError(
                f"InternData-A1 pooled stats {stats_path} contributor set no longer matches "
                f"the current {embodiment} root population (missing={missing}, extra={extra}); "
                "regenerate stats."
            )


class InternDataA1Dataset(LeRobotV3Reader):
    """Single-bucket reader for one InternData-A1 v3.0 task (LeRobot v3).

    Emits the canonical 20-D ``xyz + rot6d + gripper`` bimanual EEF with real
    action / proprio supervision. Bimanual buckets fill all 20 dims; franka
    buckets fill ``[0:10)`` and mask the rest. All window / offset / video /
    prompt machinery is inherited from :class:`LeRobotV3Reader`.
    """

    DATASET_NAME = "InternDataA1"
    ACTION_DIM = _ACTION_DIM

    NEEDED_COLS = _BIMANUAL_COLS
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True
    STATS_DIM = _ACTION_DIM
    STATS_STRICT_MINMAX = False
    DEFAULT_NORMALIZE_MODE = "quantile"

    DEPLOY_ACTION_MODE = None

    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("trim_csv",)

    def __init__(
        self,
        dataset_dir,
        *,
        a1_stats_root: Optional[str] = None,
        trim_csv: Optional[str] = None,
        _trim_snapshot: Optional[_A1TrimSnapshot] = None,
        **kwargs,
    ):
        """
        Args:
            a1_stats_root: directory holding ``meta/stats_<embodiment>.json``.
                Defaults to ``dataset_dir`` itself (single-bucket use); in root
                mode :meth:`from_config` passes the dataset root so every bucket
                shares one per-embodiment stats file.
            trim_csv: optional path to a quality-audit trim list; see
                :meth:`_filter_episodes`. ``None`` (default) disables trimming.
        """

        self._a1_stats_root = Path(a1_stats_root) if a1_stats_root else Path(dataset_dir)
        self._trim_csv = trim_csv
        trim_snapshot_was_provided = _trim_snapshot is not None
        if _trim_snapshot is not None:
            if trim_csv is None or not isinstance(_trim_snapshot, _A1TrimSnapshot):
                raise ValueError("_trim_snapshot requires a matching non-null trim_csv")
            if _trim_snapshot.path != str(trim_csv):
                raise ValueError(
                    f"_trim_snapshot path {_trim_snapshot.path!r} does not match trim_csv {str(trim_csv)!r}"
                )
        elif trim_csv is not None:
            _trim_snapshot = _load_trim_snapshot(trim_csv)
        self._trim_snapshot = _trim_snapshot

        self._a1_excluded_episode_indices = tuple(sorted(load_excluded_episodes(dataset_dir)))
        super().__init__(dataset_dir, **kwargs)
        current_exclusions = tuple(sorted(load_excluded_episodes(dataset_dir)))
        if current_exclusions != self._a1_excluded_episode_indices:
            raise DataContractError(
                f"InternDataA1 bucket {self._dataset_id}: excluded_episodes.json changed during reader "
                f"construction; expected {list(self._a1_excluded_episode_indices)}, "
                f"got {list(current_exclusions)}."
            )
        if self._trim_snapshot is not None and not trim_snapshot_was_provided:
            _assert_trim_snapshot_current(self._trim_snapshot, context="reader construction")

        self._trim_snapshot = None

    def _load_excluded_episode_indices(self) -> set[int]:
        """Use the strict exclusion snapshot pinned before base construction."""
        return set(self._a1_excluded_episode_indices)

    def _resolve_cameras(self, info: dict):
        """Detect the arm layout, then pick cameras and per-bucket columns.

        Runs before the episode index / stats / data reads, which is the
        supported point to set instance ``NEEDED_COLS`` and ``ACTION_DIM_MASK``
        (the base consults both after this hook, including for the unify mask).
        """

        features = info.get("features", {}) or {}
        self._arm_layout = detect_arm_layout(features)
        self._robot_type = info.get("robot_type", "unknown")
        self._embodiment = embodiment_key(self._robot_type, self._arm_layout)

        if self._arm_layout == "bimanual":
            self.NEEDED_COLS = _BIMANUAL_COLS
            self._sides = _BIMANUAL_SIDES

            self.ACTION_DIM_MASK = None
            left_wrist, right_wrist = _LEFT_WRIST_BIMANUAL, _RIGHT_WRIST_BIMANUAL
        else:
            self.NEEDED_COLS = _SINGLE_ARM_COLS
            self._sides = _SINGLE_ARM_SIDES

            self.ACTION_DIM_MASK = LEFT_ARM_DIM_MASK

            left_wrist, right_wrist = _WRIST_SINGLE_ARM, None

        missing = [c for c in self.NEEDED_COLS if c != "task_index" and c not in features]
        if missing:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {self._arm_layout} layout is missing "
                f"feature(s) {missing} in info.json."
            )

        self._grip_scale = tuple(
            resolve_gripper_scale(self._dataset_dir, self._embodiment, spec[1]) if spec is not None else 1.0
            for spec in self._sides["state"]
        )

        head = _HEAD_CAMERA if _HEAD_CAMERA in features else None
        if head is None:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: no {_HEAD_CAMERA!r} camera in info.features "
                f"(has {sorted(k for k in features if k.startswith('images.'))})"
            )

        if left_wrist is not None and left_wrist not in features:
            left_wrist = None
        if right_wrist is not None and right_wrist not in features:
            right_wrist = None
        return head, left_wrist, right_wrist

    def _add_data_offsets(self, eps) -> None:
        """Resolve data-shard and row offsets from physical parquet lengths.

        Manifest file indices can be stale at shard boundaries.  Global
        ``dataset_from_index`` values are therefore located against cumulative
        physical shard lengths.  The method validates manifest ranges, total
        row coverage, per-episode shard capacity, and the shard's advertised
        episode envelope before publishing offsets.
        """

        from concurrent.futures import ThreadPoolExecutor

        import pyarrow.parquet as pq

        shards = iter_data_shards(self._dataset_dir)
        if not shards:
            raise FileNotFoundError(f"No data parquet files under {self._dataset_dir}/data")

        def _read_meta(entry):
            chunk, file_idx, path = entry
            pf = pq.ParquetFile(path)
            return chunk, file_idx, pf.metadata.num_rows, _shard_episode_bounds(pf)

        with ThreadPoolExecutor(max_workers=min(len(shards), 4)) as pool:
            data_files = list(pool.map(_read_meta, shards))

        who = f"{self.DATASET_NAME}({self._dataset_id})"

        validate_manifest_ranges(
            eps["dataset_from_index"].to_numpy(),
            eps["dataset_to_index"].to_numpy(),
            eps["length"].to_numpy(),
            eps["episode_index"].to_numpy(),
            who,
        )

        starts = np.concatenate([[0], np.cumsum([n for _, _, n, _ in data_files])]).astype(np.int64)

        manifest_end = int(eps["dataset_to_index"].to_numpy().max())
        if manifest_end != int(starts[-1]):
            raise ValueError(
                f"{who}: the data shards hold {int(starts[-1])} rows but the manifest ends at "
                f"{manifest_end}. A shard is missing, truncated or out of order — resolving "
                "offsets against this would map episodes onto another episode's rows."
            )

        global_starts = eps["dataset_from_index"].to_numpy().astype(np.int64)
        file_pos = np.searchsorted(starts, global_starts, side="right") - 1
        if (file_pos < 0).any() or (file_pos >= len(data_files)).any():
            raise ValueError(f"{who}: dataset_from_index outside the data parquet row range")

        offsets = global_starts - starts[file_pos]
        lengths = eps["dataset_to_index"].to_numpy().astype(np.int64) - global_starts
        capacity = np.array([data_files[i][2] for i in file_pos], dtype=np.int64)
        overflow = np.flatnonzero(offsets + lengths > capacity)
        if overflow.size:
            i = int(overflow[0])
            raise ValueError(
                f"{who}: episode {int(eps['episode_index'].to_numpy()[i])} needs rows "
                f"[{int(offsets[i])}, {int(offsets[i] + lengths[i])}) of a shard holding only "
                f"{int(capacity[i])}. The shard set does not match the manifest — resolving "
                "offsets against it would map episodes onto another episode's rows."
            )

        ep_vals = eps["episode_index"].to_numpy()
        for i, pos in enumerate(file_pos):
            bounds = data_files[pos][3]
            if bounds is None:
                continue
            ep = int(ep_vals[i])
            if not bounds[0] <= ep <= bounds[1]:
                raise ValueError(
                    f"{who}: the manifest puts episode {ep} in shard "
                    f"chunk-{data_files[pos][0]:03d}/file-{data_files[pos][1]:03d}, but that "
                    f"shard only holds episodes {bounds[0]}..{bounds[1]}. The shard set does "
                    "not match the manifest (a copied, reordered or substituted shard passes "
                    "the row-count checks above while holding another episode's data)."
                )

        eps["data/chunk_index"] = np.array([data_files[i][0] for i in file_pos], dtype=np.int64)
        eps["data/file_index"] = np.array([data_files[i][1] for i in file_pos], dtype=np.int64)
        eps["_data_row_offset"] = offsets

    def _add_episode_offsets(self, eps) -> None:
        """Rebuild camera-frame offsets from file-local timestamps.

        ``round(from_timestamp * fps)`` is independent of which manifest rows
        survived filtering, so dropped episodes cannot shift later video away
        from its action/state rows.  Missing timestamps and invalid fps values
        fail explicitly rather than leaving one camera on a stale offset.
        """

        super()._add_episode_offsets(eps)
        for cam in self._video_cameras():
            col = f"videos/{cam}/from_timestamp"
            if col not in eps.columns:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has no "
                    f"{col!r} in meta/episodes, so its frame offset cannot be resolved."
                )
            if not np.isfinite(self._fps) or self._fps <= 0:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): fps={self._fps!r} is not a "
                    "positive finite number, so timestamps cannot be converted to frames."
                )
            ts = eps[col].to_numpy().astype(np.float64)
            if not np.isfinite(ts).all() or (ts < 0).any():
                bad = int((~np.isfinite(ts)).sum() + (ts < 0).sum())
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has {bad} "
                    "non-finite or negative from_timestamp values; casting those would "
                    "produce INT64_MIN or negative frame indices and read black slots."
                )
            frames = ts * self._fps
            if np.abs(frames - np.rint(frames)).max() > 0.25:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has "
                    "from_timestamp values that are not on frame boundaries at "
                    f"fps={self._fps}; the offsets would be rounded onto neighbouring frames."
                )
            if frames.max() > 2**53:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} timestamp * fps "
                    "exceeds the exactly-representable integer range; the cast would wrap."
                )
            off = np.rint(frames).astype(np.int64)

            ck = f"videos/{cam}/chunk_index"
            fk = f"videos/{cam}/file_index"
            if fk in eps.columns:
                ep_len = eps["length"].to_numpy().astype(np.int64)
                chunk_of = (
                    eps[ck].to_numpy().astype(np.int64) if ck in eps.columns else np.zeros(len(off), dtype=np.int64)
                )
                file_of = eps[fk].to_numpy().astype(np.int64)
                ep_vals = eps["episode_index"].to_numpy()
                shard_key = np.stack([chunk_of, file_of], axis=1)
                for shard in np.unique(shard_key, axis=0):
                    m = np.flatnonzero((chunk_of == shard[0]) & (file_of == shard[1]))
                    order = m[np.argsort(off[m], kind="stable")]
                    for a, b in zip(order, order[1:]):
                        if off[a] + ep_len[a] > off[b]:
                            raise ValueError(
                                f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r}, video "
                                f"shard chunk-{int(shard[0]):03d}/file-{int(shard[1]):03d}: "
                                f"episode {int(ep_vals[a])} covers frames "
                                f"[{int(off[a])}, {int(off[a] + ep_len[a])}) which overlaps "
                                f"episode {int(ep_vals[b])} starting at {int(off[b])}. One of "
                                "them would read the other's frames."
                            )

            tcol = f"videos/{cam}/to_timestamp"
            if tcol in eps.columns:
                span = (eps[tcol].to_numpy().astype(np.float64) - ts) * self._fps
                declared = eps["length"].to_numpy().astype(np.int64)

                bad = np.flatnonzero(~np.isfinite(span) | (np.abs(span - declared) > 0.5))
                if bad.size:
                    i = int(bad[0])
                    raise ValueError(
                        f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} episode "
                        f"{int(eps['episode_index'].to_numpy()[i])} spans {span[i]:.2f} video "
                        f"frames between from/to_timestamp but declares length="
                        f"{int(declared[i])}. The manifest describes two different episodes."
                    )
            eps[self._video_offset_col(cam)] = off

    def _trim_min_len(self) -> int:
        """Minimum frames an episode must keep to be worth trimming.

        Named so `_load_stats` can quote the same number the stats generator
        must have been given (`--min_keep`): the same trim CSV under a different
        bound produces a different kept population.
        """

        return self._num_frames if self._split == "val" else self._train_min_window_len()

    def _match_bucket_key(self, keys, what: str) -> Optional[str]:
        """Map this bucket onto a key in a mapping keyed by bucket path.

        Both the trim CSV's ``dataset`` column and the stats file's
        ``exclusions`` map are keyed by the bucket path **relative to the
        dataset root**, which is exactly what root mode already uses as
        ``dataset_id`` (see ``_per_bucket`` in :meth:`from_config`) — so the
        common case is a direct hit.

        Single-bucket mode is the awkward one: ``_dataset_id`` falls back to the
        bare directory name. A1 bucket names repeat across embodiments
        (``pick_beef_sandwich_on_conveyor`` exists under both lift2 and
        split_aloha), so a bare name is accepted only when it resolves
        uniquely — an ambiguous one is refused rather than guessed, since
        matching the wrong bucket would apply another embodiment's numbers.
        """

        return resolve_bucket_key(
            keys, self._dataset_id, self._dataset_dir, what=what, source=f"InternDataA1({self._dataset_id})"
        )

    def _trim_key(self) -> Optional[str]:
        spec = self._get_trim_snapshot().spec
        if not spec:
            return None
        return self._match_bucket_key(spec, "trim_csv")

    def _get_trim_snapshot(self) -> _A1TrimSnapshot:
        snapshot = getattr(self, "_trim_snapshot", None)
        if snapshot is None:
            if not self._trim_csv:
                raise ValueError("InternDataA1 trim snapshot requested while trim_csv is disabled")
            snapshot = _load_trim_snapshot(self._trim_csv)
            self._trim_snapshot = snapshot
        return snapshot

    def _filter_episodes(self, eps_df):
        """Trim motionless episode heads/tails from an optional trim artifact.

        Trimming adjusts data and video frame offsets together and shortens the
        episode length; it does not splice mid-episode pauses.  Bounds are
        integer frame offsets, avoiding timestamp rounding skew.  Entries whose
        recorded length no longer matches the manifest are rejected as stale,
        and the exact retained population is bound into normalization stats.
        """

        eps_df = super()._filter_episodes(eps_df)
        spec = None
        if self._trim_csv:
            key = self._trim_key()
            if key is not None:
                spec = self._get_trim_snapshot().spec.get(key)

        min_len = self._trim_min_len()

        self._a1_effective_population = effective_a1_population_provenance(
            eps_df,
            spec,
            min_len,
        )
        if not spec:
            return eps_df

        cam_cols = [c for c in eps_df.columns if c.startswith("_video_frame_offset/")]

        lengths = eps_df["length"].to_numpy().copy()
        row_off = eps_df["_data_row_offset"].to_numpy().copy()
        cam_off = {c: eps_df[c].to_numpy().copy() for c in cam_cols}
        n_trim = n_skip = 0
        frames_before = int(lengths.sum())
        for pos, ep in enumerate(eps_df["episode_index"].to_numpy()):
            entry = spec.get(int(ep))
            if entry is None:
                continue
            length = int(lengths[pos])
            bounds = resolve_trim_bounds(entry, length, min_len)
            if bounds is None:
                if not (entry[0] == 0 and entry[1] in (None, length)):
                    n_skip += 1
                continue
            head, tail = bounds
            lengths[pos] = tail - head
            if head:
                row_off[pos] += head
                for c in cam_cols:
                    cam_off[c][pos] += head
            n_trim += 1

        if n_skip:
            logger.warning(
                "InternDataA1(%s): %d trim entries skipped — recorded total_frames disagrees "
                "with the manifest length, or the trim would leave < %d frames. A total_frames "
                "mismatch is what a STALE trim list looks like; regenerate it against this "
                "corpus.",
                self._dataset_id,
                n_skip,
                min_len,
            )
        if not n_trim:
            return eps_df

        eps_df = eps_df.copy()
        eps_df["length"] = lengths
        eps_df["_data_row_offset"] = row_off
        for c in cam_cols:
            eps_df[c] = cam_off[c]
        after = int(lengths.sum())
        logger.info(
            "InternDataA1(%s): trimmed %d/%d episodes, %d -> %d frames (-%.1f%%, %.2f h removed)",
            self._dataset_id,
            n_trim,
            len(eps_df),
            frames_before,
            after,
            100.0 * (frames_before - after) / max(frames_before, 1),
            (frames_before - after) / self._fps / 3600.0,
        )
        return eps_df.reset_index(drop=True)

    def _train_min_window_len(self) -> int:
        """Require >= 2 rows so every train window keeps >= 1 supervised step.

        A 1-row episode would lose its only action step to the episode-boundary
        drop in :meth:`_n_supervised_action_steps` and yield an all-masked sample.
        """

        return 2

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Drop the clamped final action of an episode-truncated window.

        ``actions[t] == states[t+1]`` holds for every row except the episode's
        last, where the relabeling has no successor and repeats the previous
        target instead. ``actual_raw_len < num_frames`` means the window ran into
        the episode end, so its last row carries that fabricated target.
        """

        if actual_raw_len >= self._num_frames:
            return actual_raw_len
        return max(0, actual_raw_len - 1)

    def _check_stats_population(self, raw: dict, stats_path) -> None:
        """Refuse stats computed over a different population than this reader reads.

        Trimmed and untrimmed stats describe different distributions — trimming
        removes the motionless head/tail, which pulls the positional means
        toward the moving span — and NOTHING in the numbers says which is which.
        The same is true of a val-derived file loaded by a train reader, of a
        different ``--min_keep``, and of a bucket that dropped out of the scan
        but still loads the shared per-embodiment file. Each was reproduced end
        to end through the real generator and reader; each mis-scales every
        action for the whole run with no error anywhere.

        The broad policy fields below catch split/trim/min-keep/coverage
        mismatches.  A trimming reader additionally requires schema-v2
        provenance: a hash of the exact trim CSV bytes and a per-bucket digest
        of the selected manifest spans after exclusions and trim resolution.
        The stats worker creates that certificate from the same immutable trim
        snapshot and strict exclusion parser used to select its parquet rows.

        Missing block -> refuse. A stats file that cannot say what it covers is
        exactly the unverifiable pairing this exists to prevent, and the fix is
        one generator re-run.
        """

        pop = raw.get("population")
        rerun = (
            "Re-run python -m openwam.dataloader.utils.stats_computation."
            f"interndata_a1_stats_computation --dataset_dir {self._dataset_dir} "
            f"--stats_root {self._a1_stats_root}"
        )
        if not isinstance(pop, dict):
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} has no 'population' "
                f"block, so there is no way to tell whether it describes the rows this reader "
                f"loads (split, trimming and keep-bound all change the distribution). {rerun}."
            )

        if pop.get("split") != "train":
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} was computed over the "
                f"{pop.get('split')!r} split. Normalization statistics must describe the "
                f"training distribution — a val-derived file would scale training by numbers "
                f"drawn from data the model never fits. {rerun} --split train."
            )

        want_trim = bool(self._trim_csv)
        if bool(pop.get("trim_active")) != want_trim:
            state = "with" if pop.get("trim_active") else "without"
            mine = "is" if want_trim else "is not"
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} was computed {state} a "
                f"trim list but this reader {mine} trimming. Trimming removes the motionless "
                f"head/tail, so the two describe different distributions. {rerun}"
                + (f" --trim_csv {self._trim_csv}" if want_trim else "")
                + "."
            )

        want_keep = self._train_min_window_len()
        try:
            recorded_keep = int(pop.get("min_keep", -1))
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} has invalid "
                f"population.min_keep={pop.get('min_keep')!r}. {rerun}."
            ) from exc
        if want_trim and recorded_keep != int(want_keep):
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} used --min_keep="
                f"{pop.get('min_keep')} but this reader keeps episodes down to {want_keep} "
                f"frames. The same trim CSV under a different bound keeps a different set of "
                f"episodes. {rerun} --trim_csv {self._trim_csv} --min_keep {want_keep}."
            )

        buckets = pop.get("buckets")
        empty_buckets = pop.get("empty_buckets", [])
        if (
            not isinstance(buckets, list)
            or not all(isinstance(key, str) and key for key in buckets)
            or not isinstance(empty_buckets, list)
            or not all(isinstance(key, str) and key for key in empty_buckets)
        ):
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} has no bucket list in "
                f"its 'population' block, or carries malformed bucket identifiers. {rerun}."
            )

        known = list(buckets) + list(empty_buckets)

        try:
            population_key = self._match_bucket_key(dict.fromkeys(known), "stats population")
        except (AmbiguousBucketKey, TypeError, ValueError) as exc:
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: cannot resolve this bucket in the "
                f"stats population from {stats_path}: {exc}. {rerun}."
            ) from exc
        if population_key is None:
            raise DataContractError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} covers {len(known)} "
                f"buckets, none of them this one — the scan refused it (unreadable metadata, "
                f"or shards that disagree with the manifest) while it still loads the shared "
                f"per-embodiment file, so its actions would be scaled by other buckets' "
                f"numbers. {rerun}."
            )

        if want_trim:
            if pop.get("schema_version") != 2:
                raise DataContractError(
                    f"InternData-A1 bucket {self._dataset_id}: {stats_path} predates fail-closed "
                    "trim/exclusion population provenance. Regenerate the normalization stats."
                )
            expected_trim = self._get_trim_snapshot().provenance
            if pop.get("trim_provenance") != expected_trim:
                raise DataContractError(
                    f"InternData-A1 bucket {self._dataset_id}: {stats_path} trim provenance does "
                    f"not match {self._trim_csv}; expected {expected_trim}, got "
                    f"{pop.get('trim_provenance')}. Regenerate normalization stats."
                )

            bucket_provenance = pop.get("bucket_provenance")
            if not isinstance(bucket_provenance, dict) or not all(
                isinstance(key, str) and key for key in bucket_provenance
            ):
                raise DataContractError(
                    f"InternData-A1 bucket {self._dataset_id}: {stats_path} has no per-bucket "
                    "population provenance; regenerate normalization stats."
                )
            try:
                provenance_key = self._match_bucket_key(bucket_provenance, "stats bucket provenance")
            except (AmbiguousBucketKey, TypeError, ValueError) as exc:
                raise DataContractError(
                    f"InternData-A1 bucket {self._dataset_id}: cannot resolve this bucket in "
                    f"{stats_path} population provenance: {exc}. Regenerate normalization stats."
                ) from exc
            if provenance_key is None or not isinstance(bucket_provenance.get(provenance_key), dict):
                raise DataContractError(
                    f"InternData-A1 bucket {self._dataset_id}: {stats_path} has no population "
                    "certificate for this bucket; regenerate normalization stats."
                )
            recorded = bucket_provenance[provenance_key]
            current_exclusions = list(self._a1_excluded_episode_indices)
            if recorded.get("excluded_episode_indices") != current_exclusions:
                raise DataContractError(
                    f"InternData-A1 bucket {self._dataset_id}: excluded_episodes.json changed after "
                    f"{stats_path} was generated; stats recorded "
                    f"{recorded.get('excluded_episode_indices')}, current={current_exclusions}."
                )

            if self._split == "train":
                current_effective = getattr(self, "_a1_effective_population", None)
                if recorded.get("effective_population") != current_effective:
                    raise DataContractError(
                        f"InternData-A1 bucket {self._dataset_id}: effective split/exclusion/trim "
                        f"population changed after {stats_path} was generated; regenerate stats."
                    )

    def _load_stats(self, info: dict):
        """Load per-embodiment 20-D EEF stats (``meta/stats_<embodiment>.json``)."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._a1_stats_root / "meta" / f"stats_{self._embodiment}.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but stats file is missing: {stats_path}. "
                "Run python -m openwam.dataloader.utils.stats_computation."
                "interndata_a1_stats_computation --dataset_dir <extracted-root> "
                f"--stats_root {self._a1_stats_root}, or set normalize_mode=null."
            )
        with open(stats_path) as f:
            raw = json.load(f)

        self._check_stats_population(raw, stats_path)

        eef_raw = raw.get("eef", {})

        for k in ("mean", "std", "min", "max", "q01", "q99"):
            if k not in eef_raw:
                continue
            width = len(eef_raw[k])
            if width != _ACTION_DIM:
                raise ValueError(
                    f"InternData-A1 bucket {self._dataset_id}: 'eef' stats {k!r} width "
                    f"{width} in {stats_path} != expected {_ACTION_DIM}. "
                    "Re-run interndata_a1_stats_computation."
                )
        stats = materialize_eef_stats(
            eef_raw,
            self._normalize_mode,
            dim=_ACTION_DIM,
            strict_minmax=False,
            source_hint=(
                f"{stats_path}: eef.* — re-run python -m openwam.dataloader.utils."
                "stats_computation.interndata_a1_stats_computation"
            ),
            force_rot6d_identity=True,
        )
        return stats

    def _post_init(self, info: dict) -> None:
        """Sanity-check the quaternion magnitude on a small sample of rows.

        Catches a wrong field routed into the pose slot / un-normalized
        quaternions. It canNOT catch a wxyz<->xyzw reorder (both are unit-norm);
        that convention is fixed by the dataset's own ``names`` metadata
        (``quaternion.w`` first) and covered by the reader's unit tests.
        """

        if len(self._eps_df) == 0:
            return
        row = self._eps_df.iloc[0]
        try:
            table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
            pose_col = self._sides["state"][0][0]
            sample = np.stack(table.slice(0, 64).to_pandas()[pose_col].values).astype(np.float32)
        except Exception as e:
            logger.debug("%s(%s): quaternion probe skipped (%s)", self.DATASET_NAME, self._dataset_id, e)
            return
        assert_unit_quaternion(sample[:, 3:7])

    def _arm10(self, win, spec, n: int, grip_scale: float) -> np.ndarray:
        """Build one arm's ``(n, 10)`` ``[xyz(3), rot6d(6), grip(1)]`` block.

        ``grip_scale`` maps the raw gripper column onto a normalized aperture in
        [0, 1] — see :func:`resolve_gripper_scale`.
        """

        pose_col, grip_col = spec
        pose = np.stack(win[pose_col].values[:n]).astype(np.float32)
        grip = np.stack(win[grip_col].values[:n]).astype(np.float32).reshape(n, 1) / grip_scale
        rot6d = quat_wxyz_to_rot6d(pose[:, 3:7])
        return np.concatenate([pose[:, 0:3], rot6d, grip], axis=-1)

    def _eef20(self, win, kind: str, n: int) -> np.ndarray:
        """Assemble the ``(n, 20)`` bimanual EEF for ``kind`` in {state, action}.

        Single-arm buckets fill ``[0:10)`` and leave ``[10:20)`` zero; those dims
        are excluded from supervision by ``ACTION_DIM_MASK`` and pinned to
        identity stats so normalization leaves the zeros untouched.
        """

        left_spec, right_spec = self._sides[kind]
        out = np.zeros((n, _ACTION_DIM), dtype=np.float32)
        out[:, :ARM10_DIM] = self._arm10(win, left_spec, n, self._grip_scale[0])
        if right_spec is not None:
            out[:, ARM10_DIM:] = self._arm10(win, right_spec, n, self._grip_scale[1])
        return out

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply this bucket's per-embodiment normalization to a ``(..., 20)`` array."""
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        n = len(win)
        return self._normalize_array(self._eef20(win, "action", n))

    def _proprio_20d(self, win) -> np.ndarray:
        return self._normalize_array(self._eef20(win, "state", 1))

    @property
    def robot_type(self) -> str:
        return self._robot_type

    @property
    def embodiment(self) -> str:
        return self._embodiment

    @property
    def arm_layout(self) -> str:
        return self._arm_layout

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiInternDataA1Dataset

    @classmethod
    def from_config(cls, config, split: str = "train") -> Any:
        """Build one bucket, or recursively discover every bucket under a root.

        Overrides the base because A1 buckets sit at variable depth (2 or 3
        levels below the embodiment dir), which the base's single-level
        ``iterdir`` scan cannot see. Everything else — the shared window kwargs,
        the ``total_hours`` water-fill, per-bucket failure tolerance — is
        delegated to the same :func:`build_multibucket` the other readers use.
        """

        from openwam.dataloader.utils import get_cfg as _get

        dataset_dir = _get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")
        root = Path(dataset_dir)

        common: Dict[str, Any] = {"split": split}
        for key in cls.CONFIG_KEYS:
            v = _get(config, key, _CONFIG_SENTINEL)
            if v is _CONFIG_SENTINEL:
                continue

            if v is None and key != "normalize_mode":
                continue
            common[key] = v

        stats_root = _get(config, "stats_root") or str(root)
        common["a1_stats_root"] = stats_root
        trim_snapshot = None
        if common.get("trim_csv"):
            trim_snapshot = _load_trim_snapshot(common["trim_csv"])
            common["_trim_snapshot"] = trim_snapshot

        if (root / "meta" / "info.json").is_file():
            kwargs = dict(common)
            dataset_id = _get(config, "dataset_id")
            if dataset_id is not None:
                kwargs["dataset_id"] = dataset_id
            total_hours = _get(config, "total_hours")
            if total_hours is not None:
                kwargs["max_hours"] = float(total_hours)
                kwargs["subsample_seed"] = int(_get(config, "seed", 42))
            dataset = cls(dataset_dir=str(root), **kwargs)
            if trim_snapshot is not None:
                _assert_trim_snapshot_current(trim_snapshot, context="from_config reader construction")
            return dataset

        if not root.is_dir():
            raise FileNotFoundError(f"{cls.__name__}: {root} does not exist")
        sub_dirs = discover_a1_buckets(root)
        if not sub_dirs:
            raise FileNotFoundError(
                f"{cls.__name__}: no buckets with meta/info.json found anywhere under {root}. "
                "Did the tar.gz archives get extracted?"
            )
        logger.info("%s.from_config: root mode, %d buckets under %s", cls.__name__, len(sub_dirs), root)

        normalize_mode = common.get("normalize_mode", cls.DEFAULT_NORMALIZE_MODE)
        if normalize_mode and normalize_mode not in ("none", "null"):
            _validate_a1_root_stats_contributors(root, Path(stats_root), sub_dirs)

        def _per_bucket(sub: Path) -> Dict[str, Any]:
            try:
                return {"dataset_id": str(sub.relative_to(root))}
            except ValueError:
                return {"dataset_id": sub.name}

        dataset = build_multibucket(
            cls,
            sub_dirs,
            common,
            base_seed=int(_get(config, "seed", 42)),
            total_hours=_get(config, "total_hours"),
            wrapper_cls=cls._multibucket_wrapper(),
            source_name=cls.__name__,
            per_bucket_kwargs=_per_bucket,
        )
        if trim_snapshot is not None:
            _assert_trim_snapshot_current(trim_snapshot, context="from_config reader construction")
        return dataset


class MultiInternDataA1Dataset(MultiLeRobotV3Reader):
    """Aggregate InternData-A1 task buckets across supported embodiments."""

    def __init__(self, buckets):
        super().__init__(buckets)
        counts: Dict[str, int] = {}
        windows: Dict[str, int] = {}
        for b in self._buckets:
            emb = getattr(b, "embodiment", "unknown")
            counts[emb] = counts.get(emb, 0) + 1
            windows[emb] = windows.get(emb, 0) + len(b)
        logger.info(
            "MultiInternDataA1Dataset: %d buckets, %d windows | %s",
            len(self._buckets),
            len(self),
            ", ".join(f"{k}: {counts[k]} buckets/{windows[k]} windows" for k in sorted(counts)),
        )
        self._embodiment_bucket_counts = counts

    @property
    def embodiment_bucket_counts(self) -> Dict[str, int]:
        return dict(self._embodiment_bucket_counts)


__all__ = [
    "InternDataA1Dataset",
    "MultiInternDataA1Dataset",
    "ROBOT_TYPE_TO_EMBODIMENT",
    "GRIPPER_FULL_OPEN",
    "GRIPPER_ALT_FULL_OPEN",
    "detect_arm_layout",
    "embodiment_key",
    "discover_a1_buckets",
    "resolve_gripper_scale",
]
