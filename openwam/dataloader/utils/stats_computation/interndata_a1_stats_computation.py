"""Generate per-embodiment 20-D EEF stats for InternData-A1.

The generator uses the reader's schema detection, quaternion conversion,
gripper scaling, exclusions, split selection, and trim bounds so the emitted
vectors match training exactly.  Both action and state streams contribute.
Stats are pooled per embodiment, not per task bucket, and rot6d plus any
single-arm padding dimensions are pinned to identity.

Output files live under ``<stats_root>/meta/stats_<embodiment>.json`` and carry
a certificate for the exact source and retained population.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.interndata_a1 import (
    _BIMANUAL_SIDES,
    _SINGLE_ARM_SIDES,
    _assert_trim_snapshot_current,
    _load_trim_snapshot,
    detect_arm_layout,
    discover_a1_buckets,
    effective_a1_population_provenance,
    embodiment_key,
    iter_data_shards,
    load_excluded_episodes,
    resolve_bucket_key,
    resolve_gripper_scale,
    resolve_trim_bounds,
    validate_manifest_ranges,
)
from openwam.dataloader.utils.eef import ARM10_DIM, quat_wxyz_to_rot6d
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    load_episodes_parquet,
    parse_info_json,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator

logger = logging.getLogger(__name__)

EEF20_DIM = 20

# Single-arm readers reserve the right-arm half as masked zero padding.
RIGHT_ARM_DIMS_EEF20: Tuple[int, ...] = tuple(range(ARM10_DIM, EEF20_DIM))


# Reuse reader-owned column tables so stats and runtime cannot drift.
_SIDES: Dict[str, Dict[str, Sequence]] = {
    "bimanual": _BIMANUAL_SIDES,
    "single_arm": _SINGLE_ARM_SIDES,
}


def _arm10(table, spec, grip_scale: float) -> np.ndarray:
    """Build one arm's ``(N, 10)`` ``[xyz, rot6d, grip]`` block from a parquet table."""
    pose_col, grip_col = spec
    pose = np.asarray(table[pose_col].to_pylist(), dtype=np.float32)
    grip = np.asarray(table[grip_col].to_pylist(), dtype=np.float32).reshape(len(pose), 1) / grip_scale
    rot6d = quat_wxyz_to_rot6d(pose[:, 3:7])
    return np.concatenate([pose[:, 0:3], rot6d, grip], axis=-1)


def _eef20(table, sides, kind: str, grip_scales) -> np.ndarray:
    """Assemble the ``(N, 20)`` EEF for one stream; right half zero for single-arm.

    Must stay bit-identical to ``InternDataA1Dataset._eef20`` — including the
    per-bucket gripper rescale — or the stats describe a different distribution
    than the reader actually emits.
    """

    left_spec, right_spec = sides[kind]
    n = table.num_rows
    out = np.zeros((n, EEF20_DIM), dtype=np.float32)
    out[:, :ARM10_DIM] = _arm10(table, left_spec, grip_scales[0])
    if right_spec is not None:
        out[:, ARM10_DIM:] = _arm10(table, right_spec, grip_scales[1])
    return out


def _bucket_info(bucket: Path) -> Tuple[str, str, str]:
    """Return ``(embodiment, robot_type, arm_layout)`` for a bucket."""
    with open(bucket / "meta" / "info.json") as f:
        info = json.load(f)
    layout = detect_arm_layout(info.get("features", {}) or {})
    robot_type = info.get("robot_type", "unknown")
    return embodiment_key(robot_type, layout), robot_type, layout


def classify_buckets(buckets: Sequence[Path]) -> Dict[str, Dict]:
    """Group every discovered bucket by embodiment, failing on unreadable metadata.

    A shared per-embodiment stats file is safe only when every discovered bucket
    is either scanned successfully or is empty by construction under the chosen
    split.  Silently omitting a bucket whose ``info.json`` cannot be classified
    would create the same partial-normalizer failure as dropping a bucket later
    in the parquet scan, only before the worker pool has a chance to report it.
    """

    groups: Dict[str, Dict] = {}
    for b in buckets:
        try:
            emb, robot_type, layout = _bucket_info(b)
        except Exception as e:
            raise RuntimeError(
                f"Refusing to compute partial InternData-A1 stats: discovered bucket {b} "
                f"has unreadable meta/info.json ({type(e).__name__}: {e})"
            ) from e
        g = groups.setdefault(emb, {"robot_type": robot_type, "arm_layout": layout, "dirs": []})
        g["dirs"].append(b)
    return groups


def _kept_episodes(bucket: Path) -> Optional[set]:
    """Episode indices the reader will actually emit for this bucket.

    ``meta/episodes`` minus ``meta/excluded_episodes.json``. Both halves matter
    for a cleaned view: ``data/`` (and ``meta/episodes`` itself) are symlinks to
    the source, so the parquet shards physically contain every deleted episode.
    Deletions live only in the exclusion list — which is precisely why the
    cleaned view cannot express them by dropping manifest rows: the reader
    derives file-local offsets as a cumsum over surviving rows, so a shortened
    manifest would slide every later episode onto the wrong frames.

    Returns ``None`` only when the manifest is unreadable — the caller treats
    that as "cannot filter". An **empty set is meaningful and different**: it
    means the manifest was read and nothing survives, so every row must be
    excluded. Conflating the two would silently pool a fully-deleted bucket's
    rows into the statistics.

    Raises when the exclusion list exists but cannot be parsed: continuing would
    quietly include episodes the reader drops, and the caller already skips
    (and logs) a bucket that raises.
    """

    eps = _manifest_episodes(bucket)
    if eps is None:
        return None

    return eps - load_excluded_episodes(bucket)


def _manifest_episodes(bucket: Path) -> Optional[set]:
    """Every episode index listed in ``meta/episodes``, or ``None`` if unreadable."""
    files = sorted((bucket / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return None
    eps: set = set()
    try:
        for f in files:
            eps.update(int(x) for x in pq.read_table(f, columns=["episode_index"]).to_pydict()["episode_index"])
    except (OSError, KeyError, ValueError):
        return None
    return eps


def _assert_reader_can_load(bucket: Path, shards: Sequence) -> None:
    """Refuse a bucket the reader would refuse, for the same reasons.

    ``len(rows) > 0`` proves rows were read; it does not prove a reader can
    consume them. A shard truncated by one row, a manifest whose ranges overlap,
    a missing middle shard — each of those still yields plenty of rows here
    while :meth:`InternDataA1Dataset._add_data_offsets` raises. Statistics
    pooled from a bucket no reader can open describe a population that will
    never be trained on, and nothing downstream distinguishes them.

    Deliberately the reader's checks, called through the reader's functions,
    rather than a parallel implementation of the same idea — a second
    implementation is how the two sides diverged in the first place.
    """

    files = sorted((bucket / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return
    cols = ["episode_index", "dataset_from_index", "dataset_to_index", "length"]
    frm, to, length, epi = [], [], [], []
    for f in files:
        names = set(pq.ParquetFile(f).schema_arrow.names)
        if not set(cols) <= names:
            return
        d = pq.read_table(f, columns=cols).to_pydict()
        epi += d["episode_index"]
        frm += d["dataset_from_index"]
        to += d["dataset_to_index"]
        length += d["length"]

    validate_manifest_ranges(frm, to, length, epi, str(bucket))

    total = sum(pq.ParquetFile(p).metadata.num_rows for _, _, p in shards)
    manifest_end = int(max(to))
    if manifest_end != total:
        raise ValueError(
            f"{bucket}: the data shards hold {total} rows but the manifest ends at "
            f"{manifest_end}. The reader refuses this bucket, so statistics pooled from "
            "it would describe rows that are never trained on."
        )


def _split_episodes(bucket: Path, split: str) -> Optional[set]:
    """Episode indices belonging to ``split``, or ``None`` when info.json has no splits.

    The reader applies `info.json`'s split ranges before anything else, so a
    corpus whose splits actually partition the episodes would otherwise have its
    val rows pooled into the train normalizer. A1's shipped splits happen to
    cover every episode, which is exactly why this gap could sit unnoticed —
    the equivalence is a property of the current data, not of the code.

    Resolved through the reader's own `apply_info_splits` rather than by parsing
    the range here, so the two cannot disagree about what a split string means.
    """

    try:
        with open(bucket / "meta" / "info.json") as f:
            splits = json.load(f).get("splits")
    except (OSError, ValueError) as e:
        raise ValueError(f"{bucket}: cannot read meta/info.json for split resolution ({e})") from e

    eps = _manifest_episodes(bucket)
    if eps is None:
        raise ValueError(f"{bucket}: meta/episodes is unreadable, so the population is unknown")
    if not splits:
        return None if split == "train" else set()
    import pandas as pd

    from openwam.dataloader.utils.lerobotv3 import apply_info_splits

    df = pd.DataFrame({"episode_index": sorted(eps)})
    try:
        return set(apply_info_splits(df, split, splits, source_name=str(bucket))["episode_index"])
    except Exception as e:
        raise ValueError(f"{bucket}: unusable splits {splits!r} ({e})") from e


def _row_mask(table, kept: Optional[set], trim: Optional[Dict[int, Tuple]], min_len: int) -> Optional[np.ndarray]:
    """Which rows belong in the statistics: kept episodes, minus trimmed head/tail.

    ``kept is None`` means "manifest unreadable, cannot filter"; an **empty set
    excludes everything** (see :func:`_kept_episodes`). Testing ``not kept``
    would collapse those two into "no filter" and pool a fully-deleted bucket
    back in.

    Returns ``None`` only when there is genuinely nothing to filter, so the
    plain path stays allocation-free.
    """

    if kept is None and not trim:
        return None
    ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    mask = np.ones(ep.shape[0], dtype=bool)
    uniq = np.unique(ep)

    if kept is not None and not set(uniq.tolist()) <= kept:
        mask &= (
            np.isin(ep, np.fromiter(kept, dtype=np.int64, count=len(kept)))
            if kept
            else np.zeros(ep.shape[0], dtype=bool)
        )

    if trim:
        order = np.argsort(ep, kind="stable")
        bounds = np.flatnonzero(np.diff(ep[order])) + 1
        for g in np.split(order, bounds):
            entry = trim.get(int(ep[g[0]]))
            if entry is None:
                continue
            n = g.shape[0]

            b = resolve_trim_bounds(entry, n, min_len)
            if b is None:
                continue
            head, tail = b
            if head:
                mask[g[:head]] = False
            if tail < n:
                mask[g[tail:]] = False
    return mask


def _scan_bucket(args) -> Tuple[str, np.ndarray, bool, dict]:
    """Read one bucket's parquet shards and return its stacked (N, 20) rows.

    Runs in a worker process; returns the raw rows rather than a per-bucket
    Accumulator so the parent can merge them into one reservoir in a fixed
    (submission) order — see ``compute_stats_for_embodiment``.

    Per-bucket Accumulators are not ruled out by determinism — a fixed-order,
    seeded, weighted reservoir merge would be deterministic too. They are ruled
    out because no such merge exists on ``Accumulator`` and writing a
    statistically correct one is not worth it here. Worth knowing if the parent's
    memory ever needs work: it currently holds every bucket's array until the
    pool drains.
    """

    bucket_str, layout, embodiment = args[:3]
    dataset_id = args[3] if len(args) > 3 else None
    trim_csv = args[4] if len(args) > 4 else None
    min_len = args[5] if len(args) > 5 else 2
    split = args[6] if len(args) > 6 else "train"
    bucket = Path(bucket_str)
    sides = _SIDES[layout]
    cols = set()
    for kind in ("action", "state"):
        for spec in sides[kind]:
            if spec is not None:
                cols.update(spec)

    grip_scales = tuple(
        resolve_gripper_scale(bucket, embodiment, spec[1]) if spec is not None else 1.0 for spec in sides["state"]
    )

    trim = None
    trim_snapshot = None
    if trim_csv:
        trim_snapshot = _load_trim_snapshot(trim_csv)
        key = resolve_bucket_key(
            trim_snapshot.spec, dataset_id or bucket.name, bucket, what="trim_csv", source=f"stats({dataset_id})"
        )
        trim = trim_snapshot.spec.get(key) if key is not None else None

    manifest = load_episodes_parquet(bucket)
    info = parse_info_json(bucket)
    selected = apply_info_splits(
        manifest,
        split,
        info.get("splits", {}) or {},
        source_name=f"InternDataA1 stats({dataset_id or bucket.name})",
    )
    excluded_snapshot = tuple(sorted(load_excluded_episodes(bucket)))
    if excluded_snapshot:
        selected = selected[~selected["episode_index"].isin(excluded_snapshot)].reset_index(drop=True)
    kept = set(int(ep) for ep in selected["episode_index"].to_numpy())
    bucket_provenance = {
        "excluded_episode_indices": list(excluded_snapshot),
        "effective_population": effective_a1_population_provenance(selected, trim, min_len),
    }
    need_ep = bool(trim) or kept is not None

    population_empty = kept is not None and len(kept) == 0

    chunks: List[np.ndarray] = []

    shards = iter_data_shards(bucket)
    if not shards:
        raise ValueError(f"{bucket}: no data shards under data/chunk-*/file-*.parquet")

    _assert_reader_can_load(bucket, shards)

    for _, _, pth in shards:
        names = set(pq.ParquetFile(pth).schema_arrow.names)
        use_ep = need_ep and "episode_index" in names
        if need_ep and not use_ep:
            raise ValueError(
                f"{pth}: trim/exclusion filtering requested but the shard has no "
                "`episode_index` column, so the rows cannot be selected."
            )
        table = pq.read_table(pth, columns=sorted(cols | {"episode_index"}) if use_ep else sorted(cols))
        if table.num_rows == 0:
            continue
        mask = _row_mask(table, kept, trim, min_len) if use_ep else None
        for kind in ("action", "state"):
            rows = _eef20(table, sides, kind, grip_scales)
            chunks.append(rows if mask is None else rows[mask])
    out = np.zeros((0, EEF20_DIM), dtype=np.float32) if not chunks else np.concatenate(chunks, axis=0)

    if len(out) == 0 and not population_empty:
        raise ValueError(
            f"{bucket}: the {split!r} split expects "
            f"{'every episode' if kept is None else f'{len(kept)} episode(s)'} but no shard row "
            "carries them. The shards do not hold the episodes the manifest assigns to this "
            "split — pooling this as an empty population would let its reader load another "
            "bucket's statistics and read another episode's rows."
        )
    current_exclusions = tuple(sorted(load_excluded_episodes(bucket)))
    if current_exclusions != excluded_snapshot:
        raise ValueError(
            f"{bucket}: excluded_episodes.json changed during stats scan; "
            f"expected {list(excluded_snapshot)}, got {list(current_exclusions)}"
        )
    if trim_snapshot is not None:
        _assert_trim_snapshot_current(trim_snapshot, context=f"stats scan for {dataset_id or bucket.name}")
    return bucket_str, out, bool(population_empty), bucket_provenance


def compute_stats_for_embodiment(
    embodiment: str,
    group: Dict,
    *,
    rot6d_identity: bool = True,
    workers: int = 16,
    root: Optional[Path] = None,
    trim_csv: Optional[str] = None,
    min_len: int = 2,
    split: str = "train",
) -> dict:
    """Pool every bucket of one embodiment into a single 20-D stats dict.

    ``trim_csv`` keys on the bucket path RELATIVE to ``root`` — the same id the
    reader uses (bucket basenames repeat across embodiments). Without ``root``
    the relative id cannot be formed, so trimming is skipped rather than matched
    on an ambiguous basename.
    """

    dirs: List[Path] = group["dirs"]
    layout = group["arm_layout"]
    acc = Accumulator(dim=EEF20_DIM)
    n_rows = 0
    n_ok = 0

    def _rel_id(d: Path) -> str:
        """Bucket key, in the exact form the reader resolves against.

        `--dataset_dir` pointing straight at a bucket is a supported mode, and
        there `relative_to(root)` is `'.'` — a key no trim CSV contains and one
        the reader (whose id is then the bare directory name) cannot
        suffix-match. Left as-is it silently skips the trim on the stats side
        while the reader applies it, and writes `exclusions: {".": ...}` that the
        reader rejects outright.
        """

        if root is None:
            return d.name
        try:
            rel = str(d.relative_to(root))
        except ValueError:
            return d.name
        return d.name if rel == "." else rel

    if trim_csv and root is None:
        logger.warning("trim_csv given without a root; bucket ids are ambiguous, not trimming.")
        trim_csv = None
    trim_snapshot = _load_trim_snapshot(trim_csv) if trim_csv else None
    tasks = [(str(d), layout, embodiment, _rel_id(d), trim_csv, min_len, split) for d in dirs]

    scanned_buckets: List[str] = []
    empty_buckets: List[str] = []
    bucket_provenance: Dict[str, dict] = {}
    scan_failures: List[Tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
        futures = {pool.submit(_scan_bucket, t): t[0] for t in tasks}

        for i, fut in enumerate(futures, 1):
            name = futures[fut]
            try:
                _, rows, population_empty, provenance = fut.result()
            except Exception as e:
                rel_id = _rel_id(Path(name))
                scan_failures.append((rel_id, f"{type(e).__name__}: {e}"))
                logger.error("  [%s] failed %s (%s)", embodiment, name, e)
                continue
            if not len(rows):
                if not population_empty:
                    rel_id = _rel_id(Path(name))
                    message = "produced 0 rows without an empty-by-construction population"
                    scan_failures.append((rel_id, message))
                    logger.error("  [%s] %s %s", embodiment, name, message)
                    continue
                logger.info("  [%s] %s has no %s rows (val-only?)", embodiment, name, split)
                rel_id = _rel_id(Path(name))
                empty_buckets.append(rel_id)
                bucket_provenance[rel_id] = provenance
                continue
            rel_id = _rel_id(Path(name))
            scanned_buckets.append(rel_id)
            bucket_provenance[rel_id] = provenance
            acc.update_batch(rows)
            n_rows += len(rows)
            n_ok += 1
            if i % 20 == 0 or i == len(tasks):
                logger.info("  [%s] %d/%d buckets, %d rows", embodiment, i, len(tasks), n_rows)

    if scan_failures:
        details = "; ".join(f"{bucket}: {error}" for bucket, error in scan_failures)
        raise RuntimeError(
            f"Refusing to write partial InternData-A1 stats for embodiment {embodiment!r}: "
            f"{len(scan_failures)} non-empty bucket scan(s) failed: {details}"
        )
    if n_rows == 0:
        raise RuntimeError(f"embodiment {embodiment!r}: every bucket yielded 0 rows")

    population = {
        "schema_version": 2,
        "split": split,
        "trim_active": bool(trim_csv),
        "min_keep": int(min_len),
        "trim_provenance": trim_snapshot.provenance if trim_snapshot is not None else None,
        "bucket_provenance": {key: bucket_provenance[key] for key in sorted(bucket_provenance)},
        "buckets": sorted(scanned_buckets),
        "empty_buckets": sorted(empty_buckets),
    }
    if trim_snapshot is not None:
        _assert_trim_snapshot_current(trim_snapshot, context=f"stats scan for embodiment {embodiment}")

    stats = acc.finalize()
    if rot6d_identity:
        pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
        if layout == "single_arm":
            pin_rot6d_identity(stats, RIGHT_ARM_DIMS_EEF20)
    return {
        "eef": stats,
        "robot_type": group["robot_type"],
        "arm_layout": layout,
        "embodiment": embodiment,
        "num_buckets": n_ok,
        "num_rows": int(n_rows),
        "population": population,
        "scanned_buckets": population["buckets"],
        "split": population["split"],
        "rot6d_identity": bool(rot6d_identity),
        "layout_doc": "[L_xyz(0:3), L_rot6d(3:9), L_grip(9), R_xyz(10:13), R_rot6d(13:19), R_grip(19)]",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset_dir", required=True, help="extracted InternData-A1 v3.0 root")
    parser.add_argument(
        "--stats_root",
        default=None,
        help="where to write meta/stats_<embodiment>.json (default: --dataset_dir). Set this to a "
        "writable directory when the dataset mount is read-only, and pass the SAME path as "
        "dataloader.stats_root in interndata_a1.yaml — the reader resolves the file as "
        "{stats_root}/meta/stats_{embodiment}.json.",
    )
    parser.add_argument("--embodiment", default=None, help="compute for a single embodiment only")
    parser.add_argument(
        "--trim_csv",
        default=None,
        help="quality-audit trim list (same file the dataloader takes). Head/tail frames it "
        "names are excluded from the statistics, so the normalizer describes what the reader "
        "actually feeds the model instead of including motionless frames it skips. Episodes "
        "listed in a bucket's meta/excluded_episodes.json are dropped regardless of this "
        "flag: a cleaned view symlinks data/ (and meta/episodes) at the source, so deleted "
        "episodes are still physically present in the parquet and listed in the manifest.",
    )
    parser.add_argument(
        "--min_keep",
        type=int,
        default=2,
        help="leave an episode untrimmed when the trim would leave fewer than this "
        "many frames. MUST match the reader's minimum window length for the split "
        "being trained (InternDataA1Dataset._train_min_window_len() == 2 for train; "
        "num_frames for val) — otherwise the stats describe episodes the reader "
        "never emits that way.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="which info.json split to pool. Normalization stats must come from the TRAINING "
        "distribution, so this defaults to train and should rarely be changed — it exists so "
        "the scan applies the same split ranges the reader does, rather than pooling every "
        "episode on disk.",
    )
    parser.add_argument("--workers", type=int, default=16, help="parallel bucket readers")
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="do NOT pin rot6d (and single-arm padding) dims to identity — they then "
        "normalize per-dim like pos/gripper, which distorts the rotation "
        "representation; see pin_rot6d_identity.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.dataset_dir)
    buckets = discover_a1_buckets(root)
    if not buckets:
        raise SystemExit(f"no buckets with meta/info.json under {root}")
    logger.info("discovered %d buckets under %s", len(buckets), root)

    groups = classify_buckets(buckets)
    logger.info("embodiments: %s", {k: len(v["dirs"]) for k, v in sorted(groups.items())})
    if args.embodiment:
        if args.embodiment not in groups:
            raise SystemExit(f"embodiment {args.embodiment!r} not found; have {sorted(groups)}")
        groups = {args.embodiment: groups[args.embodiment]}

    stats_root = Path(args.stats_root) if args.stats_root else root
    out_dir = stats_root / "meta"
    if stats_root != root:
        logger.info("writing stats to %s (dataset root %s left untouched)", out_dir, root)
    results = {}
    for emb in sorted(groups):
        logger.info("=== %s (%d buckets) ===", emb, len(groups[emb]["dirs"]))
        result = compute_stats_for_embodiment(
            emb,
            groups[emb],
            rot6d_identity=not args.no_rot6d_identity,
            workers=args.workers,
            root=root,
            trim_csv=args.trim_csv,
            min_len=args.min_keep,
            split=args.split,
        )

        result["trim_csv"] = args.trim_csv
        result["trim_min_keep"] = args.min_keep if args.trim_csv else None
        results[emb] = result

    out_dir.mkdir(parents=True, exist_ok=True)
    for emb in sorted(results):
        result = results[emb]
        out = out_dir / f"stats_{emb}.json"

        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1)
        tmp.replace(out)
        eef = result["eef"]
        logger.info(
            "wrote %s (%d rows) | L_grip q01=%.4f q99=%.4f | R_grip q01=%.4f q99=%.4f",
            out,
            result["num_rows"],
            eef["q01"][9],
            eef["q99"][9],
            eef["q01"][19],
            eef["q99"][19],
        )
    logger.info("done: %s", sorted(f"stats_{e}.json" for e in groups))


if __name__ == "__main__":
    main()


__all__ = [
    "EEF20_DIM",
    "RIGHT_ARM_DIMS_EEF20",
    "classify_buckets",
    "compute_stats_for_embodiment",
]
