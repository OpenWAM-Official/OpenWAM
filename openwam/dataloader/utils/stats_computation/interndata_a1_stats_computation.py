"""Public implementation. Dataset-specific audit notes were removed."""





































































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



    _load_trim_spec,
    detect_arm_layout,
    discover_a1_buckets,
    embodiment_key,


    exclusion_digest,
    resolve_gripper_scale,
    resolve_trim_bounds,
    trim_digest,
)
from openwam.dataloader.utils.eef import ARM10_DIM, quat_wxyz_to_rot6d
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator

logger = logging.getLogger(__name__)

EEF20_DIM = 20


RIGHT_ARM_DIMS_EEF20: Tuple[int, ...] = tuple(range(ARM10_DIM, EEF20_DIM))







_SIDES: Dict[str, Dict[str, Sequence]] = {
    "bimanual": _BIMANUAL_SIDES,
    "single_arm": _SINGLE_ARM_SIDES,
}


def _arm10(table, spec, grip_scale: float) -> np.ndarray:
    """Public implementation. Dataset-specific audit notes were removed."""
    pose_col, grip_col = spec
    pose = np.asarray(table[pose_col].to_pylist(), dtype=np.float32)
    grip = np.asarray(table[grip_col].to_pylist(), dtype=np.float32).reshape(len(pose), 1) / grip_scale
    rot6d = quat_wxyz_to_rot6d(pose[:, 3:7])
    return np.concatenate([pose[:, 0:3], rot6d, grip], axis=-1)


def _eef20(table, sides, kind: str, grip_scales) -> np.ndarray:
    """Public implementation. Dataset-specific audit notes were removed."""





    left_spec, right_spec = sides[kind]
    n = table.num_rows
    out = np.zeros((n, EEF20_DIM), dtype=np.float32)
    out[:, :ARM10_DIM] = _arm10(table, left_spec, grip_scales[0])
    if right_spec is not None:
        out[:, ARM10_DIM:] = _arm10(table, right_spec, grip_scales[1])
    return out


def _bucket_info(bucket: Path) -> Tuple[str, str, str]:
    """Public implementation. Dataset-specific audit notes were removed."""
    with open(bucket / "meta" / "info.json") as f:
        info = json.load(f)
    layout = detect_arm_layout(info.get("features", {}) or {})
    robot_type = info.get("robot_type", "unknown")
    return embodiment_key(robot_type, layout), robot_type, layout


def classify_buckets(buckets: Sequence[Path]) -> Dict[str, Dict]:
    """Public implementation. Dataset-specific audit notes were removed."""
    groups: Dict[str, Dict] = {}
    for b in buckets:
        try:
            emb, robot_type, layout = _bucket_info(b)
        except Exception as e:
            logger.warning("skipping %s (unreadable meta/info.json: %s)", b, e)
            continue
        g = groups.setdefault(emb, {"robot_type": robot_type, "arm_layout": layout, "dirs": []})
        g["dirs"].append(b)
    return groups


def _kept_episodes(bucket: Path) -> Optional[set]:
    """Public implementation. Dataset-specific audit notes were removed."""



















    files = sorted((bucket / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return None
    eps: set = set()
    try:
        for f in files:
            eps.update(int(x) for x in pq.read_table(f, columns=["episode_index"]).to_pydict()["episode_index"])
    except (OSError, KeyError, ValueError):
        return None

    excl_path = bucket / "meta" / "excluded_episodes.json"
    if excl_path.is_file():
        with open(excl_path) as fh:
            eps -= {int(x) for x in json.load(fh)["episode_indices"]}
    return eps


def _row_mask(table, kept: Optional[set], trim: Optional[Dict[int, Tuple]],
              min_len: int) -> Optional[np.ndarray]:
    """Public implementation. Dataset-specific audit notes were removed."""









    if kept is None and not trim:
        return None
    ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    mask = np.ones(ep.shape[0], dtype=bool)
    uniq = np.unique(ep)

    if kept is not None and not set(uniq.tolist()) <= kept:
        mask &= (np.isin(ep, np.fromiter(kept, dtype=np.int64, count=len(kept)))
                 if kept else np.zeros(ep.shape[0], dtype=bool))

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


def _scan_bucket(args) -> Tuple[str, np.ndarray]:
    """Public implementation. Dataset-specific audit notes were removed."""
















    bucket_str, layout, embodiment = args[:3]
    dataset_id = args[3] if len(args) > 3 else None
    trim_csv = args[4] if len(args) > 4 else None
    min_len = args[5] if len(args) > 5 else 2
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






    trim = _load_trim_spec(trim_csv).get(dataset_id) if trim_csv else None
    kept = _kept_episodes(bucket) if (trim_csv or (bucket / "meta" / "episodes").is_dir()) else None
    need_ep = bool(trim) or kept is not None

    chunks: List[np.ndarray] = []
    warned = False
    for pth in sorted((bucket / "data").rglob("*.parquet")):




        names = set(pq.ParquetFile(pth).schema_arrow.names)
        use_ep = need_ep and "episode_index" in names
        if need_ep and not use_ep and not warned:
            logger.warning("%s: no episode_index column; scanning every row unfiltered", pth.parent)
            warned = True
        table = pq.read_table(pth, columns=sorted(cols | {"episode_index"}) if use_ep else sorted(cols))
        if table.num_rows == 0:
            continue
        mask = _row_mask(table, kept, trim, min_len) if use_ep else None
        for kind in ("action", "state"):
            rows = _eef20(table, sides, kind, grip_scales)
            chunks.append(rows if mask is None else rows[mask])
    if not chunks:
        return bucket_str, np.zeros((0, EEF20_DIM), dtype=np.float32)
    return bucket_str, np.concatenate(chunks, axis=0)


def compute_stats_for_embodiment(
    embodiment: str,
    group: Dict,
    *,
    rot6d_identity: bool = True,
    workers: int = 16,
    root: Optional[Path] = None,
    trim_csv: Optional[str] = None,
    min_len: int = 2,
) -> dict:
    """Public implementation. Dataset-specific audit notes were removed."""






    dirs: List[Path] = group["dirs"]
    layout = group["arm_layout"]
    acc = Accumulator(dim=EEF20_DIM)
    n_rows = 0
    n_ok = 0

    def _rel_id(d: Path) -> str:
        """Public implementation. Dataset-specific audit notes were removed."""








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
    tasks = [(str(d), layout, embodiment, _rel_id(d), trim_csv, min_len) for d in dirs]


    exclusions = {_rel_id(d): exclusion_digest(d) for d in dirs}
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
        futures = {pool.submit(_scan_bucket, t): t[0] for t in tasks}











        for i, fut in enumerate(futures, 1):
            name = futures[fut]
            try:
                _, rows = fut.result()
            except Exception as e:
                logger.warning("  [%s] skipping %s (%s)", embodiment, name, e)
                continue
            if len(rows):
                acc.update_batch(rows)
                n_rows += len(rows)
            n_ok += 1
            if i % 20 == 0 or i == len(tasks):
                logger.info("  [%s] %d/%d buckets, %d rows", embodiment, i, len(tasks), n_rows)

    if n_rows == 0:
        raise RuntimeError(f"embodiment {embodiment!r}: every bucket yielded 0 rows")

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



        "exclusions": exclusions,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    if stats_root != root:
        logger.info("writing stats to %s (dataset root %s left untouched)", out_dir, root)
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
        )



        result["trim_csv"] = args.trim_csv


        result["trim_digest"] = trim_digest(args.trim_csv)
        result["trim_min_keep"] = args.min_keep if args.trim_csv else None
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
