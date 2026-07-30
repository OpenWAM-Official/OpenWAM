"""Public implementation. Dataset-specific audit notes were removed."""





























































from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.interndata_a1 import (
    detect_arm_layout,
    discover_a1_buckets,
    embodiment_key,
    resolve_gripper_scale,
)
from openwam.dataloader.utils.eef import ARM10_DIM, quat_wxyz_to_rot6d
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator

logger = logging.getLogger(__name__)

EEF20_DIM = 20


RIGHT_ARM_DIMS_EEF20: Tuple[int, ...] = tuple(range(ARM10_DIM, EEF20_DIM))



_SIDES: Dict[str, Dict[str, Sequence]] = {
    "bimanual": {
        "action": (
            ("actions.left_ee_to_robot_pose", "actions.left_gripper.position"),
            ("actions.right_ee_to_robot_pose", "actions.right_gripper.position"),
        ),
        "state": (
            ("states.left_ee_to_robot_pose", "states.left_gripper.position"),
            ("states.right_ee_to_robot_pose", "states.right_gripper.position"),
        ),
    },
    "single_arm": {
        "action": (("actions.ee_to_robot_pose", "actions.gripper.position"), None),
        "state": (("states.ee_to_robot_pose", "states.gripper.position"), None),
    },
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


def _scan_bucket(args) -> Tuple[str, np.ndarray]:
    """Public implementation. Dataset-specific audit notes were removed."""




    bucket_str, layout, embodiment = args
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
    chunks: List[np.ndarray] = []
    for pth in sorted((bucket / "data").rglob("*.parquet")):
        table = pq.read_table(pth, columns=sorted(cols))
        if table.num_rows == 0:
            continue
        chunks.append(_eef20(table, sides, "action", grip_scales))
        chunks.append(_eef20(table, sides, "state", grip_scales))
    if not chunks:
        return bucket_str, np.zeros((0, EEF20_DIM), dtype=np.float32)
    return bucket_str, np.concatenate(chunks, axis=0)


def compute_stats_for_embodiment(
    embodiment: str,
    group: Dict,
    *,
    rot6d_identity: bool = True,
    workers: int = 16,
) -> dict:
    """Public implementation. Dataset-specific audit notes were removed."""
    dirs: List[Path] = group["dirs"]
    layout = group["arm_layout"]
    acc = Accumulator(dim=EEF20_DIM)
    n_rows = 0
    n_ok = 0
    tasks = [(str(d), layout, embodiment) for d in dirs]
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
        futures = {pool.submit(_scan_bucket, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
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
        "rot6d_identity": bool(rot6d_identity),
        "layout_doc": "[L_xyz(0:3), L_rot6d(3:9), L_grip(9), R_xyz(10:13), R_rot6d(13:19), R_grip(19)]",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset_dir", required=True, help="extracted InternData-A1 v3.0 root")
    parser.add_argument("--embodiment", default=None, help="compute for a single embodiment only")
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

    out_dir = root / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    for emb in sorted(groups):
        logger.info("=== %s (%d buckets) ===", emb, len(groups[emb]["dirs"]))
        result = compute_stats_for_embodiment(
            emb,
            groups[emb],
            rot6d_identity=not args.no_rot6d_identity,
            workers=args.workers,
        )
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
