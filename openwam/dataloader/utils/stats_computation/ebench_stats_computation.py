#!/usr/bin/env python3
"""Compute EBench action-normalization stats from a full parquet scan.

The EBench reader (:mod:`openwam.dataloader.ebench`) normally derives its
raw-23 stats online at construction time by merging the per-episode summary
moments in each bucket's ``meta/episodes_stats.jsonl``. That derivation is
exact for min-max / z-score, but summaries carry no true quantiles (so
``normalize_mode="quantile"`` is refused without a prebuilt file) and the
scalar-gripper stats lean on the two-finger-equality invariant that the
reader only spot-checks (3 episodes x 64 rows). This offline module scans
every action row the reader actually emits and adds what the summaries
cannot provide:

  * true ``q01``/``q99`` from the row stream — unlocking
    ``normalize_mode="quantile"`` for EBench;
  * a dataset-wide certificate of the reader's data contract (finite values,
    unit quaternions, gripper commands within ``EBENCH_GRIPPER_CMD_RANGE``,
    per-hand finger commands equal within
    ``EBENCH_FINGER_GAP_TOLERANCE``) for every episode training can serve,
    instead of the sampled init check — episodes in
    ``meta/excluded_episodes.json`` are tolerated, not certified;
  * a prebuilt cache, so training ranks cache-hit instead of every rank
    re-merging the summaries at construction.

Alignment notes (family = robocoin/behavior stats modules):

  * the row stream is projected with the READER'S OWN helpers
    (``_raw23_from_frame`` -> ``_ee_pose_gripper_base_to_raw23``: wxyz
    quat->rot6d, two-finger averaging, base column passthrough) so the stats
    are computed over byte-identical numbers to what the reader feeds the
    model — zero layout drift;
  * moments accumulate in RoboCOIN's streaming :class:`Accumulator` (exact
    mean/std/min/max, reservoir q01/q99) and the 12 rot6d dims (3:9 / 13:19 —
    the eef-20 layout EBench's raw-23 prefix shares) are pinned to identity
    via the shared ``pin_rot6d_identity`` (``--no-rot6d-identity`` opt-out);
  * ``pool="action"`` — the ACTION stream only, deliberately NOT behavior's
    ``action+proprio``: the EBench reader normalizes proprio with the action
    stats by design (measured state is rendered into the command space; see
    configs/dataloader/ebench.yaml) and the online cache is action-only, so
    this module must be a drop-in replacement with identical
    mean/std/min/max. (libero is the family precedent for a documented
    action-only pool.)
  * episode coverage mirrors the reader's summary build
    (``_build_stats_from_bucket``): every episode listed in
    ``meta/episodes_stats.jsonl``, no split filtering, and excluded episodes
    still accumulated when readable (the summary merge pools their moments
    too) — both producers agree on coverage, and the cache fingerprint
    (reused verbatim from the reader) stays valid for either. Excluded
    episodes that cannot be read cleanly are warn-skipped (they never train;
    blocking the scan on them would make quantile permanently unreachable).

Output: the reader's own cache schema
``{<action_mode>: {mean,std,min,max,q01,q99}, num_timesteps,
raw_action_dim_mask, fingerprint}`` written atomically to ``--output``
(default ``<dataset_dir>/meta/ebench_normalization_stats.npy``, the ebench.yaml
convention). The reader accepts the file unchanged for every mode, and the
same file doubles as the deploy denormalization artifact (the trainer copies
it into the checkpoint as ``normalization_stats.npy``).

Usage:
    python -m openwam.dataloader.utils.stats_computation.ebench_stats_computation \
        --dataset_dir /path/to/data_lake/EBench-Dataset
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reuse the reader's own constants/helpers so the scan and the reader cannot
# drift: same column keys, same raw-23 projection, same episode-path
# resolution, same cache fingerprint/payload/atomic writer.
from openwam.dataloader.ebench import (
    EBENCH_ACTION_KEYS,
    EBENCH_FINGER_GAP_TOLERANCE,
    EBENCH_GRIPPER_CMD_RANGE,
    EBENCH_RAW_ACTION_DIM,
    EBENCH_STD_FLOOR,
    _atomic_save_npy,
    _build_stats_from_bucket,
    _column_matrix,
    _ee_pose_gripper_base_to_raw23,
    _episode_parquet_path,
    _load_excluded_indices,
    _merge_raw_stats,
    _raw23_from_frame,
    _read_jsonl,
    _stats_cache_payload,
    _stats_fingerprint,
    discover_ebench_buckets,
)
from openwam.dataloader.utils.eef import assert_unit_quaternion
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _validated_raw23(frame: pd.DataFrame, action_keys: Sequence[str], ctx: str) -> np.ndarray:
    """Project an episode's action columns to raw-23, enforcing the reader's
    data contract on EVERY row.

    The reader's init check samples 3 episodes x 64 rows and windows are never
    re-validated at __getitem__, so bad values in a training-served episode
    would both skew the stats and reach the model silently — surfacing them at
    scan time is the dataset-wide certificate this module provides. Fail-fast
    applies only to episodes training can serve; the caller handles excluded
    episodes leniently (they never train).
    """
    ee = _column_matrix(frame, action_keys[0], 14)
    gripper = _column_matrix(frame, action_keys[1], 4)
    base = _column_matrix(frame, action_keys[2], 3)
    for key, arr in ((action_keys[0], ee), (action_keys[1], gripper), (action_keys[2], base)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{ctx} {key} contains non-finite values")
    assert_unit_quaternion(ee[:, 3:7], sample_n=len(ee))
    assert_unit_quaternion(ee[:, 10:14], sample_n=len(ee))
    lo_cmd, hi_cmd = EBENCH_GRIPPER_CMD_RANGE
    eps = 1e-4  # same slack the reader's _validate_episode_rows applies
    if gripper.min() < lo_cmd - eps or gripper.max() > hi_cmd + eps:
        raise ValueError(
            f"{ctx} {action_keys[1]} outside [{lo_cmd - eps}, {hi_cmd + eps}]: "
            f"min={gripper.min():.4f}, max={gripper.max():.4f}"
        )
    finger_gap = max(
        float(np.abs(gripper[:, 0] - gripper[:, 1]).max()),
        float(np.abs(gripper[:, 2] - gripper[:, 3]).max()),
    )
    if finger_gap > EBENCH_FINGER_GAP_TOLERANCE:
        raise ValueError(
            f"{ctx} per-hand finger commands disagree by {finger_gap:.5f} m; "
            "the scalar-gripper averaging and its summary stats assume identical finger commands"
        )
    return _ee_pose_gripper_base_to_raw23(ee, gripper, base)


def _scan_bucket(bucket: Path, action_keys: Sequence[str], acc: Accumulator) -> Tuple[int, int]:
    """Stream every episode of one bucket into ``acc``.

    Returns ``(n_files, n_skipped)``. Episodes listed in
    ``meta/excluded_episodes.json`` never train, but the online summary merge
    still pools their moments (episodes_stats.jsonl carries no exclusion
    filtering), so they are accumulated for coverage parity — leniently:
    whatever got an episode excluded (missing/corrupt parquet, contract
    violations) warn-skips that episode instead of blocking the whole scan.
    Episodes training can serve keep the strict fail-fast validation.
    """
    stats_meta = bucket / "meta" / "episodes_stats.jsonl"
    if not stats_meta.exists():
        raise FileNotFoundError(f"EBench stats file missing: {stats_meta}")
    with (bucket / "meta" / "info.json").open() as f:
        info = json.load(f)
    template = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))
    excluded = _load_excluded_indices(bucket)

    n_files = 0
    n_skipped = 0
    for row in _read_jsonl(stats_meta):
        ep_idx = int(row["episode_index"])
        fpath = _episode_parquet_path(bucket, template, chunks_size, ep_idx)
        ctx = f"EBench({bucket.name}) episode {ep_idx}"
        if ep_idx in excluded:
            try:
                frame = pq.read_table(fpath, columns=list(action_keys)).to_pandas()
                raw23 = _raw23_from_frame(frame, action_keys)
                if not np.isfinite(raw23).all():
                    raise ValueError("non-finite values")
            except Exception as e:  # noqa: BLE001 — excluded data must not block the scan
                print(f"  Warning: skipping excluded {ctx}: {e} (stats will drift slightly from the summary merge)")
                n_skipped += 1
                continue
        else:
            frame = pq.read_table(fpath, columns=list(action_keys)).to_pandas()
            raw23 = _validated_raw23(frame, action_keys, ctx=ctx)
        acc.update_batch(raw23)
        n_files += 1
    return n_files, n_skipped


def compute_ebench_stats(
    buckets: Sequence[Path],
    action_keys: Sequence[str],
    *,
    rot6d_identity: bool = True,
) -> Tuple[dict, int, int]:
    """Full parquet scan over ``buckets`` -> raw-23 action stats.

    Returns ``(stats, num_timesteps, num_files)`` where ``stats`` carries all
    six keys ``{mean, std, min, max, q01, q99}`` as float32 ``(23,)`` arrays.
    mean/std/min/max are exact (streamed over every row) and match the
    reader's summary merge; q01/q99 come from the Accumulator's bounded
    reservoir (exact below 1M rows).
    """
    acc = Accumulator(dim=EBENCH_RAW_ACTION_DIM)
    n_files = 0
    n_skipped = 0
    for bucket in buckets:
        scanned, skipped = _scan_bucket(Path(bucket), action_keys, acc)
        n_files += scanned
        n_skipped += skipped
    if acc.count == 0:
        raise RuntimeError(f"no usable episode rows under {[str(b) for b in buckets]} — nothing to compute stats from")
    if n_skipped:
        print(f"  Note: {n_skipped} excluded episode(s) skipped — expect small drift vs the summary merge.")

    stats = {k: np.asarray(v, dtype=np.float32) for k, v in acc.finalize().items()}
    # Reader parity on degenerate dims: the summary paths floor std at
    # EBENCH_STD_FLOOR, while Accumulator.finalize substitutes 1.0 below 1e-8
    # — GenManip's constant commanded base deltas would hit exactly that
    # difference. Recompute from the raw moments and apply the reader's floor.
    raw_std = np.sqrt(acc.m2 / max(acc.count, 1))
    stats["std"] = np.maximum(raw_std, EBENCH_STD_FLOOR).astype(np.float32)
    if rot6d_identity:
        # EBench raw-23's [0:20) prefix is the family eef-20 layout, so the
        # shared dim tuple applies directly; base dims 20:23 keep real stats.
        pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    return stats, int(acc.count), n_files


def build_and_save_ebench_stats(
    dataset_dir: str,
    *,
    output: Optional[str] = None,
    action_mode: str = "ebench",
    rot6d_identity: bool = True,
) -> Tuple[Path, dict]:
    """Scan, cross-check against the summary merge, and write the reader cache.

    Returns ``(output_path, stats)``.
    """
    action_keys = EBENCH_ACTION_KEYS
    bucket_paths = discover_ebench_buckets(dataset_dir)
    stats, num_timesteps, n_files = compute_ebench_stats(bucket_paths, action_keys, rot6d_identity=rot6d_identity)

    # Guardrail: the summary merge and the row scan must agree on
    # mean/std/min/max (the scan is authoritative; a large gap means
    # episodes_stats.jsonl no longer describes the parquet bytes — re-export).
    # The summary path pins rot6d unconditionally, so compare only the dims
    # both producers derive from data.
    summary = _merge_raw_stats([_build_stats_from_bucket(b, action_keys) for b in bucket_paths])
    compare = np.ones(EBENCH_RAW_ACTION_DIM, dtype=bool)
    compare[list(ROT6D_DIMS_EEF20)] = False
    for key in ("mean", "std", "min", "max"):
        drift = float(np.abs(stats[key][compare] - summary[key][compare]).max())
        if drift > 1e-3:
            print(
                f"  WARNING: {key} drifts {drift:.5f} from the episodes_stats.jsonl summary merge — "
                "the summaries may be stale relative to the parquet data."
            )

    out_path = Path(output) if output else Path(dataset_dir) / "meta" / "ebench_normalization_stats.npy"
    fingerprint = _stats_fingerprint(bucket_paths, action_keys, action_mode, dataset_dir)
    payload = _stats_cache_payload(stats, num_timesteps, fingerprint, action_mode)
    payload["pool"] = "action"
    payload["source"] = "parquet_scan"
    payload["num_files"] = int(n_files)
    _atomic_save_npy(out_path, payload)
    return out_path, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_dir", required=True, help="EBench root (contains long_horizon/ simple_pnp/ teleop_tasks/)"
    )

    parser.add_argument("--action_mode", default="eef", help="payload key; must match dataloader.action_mode")
    parser.add_argument(
        "--output",
        default=None,
        help=".npy path (default <dataset_dir>/meta/ebench_normalization_stats.npy — the ebench.yaml convention). "
        "Use a run-specific path for --groups/--buckets subsets: the cache fingerprint records the "
        "bucket set, and a training run over a different set hard-fails on it.",
    )
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="Disable pinning rot6d stats to identity (rot6d would then be per-dim normalized; "
        "generally undesirable — see pin_rot6d_identity).",
    )
    args = parser.parse_args()

    out_path, stats = build_and_save_ebench_stats(
        args.dataset_dir,
        output=args.output,
        action_mode=args.action_mode,
        rot6d_identity=not args.no_rot6d_identity,
    )

    payload = np.load(out_path, allow_pickle=True).item()
    print(f"\nEBench stats ({payload['num_files']} episode files, {payload['num_timesteps']:,} timesteps):")
    print(f"  L/R gripper mean [9],[19]: {stats['mean'][9]:.4f}, {stats['mean'][19]:.4f}")
    print(f"  base[20:23] min:  {[round(float(x), 5) for x in stats['min'][20:23]]}")
    print(f"  base[20:23] max:  {[round(float(x), 5) for x in stats['max'][20:23]]}")
    print(f"  base[20:23] q01:  {[round(float(x), 5) for x in stats['q01'][20:23]]}")
    print(f"  base[20:23] q99:  {[round(float(x), 5) for x in stats['q99'][20:23]]}")
    print(f"  rot6d pinned identity: {not args.no_rot6d_identity}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
