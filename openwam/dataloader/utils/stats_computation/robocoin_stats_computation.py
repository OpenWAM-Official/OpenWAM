#!/usr/bin/env python3
"""Compute per-robot-type EEF and hand stats for RoboCOIN.

Published Euler-XYZ pose and gripper columns are converted to the reader's
20-D EEF layout, then action and state rows are pooled together.  Dexterous
buckets additionally produce a left-then-right hand-joint stats block while
their absent scalar-grip slots remain masked.  Means, variances and extrema
are streamed exactly; quantiles use a bounded uniform reservoir.

Rot6d dimensions are pinned to identity, and each output records contributor,
exclusion, trim, and retained-population provenance required by the reader.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.robocoin import (
    MAX_HAND_DOF,
    _assert_trim_snapshot_current,
    _discover_data_parquets,
    _eef14_to_eef20,
    _excluded_episodes_provenance,
    _finger_indices,
    _load_trim_snapshot,
    _stats_population_spans,
    dex_finger_layout,
    is_robocoin_bucket_excluded,
    robocoin_bucket_exclusions_provenance,
)
from openwam.dataloader.utils.lerobotv3 import (
    DataContractError,
    assert_excluded_episodes_snapshot_current,
    load_excluded_episodes_snapshot,
    parse_info_json,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity

RESERVOIR_CAP = 1_000_000


class Accumulator:
    """Online mean/std/min/max accumulator + reservoir for q01/q99."""

    def __init__(self, dim: int = 20, reservoir_cap: int = RESERVOIR_CAP, seed: int = 0):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

        self.cap = int(reservoir_cap)
        self.rng = np.random.RandomState(seed)
        self._res = np.empty((self.cap, dim), dtype=np.float32)
        self._res_n = 0
        self._res_seen = 0

    def update(self, batch: np.ndarray):
        """Update with (N, dim) array using Welford's online algorithm."""
        for i in range(len(batch)):
            x = batch[i].astype(np.float64)
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
            self.min_val = np.minimum(self.min_val, x)
            self.max_val = np.maximum(self.max_val, x)
        self._reservoir_add(np.asarray(batch, dtype=np.float32))

    def update_batch(self, batch: np.ndarray):
        """Batch update (more efficient for large arrays)."""
        n = len(batch)
        if n == 0:
            return
        self._reservoir_add(np.asarray(batch, dtype=np.float32))
        batch = batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_var * n
            self.min_val = batch_min
            self.max_val = batch_max
            self.count = n
        else:
            total = self.count + n
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * n / total
            self.m2 = self.m2 + batch_var * n + delta**2 * self.count * n / total
            self.mean = new_mean
            self.count = total
            self.min_val = np.minimum(self.min_val, batch_min)
            self.max_val = np.maximum(self.max_val, batch_max)

    def _reservoir_add(self, batch: np.ndarray):
        """Feed (N, dim) rows into the reservoir (vectorized Algorithm R)."""
        n = len(batch)
        if n == 0:
            return

        if self._res_n < self.cap:
            take = min(self.cap - self._res_n, n)
            self._res[self._res_n : self._res_n + take] = batch[:take]
            self._res_n += take
            self._res_seen += take
            batch = batch[take:]
            if len(batch) == 0:
                return

        m = len(batch)
        t = self._res_seen + np.arange(m)
        p = self.rng.randint(0, t + 1)
        keep = p < self.cap
        self._res[p[keep]] = batch[keep]
        self._res_seen += m

    def finalize(self):
        std = np.sqrt(self.m2 / max(self.count, 1))
        std = np.where(std < 1e-8, 1.0, std)
        if self._res_n > 0:
            res = self._res[: self._res_n]
            q01 = np.quantile(res, 0.01, axis=0)
            q99 = np.quantile(res, 0.99, axis=0)
        else:
            q01 = self.min_val
            q99 = self.max_val
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": self.min_val.astype(np.float32).tolist(),
            "max": self.max_val.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }


def discover_datasets_by_robot_type(root: str) -> dict:
    """Group datasets by robot_type."""
    groups = {}
    for name in sorted(os.listdir(root)):
        if is_robocoin_bucket_excluded(name):
            print(f"  Excluding whole RoboCOIN bucket from stats discovery: {name}")
            continue
        info_path = os.path.join(root, name, "meta", "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path) as f:
            info = json.load(f)

        rtype = str(info.get("robot_type", "unknown"))
        groups.setdefault(rtype, []).append(os.path.join(root, name))
    return groups


_NEEDED_COLS = [
    "eef_sim_pose_action",
    "gripper_open_scale_action",
    "eef_sim_pose_state",
    "gripper_open_scale_state",
]
_EEF_COLS = ["eef_sim_pose_action", "eef_sim_pose_state"]
_GRIP_COLS = ("gripper_open_scale_action", "gripper_open_scale_state")

_HAND_RAW_COLS = ["action", "observation.state"]


def _classify_dataset(ds_dir: str, *, fail_closed: bool = False):
    """Classify one dataset and return its dex-hand finger layout if any.

    Reads ``meta/info.json`` ONCE and returns ``(kind, layout)``:
      * ``kind`` is one of:
          ``"grip"``   — has ``gripper_open_scale_*`` (real gripper);
          ``"dex"``    — pose, no gripper, a valid finger layout;
          ``"nogrip"`` — pose, no gripper, but the finger gate failed
                         (single-handed / >MAX_HAND_DOF / action-state disagree);
          ``"other"``  — no pose column at all (its files won't pool into `eef`).
      * ``layout`` is ``(idx_action_LR, idx_state_LR, kL, kR)`` for ``"dex"`` (the
        left-then-right ``*_hand_joint_*`` indices within the raw ``action`` /
        ``observation.state`` arrays), else ``None``.

    ``"dex"`` and ``"nogrip"`` both zero-fill the gripper slots (9/19) when pooled,
    so both must be kept out of a robot_type that also has ``"grip"`` datasets (the
    caller's mixed-bucket guard treats them alike). Uses the shared
    :func:`dex_finger_layout` gate so the stats producer and the reader agree.
    """

    try:
        with open(os.path.join(ds_dir, "meta", "info.json")) as f:
            feats = json.load(f).get("features", {})
    except (OSError, ValueError):
        if fail_closed:
            raise
        return "other", None
    if all(c in feats for c in _GRIP_COLS):
        return "grip", None
    layout = dex_finger_layout(feats)
    if layout is not None:
        aL, aR, sL, sR = layout
        return "dex", (aL + aR, sL + sR, len(aL), len(aR))
    if "eef_sim_pose_action" in feats:
        aL, aR = _finger_indices(feats.get("action", {}))
        sL, sR = _finger_indices(feats.get("observation.state", {}))
        print(
            f"  Warning: {ds_dir} looks dexterous-hand (pose, no gripper) but finger layout "
            f"failed the gate (action L/R={len(aL)}/{len(aR)}, state L/R={len(sL)}/{len(sR)}, "
            f"max={MAX_HAND_DOF}); no finger stats emitted."
        )
        return "nogrip", None
    return "other", None


def _slice_file_to_global_spans(df, file_start: int, file_end: int, kept_spans):
    """Select intersections with kept global spans using physical file offsets."""
    pieces = []
    for span_start, span_end in kept_spans:
        if span_end <= file_start:
            continue
        if span_start >= file_end:
            break
        local_start = max(span_start, file_start) - file_start
        local_end = min(span_end, file_end) - file_start
        if local_start < local_end:
            pieces.append(df.iloc[local_start:local_end])
    if not pieces:
        return df.iloc[0:0]
    return pd.concat(pieces, ignore_index=True)


def compute_stats_for_robot_type(
    rtype: str,
    dataset_dirs: list,
    rot6d_identity: bool = True,
    trim_csv=None,
    *,
    split: str = "train",
    _trim_snapshot=None,
) -> dict:
    """Compute unified action+state EEF stats across all datasets of one robot type.

    Each parquet row contributes TWO 20-D points to the pool: one from action
    columns and one from state columns. The accumulator sees their union, so
    min/max end up as the per-dim envelope across both streams and mean/std
    reflect the joint distribution.

    For dexterous-hand types (no gripper), a second accumulator pools the raw
    ``*_hand_joint_*`` finger values (action + state) into a ``hand`` stats block
    of width ``kL + kR`` (left then right), used to normalize the finger dims the
    reader scatters into the 80-D hand slots under unify_action.

    With ``trim_csv`` set, every stream is restricted to the same split,
    exclusion and manifest-based trim spans as the reader and all
    file/manifest errors fail closed.

    Returns a dict ready to dump: ``{"eef": <20-D stats>, "hand": <finger stats>?}``.
    """

    dataset_dirs = list(dataset_dirs)
    excluded_dirs = [ds_dir for ds_dir in dataset_dirs if is_robocoin_bucket_excluded(ds_dir)]
    dataset_dirs = [ds_dir for ds_dir in dataset_dirs if not is_robocoin_bucket_excluded(ds_dir)]
    if excluded_dirs:
        print(
            "  Excluding whole RoboCOIN bucket(s) from stats population: "
            + ", ".join(sorted(Path(path).name for path in excluded_dirs))
        )
    if not dataset_dirs:
        raise DataContractError(f"RoboCOIN stats for robot_type {rtype!r}: no non-excluded dataset directories")

    trim_enabled = trim_csv is not None
    if _trim_snapshot is not None and (not trim_enabled or _trim_snapshot.path != str(trim_csv)):
        raise ValueError("_trim_snapshot requires a matching non-null trim_csv")
    trim_snapshot = _trim_snapshot
    if trim_enabled and trim_snapshot is None:
        trim_snapshot = _load_trim_snapshot(trim_csv)
    trim_spec = trim_snapshot.spec if trim_enabled else {}
    exclusion_snapshots = {}
    population_datasets = {}
    acc = Accumulator(dim=20)
    total_files = 0
    grip_files = 0

    hand_acc = None
    hand_dims = None
    hand_files = 0

    grip_example = None
    nogrip_example = None

    for ds_dir in dataset_dirs:
        dataset_id = Path(ds_dir).name
        data_dir = Path(ds_dir) / "data"
        if not data_dir.is_dir():
            if trim_enabled:
                raise FileNotFoundError(f"RoboCOIN stats: missing data directory {data_dir}")
            continue
        exclusion_snapshot = None
        if trim_enabled:
            if dataset_id in exclusion_snapshots:
                raise DataContractError(
                    f"RoboCOIN stats for robot_type {rtype!r}: duplicate dataset directory "
                    f"name {dataset_id!r} cannot be represented in exclusion provenance"
                )
            exclusion_snapshot = load_excluded_episodes_snapshot(Path(ds_dir))
            exclusion_snapshots[dataset_id] = exclusion_snapshot
        kind, layout = _classify_dataset(ds_dir, fail_closed=trim_enabled)
        if kind == "grip":
            grip_example = grip_example or ds_dir
        elif kind in ("dex", "nogrip"):
            nogrip_example = nogrip_example or ds_dir
        if grip_example and nogrip_example:
            raise ValueError(
                f"robot_type {rtype!r} mixes a grippered dataset ({grip_example}) with a "
                f"no-gripper (dexterous-hand) dataset ({nogrip_example}). They share one 20-D "
                f"'eef' stats block, so the no-gripper zero-filled gripper slots (9/19) would "
                f"corrupt the grippered reader's gripper normalization. Split these into distinct "
                f"robot_types."
            )
        if layout is not None:
            idx_act_lr, idx_state_lr, kL, kR = layout
            if hand_acc is None:
                hand_dims = (kL, kR)
                hand_acc = Accumulator(dim=kL + kR)
            elif hand_dims != (kL, kR):
                raise ValueError(
                    f"robot_type {rtype!r} has heterogeneous dexterous-hand finger DOF: "
                    f"{ds_dir} has {(kL, kR)} but an earlier dataset had {hand_dims}. "
                    f"A per-robot-type 'hand' stats block cannot describe both; the reader "
                    f"slices it by each bucket's own kL/kR. Split these into distinct robot_types."
                )
        if trim_enabled:
            info = parse_info_json(Path(ds_dir))
            file_paths = [
                path
                for path, _, _ in _discover_data_parquets(
                    Path(ds_dir),
                    info["data_path"],
                )
            ]
        else:
            file_paths = []
            for chunk in sorted(os.listdir(data_dir)):
                chunk_path = data_dir / chunk
                if not chunk_path.is_dir():
                    continue
                file_paths.extend(
                    chunk_path / fname for fname in sorted(os.listdir(chunk_path)) if fname.endswith(".parquet")
                )
        if not file_paths:
            if trim_enabled:
                raise FileNotFoundError(f"RoboCOIN stats: no data parquet files under {data_dir}")
            continue

        kept_spans = None
        manifest_end = 0
        if trim_enabled:
            kept_spans, manifest_end, population_entry = _stats_population_spans(
                ds_dir,
                dataset_id=dataset_id,
                dataset_spec=trim_spec.get(dataset_id, {}),
                excluded_episode_indices=exclusion_snapshot.episode_indices,
                split=split,
                info=info,
            )
            population_datasets[dataset_id] = population_entry

        file_entries = []
        physical_end = 0
        for fpath in file_paths:
            try:
                parquet_file = pq.ParquetFile(fpath)
                present = set(parquet_file.schema_arrow.names)
                n_rows = int(parquet_file.metadata.num_rows)
                file_entries.append((fpath, physical_end, physical_end + n_rows, present))
                physical_end += n_rows
            except Exception as e:
                if trim_enabled:
                    raise
                print(f"  Warning: skipping {fpath}: {e}")
        if trim_enabled and manifest_end > physical_end:
            raise DataContractError(
                f"RoboCOIN({Path(ds_dir).name}): manifest ends at global row {manifest_end}, "
                f"past the physical parquet row count {physical_end}"
            )

        for fpath, file_start, file_end, present in file_entries:
            try:
                has_grip = all(c in present for c in _GRIP_COLS)
                cols = list(_NEEDED_COLS) if has_grip else list(_EEF_COLS)
                if layout is not None:
                    cols = cols + _HAND_RAW_COLS
                df = pd.read_parquet(fpath, columns=cols)
                if trim_enabled:
                    expected_rows = file_end - file_start
                    if len(df) != expected_rows:
                        raise DataContractError(
                            f"RoboCOIN({Path(ds_dir).name}): {fpath} metadata declares "
                            f"{expected_rows} rows but reading returned {len(df)}"
                        )
                    df = _slice_file_to_global_spans(df, file_start, file_end, kept_spans)

                total_files += 1
                grip_files += int(has_grip)
                if layout is not None:
                    hand_files += 1
                if len(df) == 0:
                    continue

                eef_a = np.stack(df["eef_sim_pose_action"].values).astype(np.float32)
                eef_s = np.stack(df["eef_sim_pose_state"].values).astype(np.float32)
                if has_grip:
                    grip_a = np.stack(df["gripper_open_scale_action"].values).astype(np.float32)
                    grip_s = np.stack(df["gripper_open_scale_state"].values).astype(np.float32)
                else:
                    grip_a = np.zeros((len(eef_a), 2), dtype=np.float32)
                    grip_s = np.zeros((len(eef_s), 2), dtype=np.float32)

                action_20d = _eef14_to_eef20(eef_a, grip_a)
                state_20d = _eef14_to_eef20(eef_s, grip_s)
                acc.update_batch(np.concatenate([action_20d, state_20d], axis=0))

                if layout is not None:
                    act_arr = np.stack(df["action"].values).astype(np.float32)
                    state_arr = np.stack(df["observation.state"].values).astype(np.float32)
                    fingers = np.concatenate([act_arr[:, idx_act_lr], state_arr[:, idx_state_lr]], axis=0)
                    hand_acc.update_batch(fingers)
            except Exception as e:
                if trim_enabled:
                    raise
                print(f"  Warning: skipping {fpath}: {e}")

    if trim_enabled and acc.count == 0:
        raise DataContractError(f"RoboCOIN stats for robot_type {rtype!r}: trimming left no EEF rows")

    stats = acc.finalize()
    if rot6d_identity:
        pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    stats["num_timesteps"] = int(acc.count)
    stats["num_datasets"] = len(dataset_dirs)
    stats["num_files"] = total_files
    stats["robot_type"] = rtype
    stats["pool"] = "action+state"

    stats["grip_present"] = total_files > 0 and grip_files == total_files

    result = {
        "eef": stats,
        "whole_bucket_exclusions_provenance": robocoin_bucket_exclusions_provenance(rtype),
    }
    if hand_acc is not None and hand_acc.count > 0:
        hand = hand_acc.finalize()
        hand["dof_left"] = hand_dims[0]
        hand["dof_right"] = hand_dims[1]
        hand["num_timesteps"] = int(hand_acc.count)
        hand["num_files"] = hand_files
        hand["pool"] = "action+state"
        hand["layout"] = "left_fingers + right_fingers"
        result["hand"] = hand
    if trim_enabled:
        result["population"] = {
            "schema_version": 2,
            "split": split,
            "policy": "info_split_then_exclusion_then_trim",
            "datasets": dict(sorted(population_datasets.items())),
        }
        result["trim_provenance"] = trim_snapshot.provenance
        result["excluded_episodes_provenance"] = _excluded_episodes_provenance(exclusion_snapshots)
        _assert_trim_snapshot_current(trim_snapshot, context=f"stats scan for robot_type {rtype!r}")
        for snapshot in exclusion_snapshots.values():
            assert_excluded_episodes_snapshot_current(
                snapshot,
                context=f"stats scan for robot_type {rtype!r}",
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--robot_type", default=None, help="Compute stats for a single robot type only")
    parser.add_argument(
        "--split",
        default="train",
        help="info.json split used for normalization statistics (default: train)",
    )
    parser.add_argument(
        "--trim_csv",
        default=None,
        help="Apply the same leading/trailing episode trims as the RoboCOIN reader",
    )
    parser.add_argument(
        "--no-rot6d-identity",
        action="store_true",
        help="Disable pinning rot6d stats to identity (rot6d would then be per-dim normalized "
        "like pos/gripper — generally undesirable; see pin_rot6d_identity).",
    )
    args = parser.parse_args()

    groups = discover_datasets_by_robot_type(args.dataset_dir)
    print(f"Found {len(groups)} robot types: {sorted(groups.keys())}")

    trim_snapshot = None
    if args.trim_csv is not None:
        trim_snapshot = _load_trim_snapshot(args.trim_csv)
        trim_spec = trim_snapshot.spec
        bucket_names = {Path(ds_dir).name for ds_list in groups.values() for ds_dir in ds_list}
        if bucket_names and bucket_names.isdisjoint(trim_spec):
            raise ValueError(
                f"RoboCOIN trim_csv {args.trim_csv}: none of its {len(trim_spec)} dataset key(s) "
                f"match any bucket directory under {args.dataset_dir}; stats would be untrimmed."
            )

    out_dir = os.path.join(args.dataset_dir, "meta")
    os.makedirs(out_dir, exist_ok=True)

    for rtype in sorted(groups.keys()):
        if args.robot_type and rtype != args.robot_type:
            continue
        ds_list = groups[rtype]
        print(f"\n{'=' * 60}")
        print(f"Computing stats for {rtype} ({len(ds_list)} datasets)...")
        result = compute_stats_for_robot_type(
            rtype,
            ds_list,
            rot6d_identity=not args.no_rot6d_identity,
            trim_csv=args.trim_csv,
            split=args.split,
            _trim_snapshot=trim_snapshot,
        )
        stats = result["eef"]

        out_path = os.path.join(out_dir, f"stats_{rtype}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  timesteps: {stats['num_timesteps']:,}")
        print(f"  mean[:5]: {stats['mean'][:5]}")
        print(f"  std[:5]:  {stats['std'][:5]}")
        print(f"  min[:5]:  {stats['min'][:5]}")
        print(f"  max[:5]:  {stats['max'][:5]}")
        if "hand" in result:
            h = result["hand"]
            print(f"  hand: dof L/R={h['dof_left']}/{h['dof_right']}, timesteps={h['num_timesteps']:,}")
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
