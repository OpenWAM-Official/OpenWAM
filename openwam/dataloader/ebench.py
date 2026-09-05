"""EBench dataloader with OpenWAM unified action-space projection.

EBench stores bimanual control in LeRobot-style per-episode parquet files:

  * ``action.ee_pose``: 14-D end-effector pose, left then right,
    ``xyz + quaternion(wxyz)`` per arm (cuRobo FK of the commanded joint
    targets, expressed in each arm's own base frame)
  * ``action.gripper``: 4 finger position targets, two per hand
    (0.0 closed, 0.044 open; both fingers of a hand receive the same command)
  * ``action.base_delta``: per-step commanded base displacement
    ``[dx_m, dy_m, dyaw_DEG]`` in the robot's spawn/odom axes — the raw
    ``base_motion`` the demo sent each step (GenManip clips it to
    ±0.015 m / ±1° before applying)
  * ``action.base``: the running cumsum of ``action.base_delta`` since
    episode start (GenManip's ``genmanip2lerobot.py``), i.e. a cumulative
    commanded odom pose with yaw in DEGREES. It is *not* a velocity.
  * matching ``state.*`` keys for proprio. ``state.base`` is the measured
    dummy-base joint qpos ``[x_m, y_m, yaw_RAD]`` — note the radian yaw.

The base slot supervises ``action.base_delta`` — the per-step displacement
command ``[dx_m, dy_m, dyaw_deg]`` in the spawn/odom axes. Proprio renders the
*measured* per-step displacement ``state.base[t] - state.base[t-1]`` (yaw
wrapped, rad→deg) so proprio and action share one raw-23 stats set in the
same space.

The raw EBench action/proprio vector is 23-D:

  * ``[0:10)`` left ``xyz + rot6d + gripper``
  * ``[10:20)`` right ``xyz + rot6d + gripper``
  * ``[20:23)`` mobile base ``x, y, yaw(deg)``

With ``unify_action=true`` (the pretrain-SFT path), the 23-D raw vector is
scattered by ``unify_action_map`` into OpenWAM's shared 80-D layout:

  * ``[0:3)`` left EEF xyz, ``[3:9)`` left EEF rot6d, ``[9]`` left gripper
  * ``[10:34)`` left dexterous hand slots
  * ``[34:37)`` right EEF xyz, ``[37:43)`` right EEF rot6d, ``[43]`` right gripper
  * ``[44:68)`` right dexterous hand slots
  * ``[68:80)`` reserved; EBench base uses ``[68:71)``

The emitted action/proprio masks are 2-D, so only mapped physical dimensions
participate in loss/conditioning. With ``unify_action=false`` the reader emits
the raw 23-D vector with all 23 dimensions visible.

Robustness mirrors ``LeRobotV3Reader``: ``__getitem__`` retries up to 64
alternate windows on decode/IO failures, and wrist-camera decode errors
degrade to a black slot (the head/target camera stays fatal → retried).
``meta/excluded_episodes.json`` (list of episode indices) is honored when
present. Corrupt data *values* still fail fast: non-finite normalized
vectors, blank prompts, and missing explicitly-configured buckets raise.

This reader intentionally extends ``BaseDataset`` directly instead of
subclassing ``LeRobotV3Reader`` (EBench is LeRobot v2.1): the standalone
implementation predates the audit and was kept to preserve its review
history; the base class's operational protections (retry, wrist tolerance,
process-global parquet LRU, excluded episodes) are ported piecemeal above.
A thin v2.1 subclass remains the better long-term shape.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout, format_prompt_for_inference
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.eef import assert_unit_quaternion, quat_xyzw_to_rot6d
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames

logger = logging.getLogger(__name__)

EBENCH_UNIFY_DIM = int(UNIFY_DIM)
EBENCH_RAW_ACTION_DIM = 23

EBENCH_ACTION_KEYS = ("action.ee_pose", "action.gripper", "action.base_delta")
EBENCH_STATE_KEYS = ("state.ee_pose", "state.gripper", "state.base")
EBENCH_DEFAULT_UNIFY_ACTION_MAP = ("0-9", "34-43", "68-70")
# All modes load from the offline-scan cache at <dataset_dir>/meta/
# ebench_normalization_stats.npy, auto-built on first use (rank 0 scans, other ranks wait;
# see _load_or_build_stats). The scan's true q01/q99 make "quantile" work out
# of the box; a legacy summary-built cache without them is still rejected
# rather than silently aliased to min/max.
EBENCH_SUPPORTED_NORMALIZE_MODES = (None, "none", "null", "min-max", "z-score", "quantile")
# GenManip commands gripper fingers in [0, 0.044] m and terminates episodes
# when a measured finger leaves [-0.01, 0.054] (env.py invalid-state guard).
EBENCH_GRIPPER_CMD_RANGE = (0.0, 0.044)
EBENCH_GRIPPER_STATE_RANGE = (-0.01, 0.054)
# Max per-hand finger-command disagreement (m) before the scalar-gripper
# averaging (and its summary-stats derivation) is considered invalid. Shared
# with the offline stats scan so both validators agree.
EBENCH_FINGER_GAP_TOLERANCE = 3e-3
# Degenerate-dim floor applied to std at stats-build time (summary merge and
# offline scan alike) so z-score never divides by ~0 on constant dims.
EBENCH_STD_FLOOR = 1e-3

EBENCH_UNIFY_DST_INDEX = parse_unify_spec(EBENCH_DEFAULT_UNIFY_ACTION_MAP, EBENCH_UNIFY_DIM)

_GETITEM_MAX_RETRIES = 64
# Wrist(-slot) decode failures degrade to a black slot instead of killing the
# job; the head/target camera stays fatal so a broken window is retried.
_WRIST_DECODE_TOLERATED: tuple = (FileNotFoundError, OSError, RuntimeError)


class EBenchDataError(ValueError):
    """Deterministic data-contract violation (non-finite values, unresolvable
    prompt). NEVER masked by the ``_safe_get`` retry loop — transient IO gets
    retried, corrupt data must surface."""


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap radian angle(s) to [-pi, pi)."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def render_ebench_state_base(cur_base: np.ndarray, prev_base: Optional[np.ndarray]) -> np.ndarray:
    """Render measured ``state.base`` ``[x_m, y_m, yaw_RAD]`` into the
    ``action.base_delta`` command space: the measured per-step displacement
    ``cur - prev`` with the yaw difference wrapped to [-pi, pi) then converted
    to DEGREES. ``prev_base=None`` (episode start) yields zeros: there is no
    previous measurement to difference. (Note this is a state-side rendering
    choice; GenManip's ``action.base_delta[0]`` keeps the raw first command
    and may be non-zero.)

    The eval bridge (``benchmarks/utils/action_conversion.py``) keeps a
    byte-identical mirror of this function; a regression test pins the two
    together. Change them in lockstep.
    """
    cur = np.asarray(cur_base, dtype=np.float64).reshape(3)
    if prev_base is None:
        return np.zeros(3, dtype=np.float32)
    prev = np.asarray(prev_base, dtype=np.float64).reshape(3)
    delta = cur - prev
    dyaw = float(wrap_angle_rad(delta[2]))
    return np.array([delta[0], delta[1], np.degrees(dyaw)], dtype=np.float32)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_plain_list(value) -> Optional[list]:
    if value is None:
        return None
    try:
        from omegaconf import ListConfig, OmegaConf

        if isinstance(value, ListConfig):
            return list(OmegaConf.to_container(value, resolve=True))
    except Exception:
        pass
    if isinstance(value, str):
        return [value]
    return list(value)


def _feature_width(info: dict, key: str) -> int:
    feat = info.get("features", {}).get(key)
    if feat is None:
        raise KeyError(f"EBench info.json missing feature {key!r}")
    shape = feat.get("shape") or []
    if not shape:
        raise ValueError(f"EBench feature {key!r} has no shape in info.json")
    return int(shape[0])


def _column_matrix(frame: pd.DataFrame, key: str, expected_width: int) -> np.ndarray:
    if key not in frame.columns:
        raise KeyError(f"EBench parquet window missing column {key!r}")
    arr = np.stack(frame[key].to_numpy()).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_width:
        raise ValueError(f"EBench column {key!r} expected shape (T, {expected_width}), got {arr.shape}")
    return arr


def _quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"EBench quaternion must be 4-D wxyz, got shape {quat_wxyz.shape}")
    leading = quat_wxyz.shape[:-1]
    flat = quat_wxyz.reshape(-1, 4)
    flat_xyzw = np.concatenate([flat[:, 1:4], flat[:, 0:1]], axis=-1)
    return quat_xyzw_to_rot6d(flat_xyzw).reshape(*leading, 6).astype(np.float32)


def _ee_pose_gripper_base_to_raw23(ee_pose: np.ndarray, gripper: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Convert EBench ``ee_pose/gripper/base`` arrays to raw 23-D EEF action.

    Input layout:
      ``ee_pose``: ``[L_xyz3, L_quat_wxyz4, R_xyz3, R_quat_wxyz4]``
      ``gripper``: two finger values per hand
      ``base``: ``[x, y, yaw]``
    """
    ee_pose = np.asarray(ee_pose, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    if ee_pose.shape[-1] != 14:
        raise ValueError(f"EBench ee_pose must be 14-D, got shape {ee_pose.shape}")
    if gripper.shape[-1] != 4:
        raise ValueError(f"EBench gripper must be 4-D, got shape {gripper.shape}")
    if base.shape[-1] != 3:
        raise ValueError(f"EBench base must be 3-D, got shape {base.shape}")

    left_grip = gripper[..., 0:2].mean(axis=-1, keepdims=True)
    right_grip = gripper[..., 2:4].mean(axis=-1, keepdims=True)
    return np.concatenate(
        [
            ee_pose[..., 0:3],
            _quat_wxyz_to_rot6d(ee_pose[..., 3:7]),
            left_grip,
            ee_pose[..., 7:10],
            _quat_wxyz_to_rot6d(ee_pose[..., 10:14]),
            right_grip,
            base,
        ],
        axis=-1,
    ).astype(np.float32)


def _raw23_from_frame(frame: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """Build raw 23-D EBench vectors from ee_pose/gripper/base columns."""
    ee_pose = _column_matrix(frame, keys[0], 14)
    gripper = _column_matrix(frame, keys[1], 4)
    base = _column_matrix(frame, keys[2], 3)
    return _ee_pose_gripper_base_to_raw23(ee_pose, gripper, base)


def _raw23_to_ebench80(raw: np.ndarray) -> np.ndarray:
    """Map raw EBench 23-D vectors into the canonical OpenWAM 80-D space."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != EBENCH_RAW_ACTION_DIM:
        raise ValueError(f"EBench raw action/state must be 23-D, got shape {raw.shape}")
    unified, _ = map_to_unify(raw, EBENCH_UNIFY_DST_INDEX, EBENCH_UNIFY_DIM)
    return unified.astype(np.float32)


def ebench80_dim_mask() -> np.ndarray:
    dst_index = EBENCH_UNIFY_DST_INDEX
    mask = np.zeros(EBENCH_UNIFY_DIM, dtype=bool)
    mask[dst_index] = True
    return mask


EBENCH80_DIM_MASK = ebench80_dim_mask()
EBENCH_RAW_DIM_MASK = np.ones(EBENCH_RAW_ACTION_DIM, dtype=bool)


def _neutral_stats(dim: int = EBENCH_RAW_ACTION_DIM) -> Dict[str, np.ndarray]:
    # No q01/q99: episodes_stats.jsonl has no true quantiles and aliasing them
    # to min/max would let a "quantile" run silently degrade to min-max. True
    # quantiles come only from the offline parquet scan
    # (ebench_stats_computation); _load_or_build_stats enforces that.
    zeros = np.zeros(dim, dtype=np.float32)
    ones = np.ones(dim, dtype=np.float32)
    return {
        "mean": zeros.copy(),
        "std": ones.copy(),
        "min": zeros.copy(),
        "max": ones.copy(),
    }


def _json_count(stats_entry: dict) -> int:
    count = stats_entry.get("count", 0)
    if isinstance(count, list):
        return int(count[0]) if count else 0
    return int(count)


def _merge_scalar_stats(entries: list[dict]) -> dict:
    """Merge per-episode stats from EBench episodes_stats.jsonl."""
    nonempty = [(e, _json_count(e)) for e in entries if _json_count(e) > 0]
    if not nonempty:
        raise ValueError("Cannot merge empty stats entries")

    count = float(sum(c for _, c in nonempty))
    means = np.stack([np.asarray(e["mean"], dtype=np.float64) for e, _ in nonempty])
    stds = np.stack([np.asarray(e["std"], dtype=np.float64) for e, _ in nonempty])
    counts = np.asarray([c for _, c in nonempty], dtype=np.float64)

    mean = (means * counts[:, None]).sum(axis=0) / count
    second = ((stds**2 + means**2) * counts[:, None]).sum(axis=0) / count
    var = np.maximum(second - mean**2, 0.0)
    return {
        "min": np.min(np.stack([np.asarray(e["min"], dtype=np.float64) for e, _ in nonempty]), axis=0),
        "max": np.max(np.stack([np.asarray(e["max"], dtype=np.float64) for e, _ in nonempty]), axis=0),
        "mean": mean,
        "std": np.sqrt(var),
        "count": int(count),
    }


def _raw_stats_to_23(stats_by_key: dict, keys: Sequence[str]) -> dict:
    """Project EBench summary stats into the raw 23-D EEF schema.

    ``episodes_stats.jsonl`` stores quaternion stats for ee_pose; rot6d stats
    cannot be derived exactly from those summary moments. Following the EEF
    readers' convention, rot6d dimensions are pinned to identity stats.
    """
    ee_key, gripper_key, base_key = keys
    ee = {k: np.asarray(stats_by_key[ee_key][k], dtype=np.float32) for k in ("mean", "std", "min", "max")}
    gripper = {k: np.asarray(stats_by_key[gripper_key][k], dtype=np.float32) for k in ("mean", "std", "min", "max")}
    base = {k: np.asarray(stats_by_key[base_key][k], dtype=np.float32) for k in ("mean", "std", "min", "max")}

    out = _neutral_stats(EBENCH_RAW_ACTION_DIM)
    for key in ("mean", "std", "min", "max"):
        arr = out[key]
        arr[0:3] = ee[key][0:3]
        arr[10:13] = ee[key][7:10]
        # Scalar gripper stats average the two per-finger marginal stats. This
        # is exact only because GenManip commands both fingers of a hand with
        # the same value (gripper_open=[0.044,0.044] / close=[0.0,0.0]) — the
        # init-time two-finger equality check (_validate_sample_values) pins
        # that assumption on real data before training starts.
        arr[9] = gripper[key][0:2].mean()
        arr[19] = gripper[key][2:4].mean()
        arr[20:23] = base[key][0:3]

    for sl in (slice(3, 9), slice(13, 19)):
        out["mean"][sl] = 0.0
        out["std"][sl] = 1.0
        out["min"][sl] = -1.0
        out["max"][sl] = 1.0

    for name in out:
        out[name] = out[name].astype(np.float32)
    out["std"] = np.maximum(out["std"], EBENCH_STD_FLOOR)
    return out


def _episode_parquet_path(dataset_dir: Path, data_path_template: str, chunks_size: int, episode_index: int) -> Path:
    """Resolve an episode's parquet path from info.json's ``data_path`` template.

    Module-level so the offline stats scan (``ebench_stats_computation``)
    resolves episode files with the exact logic the reader uses.
    """
    chunk = episode_index // chunks_size
    return dataset_dir / data_path_template.format(
        episode_chunk=chunk,
        episode_index=episode_index,
        chunk_index=chunk,
    )


@functools.lru_cache(maxsize=4)
def _load_parquet_table(path: str, columns: tuple):
    """Process-global parquet LRU shared by all bucket instances.

    A per-instance cache multiplies resident tables by (buckets × DataLoader
    workers) — the design LeRobotV3Reader removed as a memory bomb. One small
    global LRU bounds memory regardless of bucket count.
    """
    return pq.read_table(path, memory_map=True, columns=list(columns))


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_excluded_indices(dataset_dir: Path) -> set:
    """Parse ``meta/excluded_episodes.json`` into a set (missing file → empty).

    Accepts both the family/scanner schema ``{"episode_indices": [...]}``
    emitted by :class:`LeRobotV3Reader` tooling and a bare list.
    Module-level so the offline stats scan (``ebench_stats_computation``)
    applies the exact exclusion semantics the reader does.
    """
    excluded_path = Path(dataset_dir) / "meta" / "excluded_episodes.json"
    if not excluded_path.exists():
        return set()
    with excluded_path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("episode_indices", [])
    return {int(i) for i in payload}


def _build_stats_from_bucket(bucket_dir: Path, keys: Sequence[str]) -> Tuple[dict, int]:
    stats_path = bucket_dir / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"EBench stats file missing: {stats_path}")

    by_key: dict[str, list[dict]] = {k: [] for k in keys}
    total_count = 0
    for row in _read_jsonl(stats_path):
        episode_stats = row.get("stats", {})
        for key in keys:
            if key not in episode_stats:
                raise KeyError(f"{stats_path} missing stats for {key!r}")
            by_key[key].append(episode_stats[key])
        first_key = keys[0]
        total_count += _json_count(episode_stats[first_key])

    merged = {key: _merge_scalar_stats(entries) for key, entries in by_key.items()}
    return _raw_stats_to_23(merged, keys), total_count


def _merge_raw_stats(parts: list[Tuple[dict, int]]) -> dict:
    parts = [(s, int(c)) for s, c in parts if int(c) > 0]
    if not parts:
        return _neutral_stats(EBENCH_RAW_ACTION_DIM)
    counts = np.asarray([c for _, c in parts], dtype=np.float64)
    total = counts.sum()
    out = _neutral_stats(EBENCH_RAW_ACTION_DIM)
    for key in ("mean", "std", "min", "max"):
        values = np.stack([np.asarray(stats[key], dtype=np.float64) for stats, _ in parts])
        if key == "mean":
            out[key] = (values * counts[:, None]).sum(axis=0) / total
        elif key == "std":
            means = np.stack([np.asarray(stats["mean"], dtype=np.float64) for stats, _ in parts])
            second = ((values**2 + means**2) * counts[:, None]).sum(axis=0) / total
            mean = out["mean"].astype(np.float64)
            out[key] = np.sqrt(np.maximum(second - mean**2, 0.0))
        elif key == "min":
            out[key] = values.min(axis=0)
        else:
            out[key] = values.max(axis=0)
    for key in out:
        out[key] = np.asarray(out[key], dtype=np.float32)
    out["std"] = np.maximum(out["std"], EBENCH_STD_FLOOR)
    return out


def _bucket_fingerprint_paths(buckets: Sequence[Path], dataset_dir: Optional[str] = None) -> list[str]:
    root = Path(dataset_dir).resolve() if dataset_dir else None
    bucket_paths = []
    for bucket in buckets:
        resolved = Path(bucket).resolve()
        if root is not None:
            try:
                bucket_paths.append(resolved.relative_to(root).as_posix())
                continue
            except ValueError:
                pass
        bucket_paths.append(resolved.as_posix())
    return sorted(bucket_paths)


def _stats_source_digest(buckets: Sequence[Path], dataset_dir: Optional[str] = None) -> str:
    """Joint sha256 over every bucket's ``episodes_stats.jsonl`` bytes.

    An in-place dataset update (re-download, re-export) must invalidate the
    stats cache; bucket *paths* alone cannot see that. Paths are mixed into
    the digest in the same dataset_dir-relative normalization as
    ``_bucket_fingerprint_paths`` so moving the whole dataset to another
    mount does NOT invalidate a byte-identical cache.
    """
    import hashlib

    root = Path(dataset_dir).resolve() if dataset_dir else None

    def _rel(bucket: Path) -> str:
        resolved = Path(bucket).resolve()
        if root is not None:
            try:
                return resolved.relative_to(root).as_posix()
            except ValueError:
                pass
        return resolved.as_posix()

    pairs = sorted((_rel(b), Path(b).resolve()) for b in buckets)
    digest = hashlib.sha256()
    for rel, bucket in pairs:
        stats_path = bucket / "meta" / "episodes_stats.jsonl"
        digest.update(rel.encode())
        digest.update(stats_path.read_bytes())
    return digest.hexdigest()


def _stats_fingerprint(
    buckets: Sequence[Path],
    action_keys: Sequence[str],
    action_mode: str,
    dataset_dir: Optional[str] = None,
) -> dict:
    return {
        # v3: dropped q01/q99 aliases from the payload; added source_digest.
        "version": 3,
        "raw_action_dim": EBENCH_RAW_ACTION_DIM,
        "action_mode": action_mode,
        "action_keys": list(action_keys),
        "buckets": _bucket_fingerprint_paths(buckets, dataset_dir),
        "source_digest": _stats_source_digest(buckets, dataset_dir),
    }


def _validate_stats_fingerprint(path: Path, payload: dict, expected: dict) -> None:
    cached = payload.get("fingerprint")
    if cached == expected:
        return
    raise ValueError(
        "EBench normalization stats cache fingerprint mismatch for "
        f"{path}. The cache was built over a different bucket set or dataset "
        "state than this run's config — delete the stale file to rebuild.\n"
        f"cached={json.dumps(cached, sort_keys=True)}\n"
        f"expected={json.dumps(expected, sort_keys=True)}"
    )


def _atomic_save_npy(path: Path, payload: dict) -> None:
    import socket
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    # pid alone collides across nodes on a shared filesystem (same local rank
    # → same pid is common); qualify with hostname + uuid like the
    # LeRobotV3Reader deploy-stats writer.
    tmp_path = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp_path.open("wb") as f:
            np.save(f, payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _stats_cache_payload(action_stats: dict, num_timesteps: int, fingerprint: dict, action_mode: str) -> dict:
    return {
        action_mode: action_stats,
        "num_timesteps": int(num_timesteps),
        "raw_action_dim_mask": EBENCH_RAW_DIM_MASK.astype(np.bool_),
        "fingerprint": fingerprint,
    }


_EBENCH_STATS_MODULE = "openwam.dataloader.utils.stats_computation.ebench_stats_computation"


def _build_stats_cache_rank0(
    path: Path,
    dataset_dir: str,
    *,
    action_mode: str,
) -> None:
    """Build the offline-scan stats cache at ``path``, coordinated across ranks.

    Rank 0 owns the full parquet scan (minutes over the complete dataset) and
    writes atomically; other ranks poll for the file. ``dist.barrier()`` is
    deliberately avoided — a minutes-long scan would trip NCCL's collective
    timeout (the robotwin/robocasa365 multi-task precedent).
    """
    # Lazy import: the stats module imports this reader's helpers at module
    # level, so a top-level import here would be circular.
    from openwam.dataloader.utils.stats_computation.ebench_stats_computation import build_and_save_ebench_stats

    try:
        import torch.distributed as dist

        dist_ready = dist.is_available() and dist.is_initialized()
    except Exception:
        dist_ready = False
    if dist_ready:
        rank = dist.get_rank()
    else:
        # torchrun sets RANK before init_process_group; honor it so pre-init
        # constructions still elect a single builder.
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

    if rank == 0:
        logger.info("No EBench stats cache at %s — running the offline parquet scan (rank 0; other ranks wait)", path)
        build_and_save_ebench_stats(
            dataset_dir,
            output=str(path),
            action_mode=action_mode,
        )
        return

    deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
    poll_interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for rank 0 to build the EBench stats cache: {path}")
        time.sleep(poll_interval)


def _load_or_build_stats(
    buckets: Sequence[Path],
    action_keys: Sequence[str],
    *,
    action_mode: str,
    dataset_dir: str,
    normalize_mode: Optional[str] = None,
) -> Tuple[dict, str]:
    """Load raw-23 action stats from ``<dataset_dir>/meta/ebench_normalization_stats.npy``.

    The cache location is fixed (the ebench_stats_computation default); when
    the file is missing it is built in place by the offline parquet scan via
    ``_build_stats_cache_rank0``. The scan carries true q01/q99, so every
    normalize mode — including "quantile" — works without manual pre-steps.
    """
    path = Path(dataset_dir) / "meta" / "ebench_normalization_stats.npy"
    if not path.exists():
        _build_stats_cache_rank0(
            path,
            dataset_dir,
            action_mode=action_mode,
        )
    raw = np.load(path, allow_pickle=True).item()
    if not isinstance(raw, dict) or action_mode not in raw:
        raise ValueError(
            f"EBench normalization stats cache {path} uses a legacy schema without "
            f"the {action_mode!r} payload and fingerprint. Delete it and rebuild."
        )
    fingerprint = _stats_fingerprint(buckets, action_keys, action_mode, dataset_dir)
    _validate_stats_fingerprint(path, raw, fingerprint)
    stats = raw[action_mode]
    if normalize_mode == "quantile" and not ("q01" in stats and "q99" in stats):
        raise ValueError(
            f"EBench stats cache {path} was built from episodes_stats.jsonl summaries and "
            "carries no true q01/q99 — normalize_mode='quantile' needs the offline parquet "
            "scan. Delete the file; it will be rebuilt by the scan on next construction."
        )
    return stats, str(path)


def discover_ebench_buckets(dataset_dir: str) -> list[Path]:
    """Discover every EBench task bucket (``<group>/<task>/meta/info.json``)."""
    root = Path(dataset_dir)
    resolved = sorted(candidate.parent.parent for candidate in root.glob("*/*/meta/info.json") if candidate.is_file())
    if not resolved:
        raise FileNotFoundError(f"No EBench buckets with meta/info.json under {root}")
    return resolved


class EBenchDataset(BaseDataset):
    """Single EBench bucket reader that emits raw 23-D or unified 80-D action/proprio."""

    DATASET_NAME = "EBench"

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        multiview: bool = True,
        target_camera: str = "video.overlook_camera_view",
        camera_layout: Optional[Sequence[str]] = None,
        normalize_mode: Optional[str] = "min-max",
        normalization_stats_path: Optional[str] = None,
        action_stats: Optional[dict] = None,
        unify_action: bool = True,
        unify_action_map: Optional[Any] = None,
        unify_state_map: Optional[Any] = None,
        dataset_id: Optional[str] = None,
        # Optional load-time video color jitter, applied consistently across a
        # clip's frames and ONLY on the train split. None / False / {} →
        # disabled (byte-identical video). Truthy → enabled; a dict overrides
        # the per-channel strengths {brightness, contrast, saturation, hue}.
        # Mirrors LeRobotV3Reader's wiring.
        color_jitter: Optional[Any] = None,
        **_unused: Any,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = dataset_id or "/".join(self._dataset_dir.parts[-2:])
        self._split = split
        self._num_frames = int(num_frames)
        self._video_stride = max(1, int(video_stride))
        self._window_stride = max(1, int(window_stride))
        self._height = int(height)
        self._width = int(width)
        self._multiview = bool(multiview)
        self._target_camera = target_camera
        self._camera_layout = list(
            camera_layout or [target_camera, "video.left_camera_view", "video.right_camera_view"]
        )
        # Color jitter is applied in _getitem_impl to the decoded clip (same
        # random factors across all frames, via VideoColorJitter). Built only
        # for the train split; val / disabled keeps video byte-identical.
        self._color_jitter = None
        if color_jitter_enabled(color_jitter) and split == "train":
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.2)),
                contrast=float(cj_get("contrast", 0.2)),
                saturation=float(cj_get("saturation", 0.2)),
                hue=float(cj_get("hue", 0.0)),
            )
        if normalize_mode not in EBENCH_SUPPORTED_NORMALIZE_MODES:
            raise ValueError(
                f"EBench normalize_mode must be one of {EBENCH_SUPPORTED_NORMALIZE_MODES}, got "
                f"{normalize_mode!r}. Unknown spellings would silently pass through un-normalized."
            )
        if normalize_mode == "quantile" and (
            action_stats is None or not ("q01" in action_stats and "q99" in action_stats)
        ):
            raise ValueError(
                "EBench normalize_mode='quantile' but action_stats are missing or carry no q01/q99 — "
                "summary-derived stats cannot serve quantile mode. Build true quantiles with: "
                f"python -m {_EBENCH_STATS_MODULE} --dataset_dir <EBench root>"
            )
        self._normalize_mode = normalize_mode
        self.normalization_stats_path = normalization_stats_path
        self._action_keys = EBENCH_ACTION_KEYS
        self._state_keys = EBENCH_STATE_KEYS
        self._data_columns = list(dict.fromkeys((*self._action_keys, *self._state_keys, "task_index")))

        self._unify_action = bool(unify_action)
        self._unify_action_map = tuple(unify_action_map or EBENCH_DEFAULT_UNIFY_ACTION_MAP)
        if unify_state_map is not None and tuple(unify_state_map) != self._unify_action_map:
            raise ValueError(
                "unify_state_map must be null or equal to unify_action_map here: "
                "EBench state shares the action's raw-23 layout"
            )
        self._raw_action_dim = EBENCH_RAW_ACTION_DIM
        if self._unify_action:
            self._unify_dst_index = parse_unify_spec(self._unify_action_map, EBENCH_UNIFY_DIM)
            if self._unify_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"EBench({self._dataset_id}) unify_action_map maps {self._unify_dst_index.shape[0]} "
                    f"source dims but raw EBench action is {self._raw_action_dim}-D"
                )
            self._action_dim = EBENCH_UNIFY_DIM
            self._dim_mask = np.zeros(EBENCH_UNIFY_DIM, dtype=bool)
            self._dim_mask[self._unify_dst_index] = True
        else:
            self._unify_dst_index = None
            self._action_dim = EBENCH_RAW_ACTION_DIM
            self._dim_mask = EBENCH_RAW_DIM_MASK.copy()

        info_path = self._dataset_dir / "meta" / "info.json"
        with info_path.open() as f:
            info = json.load(f)
        self._fps = float(info["fps"])
        self._chunks_size = int(info.get("chunks_size", 1000))
        self._data_path_template = info["data_path"]
        self._video_path_template = info["video_path"]
        self._validate_schema(info)

        self._action_stats = action_stats
        self._fail_count = 0
        self._fail_log_every = 100

        self._episodes = list(_read_jsonl(self._dataset_dir / "meta" / "episodes.jsonl"))
        self._tasks = self._load_tasks()
        self._episode_tasks = {int(ep["episode_index"]): (ep.get("tasks") or []) for ep in self._episodes}
        self._episodes = self._filter_excluded(self._episodes)
        self._episodes = self._select_split(self._episodes, info.get("splits", {}))
        if not self._episodes:
            raise ValueError(f"EBench({self._dataset_id}) split={split!r} has no episodes")
        # Prompt completeness is checked here (episodes.jsonl is already in
        # memory) so a data bug fails at construction, not masked behind the
        # __getitem__ retry loop. Episodes without inline task text can still
        # resolve via the parquet task_index → tasks.jsonl fallback, so only
        # raise when that fallback is impossible too.
        if not self._tasks:
            missing_tasks = [
                int(ep["episode_index"])
                for ep in self._episodes
                if not any(str(t).strip() for t in (self._episode_tasks.get(int(ep["episode_index"])) or []))
            ]
            if missing_tasks:
                raise ValueError(
                    f"EBench({self._dataset_id}) episodes {missing_tasks[:8]}... have no task text and "
                    "meta/tasks.jsonl is empty — prompts cannot be resolved"
                )

        # Same formula as LeRobotV3Reader: plain arange. (No last-frame append
        # for non-divisible strides — a divergent num_video_frames would break
        # mixture collation and the Wan VAE (n-1)%4==0 contract.)
        self._video_sample_indices = np.arange(0, self._num_frames, self._video_stride, dtype=np.int64)
        self._num_video_frames = int(len(self._video_sample_indices))

        lengths = np.asarray([int(ep["length"]) for ep in self._episodes], dtype=np.int64)
        min_window_len = self._num_frames if split == "val" else 2
        n_starts = np.where(
            lengths >= min_window_len,
            (lengths - min_window_len) // self._window_stride + 1,
            0,
        ).astype(np.int64)
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        self._n_total = int(self._cum_n_starts[-1])
        if self._n_total <= 0:
            raise ValueError(f"EBench({self._dataset_id}) split={split!r} produced 0 windows")

        logger.info(
            "EBench(%s, %s): %d episodes, %d windows, fps=%.1f, multiview=%s, normalize=%s",
            self._dataset_id,
            split,
            len(self._episodes),
            self._n_total,
            self._fps,
            self._multiview,
            self._normalize_mode,
        )
        self._validate_sample_values()

    def _filter_excluded(self, episodes: list[dict]) -> list[dict]:
        """Honor ``meta/excluded_episodes.json`` (see ``_load_excluded_indices``)."""
        excluded = _load_excluded_indices(self._dataset_dir)
        if not excluded:
            return episodes
        kept = [ep for ep in episodes if int(ep["episode_index"]) not in excluded]
        if len(kept) != len(episodes):
            logger.info(
                "EBench(%s): excluded %d episodes via meta/excluded_episodes.json",
                self._dataset_id,
                len(episodes) - len(kept),
            )
        return kept

    _VALIDATE_ROWS_PER_EPISODE = 64

    def _validate_sample_values(self) -> None:
        """SAMPLED data-level convention checks before training — not a full
        dataset scan. Reads the leading 64 rows of the first, middle, and last
        selected episodes.

        Catches (a) non-unit quaternions (wrong field routed into the quat
        slots), (b) non-finite ee/gripper/base values, (c) gripper values
        outside GenManip's physical ranges, and (d) per-hand finger commands
        that disagree — which would invalidate both the scalar-gripper
        averaging and the summary-stats derivation in ``_raw_stats_to_23``
        (the check pins that assumption on a sample, it does not prove it
        dataset-wide).

        A norm check cannot catch a wxyz↔xyzw reorder; that convention is
        pinned upstream (GenManip's converter names ``ee0_qw`` first and
        cuRobo FK returns wxyz) and by the bridge round-trip tests.
        """
        picks = sorted({0, len(self._episodes) // 2, len(self._episodes) - 1})
        for ep_pos in picks:
            ep = self._episodes[ep_pos]
            self._validate_episode_rows(int(ep["episode_index"]), int(ep["length"]))

    def _validate_episode_rows(self, ep_idx: int, ep_len: int) -> None:
        frame = self._read_episode_data(ep_idx, 0, min(self._VALIDATE_ROWS_PER_EPISODE, ep_len))
        ctx = f"EBench({self._dataset_id}) episode {ep_idx}"
        for key in (self._action_keys[0], self._state_keys[0]):
            ee = _column_matrix(frame, key, 14)
            if not np.isfinite(ee).all():
                raise ValueError(f"{ctx} {key} contains non-finite values")
            assert_unit_quaternion(ee[:, 3:7])
            assert_unit_quaternion(ee[:, 10:14])
        for key in (self._action_keys[2], self._state_keys[2]):
            base = _column_matrix(frame, key, 3)
            if not np.isfinite(base).all():
                raise ValueError(f"{ctx} {key} contains non-finite values")
        lo_cmd, hi_cmd = EBENCH_GRIPPER_CMD_RANGE
        lo_state, hi_state = EBENCH_GRIPPER_STATE_RANGE
        eps = 1e-4
        for key, lo, hi in (
            (self._action_keys[1], lo_cmd - eps, hi_cmd + eps),
            (self._state_keys[1], lo_state, hi_state),
        ):
            grip = _column_matrix(frame, key, 4)
            if not np.isfinite(grip).all():
                raise ValueError(f"{ctx} {key} contains non-finite values")
            if grip.min() < lo or grip.max() > hi:
                raise ValueError(f"{ctx} {key} outside [{lo}, {hi}]: min={grip.min():.4f}, max={grip.max():.4f}")
        cmd = _column_matrix(frame, self._action_keys[1], 4)
        finger_gap = max(
            float(np.abs(cmd[:, 0] - cmd[:, 1]).max()),
            float(np.abs(cmd[:, 2] - cmd[:, 3]).max()),
        )
        if finger_gap > EBENCH_FINGER_GAP_TOLERANCE:
            raise ValueError(
                f"{ctx} per-hand finger commands disagree by {finger_gap:.5f} m; "
                "the scalar-gripper averaging and its summary stats assume identical finger commands"
            )

    def _validate_schema(self, info: dict) -> None:
        for key, width in (
            ("action.ee_pose", 14),
            ("action.gripper", 4),
            ("state.ee_pose", 14),
            ("state.gripper", 4),
            ("state.base", 3),
        ):
            got = _feature_width(info, key)
            if got != width:
                raise ValueError(f"EBench({self._dataset_id}) {key} width {got} != expected {width}")
        base_key = self._action_keys[-1]
        got = _feature_width(info, base_key)
        if got != 3:
            raise ValueError(f"EBench({self._dataset_id}) {base_key} width {got} != expected 3")
        for cam in set(self._camera_layout + [self._target_camera]):
            if cam not in info.get("features", {}):
                raise KeyError(f"EBench({self._dataset_id}) camera {cam!r} not present in info.json")

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self._dataset_dir / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            return {}
        tasks = {}
        for row in _read_jsonl(tasks_path):
            tasks[int(row["task_index"])] = str(row["task"])
        return tasks

    def _select_split(self, episodes: list[dict], splits: dict) -> list[dict]:
        if self._split not in splits:
            if self._split == "train":
                return episodes
            return []
        spec = str(splits[self._split])
        if ":" not in spec:
            return episodes
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else len(episodes)
        return [ep for ep in episodes if start <= int(ep["episode_index"]) < end]

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._n_total:
            raise IndexError(f"EBench idx {idx} out of range [0, {self._n_total})")
        return self._safe_get(idx)

    def _safe_get(self, idx: int) -> dict:
        # Adapted from LeRobotV3Reader._safe_get: a flaky NFS read or one
        # corrupt mp4 frame must not kill a multi-day multi-node run. Two
        # deliberate deviations from the base:
        #   * EBenchDataError (deterministic data corruption) re-raises
        #     immediately — retrying would mask a data bug as sample churn;
        #   * the retry walks to the NEXT EPISODE's first window, not idx+1.
        #     EBench stores one mp4 per episode, so idx+1 inside a long broken
        #     episode would reopen the same bad file 64 times and still fail.
        for attempt in range(_GETITEM_MAX_RETRIES):
            try:
                return self._getitem_impl(idx)
            except EBenchDataError:
                raise
            except Exception as e:
                if attempt == _GETITEM_MAX_RETRIES - 1:
                    raise
                self._fail_count += 1
                if self._fail_count == 1 or self._fail_count % self._fail_log_every == 0:
                    logger.warning(
                        "%s(%s): %d cumulative __getitem__ failures (latest: idx=%d, %s, attempt=%d)",
                        self.DATASET_NAME,
                        self._dataset_id,
                        self._fail_count,
                        idx,
                        type(e).__name__,
                        attempt,
                    )
                idx = self._next_episode_start(idx)
        raise RuntimeError("unreachable")

    def _next_episode_start(self, idx: int) -> int:
        """First window index of the episode after the one owning ``idx``."""
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        nxt = (ep_local + 1) % len(self._episodes)
        return int(self._cum_n_starts[nxt]) % max(1, self._n_total)

    def _getitem_impl(self, idx: int) -> dict:
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        ep = self._episodes[ep_local]
        ep_idx = int(ep["episode_index"])
        ep_len = int(ep["length"])
        actual_raw_len = min(self._num_frames, ep_len - offset)

        # One extra leading row (when it exists) so delta-mode proprio can
        # difference consecutive measured base states.
        lead = 1 if offset > 0 else 0
        frame_ext = self._read_episode_data(ep_idx, offset - lead, actual_raw_len + lead)
        frame = frame_ext.iloc[lead:].reset_index(drop=True)
        action, action_mask = self._build_action(frame, actual_raw_len)
        proprio, proprio_mask = self._build_proprio(frame_ext, lead)
        video = self._decode_window_video(ep_idx, offset, actual_raw_len)
        if self._color_jitter is not None:
            # Same jitter factors across the whole clip (temporal consistency).
            video = self._color_jitter.apply({"video": video})["video"]
        video_mask = torch.from_numpy(self._video_sample_indices < actual_raw_len)

        return {
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]] if video else [],
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "video_mask": video_mask,
            "proprio": torch.from_numpy(proprio).float(),
            "proprio_mask": torch.from_numpy(proprio_mask),
            "prompt": self._prompt_for_episode(ep_idx, frame),
        }

    def _read_episode_data(self, episode_index: int, offset: int, length: int) -> pd.DataFrame:
        table = _load_parquet_table(str(self._episode_data_path(episode_index)), tuple(self._data_columns))
        return table.slice(offset, length).to_pandas()

    def _episode_data_path(self, episode_index: int) -> Path:
        return _episode_parquet_path(self._dataset_dir, self._data_path_template, self._chunks_size, episode_index)

    def _build_action(self, frame: pd.DataFrame, actual_raw_len: int) -> tuple[np.ndarray, np.ndarray]:
        T_action = self._num_frames - 1
        action = np.zeros((T_action, self._action_dim), dtype=np.float32)
        mask = np.zeros((T_action, self._action_dim), dtype=bool)
        n_valid = min(actual_raw_len, T_action)
        if n_valid > 0:
            raw = _raw23_from_frame(frame.iloc[:n_valid], self._action_keys)
            mapped = self._finalize_raw_vector(self._normalize(raw, self._action_stats))
            action[:n_valid] = mapped
            mask[:n_valid, self._dim_mask] = True
        return action, mask

    def _build_proprio(self, frame_ext: pd.DataFrame, lead: int) -> tuple[np.ndarray, np.ndarray]:
        """Render measured state at window row 0 into the action command space.

        ``frame_ext`` carries ``lead`` extra leading row(s) (1 when the window
        does not start at the episode's first frame). The base slot is rendered
        by ``render_ebench_state_base`` so proprio and action share one raw-23
        stats set in the same space — see the module docstring.
        """
        cur_row = frame_ext.iloc[lead : lead + 1]
        ee_pose = _column_matrix(cur_row, self._state_keys[0], 14)
        gripper = _column_matrix(cur_row, self._state_keys[1], 4)
        base_rows = _column_matrix(frame_ext.iloc[: lead + 1], self._state_keys[2], 3)
        prev_base = base_rows[lead - 1] if lead > 0 else None
        base = render_ebench_state_base(base_rows[lead], prev_base)
        raw = _ee_pose_gripper_base_to_raw23(ee_pose, gripper, base[None, :])
        proprio = self._finalize_raw_vector(self._normalize(raw, self._action_stats))
        mask = np.zeros((1, self._action_dim), dtype=bool)
        mask[:, self._dim_mask] = True
        return proprio.astype(np.float32), mask

    def _finalize_raw_vector(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-1] != EBENCH_RAW_ACTION_DIM:
            raise ValueError(f"EBench raw vector must be {EBENCH_RAW_ACTION_DIM}-D, got shape {raw.shape}")
        if not self._unify_action:
            return raw.astype(np.float32)
        unified, _ = map_to_unify(raw, self._unify_dst_index, EBENCH_UNIFY_DIM)
        return unified.astype(np.float32)

    def _normalize(self, arr: np.ndarray, stats: Optional[dict]) -> np.ndarray:
        # Corrupt values must surface regardless of mode (a NaN row under
        # normalize_mode=null would otherwise flow straight into the loss).
        # EBenchDataError: _safe_get must not retry data corruption away.
        if not np.isfinite(arr).all():
            bad = np.argwhere(~np.isfinite(np.asarray(arr)))
            raise EBenchDataError(f"EBench({self._dataset_id}) non-finite raw values at indices {bad[:8].tolist()}")
        if self._normalize_mode in (None, "none", "null"):
            return arr.astype(np.float32)
        if stats is None:
            # EBenchDataError: a misconfiguration (active mode, no stats) is
            # deterministic — _safe_get must not retry it through 64 windows.
            raise EBenchDataError(
                f"EBench({self._dataset_id}) normalize_mode={self._normalize_mode!r} but no stats were provided"
            )
        normalized = apply_normalization(arr, stats, self._normalize_mode)
        if not np.isfinite(normalized).all():
            # Never silently zero corrupt values into "stop" actions — a slow
            # data-corruption event must surface, not train invisibly.
            bad = np.argwhere(~np.isfinite(normalized))
            raise EBenchDataError(
                f"EBench({self._dataset_id}) non-finite normalized values at indices {bad[:8].tolist()}"
            )
        return normalized.astype(np.float32)

    def _prompt_for_episode(self, episode_index: int, frame: pd.DataFrame) -> str:
        episode_tasks = self._episode_tasks.get(episode_index) or []
        if episode_tasks and str(episode_tasks[0]).strip():
            return format_prompt_for_inference(str(episode_tasks[0]))
        task_idx = int(frame["task_index"].iloc[0]) if "task_index" in frame.columns else None
        if task_idx is not None and task_idx in self._tasks and self._tasks[task_idx].strip():
            return format_prompt_for_inference(self._tasks[task_idx])
        # A data bug must not slip into training as a fabricated prompt
        # (family convention — BEHAVIOR fails fast on blank prompts too).
        # EBenchDataError: _safe_get must not retry this away.
        raise EBenchDataError(
            f"EBench({self._dataset_id}) episode {episode_index} has no task text in "
            "meta/episodes.jsonl or meta/tasks.jsonl"
        )

    def _decode_window_video(self, episode_index: int, offset: int, actual_raw_len: int) -> List:
        real_indices = self._video_sample_indices[self._video_sample_indices < actual_raw_len]
        real_indices = (real_indices + offset).tolist()
        if not real_indices:
            real_indices = [offset]

        frames_by_camera = {}
        cameras = self._camera_layout if self._multiview else [self._target_camera]
        for cam in cameras:
            path = self._video_path(cam, episode_index)
            h, w = self._camera_slot_size(cam)
            if self._multiview and cam != cameras[0]:
                # Wrist slots tolerate decode failures with a black slot (the
                # LeRobotV3Reader convention); the head/target camera stays
                # fatal so _safe_get retries a different window.
                try:
                    frames_by_camera[cam] = _decode_video_frames(str(path), real_indices, h, w)
                except _WRIST_DECODE_TOLERATED as e:
                    self._fail_count += 1
                    if self._fail_count == 1 or self._fail_count % self._fail_log_every == 0:
                        logger.warning(
                            "EBench(%s): wrist camera %s decode failed for episode %d (%s); black slot",
                            self._dataset_id,
                            cam,
                            episode_index,
                            type(e).__name__,
                        )
                    frames_by_camera[cam] = [Image.new("RGB", (w, h), (0, 0, 0)) for _ in real_indices]
            else:
                frames_by_camera[cam] = _decode_video_frames(str(path), real_indices, h, w)

        n_real = len(next(iter(frames_by_camera.values())))
        for cam, frames in list(frames_by_camera.items()):
            if n_real < self._num_video_frames:
                frames_by_camera[cam] = frames + [frames[-1]] * (self._num_video_frames - n_real)

        if not self._multiview:
            return frames_by_camera[self._target_camera]

        video = []
        for i in range(self._num_video_frames):
            video.append(
                assemble_multiview_layout(
                    {cam: frames_by_camera[cam][i] for cam in cameras},
                    list(cameras),
                    out_h=self._height,
                    out_w=self._width,
                )
            )
        return video

    def _video_path(self, camera: str, episode_index: int) -> Path:
        chunk = episode_index // self._chunks_size
        return self._dataset_dir / self._video_path_template.format(
            video_key=camera,
            episode_chunk=chunk,
            episode_index=episode_index,
            chunk_index=chunk,
        )

    def _camera_slot_size(self, camera: str) -> tuple[int, int]:
        if not self._multiview:
            return self._height, self._width
        top_h = int(round(self._height * 2.0 / 3.0))
        if camera == self._camera_layout[0]:
            return top_h, self._width
        bottom_h = self._height - top_h
        half_w = self._width // 2
        right_w = self._width - half_w
        if len(self._camera_layout) > 1 and camera == self._camera_layout[1]:
            return bottom_h, half_w
        return bottom_h, right_w

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._action_dim

    @property
    def normalization_stats(self) -> Optional[dict]:
        # Samples leave this reader pre-normalized; None signals downstream
        # layers to skip any re-normalization (the LeRobotV3Reader family
        # contract). The deploy artifact travels via normalization_stats_path.
        return None

    @classmethod
    def from_config(cls, config, split: str = "train"):
        dataset_dir = _cfg_get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError("EBenchDataset: missing dataloader.dataset_dir")

        buckets = discover_ebench_buckets(dataset_dir)

        normalize_mode = _cfg_get(config, "normalize_mode", "min-max")
        if normalize_mode not in EBENCH_SUPPORTED_NORMALIZE_MODES:
            raise ValueError(
                f"EBench normalize_mode must be one of {EBENCH_SUPPORTED_NORMALIZE_MODES}, got {normalize_mode!r}"
            )
        action_mode = _cfg_get(config, "action_mode", "eef")
        action_keys = EBENCH_ACTION_KEYS

        action_stats = None
        resolved_stats_path = None
        if normalize_mode not in (None, "none", "null"):
            action_stats, resolved_stats_path = _load_or_build_stats(
                buckets,
                action_keys,
                action_mode=action_mode,
                dataset_dir=dataset_dir,
                normalize_mode=normalize_mode,
            )

        common = {
            "split": split,
            "num_frames": int(_cfg_get(config, "num_frames", 33)),
            "video_stride": int(_cfg_get(config, "video_stride", 4)),
            "window_stride": int(_cfg_get(config, "window_stride", 1)),
            "height": int(_cfg_get(config, "height", 384)),
            "width": int(_cfg_get(config, "width", 320)),
            "multiview": bool(_cfg_get(config, "multiview", True)),
            "target_camera": _cfg_get(config, "target_camera", "video.overlook_camera_view"),
            "camera_layout": _as_plain_list(
                _cfg_get(
                    config,
                    "camera_layout",
                    ["video.overlook_camera_view", "video.left_camera_view", "video.right_camera_view"],
                )
            ),
            "normalize_mode": normalize_mode,
            "normalization_stats_path": resolved_stats_path,
            "action_stats": action_stats,
            "unify_action": bool(_cfg_get(config, "unify_action", True)),
            "unify_action_map": _as_plain_list(_cfg_get(config, "unify_action_map", EBENCH_DEFAULT_UNIFY_ACTION_MAP)),
            "unify_state_map": _as_plain_list(_cfg_get(config, "unify_state_map", None)),
            "color_jitter": _cfg_get(config, "color_jitter", None),
        }

        readers = [
            cls(str(bucket), dataset_id=str(bucket.relative_to(Path(dataset_dir))), **common) for bucket in buckets
        ]
        if len(readers) == 1:
            return readers[0]
        return MultiEBenchDataset(readers, normalization_stats_path=resolved_stats_path)


class MultiEBenchDataset(BaseDataset):
    """Aggregate multiple EBench task buckets."""

    def __init__(
        self,
        buckets: Sequence[EBenchDataset],
        *,
        normalization_stats_path: Optional[str] = None,
    ):
        if not buckets:
            raise ValueError("MultiEBenchDataset requires at least one bucket")
        self._buckets = list(buckets)
        action_dims = {b.action_dim for b in self._buckets}
        state_dims = {b.state_dim for b in self._buckets}
        if len(action_dims) != 1 or len(state_dims) != 1:
            raise ValueError(
                f"MultiEBenchDataset buckets have inconsistent dims: action={action_dims}, state={state_dims}"
            )
        lens = np.asarray([len(b) for b in self._buckets], dtype=np.int64)
        self._cum_lens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.normalization_stats_path = normalization_stats_path
        logger.info(
            "MultiEBenchDataset: %d buckets, %d windows",
            len(self._buckets),
            len(self),
        )

    def __len__(self) -> int:
        return int(self._cum_lens[-1])

    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        if not 0 <= idx < n:
            raise IndexError(f"MultiEBenchDataset idx {idx} out of range [0, {n})")
        bi = int(np.searchsorted(self._cum_lens, idx, side="right") - 1)
        local = idx - int(self._cum_lens[bi])
        return self._buckets[bi][local]

    @property
    def action_dim(self) -> int:
        return self._buckets[0].action_dim

    @property
    def state_dim(self) -> int:
        return self._buckets[0].state_dim

    @property
    def normalization_stats(self) -> Optional[dict]:
        # Pre-normalized samples → None (see EBenchDataset.normalization_stats).
        return None

    @property
    def buckets(self) -> List[EBenchDataset]:
        return self._buckets


__all__ = [
    "EBENCH80_DIM_MASK",
    "EBenchDataset",
    "MultiEBenchDataset",
    "discover_ebench_buckets",
    "render_ebench_state_base",
    "wrap_angle_rad",
    "_raw23_to_ebench80",
]
