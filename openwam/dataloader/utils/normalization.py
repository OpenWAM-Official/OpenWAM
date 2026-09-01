"""Shared in-reader normalization helpers.

Hosts ``apply_normalization``, the pure function form of RoboCOIN's
private ``_normalize_array``. Lifted out so OXE readers can share the same
min-max / z-score / null code paths, including quantile normalization.

Each reader keeps its own ``self._normalization_stats`` dict (loaded at
__init__ from a per-dataset stats json) and calls ``apply_normalization``
per ``_getitem_impl`` invocation. The function is stateless: returns the
input untouched whenever ``stats is None`` or ``mode`` is one of the
``no-op`` aliases.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_NO_OP_MODES = (None, "none", "null")

# The six stat vectors every materialized stats dict carries (also the keys
# readers hand to ``_write_deploy_normalizer_stats``).
STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")

# rot6d dims within the two EEF stat layouts:
#   10-D single-arm  [pos(0:3), rot6d(3:9), grip(9)]                 (OXE)
#   20-D bimanual    [L_pos, L_rot6d(3:9), L_grip, R_pos, R_rot6d(13:19), R_grip]  (RoboCOIN)
# rot6d entries are rotation-matrix basis components: already bounded in [-1, 1]
# and geometrically COUPLED (two unit 3-vectors). Per-dim affine normalization
# would scale each of the 6 independently — breaking the unit-norm structure and
# reweighting the rotation regression loss across dims, which is geometrically
# meaningless. The stats-computation scripts therefore pin these dims to
# identity (:func:`pin_rot6d_identity`) and :func:`materialize_eef_stats` warns
# when it loads a stats file that predates the pinning.
ROT6D_DIMS_ARM10 = (3, 4, 5, 6, 7, 8)
ROT6D_DIMS_EEF20 = (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)
_ROT6D_DIMS_BY_WIDTH = {10: ROT6D_DIMS_ARM10, 20: ROT6D_DIMS_EEF20}

# Identity stat values that make every normalize mode a pass-through.
_ROT6D_IDENTITY = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}

# Which materialized fields the active mode actually consumes (for the
# stale-stats warning: only complain about fields that would distort data).
_MODE_IDENTITY_FIELDS = {
    "quantile": ("q01", "q99"),
    "min-max": ("min", "max"),
    "z-score": ("mean", "std"),
}


def pin_rot6d_identity(stats: dict, dims: Sequence[int]) -> None:
    """In-place: force the rot6d dims of a flat EEF stats dict to identity.

    min=-1 / max=1 (min-max identity), q01=-1 / q99=1 (quantile identity),
    mean=0 / std=1 (z-score identity) → normalization is a pass-through on rot6d
    under EVERY mode, so the rotation representation reaches the model unchanged.
    pos / gripper dims are left untouched.

    Shared by the OXE (``dims=ROT6D_DIMS_ARM10``) and RoboCOIN
    (``dims=ROT6D_DIMS_EEF20``) stats-computation scripts.
    """
    for key, val in _ROT6D_IDENTITY.items():
        for i in dims:
            stats[key][i] = val


# Shared division-by-zero floor for read-time normalization. Referenced by
# both ``apply_normalization`` (in-reader path) and ``transforms.normalize``'s
# ``Normalizer`` default ``eps`` so the two paths agree on near-zero-variance
# dims. (The compute-time degenerate-dim handling in the *_compute_stats.py
# scripts is a separate concern with its own threshold.)
NORM_EPS = 1e-6


def apply_normalization(
    arr: np.ndarray,
    stats: Optional[dict],
    mode: Optional[str],
) -> np.ndarray:
    """Normalize ``arr`` according to ``mode`` using ``stats``.

    Args:
        arr: ``(..., D)`` float array. Last axis must match the stats vectors'
            length; broadcasting handles arbitrary leading dims.
        stats: dict carrying at least ``mean``, ``std``, ``min``, ``max`` (each
            ``(D,)`` ndarray). For ``quantile`` mode, also requires ``q01`` and
            ``q99``. ``None`` short-circuits to passthrough.
        mode: one of ``"min-max"``, ``"z-score"``, ``"quantile"``, or any of
            the no-op aliases (``None`` / ``"none"`` / ``"null"``). Unknown
            modes also pass through unchanged — matches the original
            ``_normalize_array`` behaviour.

    Returns:
        Normalized array of the same shape and dtype, or ``arr`` itself
        when the call is a no-op.

    Modes:
        * ``min-max``: ``clip((arr - min) / (max - min) * 2 - 1, -1, 1)``.
          Values within the training ``[min, max]`` map to ``[-1, 1]``;
          out-of-distribution values are clamped to the boundary (same bounded
          contract as ``quantile``).
        * ``z-score``: ``(arr - mean) / std`` with std floored at 1e-6.
          Unbounded by design (standardization), so NOT clipped.
        * ``quantile``: ``clip((arr - q01) / (q99 - q01) * 2 - 1, -1, 1)``.
          Robust to outliers (Fractal has a y-axis action range of [-5.5, 22.09]
          which would compress 99% of values into a tiny window under
          min-max). Quantile clips the outliers to the boundary.

    The min-max / z-score implementations are bit-identical to the original
    ``robocoin.RoboCOINDataset._normalize_array``.
    """
    if stats is None or mode in _NO_OP_MODES:
        return arr
    if mode == "z-score":
        mean = stats["mean"]
        std = np.maximum(stats["std"], NORM_EPS)
        return (arr - mean) / std
    if mode == "min-max":
        mn = stats["min"]
        mx = stats["max"]
        scale = np.maximum(mx - mn, NORM_EPS)
        scaled = ((arr - mn) / scale) * 2.0 - 1.0
        # Clip to [-1, 1] (consistent with quantile): values within the training
        # [min, max] map into the band; out-of-distribution values are clamped to
        # the boundary instead of extrapolating to large magnitudes.
        return np.clip(scaled, -1.0, 1.0)
    if mode == "quantile":
        if "q01" not in stats or "q99" not in stats:
            raise KeyError(
                "quantile normalization requires stats['q01'] and stats['q99']; "
                "rerun the OXE compute_stats script to populate quantile fields."
            )
        q01 = stats["q01"]
        q99 = stats["q99"]
        scale = np.maximum(q99 - q01, NORM_EPS)
        scaled = ((arr - q01) / scale) * 2.0 - 1.0
        return np.clip(scaled, -1.0, 1.0)
    return arr


def materialize_eef_stats(
    raw: dict,
    mode: Optional[str],
    *,
    dim: int,
    strict_minmax: bool,
    source_hint: str = "",
    force_rot6d_identity: bool = False,
) -> dict:
    """Validate + materialize an in-reader normalization stats dict.

    Shared by the OXE readers (per-dataset ``meta/eef_stats.json``, 10-D,
    ``strict_minmax=True``) and RoboCOIN (per-robot-type
    ``stats_<robot>.json`` ``eef`` subdict, 20-D, ``strict_minmax=False``).

    Validates that the field(s) the active ``mode`` needs are present, then
    returns a dict with all six stat vectors as float32 arrays (absent
    optional fields filled with neutral defaults).

    Args:
        raw: raw stats dict (already drilled to the eef level for RoboCOIN).
        mode: ``"min-max"`` | ``"z-score"`` | ``"quantile"``.
        dim: stat-vector length (10 single-arm OXE, 20 bimanual RoboCOIN).
        strict_minmax: when True, read ``min``/``max`` via direct indexing —
            raises ``KeyError`` if absent regardless of mode (preserves the OXE
            ``_load_eef_stats`` behavior). When False, fall back to neutral
            defaults (preserves RoboCOIN ``_load_stats`` behavior).
        source_hint: appended to the missing-field error for actionable guidance.
        force_rot6d_identity: pin the canonical rot6d slots to identity at load
            time. Readers whose action contract forbids rot6d normalization set
            this even though current generators already emit pinned files; this
            makes the loader safe against stale or hand-edited stats.
    """
    required = ("min", "max")
    if mode == "z-score":
        required = ("mean", "std")
    if mode == "quantile":
        required = ("q01", "q99")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"missing field(s) {missing} required by normalize_mode={mode!r}. {source_hint}".strip())
    if strict_minmax:
        mn = np.array(raw["min"], dtype=np.float32)
        mx = np.array(raw["max"], dtype=np.float32)
    else:
        mn = np.array(raw.get("min", [-1.0] * dim), dtype=np.float32)
        mx = np.array(raw.get("max", [1.0] * dim), dtype=np.float32)
    out = {
        "min": mn,
        "max": mx,
        "mean": np.array(raw.get("mean", [0.0] * dim), dtype=np.float32),
        "std": np.array(raw.get("std", [1.0] * dim), dtype=np.float32),
        "q01": np.array(raw.get("q01", [-1.0] * dim), dtype=np.float32),
        "q99": np.array(raw.get("q99", [1.0] * dim), dtype=np.float32),
    }

    rot6d_dims = _ROT6D_DIMS_BY_WIDTH.get(dim)
    if force_rot6d_identity and rot6d_dims is not None:
        pin_rot6d_identity(out, rot6d_dims)

    # Stale-stats guard: the rot6d identity pin normally happens at
    # stats-GENERATION time,
    # so a stats file written by a pre-pin script silently keeps the distorted
    # per-dim rot6d normalization. Warn (don't raise — --no-rot6d-identity is a
    # legitimate escape hatch) when the mode-relevant fields aren't identity.
    fields = _MODE_IDENTITY_FIELDS.get(mode)
    if rot6d_dims is not None and fields is not None:
        idx = list(rot6d_dims)
        if any(not np.allclose(out[f][idx], _ROT6D_IDENTITY[f], atol=1e-6) for f in fields):
            logger.warning(
                "rot6d dims %s of this stats file are not identity under mode=%r — the file "
                "likely predates rot6d identity pinning and normalization WILL distort the "
                "rotation representation. Rerun the matching *_stats_computation script. %s",
                idx,
                mode,
                source_hint,
            )

    return out


def load_stats_file(
    path: str | Path,
    *,
    action_mode: Optional[str],
    normalize_mode: Optional[str],
    dim: int,
) -> dict:
    """Load, materialize, and validate a training-time stats file (``.json``/``.npy``).

    Shared by the LeRobot v3 readers whose stats-computation scripts emit either
    a flat stats mapping or one nested per ``action_mode``. The raw read is
    cached per process (multi-bucket datasets and every DataLoader worker read
    the file once); materialization runs per call so each caller owns its arrays.
    """
    resolved = Path(path).expanduser().resolve()
    raw = _load_raw_stats(str(resolved), action_mode)
    stats = materialize_eef_stats(
        dict(raw),
        normalize_mode,
        dim=dim,
        strict_minmax=False,
        source_hint=f"{resolved}:{action_mode}",
    )
    bad = {key: stats[key].shape for key in STAT_KEYS if stats[key].shape != (dim,)}
    if bad:
        raise ValueError(f"normalization stats vectors must have shape ({dim},), got {bad}")
    return stats


def load_stats_metadata(path: str | Path, *, action_mode: Optional[str]) -> Mapping:
    """Return the RAW stats mapping so callers can read non-stat-vector fields.

    ``load_stats_file`` materializes only the six stat vectors; readers that
    persist a provenance/convention marker alongside them (e.g. LIBERO's
    ``gripper_convention``) read it through here. Shares ``_load_raw_stats``'s
    per-process cache, so this costs no extra I/O when the stats file has
    already been loaded.

    The returned mapping IS the cached object — treat it as read-only. Copy it
    (``dict(...)``, as ``load_stats_file`` does) before mutating, or every later
    reader in this process sees the edit.
    """
    return _load_raw_stats(str(Path(path).expanduser().resolve()), action_mode)


@functools.lru_cache(maxsize=8)
def _load_raw_stats(path: str, action_mode: Optional[str]) -> Mapping:
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    if stats_path.suffix == ".json":
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        raw = np.load(stats_path, allow_pickle=True).item()
    if not isinstance(raw, Mapping):
        raise ValueError(f"normalization stats must contain a mapping, got {type(raw).__name__}")
    if action_mode and action_mode in raw:
        return raw[action_mode]
    if not any(key in raw for key in STAT_KEYS):
        raise KeyError(f"normalization stats {stats_path} do not contain action_mode={action_mode!r}")
    return raw


__all__ = [
    "apply_normalization",
    "load_stats_file",
    "load_stats_metadata",
    "materialize_eef_stats",
    "pin_rot6d_identity",
    "ROT6D_DIMS_ARM10",
    "ROT6D_DIMS_EEF20",
    "NORM_EPS",
    "STAT_KEYS",
]
