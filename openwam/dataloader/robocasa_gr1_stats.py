"""Normalization stats helpers for RoboCasa GR1 dataloaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

_STAT_KEYS = ("min", "max", "mean", "std")


def _as_float_vector(value, *, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"normalization stat {key!r} must be a 1-D vector, got shape {arr.shape}")
    return arr


def materialize_stats(raw: Mapping) -> dict:
    """Return a float32 stats dict with required min/max/mean/std vectors."""
    missing = [key for key in _STAT_KEYS if key not in raw]
    if missing:
        raise KeyError(f"normalization stats missing required keys: {missing}")
    out = {key: _as_float_vector(raw[key], key=key) for key in _STAT_KEYS}
    dims = {len(v) for v in out.values()}
    if len(dims) != 1:
        raise ValueError(f"normalization stat vectors must share one dim, got {sorted(dims)}")
    return out


def load_stats_file(path: str | Path, *, action_mode: str | None = None) -> dict:
    """Load flat or nested normalization stats from JSON / NPY."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = np.load(path, allow_pickle=True).item()
    if action_mode and action_mode in raw:
        raw = raw[action_mode]
    return materialize_stats(raw)


def neutralize_rot6d_stats(stats: Mapping, rot6d_slices: Iterable[Sequence[int]]) -> dict:
    """Force rot6d stat slices to identity normalization.

    rot6d columns are already bounded rotation-matrix columns. They should not
    receive dataset-derived min-max or z-score statistics.
    """
    out = {key: np.asarray(value, dtype=np.float32).copy() for key, value in stats.items()}
    for start, end in rot6d_slices:
        out["min"][start:end] = -1.0
        out["max"][start:end] = 1.0
        out["mean"][start:end] = 0.0
        out["std"][start:end] = 1.0
    return out


def compute_array_stats(arrays: Iterable[np.ndarray], *, rot6d_slices: Iterable[Sequence[int]] = ()) -> dict:
    """Compute min/max/mean/std over a stream of ``(..., D)`` arrays."""
    chunks = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            continue
        chunks.append(a.reshape(-1, a.shape[-1]))
    if not chunks:
        raise ValueError("cannot compute normalization stats from an empty array stream")
    data = np.concatenate(chunks, axis=0)
    stats = {
        "min": data.min(axis=0).astype(np.float32),
        "max": data.max(axis=0).astype(np.float32),
        "mean": data.mean(axis=0).astype(np.float32),
        "std": data.std(axis=0).astype(np.float32),
    }
    return neutralize_rot6d_stats(stats, rot6d_slices)


def save_stats_file(path: str | Path, stats: Mapping) -> None:
    """Save stats as `.npy` or `.json` based on suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = materialize_stats(stats)
    if path.suffix == ".json":
        payload = {key: value.tolist() for key, value in materialized.items()}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    else:
        np.save(path, materialized)


__all__ = [
    "compute_array_stats",
    "load_stats_file",
    "materialize_stats",
    "neutralize_rot6d_stats",
    "save_stats_file",
]
