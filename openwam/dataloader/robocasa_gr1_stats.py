"""Cached normalization-stats loading for RoboCasa GR1 dataloaders."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from openwam.dataloader.utils.normalization import materialize_eef_stats

_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99")


@functools.lru_cache(maxsize=8)
def _load_raw_stats(path: str, action_mode: str | None) -> Mapping:
    """Read an immutable training-time stats file once per process."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = np.load(path, allow_pickle=True).item()
    if not isinstance(raw, Mapping):
        raise ValueError(f"normalization stats must contain a mapping, got {type(raw).__name__}")
    if action_mode and action_mode in raw:
        return raw[action_mode]
    if any(key in raw for key in _STAT_KEYS):
        return raw
    raise KeyError(f"normalization stats {path} do not contain action_mode={action_mode!r}")


def load_stats_file(
    path: str | Path,
    *,
    action_mode: str | None,
    normalize_mode: str | None,
    dim: int,
) -> dict:
    """Load and validate flat or nested JSON/NPY stats for one raw action mode."""
    resolved = str(Path(path).expanduser().resolve())
    raw = _load_raw_stats(resolved, action_mode)
    stats = materialize_eef_stats(
        dict(raw),
        normalize_mode,
        dim=dim,
        strict_minmax=False,
        source_hint=f"{resolved}:{action_mode}",
    )
    widths = {key: stats[key].shape for key in _STAT_KEYS}
    bad = {key: shape for key, shape in widths.items() if shape != (dim,)}
    if bad:
        raise ValueError(f"normalization stats vectors must have shape ({dim},), got {bad}")
    return stats


__all__ = ["load_stats_file"]
