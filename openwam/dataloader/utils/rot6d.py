"""Wuji/Astribot 58-D EEF rot6d conversion helpers.

The source stores the first two rows of each rotation matrix.  OpenWAM's
Wuji adapter stores the first two columns and places each hand immediately
after its corresponding EEF block.
"""

from __future__ import annotations

import numpy as np

_EPS = np.float32(1e-8)


def _as_float32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[-1] != 6:
        raise ValueError(f"rot6d input last dimension must be 6, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("rot6d input contains NaN or infinity")
    return arr


def row_rot6d_to_col_rot6d(values: np.ndarray) -> np.ndarray:
    """Convert first-two-row rot6d to first-two-column rot6d."""
    x = _as_float32(values)
    r0 = x[..., 0:3]
    r0 = r0 / np.maximum(np.linalg.norm(r0, axis=-1, keepdims=True), _EPS)
    r1 = x[..., 3:6] - np.sum(r0 * x[..., 3:6], axis=-1, keepdims=True) * r0
    r1 = r1 / np.maximum(np.linalg.norm(r1, axis=-1, keepdims=True), _EPS)
    r2 = np.cross(r0, r1, axis=-1)
    return np.concatenate(
        (r0[..., 0:1], r1[..., 0:1], r2[..., 0:1], r0[..., 1:2], r1[..., 1:2], r2[..., 1:2]),
        axis=-1,
    ).astype(np.float32, copy=False)


def convert_wuji_58(values: np.ndarray, *, row_rot6d: bool = True) -> np.ndarray:
    """Convert Wuji's 58-D layout to ``[L EEF, L hand, R EEF, R hand]``."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim == 0 or x.shape[-1] != 58:
        raise ValueError(f"Wuji input last dimension must be 58, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Wuji 58-D input contains NaN or infinity")
    left_eef = x[..., 0:9].copy()
    right_eef = x[..., 9:18].copy()
    if row_rot6d:
        left_eef[..., 3:9] = row_rot6d_to_col_rot6d(left_eef[..., 3:9])
        right_eef[..., 3:9] = row_rot6d_to_col_rot6d(right_eef[..., 3:9])
    return np.concatenate((left_eef, x[..., 18:38], right_eef, x[..., 38:58]), axis=-1).astype(np.float32)


def wuji_58_rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Recover ``(..., 2, 3, 3)`` matrices from converted 58-D values."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim == 0 or x.shape[-1] != 58:
        raise ValueError(f"Wuji input last dimension must be 58, got {x.shape}")
    out = []
    for start in (3, 32):
        six = x[..., start : start + 6]
        # Column convention: [c0.x,c1.x,c2.x,c0.y,c1.y,c2.y].  The z row is
        # reconstructed from the two stored columns.
        c0 = six[..., 0:3]
        c1 = six[..., 3:6]
        c2 = np.cross(c0, c1, axis=-1)
        out.append(np.stack((c0, c1, c2), axis=-1))
    return np.stack(out, axis=-3).astype(np.float32)


__all__ = ["row_rot6d_to_col_rot6d", "convert_wuji_58", "wuji_58_rot6d_to_matrix"]
