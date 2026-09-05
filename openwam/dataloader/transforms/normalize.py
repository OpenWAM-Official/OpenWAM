"""Action normalization transforms with multiple strategies.

Supported modes:
  - q99:      2 * (x - q01) / (q99 - q01) - 1   → [-1, 1]
  - min_max:  2 * (x - min) / (max - min) - 1    → [-1, 1]
  - mean_std: (x - mean) / std                    → unbounded
  - binary:   x > threshold                       → {0, 1}
  - scale:    x / max(|min|, |max|)               → ~[-1, 1]

All modes except ``binary`` are invertible for inference-time unnormalization.
"""

from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch

from openwam.dataloader.transforms.base import InvertibleModalityTransform
from openwam.dataloader.utils.normalization import NORM_EPS, apply_normalization

_NORMALIZE_STATS_KEY = "_normalize_stats"


class NormMode(str, Enum):
    Q99 = "q99"
    MIN_MAX = "min_max"
    MEAN_STD = "mean_std"
    BINARY = "binary"
    SCALE = "scale"


class Normalizer(InvertibleModalityTransform):
    """Multi-strategy normalizer for continuous data.

    Args:
        mode: Normalization strategy.
        stats: Dict with keys depending on mode:
            - q99: {q01, q99}
            - min_max: {min, max}
            - mean_std: {mean, std}
            - binary: (no stats needed)
            - scale: {min, max}
            A reserved ``_normalize_stats`` block may provide different
            statistics for ``normalize`` (proprio input); top-level statistics
            remain the source for ``unnormalize`` (model action output).
        binary_threshold: Threshold for binary mode.
        eps: Small constant to avoid division by zero.
    """

    def __init__(
        self,
        mode: str = "q99",
        stats: Optional[Dict[str, np.ndarray]] = None,
        binary_threshold: float = 0.5,
        eps: float = NORM_EPS,
    ):
        super().__init__(apply_to=["action"])
        self.mode = NormMode(mode)
        raw_stats = stats or {}
        self.stats = {key: value for key, value in raw_stats.items() if key != _NORMALIZE_STATS_KEY}
        self.normalize_stats = raw_stats.get(_NORMALIZE_STATS_KEY, self.stats)
        self.binary_threshold = binary_threshold
        self.eps = eps

        # Precompute scale/offset for fast apply/unapply; _mode_stats feeds the
        # min_max/q99 normalize delegation to the training-side formula.
        self._scale = None
        self._offset = None
        self._mode_stats = None
        self._normalize_scale = None
        self._normalize_offset = None
        self._normalize_mode_stats = None
        if stats:
            self._precompute()

    def _precompute(self):
        """Precompute scale and offset for the chosen mode."""
        s = self.stats
        if self.mode == NormMode.Q99:
            q01 = np.asarray(s["q01"], dtype=np.float32)
            q99 = np.asarray(s["q99"], dtype=np.float32)
            range_ = np.maximum(q99 - q01, self.eps)
            self._scale = 2.0 / range_
            self._offset = q01 + range_ / 2.0  # center
            self._mode_stats = {"q01": q01, "q99": q99}

        elif self.mode == NormMode.MIN_MAX:
            lo = np.asarray(s["min"], dtype=np.float32)
            hi = np.asarray(s["max"], dtype=np.float32)
            range_ = np.maximum(hi - lo, self.eps)
            self._scale = 2.0 / range_
            self._offset = lo + range_ / 2.0
            self._mode_stats = {"min": lo, "max": hi}

        elif self.mode == NormMode.MEAN_STD:
            self._offset = np.asarray(s["mean"], dtype=np.float32)
            self._scale = 1.0 / np.maximum(np.asarray(s["std"], dtype=np.float32), self.eps)

        elif self.mode == NormMode.SCALE:
            lo = np.asarray(s["min"], dtype=np.float32)
            hi = np.asarray(s["max"], dtype=np.float32)
            abs_max = np.maximum(np.abs(lo), np.abs(hi))
            self._scale = 1.0 / np.maximum(abs_max, self.eps)
            self._offset = np.zeros_like(lo)

        # Deployment can be directional: model actions are unnormalized with
        # action statistics, while raw proprio is normalized with state
        # statistics. Existing checkpoints omit the reserved state block and
        # retain the symmetric behavior above.
        ns = self.normalize_stats
        if self.mode == NormMode.Q99:
            q01 = np.asarray(ns["q01"], dtype=np.float32)
            q99 = np.asarray(ns["q99"], dtype=np.float32)
            range_ = np.maximum(q99 - q01, self.eps)
            self._normalize_scale = 2.0 / range_
            self._normalize_offset = q01 + range_ / 2.0
            self._normalize_mode_stats = {"q01": q01, "q99": q99}
        elif self.mode == NormMode.MIN_MAX:
            lo = np.asarray(ns["min"], dtype=np.float32)
            hi = np.asarray(ns["max"], dtype=np.float32)
            range_ = np.maximum(hi - lo, self.eps)
            self._normalize_scale = 2.0 / range_
            self._normalize_offset = lo + range_ / 2.0
            self._normalize_mode_stats = {"min": lo, "max": hi}
        elif self.mode == NormMode.MEAN_STD:
            self._normalize_offset = np.asarray(ns["mean"], dtype=np.float32)
            self._normalize_scale = 1.0 / np.maximum(np.asarray(ns["std"], dtype=np.float32), self.eps)
        elif self.mode == NormMode.SCALE:
            lo = np.asarray(ns["min"], dtype=np.float32)
            hi = np.asarray(ns["max"], dtype=np.float32)
            self._normalize_scale = 1.0 / np.maximum(np.maximum(np.abs(lo), np.abs(hi)), self.eps)
            self._normalize_offset = np.zeros_like(lo)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize a raw array."""
        if self.mode == NormMode.BINARY:
            return (x > self.binary_threshold).astype(np.float32)

        if self._normalize_scale is None:
            return x

        if self.mode in (NormMode.MIN_MAX, NormMode.Q99):
            # Delegate to the training-side formula so train/deploy parity
            # holds by construction — incl. the [-1,1] clip and exactness on
            # degenerate constant dims, where the precomputed (x-offset)*scale
            # form absorbs the eps into offset in float32 (|c|>=16 → 0.0
            # instead of training's -1, unbounded on out-of-range inputs).
            # apply_normalization uses NORM_EPS; self.eps is deliberately not
            # honored here — parity requires the training-side eps.
            mode = "min-max" if self.mode == NormMode.MIN_MAX else "quantile"
            return apply_normalization(x, self._normalize_mode_stats, mode).astype(np.float32)

        return ((x - self._normalize_offset) * self._normalize_scale).astype(np.float32)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.mode == NormMode.BINARY:
            return x  # Not invertible in a meaningful way

        if self._scale is None:
            return x

        return (x / self._scale) + self._offset

    def apply(self, data: dict) -> dict:
        for key in self.apply_to:
            if key in data and data[key] is not None:
                val = data[key]
                if isinstance(val, torch.Tensor):
                    data[key] = torch.from_numpy(self.normalize(val.numpy()))
                elif isinstance(val, np.ndarray):
                    data[key] = self.normalize(val)
        return data

    def unapply(self, data: dict) -> dict:
        for key in self.apply_to:
            if key in data and data[key] is not None:
                val = data[key]
                if isinstance(val, torch.Tensor):
                    data[key] = torch.from_numpy(self.unnormalize(val.numpy()))
                elif isinstance(val, np.ndarray):
                    data[key] = self.unnormalize(val)
        return data


# ---------------------------------------------------------------------------
# YAML-config-facing helpers shared by training datasets and deployment.
# ---------------------------------------------------------------------------

# Map user-facing yaml strings to the internal Normalizer modes.
#
# ``quantile`` → q99 and ``min-max`` → min_max normalize by DELEGATING to the
# training-side ``apply_normalization`` (see ``Normalizer.normalize``), so
# train/deploy parity holds by construction; unnormalization inverts the same
# linear map. Without the ``quantile`` entry, a checkpoint trained with the
# LeRobotV3 family's default ``normalize_mode=quantile`` would silently
# disable its deploy normalizer.
YAML_TO_NORM_MODE = {
    "min-max": "min_max",
    "z-score": "mean_std",
    "quantile": "q99",
}


def load_mode_stats(stats_path: str, action_mode: str) -> Optional[dict]:
    """Load ``normalization_stats.npy`` and return the sub-dict for the requested mode.

    Expected schema: ``{"joint": {...}, "eef": {...}, "num_timesteps": ...}``.
    When a sibling ``<action_mode>_state`` block exists it is attached under a
    reserved internal key so :class:`Normalizer` uses it for proprio
    normalization while preserving the action block for unnormalization.
    Returns the per-mode stats dict, or ``None`` if the file does not contain
    the requested mode.
    """
    raw = np.load(stats_path, allow_pickle=True).item()
    if action_mode in raw and isinstance(raw[action_mode], dict):
        stats = dict(raw[action_mode])
        state_key = f"{action_mode}_state"
        if state_key in raw:
            if not isinstance(raw[state_key], dict):
                raise TypeError(f"{stats_path}:{state_key} must be a stats dict")
            stats[_NORMALIZE_STATS_KEY] = raw[state_key]
        return stats
    return None
