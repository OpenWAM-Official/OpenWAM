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
from openwam.dataloader.utils.normalization import NORM_EPS


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
        self.stats = stats or {}
        self.binary_threshold = binary_threshold
        self.eps = eps

        # Precompute scale/offset for fast apply/unapply
        self._scale = None
        self._offset = None
        if stats:
            self._precompute()

    def set_stats(self, stats: Dict[str, np.ndarray]):
        """Update statistics (e.g., after loading from cache)."""
        self.stats = stats
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

        elif self.mode == NormMode.MIN_MAX:
            lo = np.asarray(s["min"], dtype=np.float32)
            hi = np.asarray(s["max"], dtype=np.float32)
            range_ = np.maximum(hi - lo, self.eps)
            self._scale = 2.0 / range_
            self._offset = lo + range_ / 2.0

        elif self.mode == NormMode.MEAN_STD:
            self._offset = np.asarray(s["mean"], dtype=np.float32)
            self._scale = 1.0 / np.maximum(np.asarray(s["std"], dtype=np.float32), self.eps)

        elif self.mode == NormMode.SCALE:
            lo = np.asarray(s["min"], dtype=np.float32)
            hi = np.asarray(s["max"], dtype=np.float32)
            abs_max = np.maximum(np.abs(lo), np.abs(hi))
            self._scale = 1.0 / np.maximum(abs_max, self.eps)
            self._offset = np.zeros_like(lo)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize a raw array."""
        if self.mode == NormMode.BINARY:
            return (x > self.binary_threshold).astype(np.float32)

        if self._scale is None:
            return x

        result = (x - self._offset) * self._scale
        if self.mode == NormMode.Q99:
            result = np.clip(result, -1.0, 1.0)
        return result.astype(np.float32)

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


class ActionNormalizer(Normalizer):
    """Convenience wrapper that normalizes the ``action`` field.

    Args:
        mode: Normalization strategy.
        stats: Action statistics dict.
        gripper_mode: Separate normalization for gripper dimension(s).
            Set to "binary" to binarize gripper, None to normalize uniformly.
        gripper_indices: Indices of gripper dimensions in the action vector.
    """

    def __init__(
        self,
        mode: str = "q99",
        stats: Optional[Dict[str, np.ndarray]] = None,
        gripper_mode: Optional[str] = None,
        gripper_indices: Optional[list] = None,
        binary_threshold: float = 0.5,
        eps: float = NORM_EPS,
    ):
        super().__init__(mode=mode, stats=stats, binary_threshold=binary_threshold, eps=eps)
        self.apply_to = ["action"]
        self.gripper_mode = gripper_mode
        self.gripper_indices = gripper_indices or []

        # Build a separate normalizer for gripper dims if needed
        self._gripper_normalizer = None
        if gripper_mode and gripper_indices:
            self._gripper_normalizer = Normalizer(
                mode=gripper_mode,
                binary_threshold=binary_threshold,
            )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        result = super().normalize(x)

        if self._gripper_normalizer and self.gripper_indices:
            for gi in self.gripper_indices:
                if gi < x.shape[-1]:
                    result[..., gi] = self._gripper_normalizer.normalize(x[..., gi])

        return result

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        result = super().unnormalize(x)

        if self._gripper_normalizer and self.gripper_indices:
            for gi in self.gripper_indices:
                if gi < x.shape[-1]:
                    result[..., gi] = self._gripper_normalizer.unnormalize(x[..., gi])

        return result


# ---------------------------------------------------------------------------
# YAML-config-facing helpers shared by training datasets and deployment.
# ---------------------------------------------------------------------------

# Map user-facing yaml strings to the internal Normalizer modes.
YAML_TO_NORM_MODE = {
    "min-max": "min_max",
    "z-score": "mean_std",
}


def load_mode_stats(stats_path: str, action_mode: str) -> Optional[dict]:
    """Load ``normalization_stats.npy`` and return the sub-dict for the requested mode.

    Expected schema: ``{"joint": {...}, "eef": {...}, "num_timesteps": ...}``.
    Returns the per-mode stats dict, or ``None`` if the file does not contain
    the requested mode.
    """
    raw = np.load(stats_path, allow_pickle=True).item()
    if action_mode in raw and isinstance(raw[action_mode], dict):
        return raw[action_mode]
    return None


def compute_extended_stats(all_actions: list) -> Dict[str, np.ndarray]:
    """Compute extended statistics: mean, std, min, max, q01, q99.

    Args:
        all_actions: List of (T_i, action_dim) numpy arrays.

    Returns:
        Dict with float32 arrays for each stat.
    """
    concatenated = np.concatenate(all_actions, axis=0).astype(np.float64)

    stats = {
        "mean": concatenated.mean(axis=0).astype(np.float32),
        "std": np.maximum(concatenated.std(axis=0), 1e-3).astype(np.float32),
        "min": concatenated.min(axis=0).astype(np.float32),
        "max": concatenated.max(axis=0).astype(np.float32),
        "q01": np.percentile(concatenated, 1, axis=0).astype(np.float32),
        "q99": np.percentile(concatenated, 99, axis=0).astype(np.float32),
    }
    return stats
