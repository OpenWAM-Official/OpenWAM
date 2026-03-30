"""MixtureDataset for multi-dataset co-training.

Combines multiple BaseActionDataset instances with configurable sampling
weights, enabling cross-dataset training (e.g., RoboTwin + DROID + BridgeV2
in a single run).

Example config (configs/data/mixture.yaml):
    type: mixture
    datasets:
      - type: robotwin_multitask
        weight: 0.5
        dataset_dir: /path/to/robotwin
        robot: arx-x5
      - type: droid
        weight: 0.3
        dataset_dir: /path/to/droid
      - type: bridge_v2
        weight: 0.2
        dataset_dir: /path/to/bridge_v2
"""

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from open_wam.data.base import BaseActionDataset

logger = logging.getLogger(__name__)


class MixtureDataset(BaseActionDataset):
    """Weighted mixture of multiple action datasets.

    Samples are drawn from sub-datasets according to normalized weights.
    The effective length is the sum of per-dataset virtual lengths (each
    sub-dataset length scaled by its weight relative to the largest).

    Args:
        datasets: List of BaseActionDataset instances.
        weights: Sampling weight per dataset (will be normalized to sum=1).
            If None, uniform weighting by dataset size.
        seed: Random seed for reproducible sampling.
        action_dim_override: If sub-datasets have different action dims, pad
            smaller actions to this size. If None, all must match.
    """

    def __init__(
        self,
        datasets: Sequence[BaseActionDataset],
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        action_dim_override: Optional[int] = None,
    ):
        if not datasets:
            raise ValueError("MixtureDataset requires at least one sub-dataset")

        self._datasets = list(datasets)
        self._seed = seed

        # Validate or determine action dim
        dims = [d.action_dim for d in self._datasets]
        if action_dim_override is not None:
            self._action_dim = action_dim_override
        elif len(set(dims)) == 1:
            self._action_dim = dims[0]
        else:
            self._action_dim = max(dims)
            logger.warning(
                "Sub-datasets have different action dims %s, padding to max=%d. "
                "Consider setting action_dim_override explicitly.",
                dims, self._action_dim,
            )

        # Normalize weights
        if weights is None:
            weights = [float(len(d)) for d in self._datasets]
        if len(weights) != len(self._datasets):
            raise ValueError(
                f"weights length ({len(weights)}) != datasets length ({len(self._datasets)})"
            )
        total_w = sum(weights)
        self._weights = [w / total_w for w in weights]

        # Build index mapping: virtual index -> (dataset_idx, sample_idx)
        self._build_index_map()

        # Aggregate action stats (weighted mean of per-dataset stats)
        self._action_stats_cache = self._compute_mixture_stats()

        logger.info(
            "MixtureDataset: %d sub-datasets, total %d samples, weights=%s",
            len(self._datasets),
            len(self),
            [f"{w:.3f}" for w in self._weights],
        )

    def _build_index_map(self):
        """Build a flat index that maps virtual indices to (dataset_idx, sample_idx).

        Each dataset contributes a number of virtual samples proportional
        to its weight. The total virtual length equals the sum of all
        sub-dataset lengths (no samples are dropped).
        """
        total_real = sum(len(d) for d in self._datasets)

        self._index_map: List[tuple] = []
        for di, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            # Each dataset gets virtual_n samples; may repeat or subsample
            virtual_n = max(1, round(total_real * w))
            ds_len = len(ds)
            for vi in range(virtual_n):
                self._index_map.append((di, vi % ds_len))

        # Deterministic shuffle so that batches mix datasets
        rng = np.random.RandomState(self._seed)
        rng.shuffle(self._index_map)

    def _compute_mixture_stats(self) -> Optional[dict]:
        """Weighted combination of per-dataset action stats."""
        all_stats = []
        for ds, w in zip(self._datasets, self._weights):
            stats = ds.action_stats
            if stats is None:
                return None
            all_stats.append((stats, w))

        # Weighted mean
        mean = np.zeros(self._action_dim, dtype=np.float64)
        for stats, w in all_stats:
            m = stats["mean"].astype(np.float64)
            if len(m) < self._action_dim:
                m = np.pad(m, (0, self._action_dim - len(m)))
            mean += w * m[:self._action_dim]

        # Weighted std (using pooled variance formula)
        var = np.zeros(self._action_dim, dtype=np.float64)
        for stats, w in all_stats:
            s = stats["std"].astype(np.float64)
            m = stats["mean"].astype(np.float64)
            if len(s) < self._action_dim:
                s = np.pad(s, (0, self._action_dim - len(s)), constant_values=1e-3)
                m = np.pad(m, (0, self._action_dim - len(m)))
            # Var = E[X^2] - E[X]^2, pooled across datasets
            var += w * (s[:self._action_dim] ** 2 + (m[:self._action_dim] - mean) ** 2)

        std = np.maximum(np.sqrt(var), 1e-3)
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }

    def __getitem__(self, idx: int) -> dict:
        di, si = self._index_map[idx]
        sample = self._datasets[di][si]

        # Pad action to unified dim if needed
        action = sample["action"]
        if isinstance(action, torch.Tensor) and action.shape[-1] < self._action_dim:
            pad_size = self._action_dim - action.shape[-1]
            action = torch.nn.functional.pad(action, (0, pad_size))
            sample["action"] = action
            if "action_trajectory" in sample:
                sample["action_trajectory"] = action

        # Tag with source dataset index for logging/debugging
        sample["_dataset_index"] = di

        return sample

    def __len__(self) -> int:
        return len(self._index_map)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._action_stats_cache

    @property
    def datasets(self) -> List[BaseActionDataset]:
        """Access underlying sub-datasets."""
        return self._datasets

    @property
    def weights(self) -> List[float]:
        """Normalized sampling weights."""
        return self._weights

    def dataset_sample_counts(self) -> Dict[int, int]:
        """Return the number of virtual samples contributed by each sub-dataset."""
        counts: Dict[int, int] = {}
        for di, _ in self._index_map:
            counts[di] = counts.get(di, 0) + 1
        return counts
