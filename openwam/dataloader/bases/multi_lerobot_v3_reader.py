"""Aggregation base class for multiple LeRobot v3 sub-datasets.

Dataset-specific readers such as
:class:`~openwam.dataloader.robocoin.MultiRobotCOINDataset` follow the same
pattern: collect N homogeneous sub-readers
("buckets"), expose ``len = Σ len(b_i)``, and dispatch ``__getitem__(idx)``
to the right bucket via binary search over cumulative lengths.

This base class captures that pattern. Subclasses can add domain-specific
metadata logging (e.g. RoboCOIN logs robot_type coverage) while inheriting
the index math + property scaffolding.

Design:
- Sub-buckets are expected to be LeRobot v3-format readers (parquet +
  per-camera mp4 + meta/info.json). The class name is intentionally
  scoped to that format to make the precondition explicit.
- ``action_dim`` defaults to the first bucket's value; subclasses can
  override if they need to assert all buckets agree.
- ``normalization_stats`` is hard-coded None because both current subclasses
  pre-normalize inside their per-bucket readers and don't expose
  aggregable stats. Subclasses that want to expose stats can override.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from openwam.dataloader.bases.dataset import BaseDataset

logger = logging.getLogger(__name__)


class MultiLeRobotV3Reader(BaseDataset):
    """Aggregate N LeRobot v3 sub-readers into a single Dataset.

    Args:
        buckets: List of homogeneous LeRobot v3 sub-readers, each a
            :class:`BaseDataset`. Must be non-empty.

    Attributes:
        buckets:    list[BaseDataset] — the sub-readers in dispatch order.
        _cum_lens:  np.int64 (N+1,) — cumulative lengths for binary search.
    """

    def __init__(self, buckets: List[BaseDataset]):
        if not buckets:
            raise ValueError(f"{type(self).__name__}: empty bucket list")
        self._buckets = list(buckets)
        lens = np.array([len(b) for b in self._buckets], dtype=np.int64)
        self._cum_lens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    def __len__(self) -> int:
        return int(self._cum_lens[-1])

    def __getitem__(self, idx: int) -> dict:
        # Bounds + negative-index normalization. ``searchsorted(side="right")``
        # on a negative idx returns 0 → local = idx (negative) → routes to a
        # nonsense slot in bucket 0. Sampling code (Mixture, default Sampler)
        # never produces negatives, but defensive validation here costs ~tens
        # of nanoseconds and makes any external misuse fail loudly.
        n = int(self._cum_lens[-1])
        if not 0 <= idx < n:
            raise IndexError(f"{type(self).__name__} idx {idx} out of range [0, {n})")
        bi = int(np.searchsorted(self._cum_lens, idx, side="right") - 1)
        local = idx - int(self._cum_lens[bi])
        return self._buckets[bi][local]

    @property
    def action_dim(self) -> int:
        """Delegate to the first bucket — all buckets share action_dim by
        construction (they're homogeneous). Not required by ``BaseDataset``,
        but every concrete subclass (MultiRobotCOIN / MultiBucketEgoDex)
        expects this attribute to be queryable by ``MixtureDataset``'s
        cross-source shape check.
        """
        return self._buckets[0].action_dim

    @property
    def normalization_stats(self):
        """Aggregated normalization stats across buckets.

        Returns None by default: sub-buckets apply their own in-reader
        normalization, and aggregating per-bucket stats into a single dict
        would erase per-source scale information. Subclasses can override
        if they want different semantics.
        """
        return None

    @property
    def buckets(self) -> List[BaseDataset]:
        return self._buckets


__all__ = ["MultiLeRobotV3Reader"]
