"""Shared fixtures for dataloader tests.

All fixtures generate test data on demand into tests/dataloader/.cache/
(gitignored) so the repo stays binary-free. The cache is reused across
test runs — delete it manually if a fixture's schema changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.dataloader.bases import BaseDataset

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_CACHE_ROOT.mkdir(exist_ok=True)


class FakeActionDataset(BaseDataset):
    """In-memory dataset returning predictable samples by index.

    Used by mixture tests where the concrete reader's IO path is not under
    test — only mixture's index dispatch, weights, set_epoch, mask
    propagation etc.

    Each sample is a dict::

        {
            "action": torch.full((T_action, action_dim), float(idx)),
            "action_mask": torch.ones(T_action, dtype=bool),
            "proprio": torch.zeros(1, action_dim),
            "proprio_mask": torch.ones(1, dtype=bool),
            "video": [object() for _ in range(num_video_frames)],
            "video_mask": torch.ones(num_video_frames, dtype=bool),
            "prompt": f"sample-{idx}",
            "_synthetic_idx": idx,
        }
    """

    def __init__(
        self,
        n: int,
        action_dim: int = 20,
        normalization_stats: dict | None = None,
        num_video_frames: int = 9,
        T_action: int = 32,
    ):
        self._n = int(n)
        self._action_dim = int(action_dim)
        self._normalization_stats = normalization_stats
        self._nv = int(num_video_frames)
        self._T = int(T_action)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        return {
            "action": torch.full((self._T, self._action_dim), float(idx)),
            "action_mask": torch.ones(self._T, dtype=torch.bool),
            "proprio": torch.zeros(1, self._action_dim),
            "proprio_mask": torch.ones(1, dtype=torch.bool),
            "video": [object() for _ in range(self._nv)],
            "video_mask": torch.ones(self._nv, dtype=torch.bool),
            "prompt": f"sample-{idx}",
            "_synthetic_idx": idx,
        }

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def normalization_stats(self):
        return self._normalization_stats


@pytest.fixture
def fake_dataset_factory():
    """Return a builder that produces FakeActionDataset instances on demand."""

    def _make(n: int, **kwargs):
        return FakeActionDataset(n, **kwargs)

    return _make


@pytest.fixture
def tiny_episodes_df():
    """Tiny eps DataFrame approximating LeRobot v3 schema for offset helpers."""
    import pandas as pd

    rows = []
    cum = 0
    for ep in range(8):
        length = 5 + ep
        chunk = ep // 4
        file = ep % 4
        rows.append(
            {
                "episode_index": ep,
                "length": length,
                "dataset_from_index": cum,
                "data/chunk_index": chunk,
                "data/file_index": file,
                "videos/cam/chunk_index": chunk,
                "videos/cam/file_index": file,
            }
        )
        cum += length
    return pd.DataFrame(rows)


@pytest.fixture
def known_stats_dict():
    """Hand-picked min/max/mean/std/q01/q99 for normalize math assertions.

    Dimension 0: range [0, 10], mean 5.0, std 2.0, q01/q99 = 1.0/9.0 (synthetic).
    Dimension 1: range [-1, 1], mean 0.0, std 0.5, q01/q99 = -0.8/0.8 (centered).
    """
    return {
        "mean": np.array([5.0, 0.0], dtype=np.float32),
        "std": np.array([2.0, 0.5], dtype=np.float32),
        "min": np.array([0.0, -1.0], dtype=np.float32),
        "max": np.array([10.0, 1.0], dtype=np.float32),
        "q01": np.array([1.0, -0.8], dtype=np.float32),
        "q99": np.array([9.0, 0.8], dtype=np.float32),
    }


@pytest.fixture
def cache_root():
    """Per-test access to the shared cache root."""
    return _CACHE_ROOT
