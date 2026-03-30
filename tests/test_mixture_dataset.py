"""Tests for MixtureDataset multi-dataset co-training."""

import numpy as np
import torch
import pytest

from open_wam.data.base import BaseActionDataset
from open_wam.data.mixture import MixtureDataset


class DummyDataset(BaseActionDataset):
    """Minimal dataset for testing."""

    def __init__(self, size: int, action_dim: int = 7, tag: str = "dummy"):
        self._size = size
        self._action_dim = action_dim
        self._tag = tag

    def __getitem__(self, idx):
        return {
            "video": [],
            "action": torch.randn(10, self._action_dim),
            "action_trajectory": torch.randn(10, self._action_dim),
            "prompt": f"{self._tag} sample {idx}",
        }

    def __len__(self):
        return self._size

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_stats(self):
        return {
            "mean": np.zeros(self._action_dim, dtype=np.float32),
            "std": np.ones(self._action_dim, dtype=np.float32),
        }


class DummyDatasetNoStats(DummyDataset):
    @property
    def action_stats(self):
        return None


def test_mixture_inherits_base():
    """MixtureDataset must be a BaseActionDataset."""
    assert issubclass(MixtureDataset, BaseActionDataset)


def test_mixture_basic():
    """Basic mixture of two datasets with explicit weights."""
    ds1 = DummyDataset(100, tag="ds1")
    ds2 = DummyDataset(50, tag="ds2")
    mix = MixtureDataset([ds1, ds2], weights=[0.7, 0.3])

    assert mix.action_dim == 7
    assert len(mix) > 0
    assert mix.weights == pytest.approx([0.7, 0.3], abs=1e-6)

    sample = mix[0]
    assert "action" in sample
    assert "_dataset_index" in sample


def test_mixture_uniform_weights():
    """Uniform weights when weights=None."""
    ds1 = DummyDataset(100)
    ds2 = DummyDataset(100)
    mix = MixtureDataset([ds1, ds2])

    assert mix.weights == pytest.approx([0.5, 0.5], abs=1e-6)


def test_mixture_single_dataset():
    """Mixture with a single dataset should work."""
    ds = DummyDataset(50)
    mix = MixtureDataset([ds], weights=[1.0])

    assert len(mix) == 50
    assert mix.action_dim == 7


def test_mixture_different_action_dims():
    """Datasets with different action dims should pad to max."""
    ds1 = DummyDataset(50, action_dim=7)
    ds2 = DummyDataset(50, action_dim=14)
    mix = MixtureDataset([ds1, ds2], weights=[0.5, 0.5])

    assert mix.action_dim == 14

    # Check that samples from ds1 are padded
    for i in range(len(mix)):
        sample = mix[i]
        assert sample["action"].shape[-1] == 14


def test_mixture_action_dim_override():
    """Explicit action_dim_override."""
    ds1 = DummyDataset(50, action_dim=7)
    ds2 = DummyDataset(50, action_dim=7)
    mix = MixtureDataset([ds1, ds2], weights=[0.5, 0.5], action_dim_override=20)

    assert mix.action_dim == 20


def test_mixture_stats_aggregation():
    """Weighted action stats are computed correctly."""
    ds1 = DummyDataset(50, action_dim=7)
    ds2 = DummyDataset(50, action_dim=7)
    mix = MixtureDataset([ds1, ds2], weights=[0.5, 0.5])

    stats = mix.action_stats
    assert stats is not None
    assert stats["mean"].shape == (7,)
    assert stats["std"].shape == (7,)
    # Both sub-datasets have mean=0, std=1, so mixture should be close
    np.testing.assert_allclose(stats["mean"], 0.0, atol=1e-5)
    np.testing.assert_allclose(stats["std"], 1.0, atol=1e-5)


def test_mixture_stats_none_if_any_missing():
    """If any sub-dataset has no stats, mixture stats should be None."""
    ds1 = DummyDataset(50)
    ds2 = DummyDatasetNoStats(50)
    mix = MixtureDataset([ds1, ds2], weights=[0.5, 0.5])

    assert mix.action_stats is None


def test_mixture_dataset_sample_counts():
    """Verify sample count distribution respects weights."""
    ds1 = DummyDataset(100, tag="large")
    ds2 = DummyDataset(100, tag="small")
    mix = MixtureDataset([ds1, ds2], weights=[0.8, 0.2])

    counts = mix.dataset_sample_counts()
    assert 0 in counts and 1 in counts
    # ds1 should have ~4x more samples than ds2
    ratio = counts[0] / counts[1]
    assert ratio > 2.0


def test_mixture_all_indices_valid():
    """Every index in the mixture should produce a valid sample."""
    ds1 = DummyDataset(10, tag="a")
    ds2 = DummyDataset(5, tag="b")
    mix = MixtureDataset([ds1, ds2], weights=[0.6, 0.4])

    for i in range(len(mix)):
        sample = mix[i]
        assert "action" in sample
        assert "prompt" in sample


def test_mixture_empty_raises():
    """Empty dataset list should raise."""
    with pytest.raises(ValueError, match="at least one"):
        MixtureDataset([])


def test_mixture_weight_length_mismatch():
    """Mismatched weights length should raise."""
    ds1 = DummyDataset(10)
    with pytest.raises(ValueError, match="weights length"):
        MixtureDataset([ds1], weights=[0.5, 0.5])


def test_mixture_deterministic_with_seed():
    """Same seed should produce same index order."""
    ds1 = DummyDataset(20, tag="a")
    ds2 = DummyDataset(20, tag="b")

    mix_a = MixtureDataset([ds1, ds2], weights=[0.5, 0.5], seed=123)
    mix_b = MixtureDataset([ds1, ds2], weights=[0.5, 0.5], seed=123)

    assert mix_a._index_map == mix_b._index_map


def test_mixture_import_from_package():
    """MixtureDataset should be importable from open_wam.data."""
    from open_wam.data import MixtureDataset as M
    assert M is MixtureDataset
