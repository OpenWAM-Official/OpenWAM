"""Tests for openwam/dataloader/mixture.py — MixtureDataset.

Uses FakeActionDataset (from conftest.FakeActionDataset) so no parquet/mp4
fixtures are needed; the tests target dispatch logic, weight strategies,
strict_action_dim, set_epoch, and the _dataset_name / get_dataset API.
"""

from __future__ import annotations

import numpy as np
import pytest

from openwam.dataloader.mixture import MixtureDataset

# ---------------------------------------------------------------------------
# Construction + basic invariants
# ---------------------------------------------------------------------------


class TestMixtureConstruction:
    def test_empty_buckets_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            MixtureDataset([])

    def test_all_zero_manual_weights_raises(self, fake_dataset_factory):
        # Regression (M2): all-zero weights must raise a clear error instead of
        # a bare ZeroDivisionError from the weight normalization.
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        with pytest.raises(ValueError, match="zero-weight"):
            MixtureDataset([a, b], weights=[0.0, 0.0])

    def test_negative_weight_raises(self, fake_dataset_factory):
        # Regression (M2): a negative weight must be rejected explicitly rather
        # than silently treated as a zero-weight skip in _build_index_map.
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        with pytest.raises(ValueError, match="negative"):
            MixtureDataset([a, b], weights=[-1.0, 2.0])

    def test_length_is_sum_of_virtual_n(self, fake_dataset_factory):
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        m = MixtureDataset([a, b], weights=[1.0, 1.0])
        # With uniform weights both sources get round(total_real * 0.5) = 12.
        # 12 + 12 = 24; close to total_real but rounding may bias by 1.
        total_real = len(a) + len(b)
        assert abs(len(m) - total_real) <= 1

    def test_default_weights_proportional_full_coverage(self, fake_dataset_factory):
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        m = MixtureDataset([a, b], weights=None)
        # weights=None uses [len(a), len(b)] → proportional → virtual_n_i = N_i.
        assert len(m) == len(a) + len(b)

        # Every (di, si) pair appears at least once.
        seen = set()
        for k in range(len(m)):
            di, si = m._index_map[k]
            seen.add((int(di), int(si)))
        expected = {(0, i) for i in range(len(a))} | {(1, i) for i in range(len(b))}
        assert seen == expected

    def test_names_default_to_source_i(self, fake_dataset_factory):
        m = MixtureDataset([fake_dataset_factory(5), fake_dataset_factory(7)], weights=[1.0, 1.0])
        assert m.names == ["source_0", "source_1"]

    def test_named_buckets(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(5), fake_dataset_factory(7)],
            weights=[1.0, 1.0],
            names=["alpha", "beta"],
        )
        assert m.names == ["alpha", "beta"]

    def test_names_length_mismatch_raises(self, fake_dataset_factory):
        with pytest.raises(ValueError, match="names length"):
            MixtureDataset(
                [fake_dataset_factory(5), fake_dataset_factory(7)],
                weights=[1.0, 1.0],
                names=["only-one"],
            )


# ---------------------------------------------------------------------------
# Strict action_dim
# ---------------------------------------------------------------------------


class TestStrictActionDim:
    def test_strict_mismatch_raises(self, fake_dataset_factory):
        a = fake_dataset_factory(5, action_dim=20)
        b = fake_dataset_factory(7, action_dim=24)
        with pytest.raises(ValueError, match="same action_dim"):
            MixtureDataset([a, b], weights=[1.0, 1.0], strict_action_dim=True)

    def test_strict_match_ok(self, fake_dataset_factory):
        a = fake_dataset_factory(5, action_dim=20)
        b = fake_dataset_factory(7, action_dim=20)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], strict_action_dim=True)
        assert m.action_dim == 20

    def test_legacy_pad_to_max(self, fake_dataset_factory):
        # strict=False, dims differ → max dim wins.
        a = fake_dataset_factory(5, action_dim=14)
        b = fake_dataset_factory(7, action_dim=20)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], strict_action_dim=False)
        assert m.action_dim == 20

    def test_normalization_stats_always_none(self, fake_dataset_factory):
        # Even if sub-buckets carry a stats dict, mixture must NOT aggregate.
        a = fake_dataset_factory(5, normalization_stats={"mean": np.zeros(20), "std": np.ones(20)})
        b = fake_dataset_factory(7, normalization_stats={"mean": np.ones(20), "std": np.ones(20) * 2})
        m = MixtureDataset([a, b], weights=[1.0, 1.0])
        assert m.normalization_stats is None


# ---------------------------------------------------------------------------
# Dispatch + name propagation
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_sample_carries_dataset_index_and_name(self, fake_dataset_factory):
        a = fake_dataset_factory(5)
        b = fake_dataset_factory(7)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], names=["alpha", "beta"])
        s = m[0]
        assert s["_dataset_index"] in (0, 1)
        assert s["_dataset_name"] in ("alpha", "beta")
        # Name and index agree.
        assert s["_dataset_name"] == m.names[s["_dataset_index"]]

    def test_get_dataset_by_name(self, fake_dataset_factory):
        a = fake_dataset_factory(5)
        b = fake_dataset_factory(7)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], names=["alpha", "beta"])
        assert m.get_dataset("alpha") is a
        assert m.get_dataset("beta") is b

    def test_get_dataset_unknown_name_raises_keyerror(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(5), fake_dataset_factory(7)],
            weights=[1.0, 1.0],
            names=["alpha", "beta"],
        )
        with pytest.raises(KeyError, match="no sub-dataset named 'gamma'"):
            m.get_dataset("gamma")

    def test_dataset_sample_counts_by_name(self, fake_dataset_factory):
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(20)
        m = MixtureDataset([a, b], weights=None, names=["alpha", "beta"])  # proportional
        counts = m.dataset_sample_counts()
        # Exact counts under proportional weights.
        assert counts == {"alpha": 10, "beta": 20}


# ---------------------------------------------------------------------------
# set_epoch
# ---------------------------------------------------------------------------


class TestSetEpoch:
    def test_set_epoch_changes_order(self, fake_dataset_factory):
        a = fake_dataset_factory(50)
        b = fake_dataset_factory(50)
        m = MixtureDataset([a, b], weights=None, seed=42, names=["a", "b"])  # proportional
        order_0 = m._index_map.copy()
        m.set_epoch(1)
        order_1 = m._index_map.copy()
        # Same total length, different sample ordering.
        assert len(order_0) == len(order_1)
        assert not (order_0 == order_1).all()

    def test_set_epoch_preserves_coverage_under_proportional(self, fake_dataset_factory):
        a = fake_dataset_factory(50)
        b = fake_dataset_factory(70)
        m = MixtureDataset([a, b], weights=None, seed=42)  # proportional
        # Coverage at epoch 0
        seen_0 = {(int(d), int(s)) for d, s in m._index_map}
        m.set_epoch(5)
        seen_5 = {(int(d), int(s)) for d, s in m._index_map}
        # Both epochs cover the full union.
        full = {(0, i) for i in range(50)} | {(1, i) for i in range(70)}
        assert seen_0 == full
        assert seen_5 == full

    def test_set_epoch_idempotent_with_same_arg(self, fake_dataset_factory):
        a = fake_dataset_factory(30)
        m = MixtureDataset([a], weights=None, seed=42)
        initial_map = m._index_map
        m.set_epoch(0)
        assert m._index_map is initial_map

        m.set_epoch(3)
        epoch_3_map = m._index_map
        m.set_epoch(3)
        assert m._index_map is epoch_3_map
