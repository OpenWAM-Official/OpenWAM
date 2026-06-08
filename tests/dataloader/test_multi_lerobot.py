"""Tests for openwam/dataloader/bases/multi_lerobot_v3_reader.py — MultiLeRobotV3Reader."""

from __future__ import annotations

import pytest

from openwam.dataloader.bases import MultiLeRobotV3Reader


class TestMultiLeRobotV3Reader:
    def test_empty_buckets_raises(self):
        with pytest.raises(ValueError, match="empty bucket list"):
            MultiLeRobotV3Reader([])

    def test_length_sums_buckets(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5), fake_dataset_factory(7), fake_dataset_factory(3)])
        assert len(m) == 5 + 7 + 3

    def test_dispatch_first_bucket(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5), fake_dataset_factory(7)])
        s = m[0]
        # FakeActionDataset's _synthetic_idx == local idx; first bucket starts at 0.
        assert s["_synthetic_idx"] == 0

    def test_dispatch_second_bucket(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5), fake_dataset_factory(7)])
        # idx=5 is the first element of the second bucket (local idx 0 there).
        s = m[5]
        assert s["_synthetic_idx"] == 0

    def test_dispatch_last_in_first_bucket(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5), fake_dataset_factory(7)])
        s = m[4]
        assert s["_synthetic_idx"] == 4

    def test_dispatch_last_overall(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5), fake_dataset_factory(7)])
        s = m[11]  # 5 + 7 - 1 = 11, local idx 6 in second bucket
        assert s["_synthetic_idx"] == 6

    def test_action_dim_from_first_bucket(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5, action_dim=20), fake_dataset_factory(7, action_dim=20)])
        assert m.action_dim == 20

    def test_normalization_stats_default_none(self, fake_dataset_factory):
        m = MultiLeRobotV3Reader([fake_dataset_factory(5), fake_dataset_factory(7)])
        # Default: aggregating per-bucket stats erases per-source scale,
        # so the base returns None regardless of bucket stats.
        assert m.normalization_stats is None

    def test_buckets_property_returns_list(self, fake_dataset_factory):
        a = fake_dataset_factory(5)
        b = fake_dataset_factory(7)
        m = MultiLeRobotV3Reader([a, b])
        assert m.buckets[0] is a
        assert m.buckets[1] is b
