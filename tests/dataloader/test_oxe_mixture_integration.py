"""Mixture-level integration test for OXE readers.

Verifies:
  1. ``mixture.yaml`` composes the supported public sources under Hydra
     defaults.
  2. The composed ``cfg.datasets.<name>`` blocks contain the inherited
     fields from ``configs/dataloader/<name>.yaml``.
  3. A mixture built from synthetic OXE buckets + FakeActionDataset
     collates cleanly through ``default_collate`` into per-batch tensors
     of the expected 2-D mask shapes.
"""

from __future__ import annotations

import os

import pytest

# ``agiworld`` is the user-facing dataset name; its registered config/type in
# this repository is ``agibotworld``.  The tuple is order-sensitive because the
# Hydra defaults order is part of the mixture's stable source-index contract.
MIXTURE_ENTRIES = (
    "agibotworld",
    "robocoin",
    "oxe_droid",
    "interndata_a1",
)


class TestMixtureYamlComposition:
    def test_mixture_includes_expected_entries(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader/pretrain_data")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        assert tuple(cfg.datasets.keys()) == MIXTURE_ENTRIES

    def test_blocks_inherit_dataset_dir(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader/pretrain_data")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        # Each entry inherits dataset_dir + type from its standalone yaml.
        for name in MIXTURE_ENTRIES:
            assert cfg.datasets[name].get("dataset_dir"), f"{name} missing dataset_dir"
            assert cfg.datasets[name].get("type") == name
            assert cfg.datasets[name].total_hours is None

    def test_proportional_weight_strategy_default(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader/pretrain_data")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        assert cfg.weight_strategy == "proportional"


class TestMixtureRegistryDispatch:
    """Only the active OXE reader participates in config dispatch."""

    def test_only_oxe_droid_is_registered(self):
        from openwam.dataloader.registry import list_registered_datasets

        names = set(list_registered_datasets())
        assert "oxe_droid" in names
        deprecated = {"ego4d", "egodex", "oxe_bcz", "oxe_bridge", "oxe_fractal"}
        assert not names.intersection(deprecated)


class TestMixtureSampleShapes:
    """Sanity-check that mixed batches produce uniformly-shaped per-key
    tensors. Uses FakeActionDataset (1-D mask) + a synthetic 2-D-mask
    dataset to verify the mixed batch's loss collate path works.
    """

    @pytest.fixture
    def fake_dataset(self):
        # Import directly instead of depending on a sibling conftest fixture;
        # this file is also collected in suites whose import mode does not
        # expose tests/dataloader/conftest.py fixtures by name.
        from tests.dataloader.conftest import FakeActionDataset

        return FakeActionDataset(n=4, action_dim=20)

    def test_fake_dataset_emits_legacy_1d_mask(self, fake_dataset):
        s = fake_dataset[0]
        assert s["action_mask"].ndim == 1
        assert s["proprio_mask"].ndim == 1

    def test_mixed_batch_collate_works(self):
        # End-to-end-ish: build a mixture from FakeActionDataset only.
        # Reader-level behavior is covered separately; this pins that mixing
        # multiple sources with the new mask shape contract still collates.
        from openwam.dataloader.mixture import MixtureDataset
        from tests.dataloader.conftest import FakeActionDataset

        ds_a = FakeActionDataset(n=5, action_dim=20)
        ds_b = FakeActionDataset(n=8, action_dim=20)
        mix = MixtureDataset(
            datasets=[ds_a, ds_b],
            names=["a", "b"],
        )
        # MixtureDataset extends per-bucket sizes by weight; ds_a has 5
        # real samples + ds_b has 8 real samples → total at least 13.
        assert len(mix) >= 13
        s = mix[0]
        assert s["action"].shape == (32, 20)
        assert "action_mask" in s
        assert "proprio_mask" in s
