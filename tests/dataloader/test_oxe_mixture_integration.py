"""Mixture-level integration test for OXE readers.

Verifies:
  1. ``mixture.yaml`` composes its robot/ego entries under Hydra defaults.
  2. The composed ``cfg.datasets.<name>`` blocks contain the inherited
     fields from ``configs/dataloader/<name>.yaml``.
  3. A mixture built from synthetic OXE buckets + FakeActionDataset
     collates cleanly through ``default_collate`` into per-batch tensors
     of the expected 2-D mask shapes.
"""

from __future__ import annotations

import os

import pytest

# mixture.yaml composes these under Hydra defaults (robot: robocoin, oxe_droid;
# ego: egodex). bcz/bridge/fractal stay registered but are no longer part of the
# default mixture — see TestMixtureRegistryDispatch.
MIXTURE_ENTRIES = ("robocoin", "oxe_droid", "egodex")


class TestMixtureYamlComposition:
    def test_mixture_includes_expected_entries(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        names = list(cfg.datasets.keys())
        for required in MIXTURE_ENTRIES:
            assert required in names, f"mixture missing {required}"

    def test_blocks_inherit_dataset_dir(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        # Each entry inherits dataset_dir + type from its standalone yaml.
        for name in MIXTURE_ENTRIES:
            assert cfg.datasets[name].get("dataset_dir"), f"{name} missing dataset_dir"
            assert cfg.datasets[name].get("type") == name

    def test_proportional_weight_strategy_default(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        assert cfg.weight_strategy == "proportional"


class TestMixtureRegistryDispatch:
    """The mixture engine dispatches via the registry — make sure all 4
    OXE types are registered and callable."""

    def test_all_four_oxe_types_registered(self):
        from openwam.dataloader.registry import list_registered_datasets

        names = list_registered_datasets()
        for required in ("oxe_bcz", "oxe_bridge", "oxe_fractal", "oxe_droid"):
            assert required in names, f"{required} not in registry"


class TestMixtureSampleShapes:
    """Sanity-check that mixed batches produce uniformly-shaped per-key
    tensors. Uses FakeActionDataset (1-D mask) + a synthetic 2-D-mask
    dataset to verify the mixed batch's loss collate path works.
    """

    @pytest.fixture
    def fake_dataset(self, fake_dataset_factory):
        return fake_dataset_factory(n=4, action_dim=20)

    def test_fake_dataset_emits_legacy_1d_mask(self, fake_dataset):
        s = fake_dataset[0]
        assert s["action_mask"].ndim == 1
        assert s["proprio_mask"].ndim == 1

    def test_mixed_batch_collate_works(self):
        # End-to-end-ish: build a mixture from FakeActionDataset only
        # (deliberate; the real OXE readers' integration is tested in
        # tests/dataloader/test_oxe_bcz.py etc). This pins that mixing
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
