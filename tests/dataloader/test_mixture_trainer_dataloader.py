"""Trainer sampling policy for the already-shuffled MixtureDataset."""

from types import SimpleNamespace

import torch

from openwam.dataloader.mixture import MixtureDataset
from openwam.train.openwam_trainer import OpenWAMTrainer


def _bare_trainer(dataset) -> OpenWAMTrainer:
    """Construct only the fields consumed by ``build_dataloader``."""
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.dataset = dataset
    trainer.cfg = SimpleNamespace(training=SimpleNamespace(dataset_num_workers=0))
    trainer._run_seed = None
    trainer._rank = 0
    return trainer


def test_mixture_uses_sequential_sampler_over_its_shuffled_index_map(fake_dataset_factory):
    mixture = MixtureDataset(
        [fake_dataset_factory(8)],
        weights=None,
        names=["tiny"],
        strict_action_dim=True,
    )

    loader = _bare_trainer(mixture).build_dataloader(batch_size=2)

    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)


def test_non_mixture_keeps_random_sampler(fake_dataset_factory):
    loader = _bare_trainer(fake_dataset_factory(8)).build_dataloader(batch_size=2)

    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
