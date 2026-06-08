"""Tests for dataset interface consistency across RoboTwin dataset adapters."""

import torch


def test_dataset_base_class_interface():
    """Verify BaseDataset's minimal abstract contract.

    Per Plan A, BaseDataset only requires __getitem__ and __len__.
    Per-source metadata like action_dim or normalization_stats is
    discovered by consumers (MixtureDataset, trainer) via duck-typed
    getattr, not enforced on the base.
    """
    import inspect

    from openwam.dataloader.bases import BaseDataset

    abstracts = {
        name for name, method in inspect.getmembers(BaseDataset) if getattr(method, "__isabstractmethod__", False)
    }
    assert abstracts == {"__getitem__", "__len__"}


def test_robotwin_dataset_imports():
    """RoboTwin datasets should be importable and follow interface."""
    from openwam.dataloader import MultiTaskRoboTwinDataset, RoboTwinDataset

    assert issubclass(RoboTwinDataset, torch.utils.data.Dataset)
    assert issubclass(MultiTaskRoboTwinDataset, torch.utils.data.Dataset)


def test_all_datasets_inherit_base():
    """All concrete datasets must inherit from BaseDataset."""
    from openwam.dataloader import (
        MultiTaskRoboTwinDataset,
        RoboTwinDataset,
    )
    from openwam.dataloader.bases import BaseDataset

    for cls in [RoboTwinDataset, MultiTaskRoboTwinDataset]:
        assert issubclass(cls, BaseDataset), f"{cls.__name__} missing BaseDataset"
