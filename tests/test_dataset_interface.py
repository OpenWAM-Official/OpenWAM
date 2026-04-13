"""Tests for dataset interface consistency across RoboTwin dataset adapters."""

import torch


def test_dataset_base_class_interface():
    """Verify BaseActionDataset has required abstract methods."""
    import inspect

    from openwam.dataloader.base_dataset import BaseActionDataset

    abstracts = {
        name for name, method in inspect.getmembers(BaseActionDataset) if getattr(method, "__isabstractmethod__", False)
    }
    assert "action_dim" in abstracts
    assert "action_stats" in abstracts
    assert "__getitem__" in abstracts
    assert "__len__" in abstracts


def test_robotwin_dataset_imports():
    """RoboTwin datasets should be importable and follow interface."""
    from openwam.dataloader import MultiTaskRoboTwinDataset, RoboTwinDataset

    assert issubclass(RoboTwinDataset, torch.utils.data.Dataset)
    assert issubclass(MultiTaskRoboTwinDataset, torch.utils.data.Dataset)


def test_all_datasets_inherit_base():
    """All concrete datasets must inherit from BaseActionDataset."""
    from openwam.dataloader import (
        MultiTaskRoboTwinDataset,
        RoboTwinDataset,
    )
    from openwam.dataloader.base_dataset import BaseActionDataset

    for cls in [RoboTwinDataset, MultiTaskRoboTwinDataset]:
        assert issubclass(cls, BaseActionDataset), f"{cls.__name__} missing BaseActionDataset"
