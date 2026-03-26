"""Tests for dataset interface consistency across all dataset adapters."""

import numpy as np
import torch


def test_dataset_base_class_interface():
    """Verify BaseActionDataset has required abstract methods."""
    from open_wam.data.base import BaseActionDataset
    import inspect

    abstracts = {
        name for name, method in inspect.getmembers(BaseActionDataset)
        if getattr(method, "__isabstractmethod__", False)
    }
    assert "action_dim" in abstracts
    assert "action_stats" in abstracts
    assert "__getitem__" in abstracts
    assert "__len__" in abstracts


def test_robotwin_dataset_imports():
    """RoboTwin datasets should be importable and follow interface."""
    from open_wam.data import RoboTwinActionDataset, MultiTaskRoboTwinActionDataset
    assert issubclass(RoboTwinActionDataset, torch.utils.data.Dataset)
    assert issubclass(MultiTaskRoboTwinActionDataset, torch.utils.data.Dataset)


def test_droid_dataset_imports():
    """DROID dataset should be importable."""
    from open_wam.data import DROIDDataset
    assert issubclass(DROIDDataset, torch.utils.data.Dataset)
    assert DROIDDataset._ACTION_DIM == 7


def test_bridge_v2_dataset_imports():
    """Bridge V2 dataset should be importable."""
    from open_wam.data import BridgeV2Dataset
    assert issubclass(BridgeV2Dataset, torch.utils.data.Dataset)
    assert BridgeV2Dataset._ACTION_DIM == 7


def test_all_datasets_inherit_base():
    """All concrete datasets must inherit from BaseActionDataset."""
    from open_wam.data.base import BaseActionDataset
    from open_wam.data import (
        RoboTwinActionDataset,
        MultiTaskRoboTwinActionDataset,
        DROIDDataset,
        BridgeV2Dataset,
    )
    for cls in [RoboTwinActionDataset, MultiTaskRoboTwinActionDataset,
                DROIDDataset, BridgeV2Dataset]:
        assert issubclass(cls, BaseActionDataset), f"{cls.__name__} missing BaseActionDataset"


def test_lerobot_utils_imports():
    """LeRobot utility functions should be importable."""
    from open_wam.data.lerobot_utils import (
        compute_dataset_action_stats,
        normalize_actions,
        discover_episodes,
    )


def test_action_stats_computation():
    """Test action stats computation utility."""
    from open_wam.data.lerobot_utils import compute_dataset_action_stats

    actions = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    ]
    stats = compute_dataset_action_stats(actions)
    assert "mean" in stats and "std" in stats
    assert stats["mean"].shape == (2,)
    assert stats["std"].shape == (2,)
    assert stats["mean"].dtype == np.float32
    # Mean of [1,3,5,7] = 4.0, [2,4,6,8] = 5.0
    np.testing.assert_allclose(stats["mean"], [4.0, 5.0], atol=1e-5)


def test_normalize_actions():
    """Test action normalization."""
    from open_wam.data.lerobot_utils import normalize_actions

    actions = np.array([[4.0, 5.0], [6.0, 7.0]])
    stats = {"mean": np.array([4.0, 5.0]), "std": np.array([2.0, 2.0])}
    normalized = normalize_actions(actions, stats)
    np.testing.assert_allclose(normalized, [[0.0, 0.0], [1.0, 1.0]])
