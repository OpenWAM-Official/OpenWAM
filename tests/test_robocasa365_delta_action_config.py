from __future__ import annotations

import os

from hydra import compose, initialize_config_dir


def test_hydra_compose_uses_independent_delta_action_contract():
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["dataloader=robocasa365_delta_action"])

    assert cfg.dataloader.type == "robocasa365_delta_action"
    assert cfg.dataloader.action_mode == "robocasa365_delta_action"
    assert list(cfg.dataloader.unify_action_map) == ["0-9", "68-72"]
    assert list(cfg.dataloader.unify_state_map) == ["0-9", "68-76"]
    assert all("robocasa365_delta_action_v3" in path for path in cfg.dataloader.dataset_dir)
    assert "robocasa365_delta_action_multitask_compact_stats.npy" in cfg.dataloader.normalization_stats_path


def test_delta_action_dataset_is_registered_separately():
    from openwam.dataloader.registry import DATASET_REGISTRY
    from openwam.dataloader.robocasa365 import MultiTaskRoboCasa365Dataset
    from openwam.dataloader.robocasa365_delta_action import MultiTaskRoboCasa365DeltaActionDataset

    assert DATASET_REGISTRY["robocasa365"] is MultiTaskRoboCasa365Dataset
    assert DATASET_REGISTRY["robocasa365_delta_action"] is MultiTaskRoboCasa365DeltaActionDataset
    assert MultiTaskRoboCasa365DeltaActionDataset is not MultiTaskRoboCasa365Dataset
