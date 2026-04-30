"""Tests for scale-oriented optimizer grouping helpers."""

import torch


class _DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))


class _DummyPipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = _DummyModule()
        self.lora_adapter = _DummyModule()


class _DummyTrainingModule:
    def __init__(self, lambda_action=1.0):
        self.lambda_action = lambda_action
        self.pipe = _DummyPipe()
        self.action_backbone = _DummyModule()


def test_build_trainable_parameters_defaults_to_flat_list():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    model = _DummyTrainingModule()
    params = build_trainable_parameters(model)
    assert isinstance(params, list)
    assert all(isinstance(param, torch.nn.Parameter) for param in params)
    assert len(params) == 3


def test_build_trainable_parameters_supports_per_group_lrs():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    model = _DummyTrainingModule()
    groups = build_trainable_parameters(
        model,
        action_lr=1e-4,
        video_lr=5e-5,
        lora_lr=2e-4,
    )

    assert len(groups) == 3
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 2e-4
    assert groups[2]["lr"] == 5e-5
    assert len(groups[0]["params"]) == 1
    assert len(groups[1]["params"]) == 1
    assert len(groups[2]["params"]) == 1


def test_build_trainable_parameters_drops_action_branch_for_video_only():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    model = _DummyTrainingModule(lambda_action=0.0)
    groups = build_trainable_parameters(model, video_lr=5e-5)

    assert len(groups) == 1
    assert groups[0]["lr"] == 5e-5
    assert len(groups[0]["params"]) == 2
    assert all(not param.requires_grad for param in model.action_backbone.parameters())
