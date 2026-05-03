"""Tests for scale-oriented optimizer grouping helpers.

Mirrors the real OpenWAMTrainer attribute layout (``self.architecture`` with
top-level ``video_backbone`` + ``action_backbone`` children) so mock drift
can't mask the optimizer-misses-video-DiT bug we hit before.
"""

import torch


class _FakeActionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        # 触发 NO_WD_PARAM_SUFFIXES 路径，与真实 ActionDiT 对齐
        self.modality_tmod_bias = torch.nn.Parameter(torch.zeros(4))


class _FakePipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = torch.nn.Linear(4, 4)  # 预期被冻结
        self.vae = torch.nn.Linear(4, 4)  # 预期被冻结
        self.dit = torch.nn.Linear(4, 4)  # 可训练
        self.lora_A = torch.nn.Linear(4, 4)  # name 含 "lora"


class _FakeVideoBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._pipe = _FakePipe()


class _FakeArchitecture(torch.nn.Module):
    """Mirrors BaseWAMArchitecture.named_children() layout."""

    def __init__(self):
        super().__init__()
        self.video_backbone = _FakeVideoBackbone()
        self.action_backbone = _FakeActionBackbone()

    def get_trainable_modules(self, freeze_list=()):
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result


class _FakeTrainer:
    """Mirrors OpenWAMTrainer post-Item-A attribute layout."""

    def __init__(self, lambda_action=1.0):
        self.lambda_action = lambda_action
        self.architecture = _FakeArchitecture()
        # 模拟 freeze_modules 已在 nested 层级把 text_encoder / vae 冻住
        self.architecture.video_backbone._pipe.text_encoder.requires_grad_(False)
        self.architecture.video_backbone._pipe.vae.requires_grad_(False)


def test_pipe_params_include_video_dit_and_lora():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    groups = build_trainable_parameters(trainer, action_lr=1e-4, video_lr=5e-5, lora_lr=2e-4)
    surfaced = {id(p) for g in groups for p in g["params"]}
    arch = trainer.architecture

    assert id(arch.video_backbone._pipe.dit.weight) in surfaced
    assert id(arch.video_backbone._pipe.dit.bias) in surfaced
    assert id(arch.video_backbone._pipe.lora_A.weight) in surfaced
    assert id(arch.video_backbone._pipe.text_encoder.weight) not in surfaced
    assert id(arch.video_backbone._pipe.vae.weight) not in surfaced

    lora_groups = [g for g in groups if g.get("lr") == 2e-4]
    assert lora_groups, "LoRA group missing"
    lora_param_ids = {id(p) for g in lora_groups for p in g["params"]}
    assert id(arch.video_backbone._pipe.lora_A.weight) in lora_param_ids


def test_modality_tmod_bias_isolated_with_zero_wd():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    groups = build_trainable_parameters(trainer, action_lr=1e-4, video_lr=5e-5)
    no_wd_groups = [g for g in groups if g.get("weight_decay") == 0.0]
    bias = trainer.architecture.action_backbone.modality_tmod_bias
    assert any(any(p is bias for p in g["params"]) for g in no_wd_groups)


def test_lambda_action_zero_freezes_action_branch():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer(lambda_action=0.0)
    groups = build_trainable_parameters(trainer, video_lr=5e-5)

    arch = trainer.architecture
    surfaced = {id(p) for g in groups for p in g["params"]}
    assert id(arch.action_backbone.proj.weight) not in surfaced
    assert all(not p.requires_grad for p in arch.action_backbone.parameters())


def test_default_returns_groups_when_no_overrides():
    """Without LR overrides but with no_wd params present, returns grouped form."""
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    result = build_trainable_parameters(trainer)

    # _FakeActionBackbone has modality_tmod_bias → no_wd params present →
    # function takes the grouped path (the flat-list shortcut needs *both*
    # no LR override AND no no_wd params).
    assert isinstance(result, list)
    assert all(isinstance(g, dict) and "params" in g for g in result)

    arch = trainer.architecture
    surfaced = {id(p) for g in result for p in g["params"]}
    # action: proj.weight, proj.bias, modality_tmod_bias
    assert id(arch.action_backbone.proj.weight) in surfaced
    assert id(arch.action_backbone.proj.bias) in surfaced
    assert id(arch.action_backbone.modality_tmod_bias) in surfaced
    # video: dit.weight, dit.bias, lora_A.weight, lora_A.bias
    assert id(arch.video_backbone._pipe.dit.weight) in surfaced
    assert id(arch.video_backbone._pipe.dit.bias) in surfaced
    assert id(arch.video_backbone._pipe.lora_A.weight) in surfaced
    assert id(arch.video_backbone._pipe.lora_A.bias) in surfaced
    # frozen: not surfaced
    assert id(arch.video_backbone._pipe.text_encoder.weight) not in surfaced
    assert id(arch.video_backbone._pipe.vae.weight) not in surfaced
