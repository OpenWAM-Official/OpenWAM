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


class _FakePipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = torch.nn.Linear(4, 4)  # expected frozen
        self.vae = torch.nn.Linear(4, 4)  # expected frozen
        self.dit = torch.nn.Linear(4, 4)  # trainable


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
        # A trainable top-level module that is neither video nor action — mirrors
        # tri_system's understanding_expert / vlm_backbone.
        self.understanding_expert = torch.nn.Linear(4, 4)

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
        # Mirror freeze_modules: text_encoder / vae already frozen at the nested level.
        self.architecture.video_backbone._pipe.text_encoder.requires_grad_(False)
        self.architecture.video_backbone._pipe.vae.requires_grad_(False)


def test_pipe_params_include_video_dit():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    groups = build_trainable_parameters(trainer, action_lr=1e-4, video_lr=5e-5)
    surfaced = {id(p) for g in groups for p in g["params"]}
    arch = trainer.architecture

    assert id(arch.video_backbone._pipe.dit.weight) in surfaced
    assert id(arch.video_backbone._pipe.dit.bias) in surfaced
    assert id(arch.video_backbone._pipe.text_encoder.weight) not in surfaced
    assert id(arch.video_backbone._pipe.vae.weight) not in surfaced

    # The video override LR lands on the group that carries the video DiT params.
    video_groups = [g for g in groups if g.get("lr") == 5e-5]
    assert video_groups, "video LR group missing"
    video_param_ids = {id(p) for g in video_groups for p in g["params"]}
    assert id(arch.video_backbone._pipe.dit.weight) in video_param_ids


def test_other_modules_ride_base_lr_not_video_lr():
    """Non-video / non-action top-level modules (e.g. understanding_expert) must NOT
    inherit video_lr — they land in a group with no 'lr' (the optimizer base LR)."""
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    groups = build_trainable_parameters(trainer, action_lr=1e-4, video_lr=5e-5)
    ue_weight = trainer.architecture.understanding_expert.weight

    # Lands in a base-LR group (no 'lr' override).
    base_ids = {id(p) for g in groups if "lr" not in g for p in g["params"]}
    assert id(ue_weight) in base_ids, "understanding_expert should ride the base LR"

    # And must NOT leak into the video_lr group.
    video_ids = {id(p) for g in groups if g.get("lr") == 5e-5 for p in g["params"]}
    assert id(ue_weight) not in video_ids
    # Sanity: the video DiT IS in the video_lr group (so the split is real).
    assert id(trainer.architecture.video_backbone._pipe.dit.weight) in video_ids


def test_action_params_land_in_action_lr_group():
    """The action override LR lands on the group carrying the action backbone params —
    guards against action params being silently routed into the base/other group."""
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    groups = build_trainable_parameters(trainer, action_lr=1e-4, video_lr=5e-5)
    arch = trainer.architecture

    action_groups = [g for g in groups if g.get("lr") == 1e-4]
    assert action_groups, "action LR group missing"
    action_ids = {id(p) for g in action_groups for p in g["params"]}
    assert id(arch.action_backbone.proj.weight) in action_ids
    assert id(arch.action_backbone.proj.bias) in action_ids
    # And action params must NOT leak into the video_lr group.
    video_ids = {id(p) for g in groups if g.get("lr") == 5e-5 for p in g["params"]}
    assert id(arch.action_backbone.proj.weight) not in video_ids


def test_freeze_is_the_sole_gate_for_action_branch():
    """A frozen action_backbone (requires_grad=False) contributes no params —
    freeze is the single source of truth for trainability."""
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer(lambda_action=0.0)
    # lambda_action no longer freezes anything; freeze must be explicit (as the
    # model yaml's ``freeze:`` list does in production).
    trainer.architecture.action_backbone.requires_grad_(False)
    groups = build_trainable_parameters(trainer, video_lr=5e-5)

    surfaced = {id(p) for g in groups for p in g["params"]}
    assert id(trainer.architecture.action_backbone.proj.weight) not in surfaced


def test_lambda_action_zero_does_not_gate_params():
    """Regression: ``lambda_action=0`` alone must NOT drop action params or flip
    requires_grad — loss weights are decoupled from optimizer membership. Only the
    freeze list (requires_grad) decides who trains."""
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer(lambda_action=0.0)  # not frozen
    result = build_trainable_parameters(trainer)

    surfaced = {id(p) for p in result}
    assert id(trainer.architecture.action_backbone.proj.weight) in surfaced
    assert all(p.requires_grad for p in trainer.architecture.action_backbone.parameters())


def test_default_returns_flat_list_when_no_overrides():
    """Without LR overrides, returns the flat parameter list."""
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    trainer = _FakeTrainer()
    result = build_trainable_parameters(trainer)

    assert isinstance(result, list)
    assert all(isinstance(p, torch.nn.Parameter) for p in result)

    arch = trainer.architecture
    surfaced = {id(p) for p in result}
    # action: proj.weight, proj.bias
    assert id(arch.action_backbone.proj.weight) in surfaced
    assert id(arch.action_backbone.proj.bias) in surfaced
    # video: dit.weight, dit.bias
    assert id(arch.video_backbone._pipe.dit.weight) in surfaced
    assert id(arch.video_backbone._pipe.dit.bias) in surfaced
    # frozen: not surfaced
    assert id(arch.video_backbone._pipe.text_encoder.weight) not in surfaced
    assert id(arch.video_backbone._pipe.vae.weight) not in surfaced
