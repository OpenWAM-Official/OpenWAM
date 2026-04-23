"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations

import types

NO_WD_PARAM_SUFFIXES: tuple[str, ...] = ("modality_tmod_bias",)


def _is_no_wd(name: str) -> bool:
    """Return True for params that must bypass weight decay.

    Zero-initialized additive biases like ``modality_tmod_bias`` have training
    target = deviate from zero; applying AdamW weight_decay would actively pull
    them back, effectively disabling the feature.
    """
    return any(name.endswith(suffix) for suffix in NO_WD_PARAM_SUFFIXES)


def _action_named_parameters(model):
    action_dit = getattr(model, "action_dit", None)
    if action_dit is None:
        return []
    return [(name, param) for name, param in action_dit.named_parameters() if param.requires_grad]


def _pipe_named_parameters(model):
    pipe = getattr(model, "pipe", None)
    if pipe is None:
        return []
    return [(name, param) for name, param in pipe.named_parameters() if param.requires_grad]


def build_trainable_parameters(
    model,
    *,
    action_lr=None,
    video_lr=None,
    lora_lr=None,
):
    """Build optimizer params/groups for the legacy training module.

    Returns optimizer param groups. Always isolates `modality_tmod_bias` (and
    any other suffix in ``NO_WD_PARAM_SUFFIXES``) into a dedicated
    ``weight_decay=0`` group, regardless of per-module LR overrides.
    """
    if getattr(model, "lambda_action", 0) <= 0 and hasattr(model, "action_dit"):
        model.action_dit.requires_grad_(False)

    action_named = _action_named_parameters(model) if getattr(model, "lambda_action", 0) > 0 else []
    pipe_named = _pipe_named_parameters(model)

    # Split no-wd params out first. They inherit the side they came from
    # (action vs video) for LR purposes via the group's lr setting below.
    action_no_wd = [p for n, p in action_named if _is_no_wd(n)]
    action_params = [p for n, p in action_named if not _is_no_wd(n)]
    pipe_no_wd = [p for n, p in pipe_named if _is_no_wd(n)]
    pipe_named = [(n, p) for n, p in pipe_named if not _is_no_wd(n)]

    # When no LR override AND no no-wd params, preserve flat-list behavior for
    # backward compatibility with existing checkpoints / test expectations.
    if action_lr is None and video_lr is None and lora_lr is None and not action_no_wd and not pipe_no_wd:
        return action_params + [param for _, param in pipe_named]

    groups = []
    if action_params:
        action_group = {"params": action_params}
        if action_lr is not None:
            action_group["lr"] = action_lr
        groups.append(action_group)

    if action_no_wd:
        group = {"params": action_no_wd, "weight_decay": 0.0}
        if action_lr is not None:
            group["lr"] = action_lr
        groups.append(group)

    if lora_lr is not None:
        lora_params = [param for name, param in pipe_named if "lora" in name.lower()]
        video_params = [param for name, param in pipe_named if "lora" not in name.lower()]
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr})
    else:
        video_params = [param for _, param in pipe_named]

    if video_params:
        video_group = {"params": video_params}
        if video_lr is not None:
            video_group["lr"] = video_lr
        groups.append(video_group)

    if pipe_no_wd:
        group = {"params": pipe_no_wd, "weight_decay": 0.0}
        if video_lr is not None:
            group["lr"] = video_lr
        groups.append(group)

    return groups


def attach_optimizer_groups(
    model,
    *,
    action_lr=None,
    video_lr=None,
    lora_lr=None,
):
    """Attach scale-oriented optimizer grouping to the legacy training module."""

    def _trainable_modules(self):
        return build_trainable_parameters(
            self,
            action_lr=action_lr,
            video_lr=video_lr,
            lora_lr=lora_lr,
        )

    model.trainable_modules = types.MethodType(_trainable_modules, model)
    return model


__all__ = ["attach_optimizer_groups", "build_trainable_parameters"]
