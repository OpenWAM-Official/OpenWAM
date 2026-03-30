"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations

import types


def _pipe_named_parameters(model):
    pipe = getattr(model, "pipe", None)
    if pipe is None:
        return []
    return [(name, param) for name, param in pipe.named_parameters() if param.requires_grad]


def _action_parameters(model):
    action_dit = getattr(model, "action_dit", None)
    if action_dit is None:
        return []
    return [param for param in action_dit.parameters() if param.requires_grad]


def build_trainable_parameters(
    model,
    *,
    action_lr=None,
    video_lr=None,
    lora_lr=None,
):
    """Build optimizer params/groups for the legacy training module.

    Returns either a flat parameter list (legacy behavior) or optimizer param
    groups when any per-module LR override is requested.
    """
    if getattr(model, "lambda_action", 0) <= 0 and hasattr(model, "action_dit"):
        model.action_dit.requires_grad_(False)

    action_params = _action_parameters(model) if getattr(model, "lambda_action", 0) > 0 else []
    pipe_named = _pipe_named_parameters(model)

    if action_lr is None and video_lr is None and lora_lr is None:
        return action_params + [param for _, param in pipe_named]

    groups = []
    if action_params:
        action_group = {"params": action_params}
        if action_lr is not None:
            action_group["lr"] = action_lr
        groups.append(action_group)

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
