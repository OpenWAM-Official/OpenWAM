"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations


def _action_named_parameters(model):
    action_backbone = model.architecture.action_backbone
    if action_backbone is None:
        return []
    return [(name, param) for name, param in action_backbone.named_parameters() if param.requires_grad]


def _pipe_named_parameters(model):
    """Yield (name, param) pairs for all trainable non-action top-level modules.

    Source of truth: ``BaseWAMArchitecture.get_trainable_modules()``. We exclude
    ``action_backbone`` (``_action_named_parameters`` handles it — its own LR group
    + lambda_action guard). Module name is prefixed onto each param name so the
    LoRA-by-name partition (``"lora" in name.lower()``) keeps working.
    """
    arch = model.architecture
    pairs = []
    for mod_name, mod in arch.get_trainable_modules(freeze_list=()).items():
        if mod_name == "action_backbone":
            continue
        for name, param in mod.named_parameters():
            if param.requires_grad:
                pairs.append((f"{mod_name}.{name}", param))
    return pairs


def build_trainable_parameters(model, *, action_lr=None, video_lr=None, lora_lr=None):
    """Build optimizer param groups for OpenWAM training.

    With no per-module LR override, returns a flat param list. Otherwise returns
    groups so action / video / LoRA can each take their own LR.
    """
    if getattr(model, "lambda_action", 0) <= 0:
        model.architecture.action_backbone.requires_grad_(False)

    action_named = _action_named_parameters(model) if getattr(model, "lambda_action", 0) > 0 else []
    pipe_named = _pipe_named_parameters(model)
    action_params = [param for _, param in action_named]

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


__all__ = ["build_trainable_parameters"]
