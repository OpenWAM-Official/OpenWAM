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
    action_backbone = model.architecture.action_backbone
    if action_backbone is None:
        return []
    return [(name, param) for name, param in action_backbone.named_parameters() if param.requires_grad]


def _pipe_named_parameters(model):
    """Yield (name, param) pairs for all trainable non-action top-level modules.

    Source of truth: ``BaseWAMArchitecture.get_trainable_modules()`` — walks
    ``named_children()`` and returns top-level modules with at least one
    ``requires_grad=True`` param. We exclude ``action_backbone`` here because
    ``_action_named_parameters`` handles it independently (its own LR group +
    lambda_action guard).

    Module name is prefixed onto each param name so the LoRA-by-name partition
    (``"lora" in name.lower()``) and the ``_is_no_wd`` suffix check both keep
    working, and any debug log of the param list is unambiguous.
    """
    arch = model.architecture
    pairs = []
    # freeze_list=() because Item B's freeze_modules has already toggled
    # requires_grad on nested submodules (e.g. video_backbone._pipe.text_encoder).
    # We only need the per-param requires_grad filter below.
    for mod_name, mod in arch.get_trainable_modules(freeze_list=()).items():
        if mod_name == "action_backbone":
            continue
        for name, param in mod.named_parameters():
            if param.requires_grad:
                pairs.append((f"{mod_name}.{name}", param))
    return pairs


def build_trainable_parameters(
    model,
    *,
    action_lr=None,
    video_lr=None,
    lora_lr=None,
):
    """Build optimizer param groups for OpenWAM training.

    Returns optimizer param groups. Always isolates `modality_tmod_bias` (and
    any other suffix in ``NO_WD_PARAM_SUFFIXES``) into a dedicated
    ``weight_decay=0`` group, regardless of per-module LR overrides.
    """
    if getattr(model, "lambda_action", 0) <= 0:
        model.architecture.action_backbone.requires_grad_(False)

    action_named = _action_named_parameters(model) if getattr(model, "lambda_action", 0) > 0 else []
    pipe_named = _pipe_named_parameters(model)

    # Split no-wd params out first. They inherit the side they came from
    # (action vs video) for LR purposes via the group's lr setting below.
    action_no_wd = [p for n, p in action_named if _is_no_wd(n)]
    action_params = [p for n, p in action_named if not _is_no_wd(n)]
    pipe_no_wd = [p for n, p in pipe_named if _is_no_wd(n)]
    pipe_named = [(n, p) for n, p in pipe_named if not _is_no_wd(n)]

    # When no LR override AND no no-wd params, preserve flat-list behavior for
    # current optimizer builder and test expectations.
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
    """Attach scale-oriented optimizer grouping to a model instance."""

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
