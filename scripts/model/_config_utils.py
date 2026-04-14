"""Shared utilities for GPU test scripts.

Loads and merges architecture + action_backbone + video_backbone YAML configs,
matching the same logic used by NativeTrainer.
"""

import os

import torch.nn as nn
import yaml


def load_architecture_config(
    arch_yaml: str = "configs/model/dual_system.yaml",
    video_yaml: str = None,
    action_yaml: str = None,
):
    """Load and merge architecture config from YAML files.

    Args:
        arch_yaml: Path to architecture YAML (dual_system / moe_expert / shared_backbone).
        video_yaml: Path to video_backbone YAML. If None, resolved from arch defaults.
        action_yaml: Path to action_backbone YAML. If None, resolved from arch defaults.

    Returns:
        dict with keys: architecture, video_backbone, action_backbone (each a dict)
    """
    with open(arch_yaml) as f:
        arch_raw = yaml.safe_load(f)

    # Resolve defaults
    defaults = {}
    for d in arch_raw.get("defaults", []):
        if isinstance(d, dict):
            defaults.update(d)

    # Video backbone
    if video_yaml is None:
        vb_name = defaults.get("video_backbone", "ti2v_5b")
        video_yaml = f"configs/model/video_backbone/{vb_name}.yaml"
    with open(video_yaml) as f:
        video_cfg = yaml.safe_load(f)

    # Action backbone (optional — moe/shared don't have one)
    action_cfg = {}
    if action_yaml is None:
        ab_name = defaults.get("action_backbone")
        if ab_name:
            action_yaml = f"configs/model/action_backbone/{ab_name}.yaml"
    if action_yaml and os.path.exists(action_yaml):
        with open(action_yaml) as f:
            action_cfg = yaml.safe_load(f)

    result = {
        "architecture": arch_raw.get("architecture", {}),
        "video_backbone": video_cfg,
        "action_backbone": action_cfg,
    }
    # Include freeze list if present in architecture yaml
    if "freeze" in arch_raw:
        result["freeze"] = arch_raw["freeze"]
    return result


def merge_arch_params(cfg: dict, video_dim: int) -> dict:
    """Merge architecture + action_backbone params + video_dim into flat dict.

    This mirrors the merging logic in NativeTrainer.
    """
    params = dict(cfg["architecture"])
    params.update(cfg.get("action_backbone", {}))
    params["video_dim"] = video_dim

    arch_type = params.get("type", "dual_system")
    bridge_layers = params.get("bridge_layers", [])

    if arch_type == "dual_system":
        params["num_layers"] = len(bridge_layers)
    elif arch_type == "moe_expert":
        params["num_experts"] = len(bridge_layers)
        params["expert_layers"] = tuple(bridge_layers)

    return params


def apply_freeze(pipe, freeze_list):
    """Freeze pipeline components specified in the list."""
    for name in freeze_list:
        module = getattr(pipe, name, None)
        if module is not None and isinstance(module, nn.Module):
            module.requires_grad_(False)


def print_freeze_status(pipe, action_model, arch_name="architecture", freeze_list=None):
    """Print trainable/frozen status for all model components.

    Args:
        pipe: WanVideoPipeline instance.
        action_model: ActionDiT / MoEExpertDiT / SharedBackboneArchitecture instance.
        arch_name: Architecture name for display.
        freeze_list: List of frozen component names (for display context).
    """

    def _param_summary(module):
        if module is None:
            return "not loaded"
        if not isinstance(module, nn.Module):
            return "not a module"
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = total - trainable
        if total == 0:
            return "no parameters"
        if trainable == 0:
            return f"FROZEN  ({total / 1e6:.1f}M params)"
        if frozen == 0:
            return f"TRAIN   ({total / 1e6:.1f}M params)"
        return f"PARTIAL ({trainable / 1e6:.1f}M train / {frozen / 1e6:.1f}M frozen)"

    print(f"\n  Freeze/Trainable Status ({arch_name}):")
    if freeze_list:
        print(f"  Config freeze list: {freeze_list}")
    print(f"  {'Component':<20s}  {'Status'}")
    print(f"  {'-' * 20}  {'-' * 40}")

    # Pipeline components
    for name in [
        "dit",
        "vace",
        "vae",
        "text_encoder",
        "image_encoder",
        "audio_encoder",
        "motion_controller",
        "animate_adapter",
    ]:
        module = getattr(pipe, name, None)
        if module is not None:
            print(f"  {'pipe.' + name:<20s}  {_param_summary(module)}")

    # Action model
    if action_model is not None:
        print(f"  {'action_model':<20s}  {_param_summary(action_model)}")

    # Grand total
    all_trainable = 0
    all_frozen = 0
    for name in [
        "dit",
        "vace",
        "vae",
        "text_encoder",
        "image_encoder",
        "audio_encoder",
        "motion_controller",
        "animate_adapter",
    ]:
        module = getattr(pipe, name, None)
        if module is not None and isinstance(module, nn.Module):
            all_trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)
            all_frozen += sum(p.numel() for p in module.parameters() if not p.requires_grad)
    if action_model is not None and isinstance(action_model, nn.Module):
        all_trainable += sum(p.numel() for p in action_model.parameters() if p.requires_grad)
        all_frozen += sum(p.numel() for p in action_model.parameters() if not p.requires_grad)

    print(f"  {'-' * 20}  {'-' * 40}")
    print(f"  {'TOTAL':<20s}  {all_trainable / 1e6:.1f}M trainable, {all_frozen / 1e6:.1f}M frozen")
