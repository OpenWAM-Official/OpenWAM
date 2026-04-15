"""Shared utilities for GPU test scripts.

Loads architecture YAML configs (which now contain video_backbone and
action_backbone inline), matching the same structure used by OpenWAMTrainer.
"""

import yaml


def load_architecture_config(arch_yaml: str = "configs/model/dual_system.yaml"):
    """Load architecture config from a single YAML file.

    Since video_backbone and action_backbone are now inline in the
    architecture YAML (no separate sub-config files), this simply
    reads and returns the relevant sections.

    Args:
        arch_yaml: Path to architecture YAML (dual_system / moe_expert / shared_backbone).

    Returns:
        dict with keys: architecture, video_backbone, action_backbone (each a dict)
    """
    with open(arch_yaml) as f:
        raw = yaml.safe_load(f)

    return {
        "architecture": raw.get("architecture", {}),
        "video_backbone": raw.get("video_backbone", {}),
        "action_backbone": raw.get("action_backbone", {}),
    }


def merge_arch_params(cfg: dict, video_dim: int) -> dict:
    """Merge architecture + action_backbone params + video_dim into flat dict.

    This mirrors the merging logic in OpenWAMTrainer.
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
