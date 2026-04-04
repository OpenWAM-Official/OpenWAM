"""Dataset registry for config-driven dataset construction.

Replaces the if-else dispatch chains in ``training/runtime.py`` with a
decorator-based registry pattern (consistent with the architecture registry).

Usage:
    # Register a dataset class
    @register_dataset("droid")
    class DROIDDataset(LeRobotBaseDataset):
        ...

    # Build from config
    dataset = build_dataset(config, split="train")
"""

from typing import Dict, Optional, Type

from open_wam.data.base import BaseActionDataset


DATASET_REGISTRY: Dict[str, Type[BaseActionDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class by name.

    Args:
        name: Config-level type name (e.g., "droid", "bridge_v2", "oxe").
    """
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper


def build_dataset(config, split: str = "train") -> BaseActionDataset:
    """Build a dataset from a config dict or DictConfig.

    Looks up the dataset class from DATASET_REGISTRY using ``config.type``
    and calls ``cls.from_config(config, split)`` if available, otherwise
    falls back to direct construction.

    Args:
        config: Dict-like config with at least a ``type`` field.
        split: "train" or "val".

    Returns:
        Instantiated dataset.
    """
    dtype = _get(config, "type")
    if dtype not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type '{dtype}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    cls = DATASET_REGISTRY[dtype]

    # Prefer from_config classmethod if available
    if hasattr(cls, "from_config"):
        return cls.from_config(config, split=split)

    # Fallback: extract kwargs from config
    return _build_from_config(cls, config, split)


def build_dataset_from_mixture_entry(entry) -> BaseActionDataset:
    """Build a single dataset from a mixture config entry.

    Args:
        entry: Dict-like entry with at least ``type`` and dataset-specific params.

    Returns:
        Instantiated dataset.
    """
    return build_dataset(entry, split="train")


def list_registered_datasets():
    """Return list of registered dataset type names."""
    return sorted(DATASET_REGISTRY.keys())


def _get(config, key, default=None):
    """Get a value from dict or DictConfig."""
    if hasattr(config, key):
        return getattr(config, key)
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def _build_from_config(cls, config, split: str):
    """Generic construction from config — extracts common parameters."""
    kwargs = {"split": split}

    # Common parameters shared across all LeRobot datasets
    for param in [
        "dataset_dir", "num_frames", "height", "width", "camera",
        "action_stats_path", "val_ratio", "seed", "action_key",
    ]:
        val = _get(config, param)
        if val is not None:
            kwargs[param] = val

    # Dataset-specific parameters
    for param in [
        "action_type",          # DROID
        "dataset_name",         # OXE
        "embodiment",           # OXE
        "canonical_action_dim", # OXE
        "hdf5_data_root",       # RoboTwin (single-task)
        "robot", "variant",     # RoboTwin
        "multiview",            # RoboTwin
        "target_camera",        # RoboTwin
        "window_stride",        # RoboTwin
        "tasks",                # RoboTwin multitask
    ]:
        val = _get(config, param)
        if val is not None:
            kwargs[param] = val

    return cls(**kwargs)


# ---- Auto-registration of built-in datasets ----
# This runs when the module is first imported.

def _register_builtins():
    """Register all built-in dataset classes."""
    from open_wam.data.droid import DROIDDataset
    from open_wam.data.bridge_v2 import BridgeV2Dataset
    from open_wam.data.oxe import OXEDataset
    from open_wam.data.robotwin import (
        RoboTwinActionDataset,
        MultiTaskRoboTwinActionDataset,
    )
    from open_wam.data.mixture import MixtureDataset

    register_dataset("droid")(DROIDDataset)
    register_dataset("bridge_v2")(BridgeV2Dataset)
    register_dataset("oxe")(OXEDataset)
    register_dataset("robotwin")(RoboTwinActionDataset)
    register_dataset("robotwin_multitask")(MultiTaskRoboTwinActionDataset)
    register_dataset("mixture")(MixtureDataset)


_register_builtins()
