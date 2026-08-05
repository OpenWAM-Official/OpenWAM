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

from typing import Dict, Type

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.utils import get_cfg

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class by name.

    Args:
        name: Config-level type name (e.g., "droid", "bridge_v2", "oxe").
    """

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def build_dataset(config, split: str = "train") -> BaseDataset:
    """Build a dataset from a config dict or DictConfig.

    Looks up the dataset class from ``DATASET_REGISTRY`` using ``config.type``
    and dispatches to ``cls.from_config(config, split)``. Every registered
    class is required to implement ``from_config`` — there is no fallback
    generic-kwargs path.

    Args:
        config: Dict-like config with at least a ``type`` field.
        split: "train" or "val".

    Returns:
        Instantiated dataset.
    """
    dtype = get_cfg(config, "type")
    if dtype not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type '{dtype}'. Available: {list(DATASET_REGISTRY.keys())}")

    cls = DATASET_REGISTRY[dtype]
    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Registered dataset class {cls.__name__} (type={dtype!r}) lacks a "
            f"from_config classmethod. All registered datasets must implement it."
        )
    return cls.from_config(config, split=split)


def list_registered_datasets():
    """Return list of registered dataset type names."""
    return sorted(DATASET_REGISTRY.keys())


# ---- Auto-registration of built-in datasets ----
# This runs when the module is first imported.


def _register_builtins():
    """Register all built-in dataset classes."""
    from openwam.dataloader.agibotworld import MultiAgiBotWorldDataset
    from openwam.dataloader.behavior import BehaviorDataset
    from openwam.dataloader.ebench import EBenchDataset
    from openwam.dataloader.ego4d import Ego4DDataset
    from openwam.dataloader.egodex import EgoDexDataset
    from openwam.dataloader.interndata_a1 import InternDataA1Dataset
    from openwam.dataloader.libero import MultiLiberoDataset
    from openwam.dataloader.mixture import MixtureDataset
    from openwam.dataloader.oxe_bcz import OxeBczDataset
    from openwam.dataloader.oxe_bridge import OxeBridgeDataset
    from openwam.dataloader.oxe_droid import OxeDroidDataset
    from openwam.dataloader.oxe_fractal import OxeFractalDataset
    from openwam.dataloader.robocasa365 import MultiTaskRoboCasa365Dataset
    from openwam.dataloader.robocasa_gr1 import MultiRoboCasaGR1Dataset
    from openwam.dataloader.robocoin import MultiRobotCOINDataset
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    register_dataset("robotwin")(MultiTaskRoboTwinDataset)
    register_dataset("agibotworld")(MultiAgiBotWorldDataset)
    register_dataset("mixture")(MixtureDataset)
    register_dataset("behavior")(BehaviorDataset)
    register_dataset("robocasa_gr1")(MultiRoboCasaGR1Dataset)
    register_dataset("robocoin")(MultiRobotCOINDataset)
    register_dataset("egodex")(EgoDexDataset)
    register_dataset("ebench")(EBenchDataset)
    register_dataset("ego4d")(Ego4DDataset)
    register_dataset("libero")(MultiLiberoDataset)
    register_dataset("oxe_bcz")(OxeBczDataset)
    register_dataset("oxe_bridge")(OxeBridgeDataset)
    register_dataset("oxe_fractal")(OxeFractalDataset)
    register_dataset("oxe_droid")(OxeDroidDataset)
    register_dataset("robocasa365")(MultiTaskRoboCasa365Dataset)
    # Registered on the SINGLE-bucket class: its from_config returns either one
    # bucket or a MultiInternDataA1Dataset depending on whether dataset_dir
    # points at a bucket or at the dataset root.
    register_dataset("interndata_a1")(InternDataA1Dataset)


_register_builtins()
