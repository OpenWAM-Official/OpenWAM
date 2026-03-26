"""WAM Architecture registry.

Provides a decorator-based registration pattern (inspired by StarVLA's
FRAMEWORK_REGISTRY) for discovering and instantiating WAM architectures
from configuration.

Usage:
    @register_architecture("dual_system")
    class DualSystemArchitecture(BaseWAMArchitecture):
        ...

    arch = build_architecture("dual_system", cfg)
"""

from typing import Dict, Type

from open_wam.models.architectures.base import BaseWAMArchitecture

ARCHITECTURE_REGISTRY: Dict[str, Type[BaseWAMArchitecture]] = {}


def register_architecture(name: str):
    """Decorator to register a WAM architecture class."""
    def decorator(cls: Type[BaseWAMArchitecture]):
        if name in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Architecture '{name}' already registered")
        ARCHITECTURE_REGISTRY[name] = cls
        return cls
    return decorator


def build_architecture(name: str, cfg=None) -> BaseWAMArchitecture:
    """Instantiate a registered WAM architecture by name.

    Args:
        name: Registry key (e.g. "dual_system", "moe_expert", "shared_backbone").
        cfg: Architecture configuration (passed to constructor).

    Returns:
        Instantiated BaseWAMArchitecture subclass.
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown architecture '{name}'. Available: {available}"
        )
    return ARCHITECTURE_REGISTRY[name](cfg)
