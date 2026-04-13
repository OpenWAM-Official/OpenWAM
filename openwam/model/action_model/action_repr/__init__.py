from typing import Dict, Type

from openwam.model.action_model.action_repr.base import BaseActionRepresentation
from openwam.model.action_model.action_repr.continuous import ContinuousActionRepresentation

ACTION_REPR_REGISTRY: Dict[str, Type[BaseActionRepresentation]] = {}


def register_action_repr(name: str):
    """Decorator to register an action representation class by name."""

    def wrapper(cls):
        if name in ACTION_REPR_REGISTRY:
            raise ValueError(f"Action representation '{name}' already registered")
        ACTION_REPR_REGISTRY[name] = cls
        return cls

    return wrapper


def build_action_representation(name: str, **kwargs) -> BaseActionRepresentation:
    """Instantiate a registered action representation by name.

    Args:
        name: Registry key (e.g. "continuous", "fast").
        **kwargs: Passed to representation constructor.

    Returns:
        Instantiated BaseActionRepresentation.
    """
    if name not in ACTION_REPR_REGISTRY:
        available = ", ".join(sorted(ACTION_REPR_REGISTRY.keys()))
        raise ValueError(f"Unknown action representation '{name}'. Available: {available}")
    return ACTION_REPR_REGISTRY[name](**kwargs)


def list_registered_action_reprs() -> list[str]:
    """Return list of registered action representation names."""
    return sorted(ACTION_REPR_REGISTRY.keys())


# ---- Auto-registration of built-in representations ----

register_action_repr("continuous")(ContinuousActionRepresentation)

try:
    from openwam.model.action_model.action_repr.fast import FASTActionRepresentation

    register_action_repr("fast")(FASTActionRepresentation)
except ImportError:
    pass  # FAST dependencies are optional


__all__ = [
    "BaseActionRepresentation",
    "ContinuousActionRepresentation",
    "ACTION_REPR_REGISTRY",
    "register_action_repr",
    "build_action_representation",
    "list_registered_action_reprs",
]
