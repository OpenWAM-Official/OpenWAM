from open_wam.models.action_repr.base import BaseActionRepresentation
from open_wam.models.action_repr.continuous import ContinuousActionRepresentation

__all__ = [
    "BaseActionRepresentation",
    "ContinuousActionRepresentation",
]

# Lazy imports for optional representations
def _import_fast():
    from open_wam.models.action_repr.fast import FASTActionRepresentation
    return FASTActionRepresentation


def build_action_representation(name: str, **kwargs) -> BaseActionRepresentation:
    """Factory for action representations.

    Args:
        name: "continuous" or "fast".
        **kwargs: Passed to representation constructor.

    Returns:
        Instantiated BaseActionRepresentation.
    """
    if name == "continuous":
        return ContinuousActionRepresentation(**kwargs)
    elif name == "fast":
        cls = _import_fast()
        return cls(**kwargs)
    else:
        raise ValueError(
            f"Unknown action representation '{name}'. "
            f"Available: 'continuous', 'fast'"
        )
