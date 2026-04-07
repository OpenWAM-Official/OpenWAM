from open_wam.models.architectures.base import ActionState, BaseWAMArchitecture

# Import concrete implementations to trigger registration
from open_wam.models.architectures.dual_system import DualSystemArchitecture  # noqa: F401
from open_wam.models.architectures.moe_expert import MoEActionExpertArchitecture  # noqa: F401
from open_wam.models.architectures.registry import (
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    build_architecture,
    get_architecture_support,
    list_experimental_architectures,
    list_supported_architectures,
    register_architecture,
)
from open_wam.models.architectures.shared_backbone import SharedBackboneArchitecture  # noqa: F401

__all__ = [
    "BaseWAMArchitecture",
    "ActionState",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "register_architecture",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "list_experimental_architectures",
    "DualSystemArchitecture",
    "MoEActionExpertArchitecture",
    "SharedBackboneArchitecture",
]
