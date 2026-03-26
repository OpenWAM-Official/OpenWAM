from open_wam.models.architectures.base import BaseWAMArchitecture, ActionState
from open_wam.models.architectures.registry import (
    ARCHITECTURE_REGISTRY,
    register_architecture,
    build_architecture,
)
# Import concrete implementations to trigger registration
from open_wam.models.architectures.dual_system import DualSystemArchitecture  # noqa: F401
from open_wam.models.architectures.moe_expert import MoEActionExpertArchitecture  # noqa: F401
from open_wam.models.architectures.shared_backbone import SharedBackboneArchitecture  # noqa: F401

__all__ = [
    "BaseWAMArchitecture",
    "ActionState",
    "ARCHITECTURE_REGISTRY",
    "register_architecture",
    "build_architecture",
    "DualSystemArchitecture",
    "MoEActionExpertArchitecture",
    "SharedBackboneArchitecture",
]
