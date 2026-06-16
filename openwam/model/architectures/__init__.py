"""Architecture package exports and side-effect registration."""

from openwam.model.architectures import dual_system, shared_backbone, tri_system  # noqa: F401
from openwam.model.architectures.architecture_base import ActionState, BaseWAMArchitecture
from openwam.model.architectures.dual_system import (
    DualSystemCrossAttnArchitecture,
    DualSystemIDMArchitecture,
    DualSystemSelfAttnArchitecture,
)
from openwam.model.architectures.registry import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    CanonicalArchitectureSpec,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
    normalize_architecture_spec,
    register_architecture,
    resolve_architecture_config,
)
from openwam.model.architectures.shared_backbone import (
    SharedBackboneMoEArchitecture,
    SharedBackboneVanillaArchitecture,
)
from openwam.model.architectures.tri_system import TriSystemJointSelfAttnArchitecture

__all__ = [
    "ActionState",
    "BaseWAMArchitecture",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "CanonicalArchitectureSpec",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "normalize_architecture_spec",
    "register_architecture",
    "resolve_architecture_config",
    "DualSystemCrossAttnArchitecture",
    "DualSystemIDMArchitecture",
    "DualSystemSelfAttnArchitecture",
    "SharedBackboneMoEArchitecture",
    "SharedBackboneVanillaArchitecture",
    "TriSystemJointSelfAttnArchitecture",
]
