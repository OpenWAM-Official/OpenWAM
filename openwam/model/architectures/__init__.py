"""Architecture package exports and side-effect registration."""

from openwam.model.architectures import dual_system, single_system, tri_system  # noqa: F401
from openwam.model.architectures.base import ActionState, BaseWAMArchitecture
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
from openwam.model.architectures.single_system import (
    SingleSystemMoEArchitecture,
    SingleSystemVanillaArchitecture,
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
    "SingleSystemMoEArchitecture",
    "SingleSystemVanillaArchitecture",
    "TriSystemJointSelfAttnArchitecture",
]
