# Import architecture packages to trigger @register_architecture decorators
from openwam.model import architectures  # noqa: F401
from openwam.model.action_backbone.separate_action_dit import ActionDiT, ActionDiTState
from openwam.model.action_backbone.shared_action_backbone import SharedMoEActionBackbone
from openwam.model.architectures import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    ActionState,
    BaseWAMArchitecture,
    CanonicalArchitectureSpec,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
    normalize_architecture_spec,
    resolve_architecture_config,
)

__all__ = [
    "ActionDiT",
    "ActionDiTState",
    "SharedMoEActionBackbone",
    "BaseWAMArchitecture",
    "ActionState",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "normalize_architecture_spec",
    "resolve_architecture_config",
    "CanonicalArchitectureSpec",
]
