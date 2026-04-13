# Import architecture modules to trigger @register_architecture decorators
import openwam.model.dual_system  # noqa: F401
import openwam.model.moe_expert  # noqa: F401
import openwam.model.shared_backbone  # noqa: F401
from openwam.model.action_model.action_dit import ActionDiT, ActionDiTState
from openwam.model.action_model.moe_expert_dit import MoEExpertDiT, MoEExpertState
from openwam.model.action_model.proprioceptive import ProprioceptiveEncoder
from openwam.model.backbone.base import BaseVideoBackbone
from openwam.model.base import ActionState, BaseWAMArchitecture
from openwam.model.registry import (
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
)

__all__ = [
    "BaseVideoBackbone",
    "ActionDiT",
    "ActionDiTState",
    "MoEExpertDiT",
    "MoEExpertState",
    "BaseWAMArchitecture",
    "ActionState",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "ProprioceptiveEncoder",
]
