"""Action backbone package: the ActionBackbone ABC + its concrete implementations.

    from openwam.model.action_backbone import ActionDiT, SharedVanillaActionBackbone

Unlike video/vlm there is no registry — each architecture constructs its action
backbone directly (dual-system builds ``ActionDiT`` with a variant; shared-backbone
builds ``SharedVanillaActionBackbone`` / ``SharedMoEActionBackbone``), so this file
only re-exports the public classes.
"""

from openwam.model.action_backbone.action_dit import ActionDiT
from openwam.model.action_backbone.base import (
    ActionBackbone,
    ActionDiTBackbone,
    SharedActionBackbone,
)
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.action_backbone.shared_moe import SharedMoEActionBackbone
from openwam.model.action_backbone.shared_vanilla import SharedVanillaActionBackbone

__all__ = [
    "ActionBackbone",
    "ActionDiT",
    "ActionScheduler",
    "ActionDiTBackbone",
    "SharedActionBackbone",
    "SharedMoEActionBackbone",
    "SharedVanillaActionBackbone",
]
