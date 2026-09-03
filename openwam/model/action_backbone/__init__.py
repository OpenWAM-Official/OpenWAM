"""Action backbone package: the action-stream ABCs + concrete implementations.

    from openwam.model.action_backbone import ActionDiT, SharedVanillaActionBackbone

Unlike video/vlm there is no registry — each architecture constructs its action
backbone directly (dual-system builds ``ActionDiT`` with a variant; single-system
builds ``SharedVanillaActionBackbone`` / ``SharedMoEActionBackbone``), so this file
only re-exports the public classes.
"""

from openwam.model.action_backbone.base import (
    ActionDiTBackbone,
    SharedActionBackbone,
)
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.action_backbone.shared_action_backbone import (
    SharedMoEActionBackbone,
    SharedVanillaActionBackbone,
)

__all__ = [
    "ActionDiT",
    "ActionScheduler",
    "ActionDiTBackbone",
    "SharedActionBackbone",
    "SharedMoEActionBackbone",
    "SharedVanillaActionBackbone",
]
