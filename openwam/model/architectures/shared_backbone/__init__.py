"""SharedBackbone architecture family.

- :class:`SharedBackboneVanillaArchitecture` — vanilla shared backbone.
- :class:`SharedBackboneMoEArchitecture` — adds expert FFN at configured layers.
"""

from openwam.model.architectures.shared_backbone.moe import SharedBackboneMoEArchitecture
from openwam.model.architectures.shared_backbone.vanilla import SharedBackboneVanillaArchitecture

__all__ = [
    "SharedBackboneMoEArchitecture",
    "SharedBackboneVanillaArchitecture",
]
