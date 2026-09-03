"""SingleSystem architecture family.

- :class:`SingleSystemVanillaArchitecture` — vanilla single system.
- :class:`SingleSystemMoEArchitecture` — adds expert FFN at configured layers.
"""

from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture
from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

__all__ = [
    "SingleSystemMoEArchitecture",
    "SingleSystemVanillaArchitecture",
]
