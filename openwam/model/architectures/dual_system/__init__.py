"""DualSystem architecture family.

Two variants split into independent classes:

- :class:`DualSystemCrossAttnArchitecture` — bridge-collection plan, ActionDiT
  runs after the video DiT.
- :class:`DualSystemSelfAttnArchitecture` — interleaved plan, joint ActionDiT
  blocks run inside the video DiT loop.
"""

from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

__all__ = [
    "DualSystemCrossAttnArchitecture",
    "DualSystemSelfAttnArchitecture",
]
