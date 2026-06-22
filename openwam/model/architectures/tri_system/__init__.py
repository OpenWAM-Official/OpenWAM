"""Side-effect import of tri_system_joint_self_attn architecture."""

from openwam.model.architectures.tri_system.joint_self_attn import (
    TriSystemJointSelfAttnArchitecture,
)
from openwam.model.architectures.tri_system.mot_driver import TriSystemMoTDriver

__all__ = ["TriSystemJointSelfAttnArchitecture", "TriSystemMoTDriver"]
