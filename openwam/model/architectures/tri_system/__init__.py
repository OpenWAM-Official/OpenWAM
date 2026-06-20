"""Side-effect import of tri_system_joint_self_attn architecture."""

from openwam.model.architectures.tri_system.joint_self_attn import (
    TriSystemJointSelfAttnArchitecture,
)
from openwam.model.architectures.utils.mot_utils import TriSystemMoTDriver

__all__ = ["TriSystemJointSelfAttnArchitecture", "TriSystemMoTDriver"]
