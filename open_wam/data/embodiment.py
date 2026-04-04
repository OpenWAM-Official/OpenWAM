"""Multi-embodiment action space abstraction.

Provides a unified EE delta action representation that maps between different
robot embodiments (varying DoFs, joint vs EEF spaces, single vs bimanual).

This allows a single model to be trained on data from multiple robots by:
1. Converting each robot's native actions to a canonical format
2. Padding/slicing to a fixed dimension
3. Converting back to robot-native format at deployment time

Canonical format: (max_dim,) vector where:
- [0:3]   = left/primary EE delta position (dx, dy, dz)
- [3:6]   = left/primary EE delta rotation (droll, dpitch, dyaw)
- [6]     = left/primary gripper
- [7:10]  = right EE delta position (bimanual only)
- [10:13] = right EE delta rotation (bimanual only)
- [13]    = right gripper (bimanual only)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


# Canonical action dimensions
CANONICAL_SINGLE_ARM_DIM = 7   # pos(3) + rot(3) + grip(1)
CANONICAL_BIMANUAL_DIM = 14    # 2 * single_arm


class EmbodimentConfig:
    """Configuration for a specific robot embodiment.

    Args:
        name: Human-readable embodiment name.
        native_action_dim: Dimension of the robot's native action space.
        action_type: "ee_delta" (EE-relative) or "joint_delta" (joint-relative).
        bimanual: Whether the robot has two arms.
        native_to_canonical: Mapping from native action indices to canonical indices.
            e.g., {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6} for 7-DoF single arm.
        action_scale: Per-dimension scale factor for normalizing across embodiments.
    """

    def __init__(
        self,
        name: str,
        native_action_dim: int,
        action_type: str = "ee_delta",
        bimanual: bool = False,
        native_to_canonical: Optional[Dict[int, int]] = None,
        action_scale: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.native_action_dim = native_action_dim
        self.action_type = action_type
        self.bimanual = bimanual
        self.canonical_dim = CANONICAL_BIMANUAL_DIM if bimanual else CANONICAL_SINGLE_ARM_DIM

        # Default: identity mapping (native[i] -> canonical[i])
        if native_to_canonical is None:
            self.native_to_canonical = {i: i for i in range(min(native_action_dim, self.canonical_dim))}
        else:
            self.native_to_canonical = native_to_canonical

        # Build reverse mapping
        self.canonical_to_native = {v: k for k, v in self.native_to_canonical.items()}

        if action_scale is not None:
            self.action_scale = np.asarray(action_scale, dtype=np.float32)
        else:
            self.action_scale = np.ones(native_action_dim, dtype=np.float32)


# Pre-defined embodiment configurations
EMBODIMENTS: Dict[str, EmbodimentConfig] = {}


def register_embodiment(name: str, config: EmbodimentConfig):
    """Register an embodiment configuration."""
    EMBODIMENTS[name] = config
    return config


def get_embodiment(name: str) -> EmbodimentConfig:
    """Get a registered embodiment config by name."""
    if name not in EMBODIMENTS:
        raise KeyError(
            f"Unknown embodiment '{name}'. Available: {list(EMBODIMENTS.keys())}"
        )
    return EMBODIMENTS[name]


# --- Register standard embodiments ---

# Franka Panda (DROID) - 7DoF EEF absolute/delta
register_embodiment("franka", EmbodimentConfig(
    name="franka",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    # (x, y, z, roll, pitch, yaw, gripper) -> canonical (same layout)
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# WidowX (Bridge V2) - 7DoF EEF delta
register_embodiment("widowx", EmbodimentConfig(
    name="widowx",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# Google Robot (SimplerEnv) - 7DoF EEF delta
register_embodiment("google_robot", EmbodimentConfig(
    name="google_robot",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# ARX-X5 (RoboTwin) - 14DoF bimanual
register_embodiment("arx-x5", EmbodimentConfig(
    name="arx-x5",
    native_action_dim=14,
    action_type="ee_delta",
    bimanual=True,
    # Left arm: native[0:7] -> canonical[0:7]
    # Right arm: native[7:14] -> canonical[7:14]
    native_to_canonical={
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
    },
))

# KUKA iiwa (OXE) - 7DoF EEF delta
register_embodiment("kuka", EmbodimentConfig(
    name="kuka",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# Universal Robots UR5 (OXE) - 6DoF + gripper
register_embodiment("ur5", EmbodimentConfig(
    name="ur5",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# Rethink Sawyer (OXE) - 7DoF EEF delta
register_embodiment("sawyer", EmbodimentConfig(
    name="sawyer",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# UFactory xArm (OXE) - 7DoF EEF delta
register_embodiment("xarm", EmbodimentConfig(
    name="xarm",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# Kinova Jaco (OXE) - 7DoF EEF delta
register_embodiment("jaco", EmbodimentConfig(
    name="jaco",
    native_action_dim=7,
    action_type="ee_delta",
    bimanual=False,
    native_to_canonical={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
))

# Aloha (bimanual) - 14DoF joint delta
register_embodiment("aloha", EmbodimentConfig(
    name="aloha",
    native_action_dim=14,
    action_type="joint_delta",
    bimanual=True,
    native_to_canonical={
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
    },
))


class ActionSpaceAdapter:
    """Converts between native and canonical action representations.

    Args:
        embodiment: Name of the embodiment or EmbodimentConfig instance.
        target_dim: Target canonical dimension. Defaults to the embodiment's
            canonical dim. Set to CANONICAL_BIMANUAL_DIM for unified training.
    """

    def __init__(
        self,
        embodiment,
        target_dim: Optional[int] = None,
    ):
        if isinstance(embodiment, str):
            self.config = get_embodiment(embodiment)
        else:
            self.config = embodiment

        self.target_dim = target_dim or self.config.canonical_dim

    def native_to_canonical(self, actions: np.ndarray) -> np.ndarray:
        """Convert native robot actions to canonical format.

        Args:
            actions: (..., native_action_dim) array.

        Returns:
            (..., target_dim) canonical actions.
        """
        shape = actions.shape[:-1]
        native_dim = actions.shape[-1]
        canonical = np.zeros((*shape, self.target_dim), dtype=np.float32)

        for native_idx, canon_idx in self.config.native_to_canonical.items():
            if native_idx < native_dim and canon_idx < self.target_dim:
                canonical[..., canon_idx] = actions[..., native_idx] * self.config.action_scale[native_idx]

        return canonical

    def canonical_to_native(self, canonical: np.ndarray) -> np.ndarray:
        """Convert canonical actions back to native robot format.

        Args:
            canonical: (..., target_dim) canonical actions.

        Returns:
            (..., native_action_dim) native actions.
        """
        shape = canonical.shape[:-1]
        native = np.zeros((*shape, self.config.native_action_dim), dtype=np.float32)

        for canon_idx, native_idx in self.config.canonical_to_native.items():
            if canon_idx < canonical.shape[-1] and native_idx < self.config.native_action_dim:
                scale = self.config.action_scale[native_idx]
                native[..., native_idx] = canonical[..., canon_idx] / max(scale, 1e-8)

        return native

    @property
    def canonical_dim(self) -> int:
        return self.target_dim

    @property
    def native_dim(self) -> int:
        return self.config.native_action_dim
