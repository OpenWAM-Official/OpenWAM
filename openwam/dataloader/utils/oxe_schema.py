"""OXE → 10-D EEF schema converters.

Each OXE dataset has its own raw state/action representation:

  * BC-Z / Bridge: ``state = [x,y,z,roll,pitch,yaw,pad,gripper] (8)``,
    ``action = [x,y,z,roll,pitch,yaw,gripper] (7)`` (Euler XYZ).
  * Fractal: ``state = [x,y,z,rx,ry,rz,rw,gripper] (8)`` (quat xyzw),
    ``action = [x,y,z,roll,pitch,yaw,gripper] (7)`` (Euler XYZ).
  * DROID: ``observation.state.cartesian_position (6) +
    observation.state.gripper_position (1) = 7`` (Euler XYZ),
    ``action.original = [x,y,z,roll,pitch,yaw,gripper] (7)`` (Euler XYZ).

These helpers normalize all four data sources to a single 10-D EEF
representation ``[pos(3) + rot6d(6) + grip(1)]`` so the reader and
stats-computation paths can share code.

The 10-D output is then passed through :func:`assemble_single_arm_left`
to slot into the canonical bimanual 20-D EEF schema.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.eef import euler_xyz_to_rot6d, quat_xyzw_to_rot6d

ARM10_DIM = 10


# ---------------------------------------------------------------------------
# BC-Z / Bridge: state has a 'pad' slot at index 6
# ---------------------------------------------------------------------------


def bcz_state_to_arm10(state: np.ndarray) -> np.ndarray:
    """``(..., 8)`` BC-Z / Bridge state → ``(..., 10)`` EEF.

    Layout: ``state[:, [0:3, 3:6, 7]]`` (drops pad at index 6).
    """
    pos = state[..., 0:3]
    euler = state[..., 3:6]
    grip = state[..., 7:8]
    rot6d = euler_xyz_to_rot6d(euler)
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


def euler7_action_to_arm10(action: np.ndarray) -> np.ndarray:
    """``(..., 7)`` ``[x,y,z,roll,pitch,yaw,gripper]`` → ``(..., 10)`` EEF.

    Used by BC-Z / Bridge / Fractal / DROID action streams (all share this layout
    after the dataset's own conversion).
    """
    pos = action[..., 0:3]
    euler = action[..., 3:6]
    grip = action[..., 6:7]
    rot6d = euler_xyz_to_rot6d(euler)
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Fractal: state has quaternion in xyzw layout at indices 3:7
# ---------------------------------------------------------------------------


def fractal_state_to_arm10(state: np.ndarray) -> np.ndarray:
    """``(..., 8)`` Fractal state → ``(..., 10)`` EEF.

    Layout: ``state[:, [0:3, 3:7, 7]]`` — quat is xyzw, no pad.
    """
    pos = state[..., 0:3]
    quat = state[..., 3:7]
    grip = state[..., 7:8]
    rot6d = quat_xyzw_to_rot6d(quat)
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# DROID: state assembled from two separate parquet columns
# ---------------------------------------------------------------------------


def droid_state_to_arm10(cartesian: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """DROID state assembly.

    Args:
        cartesian: ``(..., 6)`` from ``observation.state.cartesian_position``
            (Euler XYZ representation: ``[x,y,z,roll,pitch,yaw]``).
        gripper:   ``(..., 1)`` from ``observation.state.gripper_position``
            (scalar in ``[0, 1]``).

    Returns: ``(..., 10)`` EEF tensor.
    """
    pos = cartesian[..., 0:3]
    euler = cartesian[..., 3:6]
    rot6d = euler_xyz_to_rot6d(euler)
    return np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)


__all__ = [
    "ARM10_DIM",
    "bcz_state_to_arm10",
    "euler7_action_to_arm10",
    "fractal_state_to_arm10",
    "droid_state_to_arm10",
]
