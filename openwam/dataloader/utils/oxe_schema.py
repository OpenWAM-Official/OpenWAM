"""OXE → 10-D EEF schema converters.

Each OXE dataset has its own raw state/action representation:

  * BC-Z / Bridge: ``state = [x,y,z,roll,pitch,yaw,pad,gripper] (8)``,
    ``action = [x,y,z,roll,pitch,yaw,gripper] (7)`` (Euler XYZ).
  * Fractal: ``state = [x,y,z,rx,ry,rz,rw,gripper] (8)`` (quat xyzw),
    ``action = [x,y,z,roll,pitch,yaw,gripper] (7)`` (Euler XYZ).
  * DROID: arm-side wrist / gripper-mount pose plus a raw gripper
    **closedness** signal (``0=open``, ``1=closed``).  The re-converted bucket
    stores the achieved pose and gripper scalar in separate columns, while its
    commanded wrist stream stores both in one 7-D column.  DROID-specific
    converters assemble the two layouts and invert only the final scalar to the
    canonical open scale (``0=closed``, ``1=open``).

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

    Used by BC-Z / Bridge / Fractal action streams. Current DROID has the same
    pose layout but opposite raw gripper direction, so it must use
    :func:`droid_euler7_to_arm10` instead.
    """
    pos = action[..., 0:3]
    euler = action[..., 3:6]
    grip = action[..., 6:7]
    rot6d = euler_xyz_to_rot6d(euler)
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


def droid_euler7_to_arm10(value: np.ndarray) -> np.ndarray:
    """Current DROID commanded wrist ``(..., 7)`` -> canonical 10-D EEF.

    ``other_information.action_wrist_pose`` contains
    ``[x,y,z,roll,pitch,yaw,closedness]``.  The shared action space uses an
    openness scale, so only the gripper scalar is inverted; xyz and
    Euler->rot6d are identical to :func:`euler7_action_to_arm10`.
    """
    out = euler7_action_to_arm10(value)
    out[..., 9] = 1.0 - out[..., 9]
    return out


def droid_pose6_closedness_to_arm10(pose: np.ndarray, closedness: np.ndarray) -> np.ndarray:
    """DROID achieved wrist/gripper-mount pose + closedness -> 10-D EEF.

    Args:
        pose: ``(..., 6)`` from
            ``other_information.observation_gripper_pose6d`` in Euler XYZ.
            Despite the source's ``gripper`` spelling, this is the rigid
            arm-side mount pose, not the moving-finger/task TCP pose.
        closedness: ``(..., 1)`` copied from ``state[..., 6:7]``; DROID raw
            convention ``0=open, 1=closed``.

    The explicit two-input helper prevents the old TCP xyz stored in
    ``state[..., :6]`` from silently re-entering the unified arm-side frame.
    """
    pose = np.asarray(pose)
    closedness = np.asarray(closedness)
    if pose.shape[-1] != 6:
        raise ValueError(f"DROID gripper-mount pose must have width 6, got shape {pose.shape}")
    if closedness.shape[-1] != 1:
        raise ValueError(f"DROID closedness must have width 1, got shape {closedness.shape}")
    if pose.shape[:-1] != closedness.shape[:-1]:
        raise ValueError(
            f"DROID pose and closedness leading shapes must match, got {pose.shape} and {closedness.shape}"
        )
    return droid_euler7_to_arm10(np.concatenate([pose, closedness], axis=-1))


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
# Legacy DROID conversion: state assembled from two separate parquet columns
# ---------------------------------------------------------------------------


def droid_state_to_arm10(cartesian: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Legacy two-column DROID state assembly.

    The current re-converted bucket assembles
    ``other_information.observation_gripper_pose6d`` with ``state[6]`` through
    :func:`droid_pose6_closedness_to_arm10`; this helper remains for old
    conversion tools.

    Args:
        cartesian: ``(..., 6)`` from ``observation.state.cartesian_position``
            (Euler XYZ representation: ``[x,y,z,roll,pitch,yaw]``).
        gripper:   ``(..., 1)`` from ``observation.state.gripper_position``;
            DROID closedness in ``[0, 1]`` (``0=open``, ``1=closed``).

    Returns: ``(..., 10)`` EEF tensor.
    """
    pos = cartesian[..., 0:3]
    euler = cartesian[..., 3:6]
    rot6d = euler_xyz_to_rot6d(euler)
    openness = 1.0 - gripper
    return np.concatenate([pos, rot6d, openness], axis=-1).astype(np.float32)


__all__ = [
    "ARM10_DIM",
    "bcz_state_to_arm10",
    "droid_euler7_to_arm10",
    "droid_pose6_closedness_to_arm10",
    "euler7_action_to_arm10",
    "fractal_state_to_arm10",
    "droid_state_to_arm10",
]
