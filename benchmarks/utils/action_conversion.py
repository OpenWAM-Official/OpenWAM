"""Action-space conversions between OpenWAM EEF output and downstream robot APIs.

The OpenWAM policy server returns 20-dim EEF actions when trained with
``action_mode=eef``::

    [l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]

Many robot benchmarks (RoboTwin, SimplerEnv, etc.) consume 16-dim end-effector
actions with quaternion rotations::

    [l_xyz(3), l_quat_xyzw(4), l_grip(1), r_xyz(3), r_quat_xyzw(4), r_grip(1)]

These helpers are pure numpy and have no dependency on the ``openwam`` package,
so they can run in any benchmark client's Python environment.
"""

import numpy as np


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion(s) to 6D rotation, matching RoboTwinDataset."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must end with dimension 4, got shape {q.shape}")

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    x, y, z, w = np.moveaxis(q, -1, 0)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    col1 = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy),
        ],
        axis=-1,
    )
    col2 = np.stack(
        [
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx),
        ],
        axis=-1,
    )
    return np.concatenate([col1, col2], axis=-1).astype(np.float32)


def rot6d_to_quat_xyzw(r6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (first two columns of R) to xyzw quaternion.

    Uses Gram-Schmidt orthonormalization to recover the rotation matrix,
    then Shepperd's method to extract a unit quaternion.
    """
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    b2 = a2 - float(np.dot(b1, a2)) * b1
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-8)
    b3 = np.cross(b1, b2)
    # Columns of the rotation matrix: R[:, i] = bi
    R = np.stack([b1, b2, b3], axis=1)  # (3, 3)

    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def eef20d_to_ee16d(action: np.ndarray) -> np.ndarray:
    """Convert 20D EEF action (xyz+rot6d+grip)×2 to 16D ee action (xyz+quat+grip)×2.

    OpenWAM EEF (20D): ``[l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]``
    RoboTwin ee (16D): ``[l_xyz(3), l_quat_xyzw(4), l_grip(1), r_xyz(3), r_quat_xyzw(4), r_grip(1)]``
    """
    l_xyz, l_r6d, l_grip = action[0:3], action[3:9], action[9:10]
    r_xyz, r_r6d, r_grip = action[10:13], action[13:19], action[19:20]
    l_quat = rot6d_to_quat_xyzw(l_r6d)
    r_quat = rot6d_to_quat_xyzw(r_r6d)
    return np.concatenate([l_xyz, l_quat, l_grip, r_xyz, r_quat, r_grip]).astype(np.float32)


def robotwin_endpose_to_eef20d(
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    left_gripper: np.ndarray | float,
    right_gripper: np.ndarray | float,
) -> np.ndarray:
    """Assemble 20D OpenWAM EEF proprio from RoboTwin endpose fields.

    This mirrors ``RoboTwinDataset._read_eef_actions``:
    ``[left_xyz, left_rot6d, left_grip, right_xyz, right_rot6d, right_grip]``.
    RoboTwin endpose quaternions are xyzw.
    """
    left_ep = np.asarray(left_endpose, dtype=np.float32).reshape(-1)
    right_ep = np.asarray(right_endpose, dtype=np.float32).reshape(-1)
    if left_ep.shape[0] != 7 or right_ep.shape[0] != 7:
        raise ValueError(
            f"RoboTwin endpose fields must be 7D xyz+quat_xyzw; got left={left_ep.shape}, right={right_ep.shape}"
        )

    left_grip = np.asarray(left_gripper, dtype=np.float32).reshape(-1)
    right_grip = np.asarray(right_gripper, dtype=np.float32).reshape(-1)
    if left_grip.size < 1 or right_grip.size < 1:
        raise ValueError("RoboTwin gripper fields must contain at least one scalar value.")

    left = np.concatenate([left_ep[:3], quat_xyzw_to_rot6d(left_ep[3:]), left_grip[:1]], axis=-1)
    right = np.concatenate([right_ep[:3], quat_xyzw_to_rot6d(right_ep[3:]), right_grip[:1]], axis=-1)
    return np.concatenate([left, right], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOR-1K / R1Pro (RAW-27 ↔ OmniGibson controllers)
#
# The OpenWAM policy server serves a BEHAVIOR checkpoint trained with
# ``action_mode=unified`` (== robocoin.yaml layout). The model emits the unified
# 80-D action, but the deploy server's ``_UnifyAwareNormalizer`` (PR #17) gathers
# it back to the reader's RAW-27 layout and unnormalizes there — so the server
# RETURNS and EXPECTS the RAW-27 vector (physical units)::
#
#     [ L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6, R_grip1, base3, trunk4 ]
#
# The OmniGibson R1Pro robot consumes a flat per-controller action vector in the
# robot's ``_raw_controller_order`` = [base, trunk, arm_left, gripper_left,
# arm_right, gripper_right]. With the submission controller config
# (``benchmarks/behavior/configs/r1pro.yaml``) the arms are
# ``InverseKinematicsController, mode=absolute_pose`` (6-D each: base-frame xyz +
# absolute axis-angle), so the executed vector is 21-D::
#
#     [ base(3), trunk(4), armL_pos3+aa3 (6), gripL(1), armR_pos3+aa3 (6), gripR(1) ]
#
# Only the ARMS need a representation change (rot6d → axis-angle); base / trunk /
# grippers are the model's own native recorded commands and pass straight
# through (the controllers keep ``command_input_limits: default`` == the demos),
# clipped to the [-1, 1] normalized input range as a guard.
# ─────────────────────────────────────────────────────────────────────────────

# RAW-27 layout (the reader's pre-scatter vector == what the deploy server
# returns/expects after the _UnifyAwareNormalizer gather).
_R_L_POS = slice(0, 3)
_R_L_ROT6D = slice(3, 9)
_R_L_GRIP = 9
_R_R_POS = slice(10, 13)
_R_R_ROT6D = slice(13, 19)
_R_R_GRIP = 19
_R_BASE = slice(20, 23)  # [vx, vy, vyaw] base-frame velocity
_R_TRUNK = slice(23, 27)  # 4 absolute torso joint commands
R1PRO_RAW_DIM = 27
# Executed R1Pro vector width with IK absolute_pose arms.
R1PRO_IK_ACTION_DIM = 21


def quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3-D axis-angle (rotation vector ``axis * angle``).

    Matches OmniGibson / robosuite ``quat2axisangle``: the result, when fed back
    through ``axisangle2quat`` (what ``InverseKinematicsController`` does with
    ``command[3:6]``), reproduces the same orientation. The hemisphere is
    normalized (``w >= 0``) so the returned vector is the minimal rotation
    (``|angle| <= pi``); both hemispheres map to the same physical rotation.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quat must be 4-D xyzw, got shape {q.shape}")
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / n
    if q[3] < 0.0:  # shortest-path hemisphere: angle in [0, pi]
        q = -q
    v = q[:3]
    vn = np.linalg.norm(v)
    if vn < 1e-8:  # identity rotation
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vn, q[3])
    return ((v / vn) * angle).astype(np.float32)


def rot6d_to_axis_angle(r6d: np.ndarray) -> np.ndarray:
    """Convert a 6-D rotation (first two rotation-matrix columns) to axis-angle.

    Composes the existing ``rot6d_to_quat_xyzw`` (Gram-Schmidt) with
    ``quat_xyzw_to_axis_angle`` so the IK ``absolute_pose`` orientation command
    is bit-consistent with how the dataloader encodes orientation (rot6d).
    """
    return quat_xyzw_to_axis_angle(rot6d_to_quat_xyzw(np.asarray(r6d, dtype=np.float64).reshape(-1)))


def raw27_to_r1pro_action(action: np.ndarray, *, clip_passthrough: bool = True) -> np.ndarray:
    """Convert a denormalized RAW-27 action → the 21-D R1Pro IK vector.

    Input is the reader's raw layout (== what the deploy server returns after the
    _UnifyAwareNormalizer gather): ``[L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6,
    R_grip1, base3, trunk4]``. Output (== ``_raw_controller_order`` with IK
    absolute_pose arms)::

        [ base(3), trunk(4), armL_xyz(3)+aa(3), gripL(1), armR_xyz(3)+aa(3), gripR(1) ]

    Arms: base-frame xyz pass through (metric), rot6d → absolute axis-angle.
    base / trunk / grippers: the model's native normalized commands, optionally
    clipped to ``[-1, 1]`` (the controllers' ``command_input_limits: default``).
    Arm pose is NEVER clipped (IK absolute_pose uses ``command_*_limits: null``,
    i.e. raw metric pose + axis-angle).
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != R1PRO_RAW_DIM:
        raise ValueError(f"expected raw action of width {R1PRO_RAW_DIM}, got {a.shape[0]}")

    arm_left = np.concatenate([a[_R_L_POS], rot6d_to_axis_angle(a[_R_L_ROT6D])]).astype(np.float32)
    arm_right = np.concatenate([a[_R_R_POS], rot6d_to_axis_angle(a[_R_R_ROT6D])]).astype(np.float32)

    base = a[_R_BASE]
    trunk = a[_R_TRUNK]
    grip_l = a[_R_L_GRIP : _R_L_GRIP + 1]
    grip_r = a[_R_R_GRIP : _R_R_GRIP + 1]
    if clip_passthrough:
        base = np.clip(base, -1.0, 1.0)
        trunk = np.clip(trunk, -1.0, 1.0)
        grip_l = np.clip(grip_l, -1.0, 1.0)
        grip_r = np.clip(grip_r, -1.0, 1.0)

    return np.concatenate([base, trunk, arm_left, grip_l, arm_right, grip_r]).astype(np.float32)


# Offsets into the R1Pro 256-D proprio vector (OmniGibson ``robot_r1::proprio``,
# == the dataset's ``observation.state`` when both come from the same proprio
# config). The EEF blocks are the dataloader's reverse-engineered + unit-norm-
# validated offsets (openwam/dataloader/behavior.py) — VERIFIED on real data.
# base_vel / trunk / gripper are sourced from the ACTION command at train time
# (not the state), so their state-side offsets must be CONFIRMED on a sim box;
# ``None`` → that raw block is zero-filled (the model leans on EEF + vision).
R1PRO_PROPRIO_OFFSETS = {
    "l_pos": slice(186, 189),  # verified
    "l_quat": slice(189, 193),  # verified (xyzw, unit-norm checked by the reader)
    "r_pos": slice(225, 228),  # verified
    "r_quat": slice(228, 232),  # verified
    "trunk": slice(236, 240),  # best-estimate (PROPRIOCEPTION_INDICES) — confirm on sim
    "base_vel": None,  # UNCONFIRMED — set the slice after sim validation
    "l_grip": None,  # UNCONFIRMED — gripper width/qpos offset
    "r_grip": None,  # UNCONFIRMED
}


def r1pro_proprio_to_raw27(proprio: np.ndarray, offsets: dict | None = None) -> np.ndarray:
    """Assemble the RAW-27 proprio (physical units) from R1Pro 256-D proprio.

    Builds the reader's raw layout ``[L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6,
    R_grip1, base3, trunk4]`` (EEF pose = xyz + rot6d from the state quaternion).
    The OpenWAM server's _UnifyAwareNormalizer normalizes this raw proprio and
    scatters it into the unified space the model wants — so the bridge sends RAW,
    NOT unified. Blocks whose ``offsets`` entry is ``None`` stay zero.

    Returns an un-normalized ``(27,)`` float32 vector (the server normalizes it).
    """
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    off = dict(R1PRO_PROPRIO_OFFSETS if offsets is None else offsets)

    def _blk(name: str, width: int) -> np.ndarray:
        sl = off.get(name)
        if sl is None:
            return np.zeros(width, dtype=np.float32)
        v = p[sl].astype(np.float32)
        if v.shape[0] != width:
            raise ValueError(f"proprio offset {name!r}={sl} yielded width {v.shape[0]}, expected {width}")
        return v

    return np.concatenate(
        [
            _blk("l_pos", 3),
            quat_xyzw_to_rot6d(_blk("l_quat", 4)),
            _blk("l_grip", 1),
            _blk("r_pos", 3),
            quat_xyzw_to_rot6d(_blk("r_quat", 4)),
            _blk("r_grip", 1),
            _blk("base_vel", 3),
            _blk("trunk", 4),
        ]
    ).astype(np.float32)
