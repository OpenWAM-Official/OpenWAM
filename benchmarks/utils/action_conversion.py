"""Action-space conversions between OpenWAM EEF output and downstream robot APIs.

The OpenWAM policy server returns 20-dim EEF actions when trained with
``action_mode=eef``::

    [l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]

Many robot benchmarks (RoboTwin, SimplerEnv, etc.) consume 16-dim end-effector
actions with quaternion rotations::

    [l_xyz(3), l_quat_xyzw(4), l_grip(1), r_xyz(3), r_quat_xyzw(4), r_grip(1)]

These helpers are pure numpy and have no dependency on the ``openwam`` package,
so they can run in any benchmark client's Python environment.

Compact RoboCasa365 uses a native single-arm 10-D EEF pose directly; its
dedicated helpers intentionally avoid padding through the legacy dual-arm 20-D
layout.
"""

# Benchmark client envs can be as old as Python 3.8 (ordinary LIBERO):
# keep PEP 604 unions lazy.
from __future__ import annotations

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


def _rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D rotation (first two columns) -> 3x3 rotation matrix (Gram-Schmidt)."""
    a1, a2 = np.asarray(r6d[:3], np.float64), np.asarray(r6d[3:6], np.float64)
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    b2 = a2 - float(np.dot(b1, a2)) * b1
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns = b1,b2,b3


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle (rotation vector), pure numpy."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3, np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    return (axis * angle).astype(np.float32)


def binarize_robocasa_action12(action: np.ndarray) -> np.ndarray:
    """Project native RoboCasa gripper/mode commands to their legal two-point values.

    RoboCasa's gym wrapper interprets both native ``gripper_close`` (dim 6) and
    ``control_mode`` (dim 11) with the same inclusive threshold: values below
    ``0.5`` map to ``-1`` and values at or above ``0.5`` map to ``+1``.  Keep
    that policy at the evaluation boundary instead of encoding it in training
    dataset configuration.
    """
    out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if out.shape != (12,):
        raise ValueError(f"expected a 12-D RoboCasa action, got {out.shape}")
    out[6] = 1.0 if float(out[6]) >= 0.5 else -1.0
    out[11] = 1.0 if float(out[11]) >= 0.5 else -1.0
    return out


def eef10_to_robocasa12d(
    action: np.ndarray,
    proprio_eef_pos: np.ndarray,
    proprio_eef_rot6d: np.ndarray,
    *,
    pos_scale: float,
    rot_scale: float,
    base_motion: np.ndarray | None = None,
    control_mode: float = -1.0,
    clip: bool = True,
) -> np.ndarray:
    """Bridge a single-arm 10-D **full** EEF pose to RoboCasa's 12-D OSC-delta action.

    RoboCasa365's ``RoboCasaGymEnv`` consumes a 12-D robosuite OSC_POSE + mobile-base
    action; the compact RoboCasa365 model instead predicts
    ``[pos3, rot6d6, grip1]``: a **full pose** (not a per-step delta) expressed in the robot base
    frame (``robot0_base_to_eef_*``). The OSC controller takes normalized delta commands, so this
    conversion uses the current achieved EEF observation and the source controller scales.

    Output is the flat 12-D in the SERVER/``slice_action`` order (NOT modality.json
    order)::

        [eef_pos_cmd(3), eef_rot_cmd(3), gripper(1), base_motion(4), control_mode(1)]

    Args:
        action: 10-D EEF action ``[pos3, rot6d6, grip1]``.
        proprio_eef_pos: (3,) current base-frame EEF position (from ``state.end_effector_position_relative``
            = ``robot0_base_to_eef_pos``).
        proprio_eef_rot6d: (6,) current EEF rotation as rot6d (quat->rot6d of ``state.end_effector_rotation_relative``).
        pos_scale: robosuite OSC position ``output_max`` (metres mapped to action 1.0). **REQUIRED, env-specific** —
            read it from the eval env's OSC_POSE controller config; a wrong value drives wrong-magnitude motions.
        rot_scale: robosuite OSC rotation ``output_max`` (radians mapped to action 1.0). Same caveat as ``pos_scale``.
        base_motion: (4,) base command ``[x/y/yaw velocity, torso]``; defaults to zeros.
        control_mode: scalar native mode command; defaults to -1.0.
        clip: clip the scaled eef commands to ``[-1, 1]`` (OSC action bounds).

    Gripper: model dim ``act[9]`` uses the fixed compact-representation open scale
    ``[-1, +1]`` (**-1=close, +1=open**).  Convert it to RoboCasa's native
    close scale with the sign-reversed official threshold: ``act[9] <= -0.5``
    becomes close ``+1`` and larger values become open ``-1``. No width
    binarization — the model predicts the command directly, so there is no actuation-lag delay
    (unlike deriving open/close from the achieved finger-separation width).

    Mode: ``control_mode`` keeps RoboCasa's native sign and is projected with
    the official inclusive threshold: ``>=0.5`` becomes ``+1`` and lower
    values become ``-1``.

    Dataset/controller contract: the recorded action is a normalized OSC delta. RoboCasa365's
    controller metadata declares ``output_max=[0.05]*3+[0.5]*3``; pass those values when evaluating
    the converted dataset. Smaller measured one-step achieved motion is controller dynamics, not a
    replacement command scale.
    """
    act = np.asarray(action, dtype=np.float64).reshape(-1)
    if act.shape[0] != 10:
        raise ValueError(f"expected a 10-D EEF action, got {act.shape[0]}")
    cur_pos = np.asarray(proprio_eef_pos, np.float64).reshape(-1)
    if cur_pos.shape[0] != 3:
        raise ValueError(f"proprio_eef_pos must be 3-D, got {cur_pos.shape[0]}")
    if not (float(pos_scale) > 0.0 and float(rot_scale) > 0.0):
        raise ValueError(
            f"pos_scale and rot_scale must be > 0 (got pos={pos_scale}, rot={rot_scale}); a "
            "non-positive scale would silently mask a misconfigured OSC controller (clamping it "
            "to ~0 emits huge/garbage deltas). Set them from the eval env's OSC_POSE output_max."
        )
    tgt_pos, tgt_r6d = act[0:3], act[3:9]  # gripper (act[9]) handled below

    # Position: absolute target -> scaled OSC delta.
    pos_cmd = (tgt_pos - cur_pos) / float(pos_scale)

    # Rotation: relative rotation R_target @ R_current^-1 -> axis-angle -> scaled.
    R_t = _rot6d_to_matrix(tgt_r6d)
    R_c = _rot6d_to_matrix(np.asarray(proprio_eef_rot6d, np.float64).reshape(-1))
    rot_cmd = _matrix_to_axis_angle(R_t @ R_c.T).astype(np.float64) / float(rot_scale)

    if clip:
        pos_cmd = np.clip(pos_cmd, -1.0, 1.0)
        rot_cmd = np.clip(rot_cmd, -1.0, 1.0)

    # Policy gripper is -1=close,+1=open; RoboCasa consumes +1=close,-1=open.
    # The inclusive boundary exactly mirrors the official native >=0.5 rule
    # after reversing the sign.
    gripper_cmd = 1.0 if float(act[9]) <= -0.5 else -1.0

    base = np.zeros(4, np.float64) if base_motion is None else np.asarray(base_motion, np.float64).reshape(-1)
    if base.shape[0] != 4:
        raise ValueError(f"base_motion must be 4-D, got {base.shape[0]}")
    mode_cmd = 1.0 if float(control_mode) >= 0.5 else -1.0
    # SERVER / slice_action order: eef_pos, eef_rot, grip, base_motion, control_mode.
    return np.concatenate([pos_cmd, rot_cmd, [gripper_cmd], base, [mode_cmd]]).astype(np.float32)


def eef20d_to_robocasa12d(
    action: np.ndarray,
    proprio_eef_pos: np.ndarray,
    proprio_eef_rot6d: np.ndarray,
    *,
    pos_scale: float,
    rot_scale: float,
    base_motion: np.ndarray | None = None,
    control_mode: float = -1.0,
    clip: bool = True,
) -> np.ndarray:
    """Legacy dual-arm wrapper; RoboCasa uses only the first arm's 10 dimensions."""
    act = np.asarray(action).reshape(-1)
    if act.shape[0] != 20:
        raise ValueError(f"expected a 20-D EEF action, got {act.shape[0]}")
    return eef10_to_robocasa12d(
        act[:10],
        proprio_eef_pos,
        proprio_eef_rot6d,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        base_motion=base_motion,
        control_mode=control_mode,
        clip=clip,
    )


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


# Gripper render (MUST stay in lockstep with openwam.dataloader.robocasa365._gripper_width_to_cmd):
# achieved finger-separation width → [-1,+1] PRETRAIN open-scale (closed 0 → -1, open width → +1).
_RC365_GRIPPER_WIDTH_OPEN = 0.1


def rc365_gripper_width_to_cmd(width) -> float:
    """Achieved finger-separation width → [-1,+1] gripper OPEN-SCALE (closed→-1, open→+1) — the
    pretrain convention (-1=close, +1=open). Bit-identical to the dataloader's
    ``_gripper_width_to_cmd`` so the proprio gripper the client sends matches training."""
    return float(np.clip(2.0 * float(width) / _RC365_GRIPPER_WIDTH_OPEN - 1.0, -1.0, 1.0))


def robocasa_state_to_eef10(
    eef_pos_rel: np.ndarray,
    eef_rot_rel_quat_xyzw: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Assemble compact **10-D EEF proprio** from a RoboCasa365 observation.

    Bit-identical to the compact dataloader's EEF conversion. The server normalizes this raw value.

        arm10 = [eef_pos_rel(3), rot6d(eef_rot_rel quat xyzw, 6), gripper(1)]
        gripper = rc365_gripper_width_to_cmd(gripper_qpos[0] - gripper_qpos[1])   ([-1,+1] command space)
    """
    pos = np.asarray(eef_pos_rel, np.float32).reshape(-1)
    quat = np.asarray(eef_rot_rel_quat_xyzw, np.float32).reshape(-1)
    qpos = np.asarray(gripper_qpos, np.float32).reshape(-1)
    if pos.shape[0] != 3 or quat.shape[0] != 4 or qpos.shape[0] != 2:
        raise ValueError(
            f"robocasa proprio dims: eef_pos_rel must be 3 (got {pos.shape[0]}), "
            f"eef_rot_rel quat 4 (got {quat.shape[0]}), gripper_qpos 2 (got {qpos.shape[0]})"
        )
    grip = np.array([rc365_gripper_width_to_cmd(qpos[0] - qpos[1])], np.float32)
    return np.concatenate([pos, quat_xyzw_to_rot6d(quat), grip], axis=-1).astype(np.float32)


def robocasa_state_to_eef20d(
    eef_pos_rel: np.ndarray,
    eef_rot_rel_quat_xyzw: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Legacy wrapper that zero-pads compact RoboCasa EEF state to dual-arm 20-D."""
    out = np.zeros(20, np.float32)
    out[:10] = robocasa_state_to_eef10(eef_pos_rel, eef_rot_rel_quat_xyzw, gripper_qpos)
    return out


def base_velocity_body(prev_base_pose: np.ndarray, cur_base_pose: np.ndarray) -> np.ndarray:
    """Body-frame base velocity from two consecutive base poses (finite difference), per-frame.

    The low-level building block for ``base_velocity_cmd`` (which applies the A′ rescale on top).
    Bit-identical to the dataloader's ``_base_velocity_body`` (``openwam.dataloader.robocasa365``).
    Each pose is ``base_position(3, world) + base_rotation(4, world quat xyzw)``.

    Returns ``(3,)`` = ``[vx, vy, vyaw]`` in the robot's body frame at ``cur`` (per-step displacement,
    m/frame + rad/frame). SE(2): z + roll/pitch are ignored (ground base); Δyaw is wrapped to (-pi, pi].
    """
    prev = np.asarray(prev_base_pose, np.float64).reshape(-1)
    cur = np.asarray(cur_base_pose, np.float64).reshape(-1)
    if prev.shape[0] < 7 or cur.shape[0] < 7:
        raise ValueError(f"base pose must be >=7D (pos3+quat4); got prev={prev.shape}, cur={cur.shape}")

    def _yaw(q):  # yaw about world +z from a quaternion (x, y, z, w)
        x, y, z, w = (float(v) for v in q[:4])
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    d = cur[0:2] - prev[0:2]  # world planar displacement
    yaw_cur, yaw_prev = _yaw(cur[3:7]), _yaw(prev[3:7])
    c, s = np.cos(yaw_cur), np.sin(yaw_cur)
    vx = c * d[0] + s * d[1]  # R(-yaw_cur) @ d -> body frame
    vy = -s * d[0] + c * d[1]
    d_yaw = np.arctan2(np.sin(yaw_cur - yaw_prev), np.cos(yaw_cur - yaw_prev))  # wrapped Δyaw
    return np.array([vx, vy, d_yaw], np.float32)


# A′ base-velocity rescale (MUST stay in lockstep with openwam.dataloader.robocasa365):
# _BASE_VEL_PHYS_MAX = per-axis base max speed at command saturation, DATASET_FPS = v3 rate.
_RC365_BASE_VEL_PHYS_MAX = np.array([0.75, 0.88, 1.33], dtype=np.float32)
_RC365_FPS = 20


def base_velocity_cmd(prev_base_pose: np.ndarray, cur_base_pose: np.ndarray, fps: int = _RC365_FPS) -> np.ndarray:
    """Body-frame base velocity finite-diff rescaled into the action's [-1, 1] command space (A′).

    Bit-identical to the dataloader's ``base_velocity_cmd`` (``openwam.dataloader.robocasa365``): the
    proprio base velocity the client sends for a mobile ckpt must be derived — AND rescaled — the SAME
    way it was at train time (``× fps / _BASE_VEL_PHYS_MAX``), so the achieved proprio velocity lands
    in the same space as the recorded action base command and shares its stats. No train/eval mismatch.
    Each pose = ``base_position(3, world) + base_rotation(4, world quat xyzw)``. Sent RAW; the server
    normalizes with the combined ``eef_base`` stats block."""
    return (base_velocity_body(prev_base_pose, cur_base_pose) * float(fps) / _RC365_BASE_VEL_PHYS_MAX).astype(
        np.float32
    )


def base_pose_planar5(base_pose: np.ndarray) -> np.ndarray:
    """``(5,)`` planar world base pose ``[x, y, sin(yaw), cos(yaw), 0]`` from a 7-D base pose.

    Bit-identical to the dataloader's ``base_proprio="global_pose"`` proprio (the quantity a
    global-pose ckpt trains on): sin/cos instead of raw yaw = no ±π seam (the planar reduction of a
    rot6d yaw rotation). ``base_pose`` = ``base_position(3, world) + base_rotation(4, world quat
    xyzw)``. Sent RAW; the server normalizes with the ``eef_base_pose_proprio`` stats block."""
    p = np.asarray(base_pose, np.float64).reshape(-1)
    if p.shape[0] < 7:
        raise ValueError(f"base pose must be >=7D (pos3+quat4); got {p.shape}")
    qx, qy, qz, qw = (float(v) for v in p[3:7])
    yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return np.array([p[0], p[1], np.sin(yaw), np.cos(yaw), 0.0], np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers (rot6d / quaternion / axis-angle), shared by eval bridges.
# ─────────────────────────────────────────────────────────────────────────────


def quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3-D axis-angle (rotation vector ``axis * angle``).

    Matches the robosuite ``quat2axisangle`` convention: feeding the result
    back through ``axisangle2quat`` reproduces the same orientation. The
    hemisphere is
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
    ``quat_xyzw_to_axis_angle``, so the axis-angle output is bit-consistent
    with how the dataloaders encode orientation (rot6d).
    """
    return quat_xyzw_to_axis_angle(rot6d_to_quat_xyzw(np.asarray(r6d, dtype=np.float64).reshape(-1)))


# --------------------------------------------------------------------------- #
# LIBERO (single-arm Panda, 7-D OSC delta)                        #
# --------------------------------------------------------------------------- #
# Shared rendering helpers for the canonical native-action LIBERO reader and
# benchmark client. The checkpoint predicts native EEF10 action commands and
# the benchmark-specific rot6d-to-rotvec bridge lives in
# benchmarks/libero/openwam2libero_interface.py.
#
# The trained gripper channel (EEF10 dim 9) is an OPEN-SCALE: -1 = closed,
# +1 = open (the pretraining-mixture direction). LIBERO's own action[6] runs the
# other way (+1 = close), so this bridge negates on the way out. Keep in lockstep
# with openwam.dataloader.libero.GRIPPER_CONVENTION.

LIBERO_EEF10_DIM = 10
LIBERO_ACTION7_DIM = 7
# Panda finger separation at fully open (m). MUST stay in lockstep with
# openwam.dataloader.libero.LIBERO_GRIPPER_WIDTH_OPEN.
LIBERO_GRIPPER_WIDTH_OPEN = 0.08


def libero_gripper_qpos_to_cmd(width) -> float:
    """Achieved finger-separation width -> [-1, +1] OPEN-SCALE (+1 = open).

    Bit-identical to the dataloader's ``gripper_qpos_to_cmd`` so the proprio
    gripper the client sends matches training. Note the trained channel runs
    OPPOSITE to LIBERO's own ``action[6]`` command (+1 = close); converting back
    is :func:`libero_open_scale_to_gripper_cmd`.
    """
    return float(np.clip(2.0 * float(width) / LIBERO_GRIPPER_WIDTH_OPEN - 1.0, -1.0, 1.0))


def libero_open_scale_to_gripper_cmd(value) -> float:
    """Trained open-scale gripper (+1 = open) -> LIBERO env command (+1 = close).

    The canonical dataset stores ``-1 = closed, +1 = open`` while robosuite's
    environment command uses the opposite sign, so this is a fixed negation.
    The env's gripper stays continuous in [-1, +1].
    """
    return float(np.clip(-float(value), -1.0, 1.0))


def libero_obs_to_eef10(
    eef_pos: np.ndarray,
    eef_quat_xyzw: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Assemble the RAW 10-D EEF proprio from a live LIBERO obs (unnormalized).

    Matches the EEF10 state written by the canonical dataset converter on the
    same physical state. Converting the live quaternion directly to rot6d avoids
    an unnecessary axis-angle round trip.

        eef10 = [robot0_eef_pos(3), rot6d(robot0_eef_quat xyzw, 6),
                 gripper_cmd(robot0_gripper_qpos[0] - [1], 1)]
    """
    pos = np.asarray(eef_pos, np.float32).reshape(-1)
    quat = np.asarray(eef_quat_xyzw, np.float32).reshape(-1)
    qpos = np.asarray(gripper_qpos, np.float32).reshape(-1)
    if pos.shape[0] != 3 or quat.shape[0] != 4 or qpos.shape[0] != 2:
        raise ValueError(
            f"LIBERO proprio dims: eef_pos must be 3 (got {pos.shape[0]}), "
            f"eef_quat 4 (got {quat.shape[0]}), gripper_qpos 2 (got {qpos.shape[0]})"
        )
    grip = np.array([libero_gripper_qpos_to_cmd(qpos[0] - qpos[1])], np.float32)
    return np.concatenate([pos, quat_xyzw_to_rot6d(quat[None])[0], grip], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# EBench (GenManip lift2/R5a dual-arm mobile manipulator)                      #
# --------------------------------------------------------------------------- #
# Mirrors of the trainer's rendering in openwam/dataloader/ebench.py — the
# eval and training ends of the same contract. A regression test
# (tests/benchmarks/test_ebench_bridge.py) pins each pair together byte-for-
# byte; change them in lockstep.

EBENCH_RAW_DIM = 23
# GenManip lift2 gripper: 0.0 closed .. 0.044 open per finger (both fingers of
# a hand are commanded identically).
EBENCH_GRIPPER_OPEN = 0.044


def ebench_wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap radian angle(s) to [-pi, pi). Mirror of ebench.wrap_angle_rad."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def ebench_render_state_base(cur_base, prev_base) -> np.ndarray:
    """Render measured ``state.base`` ``[x_m, y_m, yaw_RAD]`` into the
    ``action.base_delta`` command space: measured per-step displacement (yaw
    wrapped, rad→deg); ``prev_base=None`` (episode start) → zeros. Mirror of
    ebench.render_ebench_state_base.
    """
    cur = np.asarray(cur_base, dtype=np.float64).reshape(3)
    if prev_base is None:
        return np.zeros(3, dtype=np.float32)
    prev = np.asarray(prev_base, dtype=np.float64).reshape(3)
    delta = cur - prev
    dyaw = float(ebench_wrap_angle_rad(delta[2]))
    return np.array([delta[0], delta[1], np.degrees(dyaw)], dtype=np.float32)


def ebench_quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion(s) → rot6d. Mirror of ebench._quat_wxyz_to_rot6d.

    Inlines ``openwam.dataloader.utils.eef.quat_xyzw_to_rot6d`` (float32, no
    re-normalization — GenManip quats are unit by construction and the reader
    checks a sample at init) rather than this module's normalizing
    ``quat_xyzw_to_rot6d``, so bridge proprio is byte-identical to training.
    """
    q = np.asarray(quat_wxyz, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must be 4-D wxyz, got shape {q.shape}")
    leading = q.shape[:-1]
    flat = q.reshape(-1, 4)
    flat_xyzw = np.concatenate([flat[:, 1:4], flat[:, 0:1]], axis=-1)
    x, y, z, w = flat_xyzw[:, 0], flat_xyzw[:, 1], flat_xyzw[:, 2], flat_xyzw[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    c0 = np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)], axis=-1)
    c1 = np.stack([2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)], axis=-1)
    r6d = np.concatenate([c0, c1], axis=-1).astype(flat_xyzw.dtype)
    return r6d.reshape(*leading, 6).astype(np.float32)


def ebench_obs_to_raw23(state_ee_pose, state_gripper, rendered_base) -> np.ndarray:
    """EBench eval obs (measured state) → RAW-23 proprio in the reader layout.

    ``state_ee_pose``: the obs ``state.ee_pose`` nested pairs
    ``[[L_pos3, L_quat_wxyz4], [R_pos3, R_quat_wxyz4]]`` (or an already-flat
    14-vector). ``state_gripper``: 4 finger positions ``[L,L,R,R]``.
    ``rendered_base``: the 3-vector from :func:`ebench_render_state_base`.

    Output layout (== ebench._ee_pose_gripper_base_to_raw23)::

        [L_xyz3, L_rot6d6, L_grip1, R_xyz3, R_rot6d6, R_grip1, base3]

    Sent RAW — the OpenWAM server normalizes with the checkpoint stats and
    scatters into the unified 80-D space.
    """

    def _leaves(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                yield from _leaves(item)
        else:
            yield np.asarray(node, dtype=np.float32).reshape(-1)

    flat = np.concatenate(list(_leaves(state_ee_pose)))
    if flat.shape[0] != 14:
        raise ValueError(f"state.ee_pose must flatten to 14 values, got {flat.shape[0]}")
    grip = np.asarray(state_gripper, dtype=np.float32).reshape(-1)
    if grip.shape[0] != 4:
        raise ValueError(f"state.gripper must have 4 values, got {grip.shape[0]}")
    base = np.asarray(rendered_base, dtype=np.float32).reshape(-1)
    if base.shape[0] != 3:
        raise ValueError(f"rendered base must have 3 values, got {base.shape[0]}")
    return np.concatenate(
        [
            flat[0:3],
            ebench_quat_wxyz_to_rot6d(flat[3:7]),
            grip[0:2].mean(keepdims=True),
            flat[7:10],
            ebench_quat_wxyz_to_rot6d(flat[10:14]),
            grip[2:4].mean(keepdims=True),
            base,
        ]
    ).astype(np.float32)


def raw23_to_ebench_action(action: np.ndarray) -> dict:
    """OpenWAM RAW-23 physical action → GenManip EvalClient action dict.

    The server already inverted normalization and the 80-D unify scatter, so
    ``action`` is the raw 23-D vector in physical units. Arms go out as
    absolute ``ee_pose`` targets (GenManip runs cuRobo IK per arm in the same
    per-arm base frames that produced the training FK); the scalar gripper is
    duplicated to both fingers and clipped to the physical range; the base
    slot goes out as ``base_motion=[dx_m, dy_m, dyaw_deg]`` with
    ``base_is_rel=True`` (GenManip clips each step to ±0.015 m / ±1° — the
    same clamps the demos obeyed — and converts the degree yaw internally).

    Positions/quaternions are plain Python lists ON PURPOSE: the GenManip
    server concatenates ``position + orientation`` (list concat) before IK —
    numpy arrays would broadcast-add and crash it.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != EBENCH_RAW_DIM:
        raise ValueError(f"expected raw {EBENCH_RAW_DIM}-D EBench action, got {a.shape[0]}")

    def _arm(xyz: np.ndarray, r6d: np.ndarray, grip: float) -> tuple:
        quat_xyzw = rot6d_to_quat_xyzw(r6d.astype(np.float32))
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
        g = float(np.clip(grip, 0.0, EBENCH_GRIPPER_OPEN))
        return ([float(v) for v in xyz], quat_wxyz, [g, g])

    return {
        "action": [
            _arm(a[0:3], a[3:9], a[9]),
            _arm(a[10:13], a[13:19], a[19]),
        ],
        "control_type": "ee_pose",
        "is_rel": False,
        "base_motion": [float(v) for v in a[20:23]],
        "base_is_rel": True,
    }


# ---------------------------------------------------------------------------
# VLABench (MuJoCo / dm_control, Franka Panda single arm, absolute EE pose)
# ---------------------------------------------------------------------------
VLABENCH_EEF10_DIM = 10
# Franka finger separation per finger at fully open (m). VLABench's evaluator
# consumes a 2-finger gripper target, so an "open" command is [0.04, 0.04].
VLABENCH_GRIPPER_OPEN_WIDTH = 0.04
# The dataset's ACTION gripper column is the commanded finger width binarized by
# VLABench's converter with ``> 0.03 -> 1`` against that 0.04 open width, so
# 1 = OPEN. (The STATE column uses the opposite polarity — see
# openwam/dataloader/vlabench.py. Both are passed through verbatim.)
VLABENCH_GRIPPER_OPEN_THRESHOLD = 0.5
# Fallback robot base position used by VLABench's LeRobot converter when an
# episode config carries no explicit ``robot.position``.
VLABENCH_ROBOT_BASE_DEFAULT = (0.0, -0.4, 0.78)


def rot6d_to_euler_xyz(r6d: np.ndarray) -> np.ndarray:
    """``(6,)`` rot6d -> ``(3,)`` extrinsic Euler XYZ ``[roll, pitch, yaw]`` (rad).

    Exact inverse of the dataloader's ``euler_xyz_to_rot6d``, which builds
    ``R = Rz @ Ry @ Rx`` and keeps its first two columns. Decomposing that R::

        pitch_y = asin(-R[2, 0])
        roll_x  = atan2(R[2, 1], R[2, 2])
        yaw_z   = atan2(R[1, 0], R[0, 0])

    At gimbal lock (``|R[2,0]| -> 1``, i.e. ``cos(pitch) -> 0``) roll and yaw are
    degenerate; roll is pinned to 0 and the combined rotation folded into yaw.
    Matches ``scipy.spatial.transform.Rotation.as_euler("xyz")`` (lowercase =
    extrinsic), the convention VLABench's own converter used.
    """
    R = _rot6d_to_matrix(np.asarray(r6d, np.float64).reshape(-1))
    sy = -R[2, 0]
    cy_sq = R[0, 0] ** 2 + R[1, 0] ** 2
    if cy_sq < 1e-12:  # gimbal lock
        pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
        return np.array([0.0, pitch, np.arctan2(-R[0, 1], R[1, 1])], np.float64)
    return np.array(
        [
            np.arctan2(R[2, 1], R[2, 2]),
            np.arcsin(np.clip(sy, -1.0, 1.0)),
            np.arctan2(R[1, 0], R[0, 0]),
        ],
        np.float64,
    )


def vlabench_obs_to_eef10(ee_state: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
    """Live VLABench ``ee_state`` -> RAW 10-D EEF proprio (unnormalized).

    Mirrors the dataloader on the same physical state. VLABench's
    ``robot.get_ee_state()`` returns ``[pos(3), quat_wxyz(4), open_state(1)]`` in
    the WORLD frame; the training converter subtracted the robot base position
    and stored Euler XYZ, so this does the same:

        eef10 = [pos - robot_base (3), rot6d(quat) (6), open_state (1)]

    ``open_state`` is forwarded verbatim, including the upstream Franka polarity
    inversion (1 = closed) — the dataset's state column carries the same value
    from the same accessor, so consistency is what matters.
    """
    ee = np.asarray(ee_state, np.float64).reshape(-1)
    if ee.shape[0] < 8:
        raise ValueError(f"VLABench ee_state must be at least 8-D [pos3, quat_wxyz4, grip1], got {ee.shape[0]}")
    base = np.asarray(robot_base, np.float64).reshape(-1)
    if base.shape[0] != 3:
        raise ValueError(f"robot_base must be 3-D, got {base.shape[0]}")
    pos = ee[0:3] - base
    quat_wxyz = ee[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], np.float32)
    rot6d = quat_xyzw_to_rot6d(quat_xyzw[None])[0]
    return np.concatenate([pos.astype(np.float32), rot6d, np.float32([ee[7]])], axis=-1).astype(np.float32)


def eef10_to_vlabench_ee(
    action: np.ndarray,
    robot_base: np.ndarray,
    *,
    gripper_open_threshold: float = VLABENCH_GRIPPER_OPEN_THRESHOLD,
    gripper_open_width: float = VLABENCH_GRIPPER_OPEN_WIDTH,
) -> tuple:
    """Model EEF10 -> the ``(pos, euler, gripper_state)`` VLABench's evaluator wants.

    The evaluator runs IK on an ABSOLUTE WORLD-frame target, so the robot base
    offset the dataloader removed is added back here. The gripper is thresholded
    to VLABench's 2-finger command: open = ``[w, w]``, closed = ``[0, 0]``, with
    1 = OPEN per the dataset's action convention.

    Args:
        action: ``(10,)`` EEF10 ``[xyz3, rot6d6, grip1]``, robot-base frame,
            already denormalized to physical units by the server.
        robot_base: ``(3,)`` robot base position in the world frame
            (``env.get_robot_frame_position()``, injected as ``obs["robot_frame"]``).

    Returns:
        ``(target_pos(3,), target_euler(3,), gripper_state(2,))``, all float64
        world-frame, ready for ``agent.predict``'s ``control_mode="ee"`` contract.
    """
    act = np.asarray(action, np.float64).reshape(-1)
    if act.shape[0] != VLABENCH_EEF10_DIM:
        raise ValueError(f"expected a {VLABENCH_EEF10_DIM}-D EEF action, got {act.shape[0]}")
    base = np.asarray(robot_base, np.float64).reshape(-1)
    if base.shape[0] != 3:
        raise ValueError(f"robot_base must be 3-D, got {base.shape[0]}")
    target_pos = act[0:3] + base
    target_euler = rot6d_to_euler_xyz(act[3:9])
    is_open = float(act[9]) >= float(gripper_open_threshold)
    gripper_state = np.full(2, float(gripper_open_width)) if is_open else np.zeros(2)
    return target_pos, target_euler, gripper_state
