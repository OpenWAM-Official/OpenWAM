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
    """Bridge the model's 20-D **full** EEF pose to RoboCasa's 12-D **OSC delta** action.

    RoboCasa365's ``RoboCasaGymEnv`` consumes a 12-D robosuite OSC_POSE + mobile-base
    action; the OpenWAM model trained by ``RoboCasa365Dataset`` instead predicts a 20-D single-arm
    EEF pose (left half ``[pos3, rot6d6, grip1]``, right half 0) that is a **full pose** (not a
    per-step delta) expressed in the robot **base frame** (``robot0_base_to_eef_*``) — base-relative,
    NOT world-frame. (Loosely called "absolute" elsewhere = full-not-delta; do not read it as
    world-frame.) This is the dual of robotwin's client-side ``eef20d_to_ee16d`` — except robotwin's
    env takes full 16-D poses, whereas RoboCasa's OSC controller takes *delta* commands scaled into
    ``[-1, 1]``, so the conversion needs the current proprio (to form the delta) and the controller's
    scaling.

    Output is the flat 12-D in the SERVER/``slice_action`` order (NOT modality.json
    order)::

        [eef_pos_cmd(3), eef_rot_cmd(3), gripper(1), base_motion(4), control_mode(1)]

    Args:
        action: 20-D EEF action; only the left-arm 10 dims ``[pos3, rot6d6, grip1]`` are used.
        proprio_eef_pos: (3,) current base-frame EEF position (from ``state.end_effector_position_relative``
            = ``robot0_base_to_eef_pos``).
        proprio_eef_rot6d: (6,) current EEF rotation as rot6d (quat->rot6d of ``state.end_effector_rotation_relative``).
        pos_scale: robosuite OSC position ``output_max`` (metres mapped to action 1.0). **REQUIRED, env-specific** —
            read it from the eval env's OSC_POSE controller config; a wrong value drives wrong-magnitude motions.
        rot_scale: robosuite OSC rotation ``output_max`` (radians mapped to action 1.0). Same caveat as ``pos_scale``.
        base_motion: (4,) base command [x/y/yaw vel, torso]; defaults to zeros — the arm-only (non-mobile)
            ckpt fallback. A mobile_base ckpt passes the real base command through here.
        control_mode: scalar; defaults to -1.0 ("achieved" mode) — the arm-only fallback. A mobile_base
            ckpt passes the model's real control_mode (gym thresholds it at 0.5 → -1/+1).
        clip: clip the scaled eef commands to ``[-1, 1]`` (OSC action bounds).

    Gripper: the model dim ``act[9]`` is the PRETRAIN open-scale in ``[-1, +1]`` (**-1=close, +1=open**
    — the training reader negates RoboCasa's native ``gripper_close`` so post-train matches the
    pretrained gripper space; robotwin/BEHAVIOR/robocoin all train +1=open). Close only on a
    CONFIDENT command: ``close (1.0) iff act[9] < -0.5``, else open — an uncertain / neutral output
    (~0, the flow-matching prior mean) defaults to open, avoiding spurious grasps. No width
    binarization — the model predicts the command directly, so there is no actuation-lag delay
    (unlike deriving open/close from the achieved finger-separation width). Downstream, the robocasa
    gym wrapper re-binarizes at 0.5 (>=0.5 -> close, <0.5 -> open), so the emitted {1.0, 0.0} are
    real close/open commands.

    ENV CONTRACT (MEASURED on the real robocasa/OpenDrawer env, PandaOmron / default_pandaomron.json):
    the eef action convention is **delta** (zero action -> no EEF motion; constant action -> constant
    per-step displacement), matching the (target-current)/scale here. Steady per-step motion per
    action 1.0: ~0.0126 m (pos) / ~0.102 rad (rot) -> use as pos_scale/rot_scale. control_mode -1 +
    base 0 hold the fixed base. (Probed via env.step with known actions; see e2e plan.)
    """
    act = np.asarray(action, dtype=np.float64).reshape(-1)
    if act.shape[0] != 20:
        raise ValueError(f"expected a 20-D EEF action, got {act.shape[0]}")
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

    # Gripper: model dim [9] is the PRETRAIN open-scale in [-1,+1] (-1=close, +1=open — the reader
    # NEGATES RoboCasa's native +1=close so post-train matches the pretrained gripper space). Close
    # only on a CONFIDENT command: close (1.0) iff act[9] < -0.5, else open (0.0). An uncertain /
    # neutral output (~0, the flow-matching prior mean) defaults to open, avoiding spurious grasps.
    # ENV CONTRACT (measured + gym_wrapper.py:125): the wrapper binarizes at 0.5 — >=0.5 -> +1
    # (close), <0.5 -> -1 (open) — so 0.0 here IS a real open command, not a hold.
    gripper_cmd = 1.0 if float(act[9]) < -0.5 else 0.0

    base = np.zeros(4, np.float64) if base_motion is None else np.asarray(base_motion, np.float64).reshape(-1)
    if base.shape[0] != 4:
        raise ValueError(f"base_motion must be 4-D, got {base.shape[0]}")
    # SERVER / slice_action order: eef_pos, eef_rot, grip, base_motion, control_mode.
    return np.concatenate([pos_cmd, rot_cmd, [gripper_cmd], base, [float(control_mode)]]).astype(np.float32)


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


def robocasa_state_to_eef20d(
    eef_pos_rel: np.ndarray,
    eef_rot_rel_quat_xyzw: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Assemble the **20-D single-arm EEF proprio** from a RoboCasa365 obs, RAW (unnormalized).

    Bit-identical to the dataloader's ``state_to_arm10`` + ``assemble_single_arm_left``
    (``openwam.dataloader.robocasa365``): the eval client must send proprio in the SAME 20-D
    representation the model was trained on (the env outputs a 16-D raw state; the client converts).
    The server normalizes; send RAW here. Right-arm 10 dims are zero-padded.

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
    arm10 = np.concatenate([pos, quat_xyzw_to_rot6d(quat), grip], axis=-1)  # (10,)
    out = np.zeros(20, np.float32)
    out[:10] = arm10  # single-arm LEFT; right half stays 0 (masked at train time)
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


# ── R1Pro 256-D proprio → RAW-27 (ACHIEVED state, rendered into the action's raw
#    space) ─────────────────────────────────────────────────────────────────────
# The bridge sends the model's proprio from the ONLY thing that exists at deploy:
# the robot's measured 256-D ``observation.state`` (OmniGibson ``robot_r1::proprio``
# == the dataset's ``observation.state``). It must be rendered EXACTLY as the trainer
# does (openwam.dataloader.behavior._state_to_raw_proprio_eef), else the model sees a
# proprio distribution it never trained on → pose drift. This module stays pure-numpy
# (no ``openwam`` import) so the deploy client can run it anywhere, so the rendering
# is duplicated here; ``test_behavior_bridge`` cross-checks the two implementations
# produce byte-identical output on the same 256-D state.
#
# Offsets decoded from the robot's ``proprio_obs`` list (each episode's
# meta/episodes/*.json → ``config`` → ``robots[0].proprio_obs``) and verified
# through redundant relationships (sin(qpos) agrees with the redundant sine block; quat ‖·‖==1; base_qvel
# == d(base_qpos)/dt). eef pose is achieved; gripper/base/trunk are mapped into the
# action's normalized command space (see the two helpers below).
_PP_L_POS = slice(186, 189)
_PP_L_QUAT = slice(189, 193)  # xyzw
_PP_R_POS = slice(225, 228)
_PP_R_QUAT = slice(228, 232)
_PP_L_GRIP_QPOS = slice(193, 195)  # left MultiFinger gripper: 2 finger positions (m)
_PP_R_GRIP_QPOS = slice(232, 234)  # right gripper: 2 finger positions (m)
_PP_TRUNK_QPOS = slice(236, 240)  # achieved trunk joint positions (rad)
_PP_BASE_QVEL = slice(253, 256)  # base joint velocity [vx,vy,vyaw] in the WORLD frame
_PP_BASE_YAW = 246  # base_qpos yaw (world), for the world→base-frame rotation
R1PRO_PROPRIO_DIM = 256
# Controller limits mapping achieved (physical) proprio → the action's [-1,1] cmd
# space (kept in lockstep with openwam.dataloader.behavior).
_GRIPPER_OPEN_QPOS = 0.05
_BASE_VEL_OUTPUT_SCALE = np.array([0.75, 0.75, 1.0], dtype=np.float32)


def _proprio_grip_open_scale(grip_qpos: np.ndarray) -> np.ndarray:
    """``(2,)`` finger positions → ``(1,)`` open-scale in ``[-1,+1]`` (mean of the two
    fingers through the gripper cmd→qpos limits; +1 open, -1 closed)."""
    opening = np.asarray(grip_qpos, dtype=np.float32).mean(axis=-1, keepdims=True)
    return np.clip(2.0 * opening / _GRIPPER_OPEN_QPOS - 1.0, -1.0, 1.0).astype(np.float32)


def _proprio_base_vel_local(proprio: np.ndarray) -> np.ndarray:
    """``(256,)`` proprio → ``(3,)`` achieved base velocity in the BASE frame,
    normalized to the ``[-1,1]`` command scale (world ``base_qvel`` rotated by ``-yaw``,
    then divided by the controller output limits)."""
    qv = proprio[_PP_BASE_QVEL]
    yaw = float(proprio[_PP_BASE_YAW])
    cos, sin = np.cos(yaw), np.sin(yaw)
    vx = cos * qv[0] + sin * qv[1]
    vy = -sin * qv[0] + cos * qv[1]
    return (np.array([vx, vy, qv[2]], dtype=np.float32) / _BASE_VEL_OUTPUT_SCALE).astype(np.float32)


def r1pro_proprio_to_raw27(proprio: np.ndarray) -> np.ndarray:
    """Render the R1Pro 256-D measured proprio → RAW-27 in the reader's action layout.

    Output (== ``openwam.dataloader.behavior._state_to_raw_proprio_eef``)::

        [L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6, R_grip1, base3, trunk4]

    EEF pose + rot6d from the state quaternions (achieved), gripper open-scale from
    the finger qpos, base-frame velocity, achieved trunk qpos. The OpenWAM server's
    _UnifyAwareNormalizer then normalizes this raw proprio (shared stats) and scatters
    it into the unified space — so the bridge sends RAW, NOT unified.

    Returns an un-normalized ``(27,)`` float32 vector (the server normalizes it).
    """
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if p.shape[0] != R1PRO_PROPRIO_DIM:
        raise ValueError(f"expected R1Pro proprio of width {R1PRO_PROPRIO_DIM}, got {p.shape[0]}")

    return np.concatenate(
        [
            p[_PP_L_POS],
            quat_xyzw_to_rot6d(p[_PP_L_QUAT]),
            _proprio_grip_open_scale(p[_PP_L_GRIP_QPOS]),
            p[_PP_R_POS],
            quat_xyzw_to_rot6d(p[_PP_R_QUAT]),
            _proprio_grip_open_scale(p[_PP_R_GRIP_QPOS]),
            _proprio_base_vel_local(p),
            p[_PP_TRUNK_QPOS],
        ]
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# LIBERO (single-arm Panda, 7-D OSC delta)                        #
# --------------------------------------------------------------------------- #
# Mirrors of the trainer's rendering in openwam/dataloader/libero.py — the eval
# and training ends of the same contract. tests/benchmarks/test_libero_bridge.py
# pins each pair together; change them in lockstep.
#
# The OpenWAM LIBERO checkpoint predicts a raw 10-D single-arm EEF pose
# (world frame, FULL pose — not the env's per-step OSC delta)::
#
#     eef10 = [xyz(3), rot6d(6), gripper_cmd(1)]
#
# The deploy server's _UnifyAwareNormalizer already gathered the model's 80-D
# unified output back to this raw 10-D and unnormalized it, so the client
# receives/sends physical EEF10 and only bridges representation:
#   * proprio: live obs (robot0_eef_pos / robot0_eef_quat / robot0_gripper_qpos)
#     -> EEF10, byte-consistent with the canonical dataset converter.
#   * action: EEF10 full pose -> 7-D OSC delta using the live controller scales.
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
# robosuite OSC_POSE defaults for the LIBERO pin (output_max = 0.05 m / 0.5 rad
# per unit action). Prefer reading the LIVE controller's output_max; these are
# the documented fallbacks the client uses when the env probe is unavailable.
LIBERO_OSC_POS_SCALE_DEFAULT = 0.05
LIBERO_OSC_ROT_SCALE_DEFAULT = 0.5


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


def eef10_to_libero7d(
    action: np.ndarray,
    ref_pos: np.ndarray,
    ref_rot6d: np.ndarray,
    *,
    pos_scale: float = LIBERO_OSC_POS_SCALE_DEFAULT,
    rot_scale: float = LIBERO_OSC_ROT_SCALE_DEFAULT,
    clip: bool = True,
) -> np.ndarray:
    """Bridge the model's 10-D FULL EEF pose to LIBERO's 7-D OSC delta action.

    The dual of ``eef20d_to_robocasa12d`` reduced to a fixed-base single arm:
    position difference divided by the controller position scale, rotation via
    ``R_target @ R_ref.T`` -> axis-angle divided by the rotation scale, gripper
    NEGATED from the trained open-scale (+1 = open) back into LIBERO's own
    command space (+1 = close) and clipped. LIBERO's gripper stays CONTINUOUS in
    [-1, +1] — no RoboCasa-style 0/1 thresholding.

    Args:
        action: (10,) EEF10 ``[xyz3, rot6d6, grip1]`` (world frame, physical).
        ref_pos: (3,) current achieved EEF position (world frame).
        ref_rot6d: (6,) current achieved EEF rotation as rot6d.
        pos_scale / rot_scale: robosuite OSC ``output_max`` (metres / radians
            mapped to action 1.0). Read from the live controller when possible.
        clip: clip the scaled pos/rot commands to [-1, 1] (OSC action bounds).
    """
    act = np.asarray(action, np.float64).reshape(-1)
    if act.shape[0] != LIBERO_EEF10_DIM:
        raise ValueError(f"expected a {LIBERO_EEF10_DIM}-D EEF action, got {act.shape[0]}")
    ref_pos = np.asarray(ref_pos, np.float64).reshape(-1)
    if ref_pos.shape[0] != 3:
        raise ValueError(f"ref_pos must be 3-D, got {ref_pos.shape[0]}")
    if not (float(pos_scale) > 0.0 and float(rot_scale) > 0.0):
        raise ValueError(
            f"pos_scale and rot_scale must be > 0 (got pos={pos_scale}, rot={rot_scale}); "
            "set them from the eval env's OSC_POSE output_max."
        )

    pos_cmd = (act[0:3] - ref_pos) / float(pos_scale)

    R_t = _rot6d_to_matrix(act[3:9])
    R_c = _rot6d_to_matrix(np.asarray(ref_rot6d, np.float64).reshape(-1))
    rot_cmd = _matrix_to_axis_angle(R_t @ R_c.T).astype(np.float64) / float(rot_scale)

    if clip:
        pos_cmd = np.clip(pos_cmd, -1.0, 1.0)
        rot_cmd = np.clip(rot_cmd, -1.0, 1.0)

    grip_cmd = libero_open_scale_to_gripper_cmd(act[9])
    return np.concatenate([pos_cmd, rot_cmd, [grip_cmd]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# EBench (GenManip lift2/R5a dual-arm mobile manipulator)                      #
# --------------------------------------------------------------------------- #
# Mirrors of the trainer's rendering in openwam/dataloader/ebench.py — the
# eval and training ends of the same contract. A regression test
# (tests/benchmarks/test_ebench_bridge.py) pins each pair together byte-for-
# byte; change them in lockstep.

EBENCH_RAW_DIM = 23
EBENCH_BASE_SOURCES = ("delta", "cumulative")
# GenManip lift2 gripper: 0.0 closed .. 0.044 open per finger (both fingers of
# a hand are commanded identically).
EBENCH_GRIPPER_OPEN = 0.044


def ebench_wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap radian angle(s) to [-pi, pi). Mirror of ebench.wrap_angle_rad."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def ebench_render_state_base(cur_base, prev_base, base_action_source: str) -> np.ndarray:
    """Render measured ``state.base`` ``[x_m, y_m, yaw_RAD]`` into the trained
    base-action command space. Mirror of ebench.render_ebench_state_base.

    ``delta``: measured per-step displacement (yaw wrapped, rad→deg);
    ``prev_base=None`` (episode start) → zeros. ``cumulative``: pose with yaw
    rad→deg.
    """
    cur = np.asarray(cur_base, dtype=np.float64).reshape(3)
    if base_action_source == "delta":
        if prev_base is None:
            return np.zeros(3, dtype=np.float32)
        prev = np.asarray(prev_base, dtype=np.float64).reshape(3)
        delta = cur - prev
        dyaw = float(ebench_wrap_angle_rad(delta[2]))
        return np.array([delta[0], delta[1], np.degrees(dyaw)], dtype=np.float32)
    if base_action_source == "cumulative":
        return np.array([cur[0], cur[1], np.degrees(cur[2])], dtype=np.float32)
    raise ValueError(f"base_action_source must be one of {EBENCH_BASE_SOURCES}, got {base_action_source!r}")


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


def raw23_to_ebench_action(action: np.ndarray, base_action_source: str) -> dict:
    """OpenWAM RAW-23 physical action → GenManip EvalClient action dict.

    The server already inverted normalization and the 80-D unify scatter, so
    ``action`` is the raw 23-D vector in physical units. Arms go out as
    absolute ``ee_pose`` targets (GenManip runs cuRobo IK per arm in the same
    per-arm base frames that produced the training FK); the scalar gripper is
    duplicated to both fingers and clipped to the physical range; the base
    slot goes out per the trained ``base_action_source``:

    * ``delta`` → ``base_motion=[dx_m, dy_m, dyaw_deg]``, ``base_is_rel=True``
      (GenManip clips each step to ±0.015 m / ±1° — the same clamps the demos
      obeyed — and converts the degree yaw internally).
    * ``cumulative`` → absolute ``base_motion=[x_m, y_m, yaw_deg]``,
      ``base_is_rel=False`` (GenManip applies ``deg2rad`` to index 2).

    Positions/quaternions are plain Python lists ON PURPOSE: the GenManip
    server concatenates ``position + orientation`` (list concat) before IK —
    numpy arrays would broadcast-add and crash it.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != EBENCH_RAW_DIM:
        raise ValueError(f"expected raw {EBENCH_RAW_DIM}-D EBench action, got {a.shape[0]}")
    if base_action_source not in EBENCH_BASE_SOURCES:
        raise ValueError(f"base_action_source must be one of {EBENCH_BASE_SOURCES}, got {base_action_source!r}")

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
        "base_is_rel": base_action_source == "delta",
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
