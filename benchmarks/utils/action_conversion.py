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
    gripper_close_threshold: float = 0.05,
    clip: bool = True,
) -> np.ndarray:
    """Bridge the model's 20-D **absolute** EEF pose to RoboCasa's 12-D **OSC** action.

    RoboCasa365's ``RoboCasaGymEnv`` consumes a 12-D robosuite OSC_POSE + mobile-base
    action; the OpenWAM model trained by ``RoboCasa365Dataset`` instead predicts a
    20-D *absolute* single-arm EEF pose (left half ``[pos3, rot6d6, grip1]``, right
    half 0). This is the dual of robotwin's client-side ``eef20d_to_ee16d`` — except
    robotwin's env takes absolute 16-D poses, whereas RoboCasa's OSC controller takes
    *delta* commands scaled into ``[-1, 1]``, so the conversion needs the current
    proprio (to form the delta) and the controller's scaling.

    Output is the flat 12-D in the SERVER/``slice_action`` order (NOT modality.json
    order)::

        [eef_pos_cmd(3), eef_rot_cmd(3), gripper(1), base_motion(4), control_mode(1)]

    Args:
        action: 20-D EEF action; only the left-arm 10 dims ``[pos3, rot6d6, grip1]`` are used.
        proprio_eef_pos: (3,) current absolute EEF position (from ``state.end_effector_position_relative``).
        proprio_eef_rot6d: (6,) current EEF rotation as rot6d (quat->rot6d of ``state.end_effector_rotation_relative``).
        pos_scale: robosuite OSC position ``output_max`` (metres mapped to action 1.0). **REQUIRED, env-specific** —
            read it from the eval env's OSC_POSE controller config; a wrong value drives wrong-magnitude motions.
        rot_scale: robosuite OSC rotation ``output_max`` (radians mapped to action 1.0). Same caveat as ``pos_scale``.
        base_motion: (4,) base command; defaults to zeros (fixed-base subset — base is dropped at train time).
        control_mode: scalar; defaults to -1.0 (the near-constant value observed in the fixed-base data).
        gripper_close_threshold: finger-separation (metres) below which the gripper is commanded CLOSED.
            The model's gripper dim is finger separation (large=open, ~0.013–0.081); ``RoboCasaGymEnv``
            binarizes ``gripper_close`` at 0.5 (``-1`` open / ``+1`` close), so a raw pass-through (as in
            robotwin, whose env accepts the value directly) would never cross 0.5 and the gripper would
            never close. We map separation -> {open=0, close=1}: ``close iff sep < threshold``. Default
            0.05 sits between the empirical open (~0.078) and closed (~0.034) means; override per env.
        clip: clip the scaled eef commands to ``[-1, 1]`` (OSC action bounds).

    ENV CONTRACT (MEASURED on the real robocasa/OpenDrawer env, PandaOmron / default_pandaomron.json):
    the action convention is **delta** (zero action -> no EEF motion; constant action -> constant
    per-step displacement), matching the (target-current)/scale here. Steady per-step motion per
    action 1.0: ~0.0126 m (pos) / ~0.102 rad (rot) -> use as pos_scale/rot_scale. gripper command
    g=1 closes (separation 0.078 open -> 0.006 closed), so threshold 0.05 straddles them. control_mode
    -1 + base 0 hold the fixed base. (Probed via env.step with known actions; see e2e plan.)
    """
    act = np.asarray(action, dtype=np.float64).reshape(-1)
    if act.shape[0] != 20:
        raise ValueError(f"expected a 20-D EEF action, got {act.shape[0]}")
    cur_pos = np.asarray(proprio_eef_pos, np.float64).reshape(-1)
    if cur_pos.shape[0] != 3:
        raise ValueError(f"proprio_eef_pos must be 3-D, got {cur_pos.shape[0]}")
    tgt_pos, tgt_r6d, grip = act[0:3], act[3:9], act[9:10]

    # Position: absolute target -> scaled OSC delta.
    pos_cmd = (tgt_pos - cur_pos) / max(float(pos_scale), 1e-8)

    # Rotation: relative rotation R_target @ R_current^-1 -> axis-angle -> scaled.
    R_t = _rot6d_to_matrix(tgt_r6d)
    R_c = _rot6d_to_matrix(np.asarray(proprio_eef_rot6d, np.float64).reshape(-1))
    rot_cmd = _matrix_to_axis_angle(R_t @ R_c.T).astype(np.float64) / max(float(rot_scale), 1e-8)

    if clip:
        pos_cmd = np.clip(pos_cmd, -1.0, 1.0)
        rot_cmd = np.clip(rot_cmd, -1.0, 1.0)

    # Gripper: model dim is finger separation (large=open); env binarizes gripper_close at 0.5
    # (-1 open / +1 close). Map separation -> command: close (1.0) iff separation < threshold, else open (0.0).
    gripper_cmd = 1.0 if float(act[9]) < float(gripper_close_threshold) else 0.0

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


def robocasa_state_to_eef20d(
    eef_pos_rel: np.ndarray,
    eef_rot_rel_quat_xyzw: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Assemble the **20-D single-arm EEF proprio** from a RoboCasa365 obs, RAW (unnormalized).

    Bit-identical to the dataloader's ``state_to_arm10`` + ``assemble_single_arm_left``
    (``openwam.dataloader.robocasa365``): the eval client must send proprio in the SAME 20-D
    representation the model was trained on (the env outputs a 16-D raw state; the client converts).
    The server normalizes; send RAW physical units here. Right-arm 10 dims are zero-padded.

        arm10 = [eef_pos_rel(3), rot6d(eef_rot_rel quat xyzw, 6), gripper_separation(1)]
        gripper_separation = gripper_qpos[0] - gripper_qpos[1]   (finger width; large=open)
    """
    pos = np.asarray(eef_pos_rel, np.float32).reshape(-1)
    quat = np.asarray(eef_rot_rel_quat_xyzw, np.float32).reshape(-1)
    qpos = np.asarray(gripper_qpos, np.float32).reshape(-1)
    if pos.shape[0] != 3 or quat.shape[0] != 4 or qpos.shape[0] != 2:
        raise ValueError(
            f"robocasa proprio dims: eef_pos_rel must be 3 (got {pos.shape[0]}), "
            f"eef_rot_rel quat 4 (got {quat.shape[0]}), gripper_qpos 2 (got {qpos.shape[0]})"
        )
    grip = np.array([qpos[0] - qpos[1]], np.float32)
    arm10 = np.concatenate([pos, quat_xyzw_to_rot6d(quat), grip], axis=-1)  # (10,)
    out = np.zeros(20, np.float32)
    out[:10] = arm10  # single-arm LEFT; right half stays 0 (masked at train time)
    return out
