"""Shared EEF (end-effector) conversion helpers.

Hosts the small numpy helpers used by readers that emit the canonical
bimanual 20-D EEF schema::

    [L_pos(3) + L_rot6d(6) + L_grip(1) + R_pos(3) + R_rot6d(6) + R_grip(1)]

Originally lived as private helpers inside ``robocoin.py``; extracted so
the upcoming OXE readers (BC-Z, Bridge, Fractal, DROID — all single-arm)
can reuse the same rot6d / 20-D assembly logic without copy-paste.

Single-arm OXE readers fill the **left** half ``[0:10]`` with real data
and zero-pad the right half ``[10:20]``; the per-dim portion of the
2-D action_mask / proprio_mask handles the right-arm exclusion downstream.

For active real-robot mixture sources, ``pos + rot6d`` follows one verifiable
endpoint *category*: a terminal-arm frame rigidly attached to the arm chain and
independent of gripper/finger articulation. This is intentionally broader and
more accurate than either ``flange`` or ``TCP``: depending on the published
robot model the exact frame is an arm flange, wrist-yaw link, hand base, last
arm link, gripper mount, or potentially a fixed tool frame. It does not assert
that different robots share a literal local origin/axis calibration; their
poses remain expressed in their dataset's documented robot-base frame.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.normalization import apply_normalization

EEF_DIM = 20
ARM10_DIM = 10  # pos(3) + rot6d(6) + grip(1)
EEF_POSE_FRAME_CONTRACT = "rigid_terminal_arm_frame_independent_of_gripper_motion"


def euler_xyz_to_rot6d(euler: np.ndarray) -> np.ndarray:
    """Convert euler XYZ extrinsic angles to 6-D rotation representation.

    Args:
        euler: (T, 3) float array — ``[roll_x, pitch_y, yaw_z]`` in radians.

    Returns:
        (T, 6) float array — first two columns of the rotation matrix
        ``R = Rz @ Ry @ Rx``, concatenated as ``[col0(3), col1(3)]``.

    Bit-identical to the original ``robocoin._euler_to_rot6d``.
    """
    cx, sx = np.cos(euler[:, 0]), np.sin(euler[:, 0])
    cy, sy = np.cos(euler[:, 1]), np.sin(euler[:, 1])
    cz, sz = np.cos(euler[:, 2]), np.sin(euler[:, 2])
    c0 = np.stack([cy * cz, cy * sz, -sy], axis=-1)
    c1 = np.stack(
        [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy],
        axis=-1,
    )
    return np.concatenate([c0, c1], axis=-1)  # (T, 6)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert (T, 4) quaternion [x, y, z, w] → (T, 6) rot6d.

    Args:
        quat: (T, 4) float array — unit quaternions in xyzw convention
            (scipy.spatial.transform.Rotation default). Caller is responsible
            for unit-norm guarantees; ``assert_unit_quaternion`` can sanity
            check a sample batch at reader __init__.

    Returns:
        (T, 6) float array — first two columns of the rotation matrix
        derived from the quaternion, concatenated as ``[col0(3), col1(3)]``.

    The rotation matrix is the standard quat→matrix formula::

        R = | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  |
            | 2(xy+wz)     1-2(x²+z²)  2(yz-wx)  |
            | 2(xz-wy)     2(yz+wx)    1-2(x²+y²)|

    Column 0 = ``[1-2(y²+z²), 2(xy+wz), 2(xz-wy)]``
    Column 1 = ``[2(xy-wz), 1-2(x²+z²), 2(yz+wx)]``
    """
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    c0 = np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)], axis=-1)
    c1 = np.stack([2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)], axis=-1)
    return np.concatenate([c0, c1], axis=-1).astype(quat.dtype)


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert (..., 4) quaternion [w, x, y, z] → (..., 6) rot6d.

    Thin convention adapter over :func:`quat_xyzw_to_rot6d` for sources that
    store the scalar part FIRST (Isaac / cuRobo / LeRobot ``quaternion.w`` field
    order) rather than scipy's xyzw. Routing a wxyz array straight into
    ``quat_xyzw_to_rot6d`` silently yields a wrong-but-unit rotation, which no
    norm check can catch (see :func:`assert_unit_quaternion`) — so the reorder
    must be explicit at every wxyz call site.

    Accepts arbitrary leading dims; the last axis must be 4.
    """
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"quaternion must be 4-D wxyz, got shape {quat_wxyz.shape}")
    leading = quat_wxyz.shape[:-1]
    flat = quat_wxyz.reshape(-1, 4)
    flat_xyzw = np.concatenate([flat[:, 1:4], flat[:, 0:1]], axis=-1)
    return quat_xyzw_to_rot6d(flat_xyzw).reshape(*leading, 6).astype(np.float32)


def assert_unit_quaternion(quat: np.ndarray, tol: float = 0.05, sample_n: int = 64) -> None:
    """Sanity check that ``quat`` entries are roughly unit-norm.

    Reads up to ``sample_n`` leading rows from ``quat`` and asserts
    ``|‖q‖₂ - 1| < tol`` element-wise. Used in the Fractal reader __init__ to
    catch quaternion data that is un-normalized or otherwise wrong-magnitude
    (e.g. raw axis-angle / a 4-vector that isn't a unit quaternion at all)
    before training starts.

    Note: a norm check CANNOT detect a wxyz-vs-xyzw component reordering —
    both conventions are unit-norm, so a mis-routed-but-normalized quaternion
    passes this check. Convention correctness must be guaranteed upstream.
    """
    n = int(min(sample_n, len(quat)))
    if n == 0:
        return
    norms = np.linalg.norm(quat[:n].astype(np.float64), axis=-1)
    bad = np.abs(norms - 1.0) > tol
    if bad.any():
        idx = int(np.where(bad)[0][0])
        raise ValueError(
            f"Quaternion norm check failed at row {idx}: ‖q‖₂={norms[idx]:.4f}, "
            f"expected ≈ 1.0 (tol={tol}). Likely the data uses a different "
            f"quaternion convention (wxyz vs xyzw) or is not normalized."
        )


def assemble_single_arm_left(arm10: np.ndarray) -> np.ndarray:
    """Pad a single-arm (..., 10) EEF tensor into a bimanual (..., 20) tensor.

    Real data fills slots ``[0:10]`` (left arm); slots ``[10:20]``
    (right arm) are zero-padded. Used by OXE readers that emit 10-D
    single-arm EEF and need to slot it into the canonical 20-D schema.
    The per-dim portion of ``action_mask`` / ``proprio_mask`` makes the
    right-arm dims invisible to loss / encoder.
    """
    out = np.zeros(arm10.shape[:-1] + (EEF_DIM,), dtype=arm10.dtype)
    out[..., :ARM10_DIM] = arm10
    return out


def assemble_single_arm_right(arm10: np.ndarray) -> np.ndarray:
    """Mirror of :func:`assemble_single_arm_left` for right-arm placement.

    Currently unused (all OXE readers use the left slot) — kept as
    documentation / future-proofing for asymmetric mixture strategies.
    """
    out = np.zeros(arm10.shape[:-1] + (EEF_DIM,), dtype=arm10.dtype)
    out[..., ARM10_DIM:] = arm10
    return out


def single_arm_20d(arm10: np.ndarray, stats: dict | None, mode: str | None) -> np.ndarray:
    """Normalize a single-arm ``(..., 10)`` EEF and slot it into the left half
    of the canonical bimanual ``(..., 20)`` schema (right half zero-padded).

    The single-arm readers (OXE BC-Z / Bridge / Fractal / DROID) normalize on the
    10-D arm (their ``eef_stats.json`` is 10-D) *before* assembly. This is the
    exact ``apply_normalization`` → ``assemble_single_arm_left`` sequence those
    readers ran inline; factored here so each reader's action/proprio hook is a
    one-liner instead of a copy-pasted block.
    """
    return assemble_single_arm_left(apply_normalization(arm10, stats, mode))


def eef14_to_eef20(eef12: np.ndarray, grip2: np.ndarray) -> np.ndarray:
    """Convert (T, 12) EEF + (T, 2) gripper → (T, 20) unified action/state.

    Input layout::

        eef12: [L_pos(3), L_euler(3), R_pos(3), R_euler(3)]
        grip2: [L_grip, R_grip]

    Output layout::

        [L_pos(3), L_rot6d(6), L_grip(1), R_pos(3), R_rot6d(6), R_grip(1)]

    Bit-identical to the original ``robocoin._eef14_to_eef20``.
    """
    l_pos = eef12[:, 0:3]
    l_euler = eef12[:, 3:6]
    r_pos = eef12[:, 6:9]
    r_euler = eef12[:, 9:12]
    l_grip = grip2[:, 0:1]
    r_grip = grip2[:, 1:2]
    l_rot6d = euler_xyz_to_rot6d(l_euler)
    r_rot6d = euler_xyz_to_rot6d(r_euler)
    return np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (T, 20)


# ---------------------------------------------------------------------------
# 2-D mask helpers
# ---------------------------------------------------------------------------
# Every reader that emits the 20-D EEF schema produces a 2-D
# ``action_mask`` of shape ``(T_action, action_dim)`` and a 2-D
# ``proprio_mask`` of shape ``(1, action_dim)``. ``mask[t, d] = True``
# means "this (time, dim) element is real data and should contribute
# to loss / encoder gradients".
#
# Single-arm OXE readers fill the left half ``[0:10]`` with real data
# and leave the right half ``[10:20]`` zero-padded; the per-dim portion
# of the mask makes the right-arm dims invisible to loss.

# Canonical single-arm dim mask: front 10 dims valid, back 10 invalid.
LEFT_ARM_DIM_MASK = np.concatenate([np.ones(ARM10_DIM, dtype=bool), np.zeros(EEF_DIM - ARM10_DIM, dtype=bool)])
RIGHT_ARM_DIM_MASK = np.concatenate([np.zeros(EEF_DIM - ARM10_DIM, dtype=bool), np.ones(ARM10_DIM, dtype=bool)])


def build_action_mask_2d(
    T_action: int,
    action_dim: int,
    n_valid_time: int,
    dim_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``(T_action, action_dim) bool`` action_mask.

    ``mask[t, d] = (t < n_valid_time) AND (dim_mask[d] if dim_mask else True)``

    Args:
        T_action: window action horizon (``num_frames - 1``).
        action_dim: per-step action dim (20 for EEF, 14 for joint, etc.).
        n_valid_time: count of leading time slots within the episode.
            ``n_valid_time == 0`` → entire mask is False (useful for
            ``enable_action_supervision=False`` paths).
        dim_mask: optional ``(action_dim,) bool``. When None, all dims
            are valid. Pass ``LEFT_ARM_DIM_MASK`` for OXE single-arm.

    Returns:
        ``(T_action, action_dim)`` bool ndarray.
    """
    mask = np.zeros((T_action, action_dim), dtype=bool)
    if n_valid_time > 0:
        if dim_mask is None:
            mask[:n_valid_time, :] = True
        else:
            mask[:n_valid_time, :] = dim_mask
    return mask


def build_proprio_mask_2d(
    action_dim: int,
    enabled: bool = True,
    dim_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``(1, action_dim) bool`` proprio_mask.

    Args:
        action_dim: per-step dim of proprio (matches reader's EEF dim).
        enabled: ``False`` → entire mask is False (used when
            ``enable_action_supervision=False`` or in EgoDex-style
            video-only readers).
        dim_mask: optional ``(action_dim,) bool``. None → all True.

    Returns:
        ``(1, action_dim)`` bool ndarray.
    """
    if not enabled:
        return np.zeros((1, action_dim), dtype=bool)
    if dim_mask is None:
        return np.ones((1, action_dim), dtype=bool)
    return dim_mask.astype(bool).reshape(1, action_dim).copy()


__all__ = [
    "EEF_DIM",
    "ARM10_DIM",
    "EEF_POSE_FRAME_CONTRACT",
    "LEFT_ARM_DIM_MASK",
    "RIGHT_ARM_DIM_MASK",
    "euler_xyz_to_rot6d",
    "quat_xyzw_to_rot6d",
    "quat_wxyz_to_rot6d",
    "assert_unit_quaternion",
    "assemble_single_arm_left",
    "assemble_single_arm_right",
    "single_arm_20d",
    "eef14_to_eef20",
    "build_action_mask_2d",
    "build_proprio_mask_2d",
]
