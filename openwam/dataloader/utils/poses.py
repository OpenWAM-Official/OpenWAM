"""Generic NumPy pose-frame and bimanual EEF20 conversions.

These helpers express a pose whose translation is relative to an environment
origin and whose orientation is still world-frame, into a robot-base frame
given that arm's base pose. They also pack / unpack the canonical 20-D EEF
layout. Dataset-specific numbers (for example dual-X5 base poses) stay with
the reader that owns them.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d as shared_quat_wxyz_to_rot6d

_QUATERNION_ATOL = 1e-6
_ROT6D_DEGENERACY_EPS = 1e-8


def _numeric_array(value, name: str) -> tuple[np.ndarray, np.dtype]:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values, got dtype {array.dtype}")
    output_dtype = np.result_type(array.dtype, np.float32)
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array, output_dtype


def _require_last_dim(array: np.ndarray, size: int, name: str) -> None:
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}), got {array.shape}")


def _validated_quaternion(value, name: str = "quaternion") -> tuple[np.ndarray, np.dtype]:
    quaternion, output_dtype = _numeric_array(value, name)
    _require_last_dim(quaternion, 4, name)
    norms = np.linalg.norm(quaternion, axis=-1)
    bad = ~np.isclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_ATOL)
    if np.any(bad):
        flat_index = int(np.flatnonzero(bad)[0])
        bad_norm = float(norms.reshape(-1)[flat_index])
        raise ValueError(f"{name} must be unit quaternion(s) in wxyz order; entry {flat_index} has norm {bad_norm:.8g}")
    return quaternion / norms[..., None], output_dtype


def _validated_pose(value, name: str) -> tuple[np.ndarray, np.dtype]:
    pose, output_dtype = _numeric_array(value, name)
    _require_last_dim(pose, 7, name)
    quaternion, _ = _validated_quaternion(pose[..., 3:7], f"{name} quaternion")
    normalized = pose.copy()
    normalized[..., 3:7] = quaternion
    return normalized, output_dtype


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to canonical (non-negative-w) wxyz."""
    flat_matrices = matrix.reshape(-1, 3, 3)
    flat_quaternions = np.empty((len(flat_matrices), 4), dtype=np.float64)
    for index, rotation in enumerate(flat_matrices):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif rotation[1, 1] > rotation[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        flat_quaternions[index] = quaternion
    return flat_quaternions.reshape(matrix.shape[:-2] + (4,))


def quat_wxyz_to_rot6d(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert unit wxyz quaternion(s) to first-two-column rotation 6D.

    Unit-quaternion checks stay here. The 6D columns come from
    :func:`openwam.dataloader.utils.eef.quat_wxyz_to_rot6d` so readers cannot
    drift in unified slots 3-8 / 37-42. That helper casts to float32
    internally, so even float64 input is returned at float32 precision
    (then upcast to ``output_dtype``).
    """
    quaternion, output_dtype = _validated_quaternion(quaternion_wxyz, "quaternion_wxyz")
    return shared_quat_wxyz_to_rot6d(quaternion).astype(output_dtype, copy=False)


def _validated_rot6d(value) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    rot6d, output_dtype = _numeric_array(value, "rotation_6d")
    _require_last_dim(rot6d, 6, "rotation_6d")
    first = rot6d[..., :3]
    second = rot6d[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    bad_first = first_norm[..., 0] <= _ROT6D_DEGENERACY_EPS
    if np.any(bad_first):
        flat_index = int(np.flatnonzero(bad_first)[0])
        raise ValueError(f"rotation_6d is degenerate: entry {flat_index} has a near-zero first direction")
    first_unit = first / first_norm
    second_orthogonal = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    bad_second = second_norm[..., 0] <= _ROT6D_DEGENERACY_EPS
    if np.any(bad_second):
        flat_index = int(np.flatnonzero(bad_second)[0])
        raise ValueError(f"rotation_6d is degenerate: entry {flat_index} has parallel or near-zero directions")
    second_unit = second_orthogonal / second_norm
    matrix = np.stack((first_unit, second_unit, np.cross(first_unit, second_unit)), axis=-1)
    return rot6d, matrix, output_dtype


def rot6d_to_quat_wxyz(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert non-degenerate rotation 6D value(s) to unit wxyz quaternion(s)."""
    _, matrix, output_dtype = _validated_rot6d(rotation_6d)
    return _matrix_to_quaternion(matrix).astype(output_dtype, copy=False)


def _validated_base_transform(
    base_pos_relative_to_env_origin,
    base_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    base_position, position_dtype = _numeric_array(
        base_pos_relative_to_env_origin,
        "base_pos_relative_to_env_origin",
    )
    if base_position.shape != (3,):
        raise ValueError(f"base_pos_relative_to_env_origin must have exact shape (3,), got {base_position.shape}")
    base_quaternion, quaternion_dtype = _validated_quaternion(
        base_quat_wxyz,
        "base_quat_wxyz",
    )
    if base_quaternion.shape != (4,):
        raise ValueError(f"base_quat_wxyz must have exact shape (4,), got {base_quaternion.shape}")
    return base_position, base_quaternion, np.result_type(position_dtype, quaternion_dtype)


def env_relative_world_to_robot_base(
    pose_env_relative_world_wxyz: np.ndarray,
    base_pos_relative_to_env_origin: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Express an env-relative/world-oriented source pose in robot-base axes.

    The source translation is ``world_position - scene.env_origins[env_idx]``;
    its wxyz orientation is still a world-frame orientation.
    """
    pose, pose_dtype = _validated_pose(
        pose_env_relative_world_wxyz,
        "pose_env_relative_world_wxyz",
    )
    base_position, base_quaternion, base_dtype = _validated_base_transform(
        base_pos_relative_to_env_origin,
        base_quat_wxyz,
    )
    base_rotation = _quaternion_to_matrix(base_quaternion)
    position = np.einsum(
        "ij,...j->...i",
        base_rotation.T,
        pose[..., :3] - base_position,
    )
    inverse_base_quaternion = base_quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    quaternion = _quaternion_multiply(inverse_base_quaternion, pose[..., 3:7])
    result = np.concatenate((position, quaternion), axis=-1)
    return result.astype(np.result_type(pose_dtype, base_dtype), copy=False)


def robot_base_to_env_relative_world(
    pose_robot_base_wxyz: np.ndarray,
    base_pos_relative_to_env_origin: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Invert :func:`env_relative_world_to_robot_base`."""
    pose, pose_dtype = _validated_pose(pose_robot_base_wxyz, "pose_robot_base_wxyz")
    base_position, base_quaternion, base_dtype = _validated_base_transform(
        base_pos_relative_to_env_origin,
        base_quat_wxyz,
    )
    base_rotation = _quaternion_to_matrix(base_quaternion)
    position = np.einsum("ij,...j->...i", base_rotation, pose[..., :3]) + base_position
    quaternion = _quaternion_multiply(base_quaternion, pose[..., 3:7])
    result = np.concatenate((position, quaternion), axis=-1)
    return result.astype(np.result_type(pose_dtype, base_dtype), copy=False)


def arms_to_eef20(
    left_pose_wxyz: np.ndarray,
    left_gripper: np.ndarray,
    right_pose_wxyz: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    """Pack two ``xyz + quaternion wxyz`` poses and grippers into EEF20."""
    left_pose_raw, left_pose_dtype = _numeric_array(left_pose_wxyz, "left_pose_wxyz")
    right_pose_raw, right_pose_dtype = _numeric_array(right_pose_wxyz, "right_pose_wxyz")
    left_gripper_raw, left_gripper_dtype = _numeric_array(left_gripper, "left_gripper")
    right_gripper_raw, right_gripper_dtype = _numeric_array(right_gripper, "right_gripper")
    _require_last_dim(left_pose_raw, 7, "left_pose_wxyz")
    _require_last_dim(right_pose_raw, 7, "right_pose_wxyz")
    _require_last_dim(left_gripper_raw, 1, "left_gripper")
    _require_last_dim(right_gripper_raw, 1, "right_gripper")

    leading_shapes = {
        left_pose_raw.shape[:-1],
        left_gripper_raw.shape[:-1],
        right_pose_raw.shape[:-1],
        right_gripper_raw.shape[:-1],
    }
    if len(leading_shapes) != 1:
        raise ValueError(
            "left/right pose and gripper leading shapes must match exactly; "
            f"got {left_pose_raw.shape[:-1]}, {left_gripper_raw.shape[:-1]}, "
            f"{right_pose_raw.shape[:-1]}, and {right_gripper_raw.shape[:-1]}"
        )

    left_pose, _ = _validated_pose(left_pose_raw, "left_pose_wxyz")
    right_pose, _ = _validated_pose(right_pose_raw, "right_pose_wxyz")
    eef20 = np.concatenate(
        (
            left_pose[..., :3],
            quat_wxyz_to_rot6d(left_pose[..., 3:7]),
            left_gripper_raw,
            right_pose[..., :3],
            quat_wxyz_to_rot6d(right_pose[..., 3:7]),
            right_gripper_raw,
        ),
        axis=-1,
    )
    output_dtype = np.result_type(
        left_pose_dtype,
        left_gripper_dtype,
        right_pose_dtype,
        right_gripper_dtype,
    )
    return eef20.astype(output_dtype, copy=False)


def eef20_to_arms(
    eef20: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack EEF20 into left pose/gripper then right pose/gripper."""
    value, output_dtype = _numeric_array(eef20, "EEF20")
    _require_last_dim(value, 20, "EEF20")
    left_pose = np.concatenate(
        (value[..., 0:3], rot6d_to_quat_wxyz(value[..., 3:9])),
        axis=-1,
    ).astype(output_dtype, copy=False)
    left_gripper = value[..., 9:10].astype(output_dtype, copy=False)
    right_pose = np.concatenate(
        (value[..., 10:13], rot6d_to_quat_wxyz(value[..., 13:19])),
        axis=-1,
    ).astype(output_dtype, copy=False)
    right_gripper = value[..., 19:20].astype(output_dtype, copy=False)
    return left_pose, left_gripper, right_pose, right_gripper


# Descriptive aliases for call sites that prefer explicit "pose" naming.
world_pose_to_robot_base = env_relative_world_to_robot_base
robot_base_pose_to_world = robot_base_to_env_relative_world


__all__ = [
    "arms_to_eef20",
    "eef20_to_arms",
    "env_relative_world_to_robot_base",
    "quat_wxyz_to_rot6d",
    "robot_base_pose_to_world",
    "robot_base_to_env_relative_world",
    "rot6d_to_quat_wxyz",
    "world_pose_to_robot_base",
]
