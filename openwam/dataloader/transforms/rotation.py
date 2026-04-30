"""Rotation representation transforms.

Converts between rotation representations in the action vector.
All conversions go through rotation matrices as the intermediate format.

Supported representations:
  - axis_angle:   (3,) axis-angle vector
  - quaternion:   (4,) wxyz quaternion
  - rotation_6d:  (6,) first two columns of rotation matrix
  - euler_xyz:    (3,) Euler angles (XYZ convention)
  - euler_zyx:    (3,) Euler angles (ZYX convention)

Uses scipy for conversions (no pytorch3d dependency).
"""

from enum import Enum
from typing import List, Optional

import numpy as np

from openwam.dataloader.transforms.base import InvertibleModalityTransform


class RotationType(str, Enum):
    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    EULER_XYZ = "euler_xyz"
    EULER_ZYX = "euler_zyx"


# Dimension of each rotation representation
_REPR_DIM = {
    RotationType.AXIS_ANGLE: 3,
    RotationType.QUATERNION: 4,
    RotationType.ROTATION_6D: 6,
    RotationType.EULER_XYZ: 3,
    RotationType.EULER_ZYX: 3,
}


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle → (N, 3, 3) rotation matrices."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(aa).as_matrix()


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternion → (N, 3, 3) rotation matrices."""
    from scipy.spatial.transform import Rotation

    # scipy uses xyzw, our convention is wxyz
    xyzw = np.concatenate([q[..., 1:], q[..., :1]], axis=-1)
    return Rotation.from_quat(xyzw).as_matrix()


def _euler_to_matrix(e: np.ndarray, convention: str) -> np.ndarray:
    """(N, 3) euler angles → (N, 3, 3) rotation matrices."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(convention.upper(), e).as_matrix()


def _rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """(N, 6) 6D rotation → (N, 3, 3) rotation matrices via Gram-Schmidt."""
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)


def _matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices → (N, 3) axis-angle."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(mat).as_rotvec()


def _matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices → (N, 4) wxyz quaternion."""
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(mat).as_quat()
    return np.concatenate([xyzw[..., 3:], xyzw[..., :3]], axis=-1)


def _matrix_to_euler(mat: np.ndarray, convention: str) -> np.ndarray:
    """(N, 3, 3) rotation matrices → (N, 3) euler angles."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(mat).as_euler(convention.upper())


def _matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices → (N, 6) 6D rotation (first two columns)."""
    # mat[..., :, 0] = first column, mat[..., :, 1] = second column
    col1 = mat[..., :, 0]  # (..., 3)
    col2 = mat[..., :, 1]  # (..., 3)
    return np.concatenate([col1, col2], axis=-1)  # (..., 6)


# Dispatch tables
_TO_MATRIX = {
    RotationType.AXIS_ANGLE: _axis_angle_to_matrix,
    RotationType.QUATERNION: _quaternion_to_matrix,
    RotationType.ROTATION_6D: _rotation_6d_to_matrix,
    RotationType.EULER_XYZ: lambda x: _euler_to_matrix(x, "xyz"),
    RotationType.EULER_ZYX: lambda x: _euler_to_matrix(x, "zyx"),
}

_FROM_MATRIX = {
    RotationType.AXIS_ANGLE: _matrix_to_axis_angle,
    RotationType.QUATERNION: _matrix_to_quaternion,
    RotationType.ROTATION_6D: _matrix_to_rotation_6d,
    RotationType.EULER_XYZ: lambda x: _matrix_to_euler(x, "xyz"),
    RotationType.EULER_ZYX: lambda x: _matrix_to_euler(x, "zyx"),
}


def convert_rotation(
    data: np.ndarray,
    source: RotationType,
    target: RotationType,
) -> np.ndarray:
    """Convert between rotation representations via rotation matrix.

    Args:
        data: (..., source_dim) rotation data.
        source: Source representation type.
        target: Target representation type.

    Returns:
        (..., target_dim) converted rotation data.
    """
    if source == target:
        return data

    original_shape = data.shape[:-1]
    flat = data.reshape(-1, _REPR_DIM[source])

    mat = _TO_MATRIX[source](flat)
    result = _FROM_MATRIX[target](mat)

    target_dim = _REPR_DIM[target]
    return result.reshape(*original_shape, target_dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience helpers for RoboTwin HDF5 xyzw quaternion ↔ 6D rotation
#
# RoboTwin HDF5 stores quaternions as xyzw (scipy default) while
# OpenWAM's convert_rotation() uses wxyz.  These thin wrappers handle
# the convention swap so callers don't have to think about it.
# ---------------------------------------------------------------------------


def quat_xyzw_to_rotation_6d(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 6D rotation representation.

    Args:
        q: (..., 4) quaternion in xyzw format (scipy / RoboTwin convention).

    Returns:
        (..., 6) 6D rotation (first two columns of the rotation matrix).
    """
    wxyz = np.concatenate([q[..., 3:], q[..., :3]], axis=-1)
    return convert_rotation(wxyz, RotationType.QUATERNION, RotationType.ROTATION_6D)


def rotation_6d_to_quat_xyzw(r6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to xyzw quaternion.

    Args:
        r6d: (..., 6) 6D rotation representation.

    Returns:
        (..., 4) quaternion in xyzw format (scipy / RoboTwin convention).
    """
    wxyz = convert_rotation(r6d, RotationType.ROTATION_6D, RotationType.QUATERNION)
    return np.concatenate([wxyz[..., 1:], wxyz[..., :1]], axis=-1)


class RotationTransform(InvertibleModalityTransform):
    """Convert rotation components in action vectors between representations.

    Args:
        source_repr: Current rotation representation in the data.
        target_repr: Desired rotation representation after transform.
        keys: Sample dict keys to transform (default: ["action"]).
        rotation_slice: Slice into the action vector for rotation dimensions.
            e.g., slice(3, 6) for standard 7D EEF actions [pos(3), rot(3), grip(1)].
    """

    def __init__(
        self,
        source_repr: str,
        target_repr: str,
        keys: Optional[List[str]] = None,
        rotation_slice: slice = slice(3, 6),
    ):
        super().__init__(apply_to=keys or ["action"])
        self.source = RotationType(source_repr)
        self.target = RotationType(target_repr)
        self.rotation_slice = rotation_slice
        self.source_dim = _REPR_DIM[self.source]
        self.target_dim = _REPR_DIM[self.target]

    def _transform_array(self, x: np.ndarray, source: RotationType, target: RotationType) -> np.ndarray:
        """Transform rotation components within an action array."""
        s = self.rotation_slice
        rot_data = x[..., s]

        converted = convert_rotation(rot_data, source, target)

        # Reconstruct the action vector with new rotation dim
        pre = x[..., : s.start]
        post = x[..., s.stop :]
        return np.concatenate([pre, converted, post], axis=-1).astype(np.float32)

    def apply(self, data: dict) -> dict:
        import torch as _torch

        for key in self.apply_to:
            if key in data and data[key] is not None:
                val = data[key]
                is_tensor = isinstance(val, _torch.Tensor)
                arr = val.numpy() if is_tensor else val
                arr = self._transform_array(arr, self.source, self.target)
                data[key] = _torch.from_numpy(arr) if is_tensor else arr
        return data

    def unapply(self, data: dict) -> dict:
        import torch as _torch

        for key in self.apply_to:
            if key in data and data[key] is not None:
                val = data[key]
                is_tensor = isinstance(val, _torch.Tensor)
                arr = val.numpy() if is_tensor else val
                # Reverse: target → source, but with adjusted slice for target dim
                reverse_slice = slice(
                    self.rotation_slice.start,
                    self.rotation_slice.start + self.target_dim,
                )
                rot_data = arr[..., reverse_slice]
                converted = convert_rotation(rot_data, self.target, self.source)
                pre = arr[..., : reverse_slice.start]
                post = arr[..., reverse_slice.stop :]
                arr = np.concatenate([pre, converted, post], axis=-1).astype(np.float32)
                data[key] = _torch.from_numpy(arr) if is_tensor else arr
        return data
