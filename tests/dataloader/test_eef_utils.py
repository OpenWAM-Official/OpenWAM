"""Tests for the extracted EEF helpers in ``openwam.dataloader.utils.eef``.

These tests pin the byte-level behavior of the shared module so subsequent
refactors (and the upcoming OXE readers that re-use these helpers) cannot
drift away from the original RoboCOIN implementations.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    assemble_single_arm_left,
    assemble_single_arm_right,
    assert_unit_quaternion,
    eef14_to_eef20,
    euler_xyz_to_rot6d,
    quat_xyzw_to_rot6d,
)


class TestEulerToRot6d:
    def test_identity_euler_returns_identity_first_two_cols(self):
        # euler = (0, 0, 0) → R = I, first two columns = [(1,0,0), (0,1,0)]
        out = euler_xyz_to_rot6d(np.zeros((1, 3), dtype=np.float32))
        np.testing.assert_allclose(out, [[1, 0, 0, 0, 1, 0]], atol=1e-7)

    def test_pure_yaw_pi_over_2(self):
        # Pure yaw +90° → Rz rotates X axis to Y axis.
        # col0 of R = Rz(pi/2) [1,0,0]^T = [0, 1, 0]
        # col1 of R = Rz(pi/2) [0,1,0]^T = [-1, 0, 0]
        euler = np.array([[0.0, 0.0, np.pi / 2]], dtype=np.float32)
        out = euler_xyz_to_rot6d(euler)
        np.testing.assert_allclose(out, [[0, 1, 0, -1, 0, 0]], atol=1e-6)

    def test_pure_pitch_pi_over_2(self):
        # Pure pitch +90° → Ry rotates X axis to -Z axis.
        # col0 = Ry(pi/2) [1,0,0]^T = [0, 0, -1]
        # col1 = Ry(pi/2) [0,1,0]^T = [0, 1, 0]
        euler = np.array([[0.0, np.pi / 2, 0.0]], dtype=np.float32)
        out = euler_xyz_to_rot6d(euler)
        np.testing.assert_allclose(out, [[0, 0, -1, 0, 1, 0]], atol=1e-6)

    def test_batch_shape_preserved(self):
        T = 17
        euler = np.random.RandomState(0).randn(T, 3).astype(np.float32)
        out = euler_xyz_to_rot6d(euler)
        assert out.shape == (T, 6)
        assert out.dtype == np.float32

    def test_first_two_cols_are_orthogonal_unit_vectors(self):
        # For any euler input, col0 and col1 should remain orthonormal in 3D
        # (rotation matrix property). The 6D rep stacks them.
        T = 50
        euler = np.random.RandomState(42).randn(T, 3).astype(np.float32)
        out = euler_xyz_to_rot6d(euler)
        c0 = out[:, :3]
        c1 = out[:, 3:]
        # ||c0|| = ||c1|| = 1
        np.testing.assert_allclose(np.linalg.norm(c0, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(c1, axis=-1), 1.0, atol=1e-5)
        # c0 . c1 = 0
        dots = (c0 * c1).sum(axis=-1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-5)


class TestEef14ToEef20:
    def test_shape_and_dim(self):
        eef12 = np.zeros((3, 12), dtype=np.float32)
        grip2 = np.zeros((3, 2), dtype=np.float32)
        out = eef14_to_eef20(eef12, grip2)
        assert out.shape == (3, EEF_DIM)
        assert ARM10_DIM == 10  # constant pinned

    def test_pos_grip_passthrough_zero_rotation(self):
        # With euler=0, rot6d = [1,0,0,0,1,0] for both arms.
        # Pos / grip should pass through unchanged.
        eef12 = np.array([[0.1, 0.2, 0.3, 0, 0, 0, 0.4, 0.5, 0.6, 0, 0, 0]], dtype=np.float32)
        grip2 = np.array([[0.7, 0.8]], dtype=np.float32)
        out = eef14_to_eef20(eef12, grip2)
        expected = np.array(
            [
                [
                    0.1,
                    0.2,
                    0.3,  # L_pos
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,  # L_rot6d (identity)
                    0.7,  # L_grip
                    0.4,
                    0.5,
                    0.6,  # R_pos
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,  # R_rot6d (identity)
                    0.8,  # R_grip
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_layout_order_left_then_right(self):
        # Confirm the canonical [L_pos, L_rot6d, L_grip, R_pos, R_rot6d, R_grip]
        # split points are at 3 / 9 / 10 / 13 / 19 / 20.
        eef12 = np.zeros((1, 12), dtype=np.float32)
        grip2 = np.array([[0.123, 0.456]], dtype=np.float32)
        out = eef14_to_eef20(eef12, grip2)
        assert out[0, 9] == np.float32(0.123)  # L_grip slot
        assert out[0, 19] == np.float32(0.456)  # R_grip slot


class TestQuatXyzwToRot6d:
    def test_identity_quat_returns_identity_rot6d(self):
        # (0, 0, 0, 1) is the identity rotation in xyzw convention.
        out = quat_xyzw_to_rot6d(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(out, [[1, 0, 0, 0, 1, 0]], atol=1e-6)

    def test_pure_z_quat_pi_over_2(self):
        # Quaternion for 90° about Z: w = cos(pi/4), z = sin(pi/4), x = y = 0
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        q = np.array([[0.0, 0.0, s, c]], dtype=np.float64)
        out = quat_xyzw_to_rot6d(q)
        # Same physical rotation as euler [0, 0, pi/2]; same expected rot6d
        np.testing.assert_allclose(out, [[0, 1, 0, -1, 0, 0]], atol=1e-6)

    def test_quat_to_euler_agreement(self):
        # Same rotation expressed two ways → same rot6d
        from scipy.spatial.transform import Rotation as _R

        rng = np.random.RandomState(7)
        T = 20
        euler = rng.randn(T, 3).astype(np.float32)
        quat = _R.from_euler("xyz", euler, degrees=False).as_quat().astype(np.float32)
        out_euler = euler_xyz_to_rot6d(euler)
        out_quat = quat_xyzw_to_rot6d(quat)
        np.testing.assert_allclose(out_euler, out_quat, atol=1e-5)

    def test_first_two_cols_orthonormal(self):
        from scipy.spatial.transform import Rotation as _R

        rng = np.random.RandomState(11)
        T = 30
        q = _R.random(T, random_state=rng).as_quat().astype(np.float32)
        out = quat_xyzw_to_rot6d(q)
        c0 = out[:, :3]
        c1 = out[:, 3:]
        np.testing.assert_allclose(np.linalg.norm(c0, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(c1, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose((c0 * c1).sum(axis=-1), 0.0, atol=1e-5)


class TestAssertUnitQuaternion:
    def test_unit_quats_pass(self):
        q = np.array([[0.0, 0.0, 0.0, 1.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32)
        assert_unit_quaternion(q)  # no raise

    def test_non_unit_raises(self):
        # 0.5 norm is way off
        q = np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
        import pytest

        with pytest.raises(ValueError, match="Quaternion norm check failed"):
            assert_unit_quaternion(q)

    def test_within_tolerance_passes(self):
        # Slightly off but within tol
        q = np.array([[0.0, 0.0, 0.0, 1.01]], dtype=np.float32)
        assert_unit_quaternion(q, tol=0.05)

    def test_empty_array_no_op(self):
        # Edge case: 0-row quat array should not raise
        q = np.zeros((0, 4), dtype=np.float32)
        assert_unit_quaternion(q)


class TestAssembleSingleArm:
    def test_left_slot_filled_right_zero(self):
        arm10 = np.arange(10, dtype=np.float32).reshape(1, 10)
        out = assemble_single_arm_left(arm10)
        assert out.shape == (1, EEF_DIM)
        np.testing.assert_array_equal(out[0, :ARM10_DIM], np.arange(10, dtype=np.float32))
        assert (out[0, ARM10_DIM:] == 0).all()

    def test_right_slot_filled_left_zero(self):
        arm10 = np.arange(10, dtype=np.float32).reshape(1, 10) + 100
        out = assemble_single_arm_right(arm10)
        assert out.shape == (1, EEF_DIM)
        assert (out[0, :ARM10_DIM] == 0).all()
        np.testing.assert_array_equal(out[0, ARM10_DIM:], np.arange(10, dtype=np.float32) + 100)

    def test_preserves_dtype(self):
        for dt in (np.float32, np.float64):
            arm10 = np.ones((3, 10), dtype=dt)
            out = assemble_single_arm_left(arm10)
            assert out.dtype == dt

    def test_preserves_leading_dims(self):
        arm10 = np.zeros((4, 5, 10), dtype=np.float32)
        out = assemble_single_arm_left(arm10)
        assert out.shape == (4, 5, 20)
