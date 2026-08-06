"""Tests for the OXE schema conversion helpers in ``utils.oxe_schema``.

Pins the per-dataset state/action layouts against the canonical 10-D EEF
representation ``[pos(3) + rot6d(6) + grip(1)]`` consumed by the OXE
readers.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.oxe_schema import (
    bcz_state_to_arm10,
    droid_euler7_to_arm10,
    droid_pose6_closedness_to_arm10,
    droid_state_to_arm10,
    euler7_action_to_arm10,
    fractal_state_to_arm10,
)


class TestBczStateToArm10:
    def test_shape_and_dtype(self):
        state = np.zeros((4, 8), dtype=np.float32)
        out = bcz_state_to_arm10(state)
        assert out.shape == (4, 10)
        assert out.dtype == np.float32

    def test_pad_index_6_dropped(self):
        # state[:, 6] = pad is ignored. Use distinct values for grip vs pad.
        state = np.zeros((1, 8), dtype=np.float32)
        state[0, 6] = 99.0  # pad value — should NOT appear in output
        state[0, 7] = 0.5  # gripper
        out = bcz_state_to_arm10(state)
        # grip slot is the last (index 9)
        assert out[0, 9] == np.float32(0.5)
        # No 99.0 anywhere in the output
        assert not (out == 99.0).any()

    def test_position_passthrough(self):
        state = np.array([[1.5, 2.5, 3.5, 0, 0, 0, 0, 0.7]], dtype=np.float32)
        out = bcz_state_to_arm10(state)
        np.testing.assert_allclose(out[0, :3], [1.5, 2.5, 3.5], atol=1e-6)

    def test_identity_euler_yields_identity_rot6d(self):
        state = np.zeros((1, 8), dtype=np.float32)
        out = bcz_state_to_arm10(state)
        # euler = 0 → rot6d = [1, 0, 0, 0, 1, 0]
        np.testing.assert_allclose(out[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)


class TestEuler7ActionToArm10:
    def test_shape(self):
        action = np.zeros((4, 7), dtype=np.float32)
        out = euler7_action_to_arm10(action)
        assert out.shape == (4, 10)
        assert out.dtype == np.float32

    def test_pos_grip_passthrough(self):
        action = np.array([[0.1, 0.2, 0.3, 0, 0, 0, 0.9]], dtype=np.float32)
        out = euler7_action_to_arm10(action)
        np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
        assert out[0, 9] == np.float32(0.9)


class TestFractalStateToArm10:
    def test_shape(self):
        state = np.zeros((4, 8), dtype=np.float32)
        state[:, 6] = 1.0  # w = 1 → identity quat
        out = fractal_state_to_arm10(state)
        assert out.shape == (4, 10)
        assert out.dtype == np.float32

    def test_identity_quat_yields_identity_rot6d(self):
        # state[3:7] = [0, 0, 0, 1] is identity in xyzw
        state = np.array([[1.0, 2.0, 3.0, 0, 0, 0, 1, 0.5]], dtype=np.float32)
        out = fractal_state_to_arm10(state)
        # position
        np.testing.assert_allclose(out[0, :3], [1, 2, 3], atol=1e-6)
        # rot6d
        np.testing.assert_allclose(out[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
        # gripper
        assert out[0, 9] == np.float32(0.5)

    def test_quat_vs_euler_consistency(self):
        # Build BC-Z-style with euler [0, 0, pi/4]; build Fractal-style with
        # equivalent quaternion. Both should give the same rot6d.
        from scipy.spatial.transform import Rotation as _R

        T = 5
        euler = np.random.RandomState(0).randn(T, 3).astype(np.float32)
        quat = _R.from_euler("xyz", euler).as_quat().astype(np.float32)

        bcz_state = np.zeros((T, 8), dtype=np.float32)
        bcz_state[:, 3:6] = euler
        fractal_state = np.zeros((T, 8), dtype=np.float32)
        fractal_state[:, 3:7] = quat

        bcz_out = bcz_state_to_arm10(bcz_state)
        fractal_out = fractal_state_to_arm10(fractal_state)
        # Rot6d portion should match
        np.testing.assert_allclose(bcz_out[:, 3:9], fractal_out[:, 3:9], atol=1e-5)


class TestDroidStateToArm10:
    def test_shape(self):
        cart = np.zeros((4, 6), dtype=np.float32)
        grip = np.zeros((4, 1), dtype=np.float32)
        out = droid_state_to_arm10(cart, grip)
        assert out.shape == (4, 10)
        assert out.dtype == np.float32

    def test_layout(self):
        cart = np.array([[0.1, 0.2, 0.3, 0, 0, 0]], dtype=np.float32)
        grip = np.array([[0.8]], dtype=np.float32)
        out = droid_state_to_arm10(cart, grip)
        np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(out[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(out[0, 9], 0.2, atol=1e-7)

    def test_scalar_gripper_shape(self):
        # Even if gripper is shape (T, 1), the output works.
        cart = np.zeros((3, 6), dtype=np.float32)
        grip = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
        out = droid_state_to_arm10(cart, grip)
        np.testing.assert_allclose(out[:, 9], [0.9, 0.5, 0.1], atol=1e-6)


class TestDroidEuler7ToArm10:
    def test_closedness_is_inverted_to_canonical_openness(self):
        value = np.zeros((3, 7), dtype=np.float32)
        value[:, 6] = [0.0, 0.25, 1.0]

        out = droid_euler7_to_arm10(value)

        np.testing.assert_allclose(out[:, 9], [1.0, 0.75, 0.0], atol=1e-7)

    def test_pose_conversion_matches_generic_converter(self):
        value = np.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.25]], dtype=np.float32)

        droid = droid_euler7_to_arm10(value)
        generic = euler7_action_to_arm10(value)

        np.testing.assert_allclose(droid[:, :9], generic[:, :9], atol=1e-7)


class TestDroidPose6ClosednessToArm10:
    def test_assembles_arm_side_pose_and_inverts_only_closedness(self):
        pose = np.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=np.float32)
        closedness = np.array([[0.25]], dtype=np.float32)

        out = droid_pose6_closedness_to_arm10(pose, closedness)
        reference = droid_euler7_to_arm10(np.concatenate([pose, closedness], axis=-1))

        np.testing.assert_allclose(out, reference, atol=1e-7)
        np.testing.assert_allclose(out[:, :3], pose[:, :3], atol=1e-7)
        np.testing.assert_allclose(out[:, 9], [0.75], atol=1e-7)
