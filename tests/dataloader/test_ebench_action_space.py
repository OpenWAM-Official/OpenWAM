import numpy as np

from openwam.dataloader.ebench import (
    EBENCH80_DIM_MASK,
    _ee_pose_gripper_base_to_raw23,
    _raw23_to_ebench80,
)


def test_raw23_to_ebench80_mapping():
    raw = np.arange(23, dtype=np.float32)
    mapped = _raw23_to_ebench80(raw)

    assert mapped.shape == (80,)
    np.testing.assert_allclose(mapped[0:10], raw[0:10])
    np.testing.assert_allclose(mapped[34:44], raw[10:20])
    np.testing.assert_allclose(mapped[68:71], raw[20:23])

    assert int(EBENCH80_DIM_MASK.sum()) == 23
    assert np.all(mapped[~EBENCH80_DIM_MASK] == 0.0)


def test_ee_pose_gripper_base_to_raw23_uses_rot6d_and_scalar_grippers():
    ee_pose = np.zeros((1, 14), dtype=np.float32)
    ee_pose[0, 0:3] = [0.1, 0.2, 0.3]
    ee_pose[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    ee_pose[0, 7:10] = [0.4, 0.5, 0.6]
    ee_pose[0, 10:14] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    gripper = np.array([[1.0, 3.0, 5.0, 7.0]], dtype=np.float32)
    base = np.array([[0.7, 0.8, 0.9]], dtype=np.float32)

    raw = _ee_pose_gripper_base_to_raw23(ee_pose, gripper, base)

    assert raw.shape == (1, 23)
    np.testing.assert_allclose(raw[0, 0:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(raw[0, 3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert raw[0, 9] == 2.0
    np.testing.assert_allclose(raw[0, 10:13], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(raw[0, 13:19], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert raw[0, 19] == 6.0
    np.testing.assert_allclose(raw[0, 20:23], [0.7, 0.8, 0.9])
