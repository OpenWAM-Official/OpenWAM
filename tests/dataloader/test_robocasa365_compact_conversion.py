from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.convert_robocasa365_compact_v3 import (
    ACTION_DIM,
    STATE_DIM,
    StatsAccumulator,
    _stats_payload,
    convert_state_action,
    rot6d_to_matrix,
)


def test_native_delta_action_is_direct_and_roundtrips_without_controller_scale():
    state = np.zeros((2, 16), np.float64)
    state[:, 0:3] = [[1, 2, 0], [9, 8, 0]]
    state[:, 3:7] = [0, 0, 0, 1]
    state[:, 7:10] = [[0.1, 0.2, 0.3], [5.0, 6.0, 7.0]]
    state[:, 10:14] = [0, 0, 0, 1]
    state[:, 14:16] = [[0.04, -0.04], [0.0, 0.0]]
    action = np.zeros((2, 12), np.float64)
    action[:, 0:5] = [[0.2, -0.3, 0.4, 0.7, -1], [-0.2, 0.1, 0.0, 0.0, 1]]
    action[:, 5:8] = [[0.5, -0.25, 0.1], [-1.0, 1.0, 0.0]]
    action[:, 8:11] = [[0.0, 0.0, 0.8], [0.1, -0.2, 0.3]]
    action[:, 11] = [-1, 1]

    state19, action15, errors = convert_state_action(state, action)

    assert state19.shape == (2, STATE_DIM)
    assert action15.shape == (2, ACTION_DIM)
    np.testing.assert_array_equal(action15[:, 0:3], action[:, 5:8].astype(np.float32))
    np.testing.assert_allclose(
        rot6d_to_matrix(action15[:, 3:9]),
        Rotation.from_rotvec(action[:, 8:11]).as_matrix(),
        atol=1e-6,
    )
    # A hidden 0.5 scale would produce a 0.4-rad first rotation, not 0.8 rad.
    np.testing.assert_allclose(
        np.linalg.norm(Rotation.from_matrix(rot6d_to_matrix(action15[0, 3:9])).as_rotvec()),
        0.8,
        atol=1e-6,
    )
    np.testing.assert_array_equal(action15[:, 9], (-action[:, 11]).astype(np.float32))
    np.testing.assert_array_equal(action15[:, 10:15], action[:, 0:5].astype(np.float32))
    assert max(errors.values()) < 3e-6


def test_state_conversion_keeps_xyzw_base_quaternion_contract():
    state = np.zeros((1, 16), np.float64)
    state[0, 0:3] = [1, 2, 3]
    state[0, 3:7] = [0, 0, 0, 1]
    state[0, 7:10] = [0.4, 0.5, 0.6]
    state[0, 10:14] = [0, 0, 0, 1]
    state19, _, _ = convert_state_action(state, np.zeros((1, 12), np.float64))
    np.testing.assert_allclose(state19[0, 13:19], [1, 0, 0, 0, 1, 0])


def test_delta_action_and_achieved_state_statistics_are_not_pooled():
    state = np.zeros((2, STATE_DIM), np.float32)
    action = np.zeros((2, ACTION_DIM), np.float32)
    state[:, 0:3] = [[10, 20, 30], [11, 21, 31]]
    action[:, 0:3] = [[-1, -0.5, 0], [1, 0.5, 0.25]]

    action_acc = StatsAccumulator(ACTION_DIM, seed=1)
    state_acc = StatsAccumulator(STATE_DIM, seed=2)
    action_acc.update(action)
    state_acc.update(state)
    payload = _stats_payload(action_acc, state_acc)

    np.testing.assert_allclose(payload["robocasa365"]["min"][:3], [-1, -0.5, 0])
    np.testing.assert_allclose(payload["robocasa365_state"]["min"][:3], [10, 20, 30])
    np.testing.assert_allclose(payload["robocasa365"]["min"][3:9], -1)
    np.testing.assert_allclose(payload["robocasa365_state"]["min"][3:9], -1)
