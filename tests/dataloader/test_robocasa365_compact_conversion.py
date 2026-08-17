from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.convert_robocasa365_compact_v3 import (
    ACTION_DIM,
    POSITION_SCALE,
    ROTATION_SCALE,
    SHARED_EEF_STATS_DIM,
    STATE_DIM,
    StatsAccumulator,
    _shared_eef_rows,
    _stats_payload,
    convert_state_action,
    rot6d_to_matrix,
)


def test_same_row_native_command_converts_to_state19_action15_and_roundtrips():
    state = np.zeros((2, 16), np.float64)
    state[:, 0:3] = [[1, 2, 0], [9, 8, 0]]
    state[:, 3:7] = [0, 0, 0, 1]
    state[:, 7:10] = [[0.1, 0.2, 0.3], [5.0, 6.0, 7.0]]
    state[:, 10:14] = [0, 0, 0, 1]
    state[:, 14:16] = [[0.04, -0.04], [0.0, 0.0]]
    action = np.zeros((2, 12), np.float64)
    action[:, 0:5] = [[0.2, -0.3, 0.4, 0.7, -1], [-0.2, 0.1, 0.0, 0.0, 1]]
    action[:, 5:8] = [[0.5, -0.25, 0.1], [-1.0, 1.0, 0.0]]
    action[:, 8:11] = [[0.0, 0.0, 0.2], [0.1, -0.2, 0.3]]
    action[:, 11] = [-1, 1]

    state19, action15, errors = convert_state_action(state, action)
    assert state19.shape == (2, STATE_DIM)
    assert action15.shape == (2, ACTION_DIM)
    np.testing.assert_allclose(action15[:, :3], state[:, 7:10] + POSITION_SCALE * action[:, 5:8], atol=1e-7)
    # Row zero must be based on state row zero, not the very different state row one.
    assert action15[0, 0] < 1.0
    expected_rotation = Rotation.from_rotvec(ROTATION_SCALE * action[:, 8:11]).as_matrix()
    np.testing.assert_allclose(rot6d_to_matrix(action15[:, 3:9]), expected_rotation, atol=1e-6)
    np.testing.assert_allclose(action15[:, 9], -action[:, 11])
    np.testing.assert_allclose(action15[:, 10:15], action[:, 0:5])
    assert max(errors.values()) < 1e-5


def test_state_contains_eef_then_world_base_pose():
    state = np.zeros((1, 16), np.float64)
    state[0, 0:3] = [1, 2, 3]
    state[0, 3:7] = [0, 0, 0, 1]
    state[0, 7:10] = [0.4, 0.5, 0.6]
    state[0, 10:14] = [0, 0, 0, 1]
    state[0, 14:16] = [0.05, 0.0]
    state19, _, _ = convert_state_action(state, np.zeros((1, 12), np.float64))
    np.testing.assert_allclose(state19[0, :3], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(state19[0, 3:9], [1, 0, 0, 0, 1, 0])
    assert state19[0, 9] == 0.0
    np.testing.assert_allclose(state19[0, 10:13], [1, 2, 3])
    np.testing.assert_allclose(state19[0, 13:19], [1, 0, 0, 0, 1, 0])


def test_stats_pool_eef_but_keep_base_directional_and_rot6d_identity():
    state = np.zeros((2, STATE_DIM), np.float32)
    action = np.zeros((2, ACTION_DIM), np.float32)
    state[:, 0:3] = [[-2.0, 1.0, 3.0], [2.0, 5.0, 7.0]]
    action[:, 0:3] = [[-4.0, 0.0, 4.0], [4.0, 6.0, 8.0]]
    state[:, 9] = [-0.5, 0.5]
    action[:, 9] = [-1.0, 1.0]
    state[:, 10:13] = [[100.0, 200.0, 300.0], [110.0, 220.0, 330.0]]
    action[:, 10:15] = [[-0.2, -0.3, -0.4, 0.0, -1.0], [0.2, 0.3, 0.4, 0.7, 1.0]]

    action_acc = StatsAccumulator(ACTION_DIM, seed=1)
    state_acc = StatsAccumulator(STATE_DIM, seed=2)
    shared_acc = StatsAccumulator(SHARED_EEF_STATS_DIM, seed=3)
    action_acc.update(action)
    state_acc.update(state)
    shared_acc.update(_shared_eef_rows(state, action))
    payload = _stats_payload(action_acc, state_acc, shared_acc)
    action_stats = payload["robocasa365"]
    state_stats = payload["robocasa365_state"]

    # Achieved state and commanded action share one EEF xyz+gripper transform.
    np.testing.assert_allclose(action_stats["min"][:3], [-4.0, 0.0, 3.0])
    np.testing.assert_allclose(action_stats["max"][:3], [4.0, 6.0, 8.0])
    np.testing.assert_allclose(state_stats["min"][:3], action_stats["min"][:3])
    np.testing.assert_allclose(state_stats["max"][:3], action_stats["max"][:3])
    assert action_stats["min"][9] == state_stats["min"][9] == -1.0
    assert action_stats["max"][9] == state_stats["max"][9] == 1.0

    # All Rot6D slots are pass-through under min-max; no other dim is pinned.
    np.testing.assert_allclose(action_stats["min"][3:9], -1.0)
    np.testing.assert_allclose(action_stats["max"][3:9], 1.0)
    np.testing.assert_allclose(state_stats["min"][3:9], -1.0)
    np.testing.assert_allclose(state_stats["max"][3:9], 1.0)
    np.testing.assert_allclose(state_stats["min"][13:19], -1.0)
    np.testing.assert_allclose(state_stats["max"][13:19], 1.0)
    np.testing.assert_allclose(action_stats["min"][10:15], [-0.2, -0.3, -0.4, 0.0, -1.0])
    np.testing.assert_allclose(action_stats["max"][10:15], [0.2, 0.3, 0.4, 0.7, 1.0])
    np.testing.assert_allclose(state_stats["min"][10:13], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(state_stats["max"][10:13], [110.0, 220.0, 330.0])
