from __future__ import annotations

import numpy as np
import pandas as pd

from openwam.dataloader.libero import LiberoDataset
from scripts.convert_libero_to_absolute_eef10_v3 import (
    POSITION_SCALE,
    ROTATION_SCALE,
    convert_state_action,
)


def test_state_and_action_convert_to_one_eef10_contract_with_exact_roundtrip():
    rng = np.random.RandomState(19)
    state = np.zeros((128, 8), dtype=np.float32)
    state[:, 0:3] = rng.uniform([-0.45, -0.3, 0.1], [0.2, 0.4, 1.4], size=(128, 3))
    state[:, 3:6] = rng.uniform(-1.0, 1.0, size=(128, 3))
    state[:, 6] = rng.uniform(0.0, 0.04, size=128)
    state[:, 7] = rng.uniform(-0.04, 0.0, size=128)
    action = rng.uniform(-1.0, 1.0, size=(128, 7)).astype(np.float32)
    action[:, 6] = rng.randint(0, 2, size=128)

    state10, action10, errors = convert_state_action(state, action)

    assert state10.shape == action10.shape == (128, 10)
    np.testing.assert_allclose(
        action10[:, 0:3],
        state10[:, 0:3] + action[:, 0:3] * POSITION_SCALE,
        atol=2e-7,
    )
    np.testing.assert_array_equal(action10[:, 9], 2.0 * action[:, 6] - 1.0)
    assert errors["position"] < 2e-6
    assert errors["rotation"] < 2e-6
    assert errors["gripper"] == 0.0
    assert POSITION_SCALE == 0.05
    assert ROTATION_SCALE == 0.5


def test_absolute_eef10_reader_consumes_state_and_action_row_aligned():
    rng = np.random.RandomState(23)
    state10 = rng.uniform(-1.0, 1.0, size=(6, 10)).astype(np.float32)
    action10 = rng.uniform(-1.0, 1.0, size=(6, 10)).astype(np.float32)
    window = pd.DataFrame(
        {
            "observation.state": list(state10),
            "action": list(action10),
        }
    )
    reader = object.__new__(LiberoDataset)
    reader._normalization_stats = None
    reader._normalize_mode = None

    np.testing.assert_array_equal(reader._raw_state_eef10(window), state10)
    np.testing.assert_array_equal(reader._raw_action_eef10(window), action10)
    np.testing.assert_array_equal(reader._proprio_20d(window), state10[:1])
    assert reader._train_min_window_len() == 1
    assert reader._n_supervised_action_steps(len(window)) == len(window)
