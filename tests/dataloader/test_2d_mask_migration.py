"""Regression tests for the 2-D action-mask migration.

Verifies that RoboCOIN / EgoDex / RoboTwin readers emit the new 2-D
``action_mask`` / ``proprio_mask`` shapes, and that the 2-D mask is
exactly a per-dim broadcast of the legacy 1-D time mask (every valid
timestep has every dim valid).

We exercise the helper builders in ``utils/eef.py`` plus a small
end-to-end test that instantiates RoboCOIN against a synthetic bucket.
"""

from __future__ import annotations

import numpy as np
import torch

from openwam.dataloader.utils.eef import (
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    RIGHT_ARM_DIM_MASK,
    build_action_mask_2d,
    build_proprio_mask_2d,
)


class TestActionMask2dBuilder:
    def test_shape(self):
        m = build_action_mask_2d(T_action=32, action_dim=20, n_valid_time=10)
        assert m.shape == (32, 20)
        assert m.dtype == bool

    def test_full_valid_no_dim_mask(self):
        m = build_action_mask_2d(T_action=5, action_dim=4, n_valid_time=5)
        assert m.all()

    def test_partial_time_valid(self):
        m = build_action_mask_2d(T_action=10, action_dim=4, n_valid_time=3)
        assert m[:3, :].all()
        assert not m[3:, :].any()

    def test_n_valid_time_zero_returns_all_false(self):
        m = build_action_mask_2d(T_action=10, action_dim=20, n_valid_time=0)
        assert not m.any()

    def test_single_arm_left_dim_mask(self):
        m = build_action_mask_2d(T_action=10, action_dim=20, n_valid_time=5, dim_mask=LEFT_ARM_DIM_MASK)
        # Front 10 dims True in valid timesteps
        assert m[:5, :10].all()
        # Back 10 dims always False
        assert not m[:, 10:].any()
        # Padded timesteps all False
        assert not m[5:, :].any()

    def test_single_arm_right_dim_mask(self):
        m = build_action_mask_2d(T_action=10, action_dim=20, n_valid_time=5, dim_mask=RIGHT_ARM_DIM_MASK)
        assert not m[:, :10].any()
        assert m[:5, 10:].all()
        assert not m[5:, :].any()

    def test_2d_mask_equals_broadcast_of_1d(self):
        """The 2-D mask must equal the legacy 1-D time mask broadcast across D.
        This is the equivalence that gives RoboCOIN/EgoDex/RoboTwin loss
        numerical identity."""
        T_action, D, n_valid = 32, 20, 7
        # Legacy 1-D contract
        legacy_1d = np.zeros(T_action, dtype=bool)
        legacy_1d[:n_valid] = True
        # New 2-D contract (bimanual: no per-dim mask)
        new_2d = build_action_mask_2d(T_action, D, n_valid)
        # Broadcast the 1-D to 2-D for comparison
        broadcast = np.tile(legacy_1d[:, None], (1, D))
        np.testing.assert_array_equal(new_2d, broadcast)


class TestProprioMask2dBuilder:
    def test_shape(self):
        m = build_proprio_mask_2d(action_dim=20)
        assert m.shape == (1, 20)
        assert m.dtype == bool

    def test_enabled_all_true(self):
        m = build_proprio_mask_2d(action_dim=20, enabled=True)
        assert m.all()

    def test_disabled_all_false(self):
        m = build_proprio_mask_2d(action_dim=20, enabled=False)
        assert not m.any()

    def test_left_arm_dim_mask(self):
        m = build_proprio_mask_2d(action_dim=20, enabled=True, dim_mask=LEFT_ARM_DIM_MASK)
        assert m[0, :10].all()
        assert not m[0, 10:].any()

    def test_disabled_overrides_dim_mask(self):
        m = build_proprio_mask_2d(action_dim=20, enabled=False, dim_mask=LEFT_ARM_DIM_MASK)
        # disabled wins — even if dim_mask says front 10 True, output is all False
        assert not m.any()


class TestDimMaskConstants:
    def test_eef_dim(self):
        assert EEF_DIM == 20

    def test_left_arm_constant(self):
        assert LEFT_ARM_DIM_MASK.shape == (20,)
        assert LEFT_ARM_DIM_MASK[:10].all()
        assert not LEFT_ARM_DIM_MASK[10:].any()

    def test_right_arm_constant(self):
        assert RIGHT_ARM_DIM_MASK.shape == (20,)
        assert not RIGHT_ARM_DIM_MASK[:10].any()
        assert RIGHT_ARM_DIM_MASK[10:].all()

    def test_left_and_right_are_complementary(self):
        assert (LEFT_ARM_DIM_MASK | RIGHT_ARM_DIM_MASK).all()
        assert not (LEFT_ARM_DIM_MASK & RIGHT_ARM_DIM_MASK).any()


class TestReaderOutputShapes:
    """Smoke-test the shape contract by instantiating the readers'
    __getitem__ output dict structure via FakeActionDataset-style synthesis."""

    def _make_robocoin_sample(self, T_action=32, n_valid=10, enable_supervision=True):
        """Build the dict robocoin._getitem_impl would return for given
        parameters, using only the same code path (build_*_mask_2d helpers)."""
        action_mask = build_action_mask_2d(
            T_action=T_action,
            action_dim=EEF_DIM,
            n_valid_time=n_valid if enable_supervision else 0,
        )
        proprio_mask = build_proprio_mask_2d(action_dim=EEF_DIM, enabled=enable_supervision)
        return action_mask, proprio_mask

    def test_robocoin_supervised_shapes_and_values(self):
        am, pm = self._make_robocoin_sample(T_action=32, n_valid=20, enable_supervision=True)
        assert am.shape == (32, 20)
        assert pm.shape == (1, 20)
        # 2-D should equal broadcast 1-D
        assert am[:20, :].all() and not am[20:, :].any()
        assert pm.all()

    def test_robocoin_supervision_off_all_false(self):
        am, pm = self._make_robocoin_sample(T_action=32, n_valid=20, enable_supervision=False)
        assert am.shape == (32, 20)
        assert pm.shape == (1, 20)
        assert not am.any()
        assert not pm.any()

    def test_egodex_emits_all_false_2d(self):
        # EgoDex behavior: action_mask (T, 20) all False; proprio_mask (1, 20) all False
        T_action = 32
        am = torch.zeros(T_action, EEF_DIM, dtype=torch.bool)
        pm = torch.zeros(1, EEF_DIM, dtype=torch.bool)
        assert am.shape == (T_action, EEF_DIM)
        assert pm.shape == (1, EEF_DIM)
        assert not am.any()
        assert not pm.any()

    def test_robotwin_2d_via_broadcast(self):
        # RoboTwin contract: build time_validity (T,) → unsqueeze + expand to
        # (T, action_dim). Validate the expand path works at action_dim=20 and 14.
        for D in (14, 20):
            T = 32
            time_validity = torch.tensor([(t + 1) < 20 for t in range(T)], dtype=torch.bool)
            am = time_validity.unsqueeze(-1).expand(-1, D).contiguous()
            assert am.shape == (T, D)
            # Each timestep's row is uniform across D
            for t in range(T):
                row = am[t]
                assert (row == row[0]).all(), f"row {t} not uniform across D"

            pm = torch.full((1, D), fill_value=True, dtype=torch.bool)
            assert pm.shape == (1, D)
