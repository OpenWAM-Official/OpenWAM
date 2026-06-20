"""End-to-end tests of LeRobot3VideoActionModel._compute_action_loss.

Verifies that the actual method implementation (not a re-implementation)
behaves identically to the legacy formula when handed an equivalent 1-D
vs 2-D action_is_pad. Uses a thin stub that supplies just the scheduler
+ device arguments the method needs; no diffusion model is constructed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _SchedulerStub:
    def training_weight(self, timestep_ids):
        return torch.ones_like(timestep_ids, dtype=torch.float32)


class _ModelStub:
    """Subset of LeRobot3VideoActionModel just sufficient to call
    ``_compute_action_loss``."""

    # Pull in the actual implementation under test.
    from openwam.model.architectures.base import BaseWAMArchitecture

    _compute_action_loss = BaseWAMArchitecture._compute_action_loss


def _make_inputs(B, T, D, mask_1d_or_2d=None):
    """Build minimal kwargs for _compute_action_loss."""
    return {
        "noise_pred": torch.randn(B, T, D),
        "target": torch.randn(B, T, D),
        "timestep_ids": torch.randint(0, 1000, (B,)),
        "scheduler": _SchedulerStub(),
        "inputs": {"action_is_pad": mask_1d_or_2d} if mask_1d_or_2d is not None else {},
        "device": "cpu",
    }


class TestComputeActionLoss:
    def _legacy_oracle(self, pred, target, pad_1d, tw):
        """Replica of the pre-migration legacy formula for comparison."""
        per_element = F.mse_loss(pred.float(), target.float(), reduction="none")
        per_step = per_element.mean(dim=2)
        valid = (~pad_1d).float()
        per_step = per_step * valid
        valid_count = valid.sum(dim=1).clamp(min=1)
        per_sample = per_step.sum(dim=1) / valid_count
        return (per_sample * tw).mean()

    def test_no_mask_equivalent_to_legacy(self):
        torch.manual_seed(0)
        B, T, D = 3, 8, 20
        kw = _make_inputs(B, T, D)
        m = _ModelStub()
        l = m._compute_action_loss(**kw)
        oracle = kw["noise_pred"].float().sub(kw["target"].float()).pow(2).mean(dim=(1, 2)).mean()
        assert torch.allclose(l, oracle, atol=1e-7)

    def test_2d_full_true_equivalent_to_legacy_full_true(self):
        torch.manual_seed(1)
        B, T, D = 3, 8, 20
        # Use same pred/target for both paths
        torch.manual_seed(1)
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        ts = torch.randint(0, 1000, (B,))

        pad_1d = torch.zeros(B, T, dtype=torch.bool)
        pad_2d = torch.zeros(B, T, D, dtype=torch.bool)

        m = _ModelStub()
        # Path with 1D mask
        kw_1d = {
            "noise_pred": pred,
            "target": target,
            "timestep_ids": ts,
            "scheduler": _SchedulerStub(),
            "inputs": {"action_is_pad": pad_1d},
            "device": "cpu",
        }
        l_1d = m._compute_action_loss(**kw_1d)
        # Path with 2D mask
        kw_2d = dict(kw_1d, inputs={"action_is_pad": pad_2d})
        l_2d = m._compute_action_loss(**kw_2d)

        assert torch.allclose(l_1d, l_2d, atol=1e-7), f"1D vs 2D loss diverged: {float(abs(l_1d - l_2d))}"

    def test_2d_broadcast_time_equivalent_to_1d_time(self):
        torch.manual_seed(2)
        B, T, D = 4, 10, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        ts = torch.randint(0, 1000, (B,))

        # Mixed time-only pad
        pad_1d = torch.tensor(
            [
                [False] * 7 + [True] * 3,
                [False] * 5 + [True] * 5,
                [False] * 10,
                [False] * 1 + [True] * 9,
            ]
        )
        pad_2d = pad_1d.unsqueeze(-1).expand(-1, -1, D).contiguous()

        m = _ModelStub()
        kw = {
            "noise_pred": pred,
            "target": target,
            "timestep_ids": ts,
            "scheduler": _SchedulerStub(),
            "inputs": {"action_is_pad": pad_1d},
            "device": "cpu",
        }
        l_1d = m._compute_action_loss(**kw)
        kw_2d = dict(kw, inputs={"action_is_pad": pad_2d})
        l_2d = m._compute_action_loss(**kw_2d)
        assert torch.allclose(l_1d, l_2d, atol=1e-6), (
            f"1D-time vs 2D-broadcast loss diverged: {float(abs(l_1d - l_2d))}"
        )

    def test_2d_per_dim_mask_changes_loss(self):
        """Single-arm (front-10-True / back-10-False) mask gives different
        loss vs all-True; this is the OXE training contract."""
        torch.manual_seed(3)
        B, T, D = 4, 8, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        ts = torch.randint(0, 1000, (B,))

        dim_pad = torch.cat([torch.zeros(10, dtype=torch.bool), torch.ones(10, dtype=torch.bool)])
        pad_2d = dim_pad.unsqueeze(0).unsqueeze(0).expand(B, T, -1).contiguous()

        m = _ModelStub()
        l_partial = m._compute_action_loss(
            noise_pred=pred,
            target=target,
            timestep_ids=ts,
            scheduler=_SchedulerStub(),
            inputs={"action_is_pad": pad_2d},
            device="cpu",
        )
        l_full = m._compute_action_loss(
            noise_pred=pred,
            target=target,
            timestep_ids=ts,
            scheduler=_SchedulerStub(),
            inputs={"action_is_pad": torch.zeros(B, T, D, dtype=torch.bool)},
            device="cpu",
        )
        assert torch.isfinite(l_partial)
        # Different distribution over dims → different loss
        assert not torch.allclose(l_partial, l_full, atol=1e-5)

    def test_2d_all_false_mask_returns_zero(self):
        torch.manual_seed(4)
        B, T, D = 2, 4, 20
        m = _ModelStub()
        l = m._compute_action_loss(
            noise_pred=torch.randn(B, T, D),
            target=torch.randn(B, T, D),
            timestep_ids=torch.randint(0, 1000, (B,)),
            scheduler=_SchedulerStub(),
            inputs={"action_is_pad": torch.ones(B, T, D, dtype=torch.bool)},
            device="cpu",
        )
        assert torch.allclose(l, torch.tensor(0.0), atol=1e-7)
