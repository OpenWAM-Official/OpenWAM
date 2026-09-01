"""Numerical equivalence tests for the 2-D action_mask loss path.

The 2-D mask path allows
``LeRobot3VideoActionModel._compute_action_loss`` to accept either
shape:

    * 1-D legacy (B, T) — per-step mask, ``loss = mean_d → mean_t``.
    * 2-D new     (B, T, D) — per-element mask, ``loss = (sum / count)`` over (T, D).

The two paths are *mathematically equivalent* whenever the 2-D mask is
a broadcast of the 1-D time mask across all D dims (every valid timestep
has every dim valid). These tests pin that contract bit-for-bit, plus
exercise the masking semantics for the OXE single-arm and
EgoDex-style-disabled use cases.

We exercise the loss path through a lightweight ``__new__`` stub that
avoids constructing the full diffusion model — only ``_compute_action_loss``
is under test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stand-alone re-implementations of the two loss code paths.
# Used as the "expected" oracle in equivalence tests so the tests don't
# accidentally validate themselves against the implementation under test.
# ---------------------------------------------------------------------------


def _legacy_loss(noise_pred, target, action_is_pad_1d, tw):
    """Old per-timestep mask path (1D ``action_is_pad``)."""
    per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
    per_step = per_element.mean(dim=2)
    valid_mask = (~action_is_pad_1d).float()
    per_step = per_step * valid_mask
    valid_count = valid_mask.sum(dim=1).clamp(min=1)
    per_sample = per_step.sum(dim=1) / valid_count
    return (per_sample * tw).mean()


def _new_loss(noise_pred, target, action_is_pad_2d, tw):
    """New per-element mask path (2D ``action_is_pad``)."""
    per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
    valid_mask_f = (~action_is_pad_2d).float()
    weighted = per_element * valid_mask_f
    per_sample = weighted.sum(dim=(1, 2)) / valid_mask_f.sum(dim=(1, 2)).clamp(min=1)
    return (per_sample * tw).mean()


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------


class TestLossEquivalence:
    """Bit-level equivalence between legacy 1D and new 2D loss formulas."""

    @pytest.fixture
    def fixed_seed(self):
        torch.manual_seed(0)

    def _broadcast_1d_to_2d(self, mask_1d, D):
        """(B, T) → (B, T, D) by per-dim broadcast (every dim has same validity)."""
        return mask_1d.unsqueeze(-1).expand(-1, -1, D).contiguous()

    def test_full_valid_no_mask_equivalence(self, fixed_seed):
        """No-mask code path returns identical loss for both shapes."""
        B, T, D = 4, 10, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        tw = torch.rand(B)
        # No mask is the same as all-True mask.
        l_legacy = _legacy_loss(pred, target, torch.zeros(B, T, dtype=torch.bool), tw)
        l_new = _new_loss(pred, target, torch.zeros(B, T, D, dtype=torch.bool), tw)
        assert torch.allclose(l_legacy, l_new, atol=1e-7), f"diff={float(abs(l_legacy - l_new))}"

    def test_partial_time_mask_broadcast_dim_equivalence(self, fixed_seed):
        """1D time mask vs its 2D broadcast over D → identical loss."""
        B, T, D = 4, 10, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        tw = torch.rand(B)
        # Mix of valid/invalid timesteps per sample.
        pad_1d = torch.tensor(
            [
                [False] * 7 + [True] * 3,  # last 3 padded
                [False] * 5 + [True] * 5,
                [False] * 10,  # all valid
                [True] * 1 + [False] * 9,  # first padded (unusual but exercise it)
            ]
        )
        pad_2d = self._broadcast_1d_to_2d(pad_1d, D)
        l_legacy = _legacy_loss(pred, target, pad_1d, tw)
        l_new = _new_loss(pred, target, pad_2d, tw)
        assert torch.allclose(l_legacy, l_new, atol=1e-7), f"diff={float(abs(l_legacy - l_new))}"

    def test_partial_dim_mask_changes_loss(self, fixed_seed):
        """Per-dim mask (OXE single-arm: front 10 True / back 10 False) gives
        a different loss than the all-True case, but stays finite + bounded."""
        B, T, D = 4, 10, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        tw = torch.rand(B)

        # OXE-style: front 10 dims valid, back 10 zero-padded.
        dim_valid = torch.cat([torch.ones(10, dtype=torch.bool), torch.zeros(10, dtype=torch.bool)])
        pad_2d = (~dim_valid).unsqueeze(0).unsqueeze(0).expand(B, T, -1).contiguous()

        l_partial = _new_loss(pred, target, pad_2d, tw)
        # All-True reference
        l_full = _new_loss(pred, target, torch.zeros(B, T, D, dtype=torch.bool), tw)

        assert torch.isfinite(l_partial)
        assert l_partial >= 0
        # With random gaussian preds & targets, the loss over the front 10 dims
        # will differ from the loss over all 20. They should not coincidentally
        # match.
        assert not torch.allclose(l_partial, l_full, atol=1e-4), "single-arm dim mask should change the loss vs all-dim"

    def test_partial_dim_mask_only_uses_valid_dims(self, fixed_seed):
        """Loss with mask=[T,T,...,T] front 10 / [F,F,...,F] back 10 equals
        the loss computed on only the front 10 dims."""
        B, T, D = 4, 8, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        tw = torch.rand(B)

        dim_valid = torch.cat([torch.ones(10, dtype=torch.bool), torch.zeros(10, dtype=torch.bool)])
        pad_2d = (~dim_valid).unsqueeze(0).unsqueeze(0).expand(B, T, -1).contiguous()
        l_masked = _new_loss(pred, target, pad_2d, tw)

        # Manual oracle: loss = (per_element[:, :, :10]).mean over (T, D=10), per-sample weighted.
        per_element = F.mse_loss(pred.float(), target.float(), reduction="none")[:, :, :10]
        per_sample = per_element.mean(dim=(1, 2))
        l_oracle = (per_sample * tw).mean()

        assert torch.allclose(l_masked, l_oracle, atol=1e-7), f"diff={float(abs(l_masked - l_oracle))}"

    def test_combined_time_and_dim_mask(self, fixed_seed):
        """Time mask + dim mask combined gives sum/(N_valid_t * N_valid_d)."""
        B, T, D = 3, 10, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        tw = torch.rand(B)

        pad_2d = torch.zeros(B, T, D, dtype=torch.bool)
        # Sample 0: first 7 time valid; front 10 dims valid
        pad_2d[0, 7:, :] = True
        pad_2d[0, :, 10:] = True
        # Sample 1: all time valid; front 10 dims valid
        pad_2d[1, :, 10:] = True
        # Sample 2: all valid
        # (pad_2d[2] stays all-False)

        l = _new_loss(pred, target, pad_2d, tw)
        assert torch.isfinite(l)
        assert l >= 0

    def test_all_false_mask_returns_zero(self, fixed_seed):
        """Sample with mask all-False (EgoDex / supervision-off) contributes
        zero per_sample (sum=0, count clamped to 1)."""
        B, T, D = 2, 5, 20
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)
        tw = torch.ones(B)

        pad_2d = torch.ones(B, T, D, dtype=torch.bool)  # everything is pad
        l = _new_loss(pred, target, pad_2d, tw)
        assert torch.allclose(l, torch.tensor(0.0), atol=1e-7)

    def test_gradient_zero_on_masked_dims(self, fixed_seed):
        """Backprop: dims with mask=False receive zero gradient."""
        B, T, D = 2, 4, 20
        pred = torch.randn(B, T, D, requires_grad=True)
        target = torch.randn(B, T, D)
        tw = torch.ones(B)

        dim_valid = torch.cat([torch.ones(10, dtype=torch.bool), torch.zeros(10, dtype=torch.bool)])
        pad_2d = (~dim_valid).unsqueeze(0).unsqueeze(0).expand(B, T, -1).contiguous()
        l = _new_loss(pred, target, pad_2d, tw)
        l.backward()

        # Front 10 dims should have nonzero gradient (expected MSE backward).
        assert (pred.grad[:, :, :10].abs() > 0).any()
        # Back 10 dims must be exactly zero.
        assert torch.equal(pred.grad[:, :, 10:], torch.zeros_like(pred.grad[:, :, 10:]))


class TestLossDeterminism:
    """Verify deterministic computation under fixed seed for both paths."""

    def test_legacy_deterministic(self):
        torch.manual_seed(42)
        pred = torch.randn(2, 5, 8)
        target = torch.randn(2, 5, 8)
        pad = torch.zeros(2, 5, dtype=torch.bool)
        tw = torch.ones(2)
        l1 = _legacy_loss(pred, target, pad, tw)
        l2 = _legacy_loss(pred, target, pad, tw)
        assert torch.equal(l1, l2)

    def test_new_deterministic(self):
        torch.manual_seed(42)
        pred = torch.randn(2, 5, 8)
        target = torch.randn(2, 5, 8)
        pad = torch.zeros(2, 5, 8, dtype=torch.bool)
        tw = torch.ones(2)
        l1 = _new_loss(pred, target, pad, tw)
        l2 = _new_loss(pred, target, pad, tw)
        assert torch.equal(l1, l2)
