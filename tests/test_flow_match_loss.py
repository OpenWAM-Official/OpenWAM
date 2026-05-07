"""Tests for video/action loss helpers on BaseWAMArchitecture.

These tests exercise architecture._compute_video_loss and
architecture._compute_action_loss, which replaced the standalone
FlowMatchVideoActionLoss class.
"""

import pytest
import torch


def _make_arch():
    """Build a minimal architecture with a mock video backbone for loss tests."""
    from tests.test_openwam_trainer import _make_tiny_arch

    return _make_tiny_arch()


class _MockScheduler:
    num_train_timesteps = 1000
    linear_timesteps_weights = torch.ones(1000)
    timesteps = torch.linspace(0, 1, 1000)
    sigmas = torch.linspace(1, 0, 1000)

    def add_noise(self, original, noise, sigma):
        return (1 - sigma) * original + sigma * noise

    def training_target(self, original, noise):
        return noise - original

    def training_weight(self, timestep_ids):
        return self.linear_timesteps_weights[timestep_ids]

    def flow_step(self, pred, sigma, sigma_next, sample):
        return sample + pred * (sigma_next - sigma)


def test_video_loss_basic():
    """Video loss produces a positive scalar."""
    arch = _make_arch()
    noise_pred = torch.randn(2, 16, 5, 4, 4)
    target = torch.randn(2, 16, 5, 4, 4)
    timestep_ids = torch.tensor([10, 20])

    loss = arch._compute_video_loss(noise_pred, target, timestep_ids, {}, device="cpu")
    assert loss.shape == ()
    assert loss.item() > 0


def test_action_loss_basic():
    """Action loss produces a positive scalar."""
    arch = _make_arch()
    noise_pred = torch.randn(2, 49, 14)
    target = torch.randn(2, 49, 14)
    timestep_ids = torch.tensor([5, 15])

    loss = arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs={}, device="cpu")
    assert loss.shape == ()
    assert loss.item() > 0


def test_action_loss_single_sample():
    """Loss computation should handle B=1."""
    arch = _make_arch()
    noise_pred = torch.randn(1, 49, 14)
    target = torch.randn(1, 49, 14)
    timestep_ids = torch.tensor([10])

    loss = arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs={}, device="cpu")
    assert loss.shape == ()


def test_action_loss_per_token_timestep():
    """_compute_action_loss rejects (B, T) timestep_ids in FastWAM-compatible mode."""
    arch = _make_arch()
    B, T, action_dim = 2, 49, 14
    noise_pred = torch.randn(B, T, action_dim)
    target = torch.randn(B, T, action_dim)
    timestep_ids = torch.randint(0, 1000, (B, T))

    with pytest.raises(ValueError, match="per-sample"):
        arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs={}, device="cpu")


def test_action_loss_per_token_timestep_with_pad():
    """Per-token timestep is rejected even when action padding is present."""
    arch = _make_arch()
    B, T, action_dim = 2, 10, 4
    noise_pred = torch.randn(B, T, action_dim)
    target = torch.randn(B, T, action_dim)
    timestep_ids = torch.randint(0, 1000, (B, T))
    action_is_pad = torch.zeros(B, T, dtype=torch.bool)
    action_is_pad[0, 7:] = True
    action_is_pad[1, 4:] = True

    inputs = {"action_is_pad": action_is_pad}
    with pytest.raises(ValueError, match="per-sample"):
        arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs=inputs, device="cpu")
