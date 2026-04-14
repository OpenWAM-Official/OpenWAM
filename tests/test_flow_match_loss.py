"""Tests for standalone FlowMatchVideoActionLoss."""

import torch


def test_flow_match_loss_importable():
    """FlowMatchVideoActionLoss should be importable from training package."""
    from openwam.train import FlowMatchVideoActionLoss

    assert callable(FlowMatchVideoActionLoss)


def test_flow_match_loss_from_module():
    """Direct import from flow_match_loss module."""
    from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

    loss_fn = FlowMatchVideoActionLoss(lambda_video=1.0, lambda_action=1.0)
    assert loss_fn.lambda_video == 1.0
    assert loss_fn.lambda_action == 1.0
    assert loss_fn.detach_bridge is False


def test_flow_match_loss_config():
    """Loss function should accept all configuration options."""
    from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

    loss_fn = FlowMatchVideoActionLoss(
        lambda_video=0.5,
        lambda_action=2.0,
        detach_bridge=True,
    )
    assert loss_fn.lambda_video == 0.5
    assert loss_fn.lambda_action == 2.0
    assert loss_fn.detach_bridge is True


def test_flow_match_loss_video_loss_computation():
    """Test video loss computation with mock tensors."""
    from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

    loss_fn = FlowMatchVideoActionLoss()

    # Test the static helper for video loss
    noise_pred = torch.randn(2, 16, 5, 4, 4)
    target = torch.randn(2, 16, 5, 4, 4)
    timestep_ids = torch.tensor([10, 20])

    # Create mock scheduler weights
    class MockScheduler:
        linear_timesteps_weights = torch.ones(1000)
        timesteps = torch.linspace(0, 1, 1000)
        sigmas = torch.linspace(1, 0, 1000)

    class MockPipe:
        torch_dtype = torch.float32
        device = "cpu"
        scheduler = MockScheduler()

    loss = loss_fn._compute_video_loss(noise_pred, target, timestep_ids, MockPipe(), {}, B=2)
    assert loss.shape == ()
    assert loss.item() > 0


def test_flow_match_loss_action_loss_computation():
    """Test action loss computation with mock tensors."""
    from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

    loss_fn = FlowMatchVideoActionLoss()

    noise_pred = torch.randn(2, 49, 14)
    target = torch.randn(2, 49, 14)
    timestep_ids = torch.tensor([5, 15])

    class MockScheduler:
        linear_timesteps_weights = torch.ones(1000)

    class MockPipe:
        torch_dtype = torch.float32
        device = "cpu"

    loss = loss_fn._compute_action_loss(noise_pred, target, timestep_ids, MockScheduler(), MockPipe(), B=2)
    assert loss.shape == ()
    assert loss.item() > 0


def test_flow_match_loss_single_sample():
    """Loss computation should handle B=1 fast path."""
    from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

    loss_fn = FlowMatchVideoActionLoss()

    noise_pred = torch.randn(1, 49, 14)
    target = torch.randn(1, 49, 14)
    timestep_ids = torch.tensor([10])

    class MockScheduler:
        linear_timesteps_weights = torch.ones(1000)

    class MockPipe:
        torch_dtype = torch.float32
        device = "cpu"

    loss = loss_fn._compute_action_loss(noise_pred, target, timestep_ids, MockScheduler(), MockPipe(), B=1)
    assert loss.shape == ()
