"""Cosmos flow-match scheduler adapter — exposes the four fields trainer needs."""

from __future__ import annotations

import torch

from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter


def test_set_timesteps_training_populates_all_fields():
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=50, training=True)
    assert isinstance(sch.timesteps, torch.Tensor)
    assert isinstance(sch.sigmas, torch.Tensor)
    assert isinstance(sch.linear_timesteps_weights, torch.Tensor)
    assert sch.timesteps.shape == (50,)
    assert sch.sigmas.shape == (50,)
    assert sch.linear_timesteps_weights.shape == (50,)
    assert sch.num_train_timesteps == 1000
    assert sch.training is True


def test_default_shift_video_is_cosmos_upstream_value():
    """Default must match upstream
    cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:99
    (`shift: int = 5`)."""
    sch = CosmosFlowSchedulerAdapter()
    assert sch.shift_video == 5.0


def test_training_weights_are_uniform():
    """Cosmos upstream uses TrainTimeWeight("uniform") — weights must be 1.0
    everywhere (see cosmos_predict2/_src/predict2/schedulers/rectified_flow.py:21-42)."""
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=50, training=True)
    weights = sch.linear_timesteps_weights
    assert weights is not None
    assert torch.allclose(weights, torch.ones_like(weights))


def test_set_timesteps_eval_leaves_weights_none():
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=20, training=False)
    assert sch.linear_timesteps_weights is None
    assert sch.training is False


def test_add_noise_matches_flow_match_form():
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=10, training=True)
    clean = torch.zeros(2, 4)
    noise = torch.ones(2, 4)
    noisy = sch.add_noise(clean, noise, sch.timesteps[0])
    # (1 - sigma)·clean + sigma·noise with clean=0 → noisy == sigma·noise.
    assert torch.allclose(noisy, torch.full_like(noisy, float(sch.sigmas[0])))


def test_training_target_is_noise_minus_sample():
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=5, training=True)
    sample = torch.randn(2, 3)
    noise = torch.randn(2, 3)
    target = sch.training_target(sample, noise, sch.timesteps[0])
    assert torch.allclose(target, noise - sample)


def test_shift_video_changes_sigmas():
    sch_low = CosmosFlowSchedulerAdapter(shift_video=1.0)
    sch_high = CosmosFlowSchedulerAdapter(shift_video=5.0)
    sch_low.set_timesteps(num_inference_steps=20, training=False)
    sch_high.set_timesteps(num_inference_steps=20, training=False)
    # shift=1 collapses to identity; shift=5 pushes sigmas toward 1.
    assert not torch.allclose(sch_low.sigmas, sch_high.sigmas)
