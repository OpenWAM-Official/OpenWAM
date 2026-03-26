"""Tests for decoupled noise schedules."""

import torch
import numpy as np


def test_schedule_decoupled_flash_length():
    from open_wam.inference.optimizations.decoupled_schedule import schedule_decoupled_flash
    for steps in [1, 2, 4]:
        s = schedule_decoupled_flash(action_steps=steps)
        assert len(s) == steps + 1  # N steps = N+1 entries
        # All video timesteps should be 0 (clean)
        for t_v, t_a in s:
            assert t_v == 0.0
        # Last entry should be (0, 0)
        assert s[-1] == (0.0, 0.0)


def test_schedule_decoupled_flash_descending():
    from open_wam.inference.optimizations.decoupled_schedule import schedule_decoupled_flash
    s = schedule_decoupled_flash(action_steps=4)
    action_ts = [t_a for _, t_a in s]
    # Action timesteps should be descending (high → 0)
    for i in range(len(action_ts) - 1):
        assert action_ts[i] >= action_ts[i + 1]


def test_schedule_decoupled_asymmetric():
    from open_wam.inference.optimizations.decoupled_schedule import schedule_decoupled_asymmetric
    s = schedule_decoupled_asymmetric(video_steps=10, action_steps=3)
    assert len(s) == 11  # 10 steps + terminal
    # First 7 steps: action at max noise (idle)
    # Last 3 steps: action denoises
    # Final: (0, 0)
    assert s[-1] == (0.0, 0.0)


def test_schedule_decoupled_asymmetric_validation():
    from open_wam.inference.optimizations.decoupled_schedule import schedule_decoupled_asymmetric
    import pytest
    with pytest.raises(AssertionError):
        schedule_decoupled_asymmetric(video_steps=5, action_steps=10)


def test_sample_decoupled_timesteps_shapes():
    from open_wam.inference.optimizations.decoupled_schedule import sample_decoupled_timesteps
    v_t, a_t = sample_decoupled_timesteps(batch_size=8)
    assert v_t.shape == (8,)
    assert a_t.shape == (8,)
    assert v_t.min() >= 0
    assert v_t.max() < 1000
    assert a_t.min() >= 0
    assert a_t.max() < 1000


def test_sample_decoupled_timesteps_video_biased_high():
    """Video timesteps should be biased toward high values (high noise)."""
    from open_wam.inference.optimizations.decoupled_schedule import sample_decoupled_timesteps
    # With Beta(0.5, 1.0) flipped, mean should be > 500
    v_t, _ = sample_decoupled_timesteps(batch_size=1000, video_beta_a=0.5, video_beta_b=1.0)
    mean_v = v_t.mean().item()
    assert mean_v > 500, f"Expected video mean > 500, got {mean_v}"


def test_decoupled_loss_warmup():
    from open_wam.training.decoupled_loss import DecoupledFlowMatchLoss
    loss_fn = DecoupledFlowMatchLoss(warmup_steps=100)

    # During warmup: standard uniform sampling
    v_t, a_t = loss_fn.sample_timesteps(batch_size=16, current_step=50)
    assert v_t.shape == (16,)

    # After warmup: decoupled sampling
    v_t, a_t = loss_fn.sample_timesteps(batch_size=16, current_step=200)
    assert v_t.shape == (16,)


def test_optimizations_imports():
    """All optimization modules should be importable."""
    from open_wam.inference.optimizations import (
        DiTVelocityCache,
        CFGBatchMerger,
        CFGParallelExecutor,
        schedule_decoupled_flash,
        schedule_decoupled_asymmetric,
        sample_decoupled_timesteps,
        AsyncInferenceExecutor,
    )
