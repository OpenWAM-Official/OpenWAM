"""Tests for decoupled noise schedules."""


def _two_schedulers():
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    return FlowMatchScheduler("Wan"), ActionScheduler()


def test_schedule_decoupled_flash_length():
    from openwam.deploy.optimizations.decoupled_schedule import schedule_decoupled_flash

    for steps in [1, 2, 4]:
        v, a = _two_schedulers()
        s = schedule_decoupled_flash(v, a, action_steps=steps)
        assert len(s) == steps + 1  # N steps = N+1 entries
        for t_v, t_a in s:
            assert t_v == 0.0
        assert s[-1] == (0.0, 0.0)


def test_schedule_decoupled_flash_descending():
    from openwam.deploy.optimizations.decoupled_schedule import schedule_decoupled_flash

    v, a = _two_schedulers()
    s = schedule_decoupled_flash(v, a, action_steps=4)
    action_ts = [t_a for _, t_a in s]
    for i in range(len(action_ts) - 1):
        assert action_ts[i] >= action_ts[i + 1]


def test_schedule_decoupled_asymmetric():
    from openwam.deploy.optimizations.decoupled_schedule import schedule_decoupled_asymmetric

    v, a = _two_schedulers()
    s = schedule_decoupled_asymmetric(v, a, video_steps=10, action_steps=3)
    assert len(s) == 11  # 10 steps + terminal
    assert s[-1] == (0.0, 0.0)


def test_schedule_decoupled_asymmetric_validation():
    import pytest

    from openwam.deploy.optimizations.decoupled_schedule import schedule_decoupled_asymmetric

    v, a = _two_schedulers()
    with pytest.raises(AssertionError):
        schedule_decoupled_asymmetric(v, a, video_steps=5, action_steps=10)


def test_sample_decoupled_timesteps_shapes():
    from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss

    sampler = DecoupledFlowMatchLoss()
    v_t, a_t = sampler.sample_timesteps(batch_size=8)
    assert v_t.shape == (8,)
    assert a_t.shape == (8,)
    assert v_t.min() >= 0
    assert v_t.max() < 1000
    assert a_t.min() >= 0
    assert a_t.max() < 1000


def test_sample_decoupled_timesteps_video_biased_high():
    """Video timesteps should be biased toward high values (high noise)."""
    from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss

    # With Beta(0.5, 1.0) flipped, mean should be > 500
    sampler = DecoupledFlowMatchLoss(video_beta_a=0.5, video_beta_b=1.0)
    v_t, _ = sampler.sample_timesteps(batch_size=1000)
    mean_v = v_t.mean().item()
    assert mean_v > 500, f"Expected video mean > 500, got {mean_v}"


def test_decoupled_loss_warmup():
    from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss

    loss_fn = DecoupledFlowMatchLoss(warmup_steps=100)

    # During warmup: standard uniform sampling
    v_t, a_t = loss_fn.sample_timesteps(batch_size=16, current_step=50)
    assert v_t.shape == (16,)

    # After warmup: decoupled sampling
    v_t, a_t = loss_fn.sample_timesteps(batch_size=16, current_step=200)
    assert v_t.shape == (16,)


def test_optimizations_imports():
    """All optimization modules should be importable."""
