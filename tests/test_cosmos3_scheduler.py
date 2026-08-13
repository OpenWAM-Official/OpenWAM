"""Cosmos3 scheduler adapter: same math as the predict2.5 adapter, and the
``timesteps = sigmas × num_train_timesteps`` linear relation that generate() /
denoise_schedule.py invert externally."""

import torch

from openwam.model.video_backbone.cosmos3.scheduler import Cosmos3FlowSchedulerAdapter
from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter


def test_subclass_of_predict25_adapter():
    assert issubclass(Cosmos3FlowSchedulerAdapter, CosmosFlowSchedulerAdapter)


def test_training_grid_matches_shift_formula():
    shift = 5.0
    sched = Cosmos3FlowSchedulerAdapter(shift_video=shift)
    sched.set_timesteps(1000, training=True, shift=shift)

    base = torch.linspace(1.0, 0.0, 1001)[:-1]
    expected = shift * base / (1 + (shift - 1) * base)
    assert torch.allclose(sched.sigmas, expected, atol=1e-6)
    assert torch.allclose(sched.timesteps, sched.sigmas * sched.num_train_timesteps, atol=1e-4)
    assert sched.linear_timesteps_weights is not None
    assert torch.all(sched.linear_timesteps_weights == 1.0)


def test_inference_grid_respects_shift_override():
    sched = Cosmos3FlowSchedulerAdapter(shift_video=5.0)
    sched.set_timesteps(10, shift=3.0)
    base = torch.linspace(1.0, 0.0, 11)[:-1]
    expected = 3.0 * base / (1 + 2.0 * base)
    assert torch.allclose(sched.sigmas, expected, atol=1e-6)
    assert sched.linear_timesteps_weights is None
