"""Gradient-accumulation loss scaling under DeepSpeed (openwam.train.utils.grad_accumulation)."""

import pytest
import torch
import torch.nn.functional as F
from accelerate.utils import DistributedType

from openwam.train.utils import grad_accumulation as ga


class _Accelerator:
    def __init__(self, gas, distributed_type=DistributedType.DEEPSPEED):
        self.gradient_accumulation_steps = gas
        self.distributed_type = distributed_type


@pytest.fixture
def unscaled_deepspeed(monkeypatch):
    """An installed DeepSpeed whose engine.backward() does not divide by GAS."""
    monkeypatch.setattr(ga, "_installed_deepspeed_scales_gas", lambda: False)


@pytest.mark.parametrize(
    "version, scales",
    [
        ("0.17.6", True),
        ("0.18.2", True),
        ("0.18.3", False),
        ("0.18.9", False),
        ("0.19.5", False),
        ("0.19.6", True),
        ("0.20.0", True),
        ("not-a-version", True),
    ],
)
def test_deepspeed_versions_without_gas_scaling(version, scales):
    assert ga.deepspeed_backward_scales_gas(version) is scales


def test_divides_by_gas_when_deepspeed_does_not(unscaled_deepspeed):
    loss = torch.tensor(3.0)
    assert torch.equal(ga.scale_loss_for_backward(_Accelerator(4), loss), loss / 4)


def test_gas_1_returns_the_loss_itself(unscaled_deepspeed):
    loss = torch.tensor(3.0)
    assert ga.scale_loss_for_backward(_Accelerator(1), loss) is loss


def test_never_scales_twice(monkeypatch):
    loss = torch.tensor(3.0)
    monkeypatch.setattr(ga, "_installed_deepspeed_scales_gas", lambda: True)
    assert ga.scale_loss_for_backward(_Accelerator(4), loss) is loss
    # Without DeepSpeed, accelerate divides by GAS itself.
    monkeypatch.setattr(ga, "_installed_deepspeed_scales_gas", lambda: False)
    assert ga.scale_loss_for_backward(_Accelerator(4, DistributedType.MULTI_GPU), loss) is loss


def test_accumulated_gradient_is_the_full_batch_mean(unscaled_deepspeed):
    torch.manual_seed(0)
    model = torch.nn.Linear(5, 1, dtype=torch.float64)
    x = torch.randn(8, 5, dtype=torch.float64)
    y = torch.randn(8, 1, dtype=torch.float64)

    accelerator = _Accelerator(4)
    for xb, yb in zip(x.chunk(4), y.chunk(4)):
        ga.scale_loss_for_backward(accelerator, F.mse_loss(model(xb), yb)).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    F.mse_loss(model(x), y).backward()
    for grad, p in zip(accumulated, model.parameters()):
        torch.testing.assert_close(grad, p.grad)
