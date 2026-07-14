"""Regression test for the multi-GPU (DDP) crash in ``prepare_accelerate``.

After ``accelerator.prepare(...)`` the trainer's ``self.architecture`` is the
*wrapped* handle. Architecture-level helpers (``set_dtype_device`` /
``move_frozen_to_device``) must be invoked on the UNDERLYING module:

* ``DeepSpeedEngine.__getattr__`` forwards unknown attributes to the inner
  module, so the DeepSpeed path happened to work even when the helpers were
  called on the wrapper.
* ``torch.nn.parallel.DistributedDataParallel`` does NOT forward arbitrary
  attributes, so under a hypothetical plain-DDP accelerator with
  ``world_size > 1`` the wrapper has no ``set_dtype_device`` and the call
  raised ``AttributeError`` at startup.

Single-GPU never exposed this: Accelerate adds no DDP wrapper at
``world_size == 1``, so ``self.architecture`` stayed the bare module.

The fix routes the helper calls through ``accelerator.unwrap_model(...)``,
which returns the inner module for both backends (and is a no-op single-GPU).
These tests exercise the fix on CPU without any real Accelerator/DDP.
"""

import pytest
import torch

from openwam.train.openwam_trainer import OpenWAMTrainer


class _FakeArch:
    """Stand-in for a WAM architecture: records the helper calls."""

    def __init__(self):
        self.dtype = torch.float32
        self.set_dtype_device_calls = []
        self.move_frozen_calls = []

    def set_dtype_device(self, dtype, device):
        self.set_dtype_device_calls.append((dtype, device))

    def move_frozen_to_device(self, device):
        self.move_frozen_calls.append(device)


class _DDPLikeWrapper:
    """Mimics ``DistributedDataParallel``: holds ``.module`` but does NOT
    forward arbitrary attribute lookups to it. Accessing ``set_dtype_device``
    therefore raises ``AttributeError`` — the exact multi-GPU failure mode."""

    def __init__(self, module):
        self.module = module


class _FakeAccelerator:
    """Minimal Accelerator: ``prepare`` wraps the model (as DDP would for
    ``world_size > 1``); ``unwrap_model`` returns the inner module."""

    def __init__(self, wrap):
        self._wrap = wrap
        self.device = torch.device("cpu")

    def prepare(self, *args):
        model, *rest = args
        wrapped = _DDPLikeWrapper(model) if self._wrap else model
        return (wrapped, *rest)

    def unwrap_model(self, model):
        return model.module if isinstance(model, _DDPLikeWrapper) else model


def _make_trainer(*, wrap):
    """Build a trainer with only the fields ``prepare_accelerate`` touches,
    bypassing the heavy ``__init__`` (no cfg / weights / dataset needed)."""
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.architecture = _FakeArch()
    trainer.accelerator = _FakeAccelerator(wrap=wrap)
    return trainer


@pytest.mark.parametrize("wrap", [True, False], ids=["multi_gpu_ddp", "single_gpu"])
def test_prepare_accelerate_drives_unwrapped_module(wrap):
    """The helpers land on the underlying module for both the DDP-wrapped
    (multi-GPU) and bare (single-GPU) cases, and never crash."""
    trainer = _make_trainer(wrap=wrap)
    arch = trainer.architecture  # underlying module, before prepare wraps it

    optimizer, dataloader, scheduler = object(), object(), None
    trainer.prepare_accelerate(optimizer, dataloader, scheduler)

    # set_dtype_device / move_frozen_to_device ran on the real module, on device.
    assert arch.set_dtype_device_calls == [(torch.float32, torch.device("cpu"))]
    assert arch.move_frozen_calls == [torch.device("cpu")]

    # The loop still drives the *wrapped* handle for forward/backward.
    if wrap:
        assert isinstance(trainer.architecture, _DDPLikeWrapper)
        assert trainer.architecture.module is arch


def test_ddp_wrapper_would_crash_without_unwrap():
    """Pins the failure mode the fix addresses: a DDP-style wrapper does not
    forward ``set_dtype_device``, so calling it on the wrapper raises. This is
    what ``prepare_accelerate`` used to do on the multi-GPU no-DeepSpeed path."""
    wrapper = _DDPLikeWrapper(_FakeArch())
    with pytest.raises(AttributeError):
        wrapper.set_dtype_device(torch.float32, torch.device("cpu"))
