"""Loss scaling for gradient accumulation under DeepSpeed.

With ``training.gradient_accumulation_steps`` (GAS) > 1 the accumulated gradient
should be the mean over the micro-batches. ``Accelerator.backward`` leaves that
division to DeepSpeed, but DeepSpeed 0.18.3 moved it out of ``engine.backward()``
into backward hooks that ``DeepSpeedEngine.forward()`` registers on its outputs
(deepspeedai/DeepSpeed#7665; #8245 in 0.19.6 moved it back into
``engine.backward()``). The trainer computes the loss through the architecture's
own methods and never calls the engine's forward, so with those DeepSpeed
versions nothing divides by GAS: the gradient that is clipped (``max_grad_norm``)
and stepped is the sum over the micro-batches, GAS times the mean.

``scale_loss_for_backward`` divides the loss by GAS in exactly that case and
returns it unchanged otherwise (GAS = 1, other DeepSpeed versions, no DeepSpeed),
so the loss is never scaled twice. The trainer must keep computing the loss
without ``DeepSpeedEngine.forward``: that call would register DeepSpeed's own
scaling hooks.
"""

from __future__ import annotations

import functools
import importlib.metadata

from packaging.version import InvalidVersion, Version

# DeepSpeed releases in which engine.backward() does not divide by GAS.
_FIRST_UNSCALED = (0, 18, 3)
_FIRST_FIXED = (0, 19, 6)


def deepspeed_backward_scales_gas(version: str) -> bool:
    """Whether ``engine.backward()`` of this DeepSpeed version divides by GAS itself."""
    try:
        release = Version(version).release[:3]
    except InvalidVersion:
        return True
    return not (_FIRST_UNSCALED <= release < _FIRST_FIXED)


@functools.lru_cache(maxsize=1)
def _installed_deepspeed_scales_gas() -> bool:
    try:
        return deepspeed_backward_scales_gas(importlib.metadata.version("deepspeed"))
    except importlib.metadata.PackageNotFoundError:
        return True


def _uses_deepspeed(accelerator) -> bool:
    from accelerate.utils import DistributedType

    return getattr(accelerator, "distributed_type", None) == DistributedType.DEEPSPEED


def scale_loss_for_backward(accelerator, loss):
    """``loss / GAS`` when neither accelerate nor DeepSpeed will apply the 1/GAS
    gradient-accumulation scaling in ``accelerator.backward``; else ``loss`` itself."""
    gas = int(getattr(accelerator, "gradient_accumulation_steps", 1) or 1)
    if gas == 1 or not _uses_deepspeed(accelerator) or _installed_deepspeed_scales_gas():
        return loss
    return loss / gas
