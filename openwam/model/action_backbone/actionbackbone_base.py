"""Action stream backbone ABC.

Each concrete subclass exposes only the methods its architecture's
``forward()`` invokes. Required attributes on every subclass:

    action_dim:    int
    bridge_layers or expert_layers: tuple[int, ...] when used by the architecture
    action_mean:   Tensor — per-dim mean for denormalization
    action_std:    Tensor — per-dim std for denormalization
    scheduler:     ActionScheduler (provided by ``ActionBackbone.__init__``)
"""

from __future__ import annotations

from abc import ABC

from torch import nn

from openwam.model.action_backbone.scheduler import ActionScheduler


class ActionBackbone(nn.Module, ABC):
    """Minimal action-stream backbone ABC.

    Concrete subclasses inherit directly — there is no wrapping layer. The
    state_dict therefore lives at ``action_backbone.<param-name>`` with no
    extra prefix.
    """

    def __init__(self):
        super().__init__()
        self.scheduler = ActionScheduler()

    @property
    def uses_proprioception(self) -> bool:
        """Whether this backbone consumes a ``proprio_state`` input. Default False."""
        return False

    @property
    def shift_action(self):
        """Optional α-shift for the action scheduler — single source of truth read by
        the architecture for both training (``init_training_schedulers``) and inference
        (deploy schedule). ``None`` falls back to the scheduler template default.
        Symmetric to ``VideoBackbone.shift_video``. Concrete backbones store the
        resolved value into ``self._shift_action`` during ``__init__``."""
        return getattr(self, "_shift_action", None)

    def set_dtype_device(self, dtype, device) -> None:
        """Move action backbone params/buffers to (dtype, device).

        Subclasses can override for custom logic (e.g. partial freezing of
        sub-components); the default just moves everything via ``nn.Module.to``.
        """
        self.to(dtype=dtype, device=device)


__all__ = ["ActionBackbone"]
