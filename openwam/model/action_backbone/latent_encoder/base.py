"""Latent-action target provider ABC.

A latent-action encoder turns a batch of videos into clean latent-action
targets used as the regression target when ``model.action_backbone.type=latent``
(LAPA-style pretraining). It is a frozen **label generator** living on the
trainer side — not part of the architecture tree, not an action backbone
(:class:`~openwam.model.action_backbone.base.ActionDiTBackbone`). The trainer
builds it once and calls it per step to replace the dataloader's real actions.

Subclass contract:
    __init__(cfg, *, device, dtype)   construct + load frozen weights
    forward(videos) -> Tensor         (B, T_target, token_dim) clean targets
    device / dtype                    where targets are produced

The output token_dim must equal the latent ActionDiT's ``action_dim``; the
trainer validates this when wiring the provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class LatentActionEncoder(nn.Module, ABC):
    """Frozen video -> latent-action target provider."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

    @abstractmethod
    def forward(self, videos: list[Any] | torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos into clean latent-action targets.

        Returns ``(B, T_target, token_dim)`` on ``self.device`` / ``self.dtype``.
        Must be deterministic — it generates frozen regression labels.
        """
        raise NotImplementedError


__all__ = ["LatentActionEncoder"]
