"""Abstract base class for trainers."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseTrainer(ABC):
    """Trainer base class.

    Concrete trainers implement ``compute_loss`` and optionally override
    ``train_step``, ``validate``, ``save_checkpoint``, ``load_checkpoint``.
    """

    def __init__(self, cfg, model: torch.nn.Module, dataset, accelerator):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.accelerator = accelerator

    @abstractmethod
    def compute_loss(self, batch) -> dict:
        """Compute training loss.

        Returns:
            dict with at least ``total`` key and optional breakdown
            (e.g. ``video``, ``action``).
        """
        ...

    def train_step(self, batch) -> dict:
        """Generic training step: forward + backward + optimizer step.

        Override in subclass for custom logic.
        """
        losses = self.compute_loss(batch)
        self.accelerator.backward(losses["total"])
        return losses

    def validate(self, val_loader) -> dict:
        """Run validation loop and return metrics."""
        return {}

    @abstractmethod
    def save_checkpoint(self, path: str):
        ...

    @abstractmethod
    def load_checkpoint(self, path: str):
        ...
