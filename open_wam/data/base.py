"""Abstract base class for action datasets."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseActionDataset(ABC, torch.utils.data.Dataset):
    """Action dataset base class.

    Every concrete dataset must return a dict with at least ``video`` and
    ``action`` tensors.  Optional fields include ``prompt``,
    ``reference_image``, and ``context_video`` (VACE conditioning).
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Expected keys:
            video:           Tensor (T, C, H, W)
            action:          Tensor (T, action_dim)
            prompt:          str
            reference_image: Tensor (C, H, W), optional
            context_video:   Tensor (T, C, H, W), optional (VACE)
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""
        ...

    @property
    @abstractmethod
    def action_stats(self) -> Optional[dict]:
        """Return ``{"mean": ndarray, "std": ndarray}`` or ``None``."""
        ...
