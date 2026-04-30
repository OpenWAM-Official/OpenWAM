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
            video:           List[PIL.Image] — raw frames (the training
                             pipeline's ``preprocess_video`` handles
                             conversion to tensors and resizing)
            action:          Tensor (T, action_dim)
            prompt:          str
            reference_image: List[PIL.Image], optional
            context_video:   List[PIL.Image], optional (VACE conditioning)

        Note: video is returned as PIL Images (not Tensors) because the
        legacy WanVideoPipeline.preprocess_video handles cropping, resizing,
        and VAE encoding internally. Converting to Tensor prematurely would
        bypass this preprocessing.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

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
