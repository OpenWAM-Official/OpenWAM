"""Base classes for the transform pipeline.

Transforms operate on sample dicts (the output of Dataset.__getitem__)
and can be composed into pipelines. InvertibleModalityTransform supports
reverse application for inference-time unnormalization.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class ModalityTransform(ABC):
    """Abstract base for sample-level transforms.

    A transform receives a sample dict and returns a modified sample dict.
    Transforms declare which keys they operate on via ``apply_to``.
    """

    def __init__(self, apply_to: Optional[List[str]] = None, training: bool = True):
        self.apply_to = apply_to or []
        self.training = training

    @abstractmethod
    def apply(self, data: dict) -> dict:
        """Apply the transform to a sample dict."""
        ...

    def train(self):
        """Switch to training mode (e.g., random augmentation)."""
        self.training = True
        return self

    def eval(self):
        """Switch to evaluation mode (e.g., deterministic center crop)."""
        self.training = False
        return self

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class InvertibleModalityTransform(ModalityTransform):
    """Transform that can be reversed (e.g., for unnormalization at inference)."""

    @abstractmethod
    def unapply(self, data: dict) -> dict:
        """Reverse the transform on a sample dict."""
        ...


class ComposedTransform(InvertibleModalityTransform):
    """Chain of transforms applied sequentially.

    ``apply()`` runs transforms in order; ``unapply()`` runs them in
    reverse order (only for InvertibleModalityTransform instances).

    Args:
        transforms: List of transforms to compose.
    """

    def __init__(self, transforms: Optional[List[ModalityTransform]] = None):
        super().__init__()
        self.transforms = transforms or []

    def apply(self, data: dict) -> dict:
        for t in self.transforms:
            data = t.apply(data)
        return data

    def unapply(self, data: dict) -> dict:
        for t in reversed(self.transforms):
            if isinstance(t, InvertibleModalityTransform):
                data = t.unapply(data)
        return data

    def train(self):
        self.training = True
        for t in self.transforms:
            t.train()
        return self

    def eval(self):
        self.training = False
        for t in self.transforms:
            t.eval()
        return self

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def append(self, transform: ModalityTransform):
        self.transforms.append(transform)
        return self
