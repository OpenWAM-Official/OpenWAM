"""Transform pipeline for dataset preprocessing.

Provides composable, invertible transforms for actions, rotations, and video
that cleanly separate data loading from preprocessing logic.
"""

from openwam.dataloader.transforms.base import (
    ComposedTransform,
    InvertibleModalityTransform,
    ModalityTransform,
)
from openwam.dataloader.transforms.builder import build_transforms
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform
from openwam.dataloader.transforms.rotation import RotationTransform, RotationType
from openwam.dataloader.transforms.video import (
    VideoColorJitter,
    VideoHorizontalFlip,
    VideoRandomCrop,
    VideoResize,
)

__all__ = [
    "ModalityTransform",
    "InvertibleModalityTransform",
    "ComposedTransform",
    "Normalizer",
    "RotationTransform",
    "RotationType",
    "VideoResize",
    "VideoRandomCrop",
    "VideoColorJitter",
    "VideoHorizontalFlip",
    "FirstFrameConditioningTransform",
    "build_transforms",
]
