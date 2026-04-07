"""Transform pipeline for dataset preprocessing.

Provides composable, invertible transforms for actions, rotations, and video
that cleanly separate data loading from preprocessing logic.
"""

# Legacy image helpers (from RoboTwin dataset module)
from open_wam.data._robotwin_impl import (
    _crop_and_resize as crop_and_resize,
)
from open_wam.data._robotwin_impl import (
    _pad_and_resize as pad_and_resize,
)
from open_wam.data._robotwin_impl import (
    _resize_frame as resize_frame,
)
from open_wam.data.transforms.base import (
    ComposedTransform,
    InvertibleModalityTransform,
    ModalityTransform,
)
from open_wam.data.transforms.builder import build_transforms
from open_wam.data.transforms.normalize import ActionNormalizer, Normalizer
from open_wam.data.transforms.pipeline import VACEConditioningTransform
from open_wam.data.transforms.rotation import RotationTransform, RotationType
from open_wam.data.transforms.video import (
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
    "ActionNormalizer",
    "RotationTransform",
    "RotationType",
    "VideoResize",
    "VideoRandomCrop",
    "VideoColorJitter",
    "VideoHorizontalFlip",
    "VACEConditioningTransform",
    "build_transforms",
    "crop_and_resize",
    "pad_and_resize",
    "resize_frame",
]
