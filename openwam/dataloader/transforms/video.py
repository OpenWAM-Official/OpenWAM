"""Video augmentation transforms.

Operates on ``data["video"]`` which is ``List[PIL.Image]``.
All transforms apply the same random parameters across all frames
in a sample to maintain temporal consistency.
"""

import random
from typing import Optional, Tuple

from PIL import Image, ImageEnhance

from openwam.dataloader.transforms.base import ModalityTransform


class VideoResize(ModalityTransform):
    """Resize all frames to a target resolution.

    Args:
        height: Target height.
        width: Target width.
        mode: Resize mode — "lanczos", "bilinear", or "nearest".
    """

    _RESAMPLE = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }

    def __init__(self, height: int = 384, width: int = 320, mode: str = "lanczos"):
        super().__init__(apply_to=["video"])
        self.height = height
        self.width = width
        self.resample = self._RESAMPLE.get(mode, Image.LANCZOS)

    def apply(self, data: dict) -> dict:
        if "video" in data and data["video"]:
            data["video"] = [frame.resize((self.width, self.height), self.resample) for frame in data["video"]]
        return data


class VideoRandomCrop(ModalityTransform):
    """Random crop (training) or center crop (evaluation).

    Crops a random region of the frame, then resizes to target size.
    The same crop region is used for all frames in a sample.

    Args:
        height: Target output height.
        width: Target output width.
        scale: Range of crop area relative to original area.
        ratio: Range of aspect ratio of crop.
    """

    def __init__(
        self,
        height: int = 384,
        width: int = 320,
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(apply_to=["video"])
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio

    def _get_crop_params(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Get (left, top, right, bottom) crop coordinates."""
        if not self.training:
            # Center crop with the smaller scale value
            s = self.scale[0]
            cw, ch = int(w * s), int(h * s)
            left = (w - cw) // 2
            top = (h - ch) // 2
            return left, top, left + cw, top + ch

        s = random.uniform(*self.scale)
        cw, ch = int(w * s), int(h * s)
        left = random.randint(0, max(0, w - cw))
        top = random.randint(0, max(0, h - ch))
        return left, top, left + cw, top + ch

    def apply(self, data: dict) -> dict:
        if "video" not in data or not data["video"]:
            return data

        frames = data["video"]
        w, h = frames[0].size
        crop = self._get_crop_params(w, h)

        data["video"] = [frame.crop(crop).resize((self.width, self.height), Image.LANCZOS) for frame in frames]
        return data


class VideoColorJitter(ModalityTransform):
    """Random color jitter applied consistently across all frames.

    Only active during training.

    Args:
        brightness: Max brightness change factor.
        contrast: Max contrast change factor.
        saturation: Max saturation change factor.
        hue: Max hue change (not applied in PIL-only mode).
    """

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.0,
    ):
        super().__init__(apply_to=["video"])
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, data: dict) -> dict:
        if not self.training:
            return data

        if "video" not in data or not data["video"]:
            return data

        # Sample random factors (same for all frames)
        b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        s_factor = 1.0 + random.uniform(-self.saturation, self.saturation)

        augmented = []
        for frame in data["video"]:
            frame = ImageEnhance.Brightness(frame).enhance(b_factor)
            frame = ImageEnhance.Contrast(frame).enhance(c_factor)
            frame = ImageEnhance.Color(frame).enhance(s_factor)
            augmented.append(frame)

        data["video"] = augmented
        return data


class VideoHorizontalFlip(ModalityTransform):
    """Random horizontal flip with probability ``p``.

    Only active during training. All frames in a sample are flipped together.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(apply_to=["video"])
        self.p = p

    def apply(self, data: dict) -> dict:
        if not self.training:
            return data

        if random.random() >= self.p:
            return data

        if "video" in data and data["video"]:
            data["video"] = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in data["video"]]
        return data
