"""Build transform pipelines from Hydra config.

Reads the ``transforms`` section of a data config and constructs a
:class:`ComposedTransform` pipeline.

Config example::

    transforms:
      normalize:
        mode: q99          # q99 | min_max | mean_std | binary | scale
        gripper_mode: binary
        gripper_indices: [6]
      rotation:
        source: axis_angle
        target: rotation_6d
        rotation_slice: [3, 6]
      augmentation:
        random_crop:
          scale: [0.8, 1.0]
        color_jitter:
          brightness: 0.1
          contrast: 0.1
          saturation: 0.1
        horizontal_flip:
          p: 0.5
"""

from typing import Optional

from openwam.dataloader.transforms.base import ComposedTransform
from openwam.dataloader.transforms.normalize import ActionNormalizer
from openwam.dataloader.transforms.rotation import RotationTransform
from openwam.dataloader.transforms.video import (
    VideoColorJitter,
    VideoHorizontalFlip,
    VideoRandomCrop,
)


def build_transforms(
    transform_cfg,
    action_stats: Optional[dict] = None,
    height: int = 480,
    width: int = 832,
) -> Optional[ComposedTransform]:
    """Build a transform pipeline from config.

    Args:
        transform_cfg: Config dict/DictConfig with transform definitions.
            If None, returns None (no transforms).
        action_stats: Precomputed action statistics for normalization.
        height: Video target height (for augmentation).
        width: Video target width (for augmentation).

    Returns:
        ComposedTransform or None.
    """
    if transform_cfg is None:
        return None

    transforms = []

    # 1. Rotation conversion (before normalization — operates on raw values)
    rot_cfg = _get(transform_cfg, "rotation")
    if rot_cfg is not None:
        source = _get(rot_cfg, "source", "axis_angle")
        target = _get(rot_cfg, "target", "rotation_6d")
        rot_slice = _get(rot_cfg, "rotation_slice", [3, 6])
        if isinstance(rot_slice, (list, tuple)):
            rot_slice = slice(int(rot_slice[0]), int(rot_slice[1]))
        transforms.append(
            RotationTransform(
                source_repr=source,
                target_repr=target,
                rotation_slice=rot_slice,
            )
        )

    # 2. Action normalization
    norm_cfg = _get(transform_cfg, "normalize")
    if norm_cfg is not None:
        mode = _get(norm_cfg, "mode", "q99")
        gripper_mode = _get(norm_cfg, "gripper_mode", None)
        gripper_indices = _get(norm_cfg, "gripper_indices", None)
        if gripper_indices is not None:
            gripper_indices = list(gripper_indices)

        normalizer = ActionNormalizer(
            mode=mode,
            stats=action_stats,
            gripper_mode=gripper_mode,
            gripper_indices=gripper_indices,
        )
        transforms.append(normalizer)

    # 3. Video augmentation (applied last, only to video frames)
    aug_cfg = _get(transform_cfg, "augmentation")
    if aug_cfg is not None:
        crop_cfg = _get(aug_cfg, "random_crop")
        if crop_cfg is not None:
            scale = _get(crop_cfg, "scale", [0.8, 1.0])
            if not isinstance(scale, (list, tuple)):
                scale = [scale, 1.0]
            transforms.append(
                VideoRandomCrop(
                    height=height,
                    width=width,
                    scale=tuple(scale),
                )
            )

        jitter_cfg = _get(aug_cfg, "color_jitter")
        if jitter_cfg is not None:
            transforms.append(
                VideoColorJitter(
                    brightness=float(_get(jitter_cfg, "brightness", 0.1)),
                    contrast=float(_get(jitter_cfg, "contrast", 0.1)),
                    saturation=float(_get(jitter_cfg, "saturation", 0.1)),
                    hue=float(_get(jitter_cfg, "hue", 0.0)),
                )
            )

        flip_cfg = _get(aug_cfg, "horizontal_flip")
        if flip_cfg is not None:
            transforms.append(
                VideoHorizontalFlip(
                    p=float(_get(flip_cfg, "p", 0.5)),
                )
            )

    if not transforms:
        return None

    return ComposedTransform(transforms)


def _get(config, key, default=None):
    """Get a value from dict or DictConfig."""
    if config is None:
        return default
    if hasattr(config, key):
        val = getattr(config, key)
        return val if val is not None else default
    if hasattr(config, "get"):
        return config.get(key, default)
    return default
