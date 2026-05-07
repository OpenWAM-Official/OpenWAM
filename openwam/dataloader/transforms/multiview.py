"""Dataset-neutral multi-view image composition and prompt formatting.

These helpers used to live inside ``robotwin_dataset.py`` but are pure image /
text utilities reused by both the RoboTwin dataloader and the deployment
``policy_server``. Keeping them in a transform module avoids deploy → dataset
hard-coupling.
"""

from PIL import Image

# Single source of truth for the L-shape multiview layout used by RoboTwin
# (and by the deploy server as a fallback when ``cfg.dataloader.camera_layout``
# is missing). The matching yaml key in ``configs/dataloader/robotwin.yaml`` is
# ``camera_layout`` and should mirror this list byte-for-byte.
DEFAULT_MULTIVIEW_CAMERA_LAYOUT = ("head_camera", "left_camera", "right_camera")


def crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Center-crop and resize an image to ``(target_width, target_height)``.

    Scales the image so the shorter side matches the target, then center-crops
    to the exact target resolution. Preserves aspect ratio.
    """
    img_w, img_h = image.size
    scale = max(target_width / img_w, target_height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


def _stretch_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Direct BILINEAR resize without aspect-ratio preservation.

    Matches FastWAM's per-camera ``torchvision.transforms.functional.resize(...,
    BILINEAR, antialias=True)`` step in its multi-view composition.
    """
    return image.resize((target_width, target_height), Image.BILINEAR)


def assemble_multiview_layout(
    frames_by_camera: dict,
    camera_layout: list,
    out_h: int,
    out_w: int,
    top_height_ratio: float = 2.0 / 3.0,
) -> Image.Image:
    """3-camera L-shape composition (FastWAM / RoboTwin compatible).

    Layout with the default ratio:
        top    -> (out_h * 2/3, out_w)          full width, 2/3 of height
        bot-L  -> (out_h * 1/3, out_w / 2)      half width, 1/3 of height
        bot-R  -> (out_h * 1/3, out_w / 2)      half width, 1/3 of height

    Each camera is BILINEAR-resized directly to its slot with no aspect-ratio
    preservation (slight horizontal/vertical stretch is accepted), then pasted
    without gaps. At ``out_h=384, out_w=320`` this produces a canvas with
    top=256x320 and each bottom=128x160, matching FastWAM exactly.

    Args:
        frames_by_camera: ``{camera_name: PIL.Image}``. Missing key -> black region.
        camera_layout: ordered list of 3 camera names (top, bot-left, bot-right).
        out_h, out_w: final canvas size in pixels.
        top_height_ratio: fraction of height allocated to the top camera.

    Returns:
        PIL Image of size ``(out_w, out_h)``.
    """
    if len(camera_layout) != 3:
        raise ValueError(f"multiview layout expects 3 cameras, got {len(camera_layout)}: {camera_layout}")

    top_h = int(round(out_h * top_height_ratio))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    right_w = out_w - half_w

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))

    top_frame = frames_by_camera.get(camera_layout[0])
    if top_frame is not None:
        canvas.paste(_stretch_resize(top_frame, top_h, out_w), (0, 0))

    bl_frame = frames_by_camera.get(camera_layout[1])
    if bl_frame is not None:
        canvas.paste(_stretch_resize(bl_frame, bottom_h, half_w), (0, top_h))

    br_frame = frames_by_camera.get(camera_layout[2])
    if br_frame is not None:
        canvas.paste(_stretch_resize(br_frame, bottom_h, right_w), (half_w, top_h))

    return canvas


def format_prompt_for_inference(base_prompt: str) -> str:
    """Deploy prompt template.

    Must match :meth:`RoboTwinDataset._get_prompt` byte-for-byte so that the
    deployment-time prompt stays in-distribution with training.
    """
    return "A video recorded from a robot's point of view executing the following instruction: " + base_prompt


__all__ = [
    "DEFAULT_MULTIVIEW_CAMERA_LAYOUT",
    "crop_and_resize",
    "assemble_multiview_layout",
    "format_prompt_for_inference",
]
