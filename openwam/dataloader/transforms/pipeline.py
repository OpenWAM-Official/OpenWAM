"""Pipeline-specific transforms that bridge between model-agnostic data and
pipeline-specific conditioning fields.

These transforms are applied at the training pipeline level (not in the
dataset), keeping the dataset output model-agnostic.
"""

from openwam.dataloader.transforms.base import ModalityTransform


class FirstFrameConditioningTransform(ModalityTransform):
    """Add first-frame conditioning fields to a sample.

    Reads ``data["video"]`` and derives:
    - ``first_frame_image``: First frame, used as TI2V first-frame condition
      on Wan2.2-TI2V backbones, or as VACE spatial reference on Wan2.1-VACE
      backbones. The OpenWAM layer stays backend-agnostic; the mapping to
      the Wan pipeline's ``vace_reference_image`` input happens at the pipeline
      boundary (see the video backbone adapter used by deploy and training).
    - ``vace_video``: Set to None (inactive conditioning by default).

    Args:
        use_first_frame_as_reference: If True, set ``first_frame_image``
            to ``[video[0]]``. If False, set to None.
    """

    def __init__(self, use_first_frame_as_reference: bool = True):
        super().__init__(apply_to=["video"])
        self.use_first_frame_as_reference = use_first_frame_as_reference

    def apply(self, data: dict) -> dict:
        if "vace_video" not in data:
            data["vace_video"] = None

        if "first_frame_image" not in data:
            video = data.get("video")
            # ``len`` instead of truthiness: same for frame lists, and also valid for
            # the uint8 (T, H, W, 3) clip tensors of openwam.dataloader.collate.
            if self.use_first_frame_as_reference and video is not None and len(video) > 0:
                data["first_frame_image"] = [data["video"][0]]
            else:
                data["first_frame_image"] = None

        return data
