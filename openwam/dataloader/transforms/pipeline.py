"""Pipeline-specific transforms that bridge between model-agnostic data and
pipeline-specific conditioning fields.

These transforms are applied at the training pipeline level (not in the
dataset), keeping the dataset output model-agnostic.
"""

from openwam.dataloader.transforms.base import ModalityTransform


class VACEConditioningTransform(ModalityTransform):
    """Add VACE-specific conditioning fields to a sample.

    Reads ``data["video"]`` and derives:
    - ``vace_reference_image``: First frame as reference for spatial grounding.
    - ``vace_video``: Set to None (inactive conditioning by default).

    This keeps the dataset layer model-agnostic while providing the fields
    that the VACE video pipeline expects.

    Args:
        use_first_frame_as_reference: If True, set ``vace_reference_image``
            to ``[video[0]]``. If False, set to None.
    """

    def __init__(self, use_first_frame_as_reference: bool = True):
        super().__init__(apply_to=["video"])
        self.use_first_frame_as_reference = use_first_frame_as_reference

    def apply(self, data: dict) -> dict:
        # Only add if not already present (don't overwrite RoboTwin's explicit values)
        if "vace_video" not in data:
            data["vace_video"] = None

        if "vace_reference_image" not in data:
            if self.use_first_frame_as_reference and "video" in data and data["video"]:
                data["vace_reference_image"] = [data["video"][0]]
            else:
                data["vace_reference_image"] = None

        return data
