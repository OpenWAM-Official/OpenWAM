"""DROID dataset adapter for WAM training.

DROID (DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset):
- 76k demonstrations, 350 hours of interaction
- 7-DoF EEF actions (absolute pose + gripper)
- 3 cameras: 2 exterior Zed 2 + 1 wrist Zed Mini
- Control frequency: 15 Hz
- Robot: Franka Panda

Data format: LeRobot v2/v3 (HuggingFace datasets with parquet + mp4).
Download: ``huggingface-cli download lerobot/droid --repo-type dataset``
"""

from typing import Optional

from open_wam.data.lerobot_base import LeRobotBaseDataset
from open_wam.data.transforms.base import ModalityTransform


class DROIDDataset(LeRobotBaseDataset):
    """DROID dataset for WAM training.

    Thin subclass of :class:`LeRobotBaseDataset` with DROID-specific defaults.

    Args:
        dataset_dir: Root directory of the DROID dataset.
        action_type: "absolute" (raw EEF pose) or "delta" (relative changes).
        **kwargs: Passed to LeRobotBaseDataset.
    """

    _DEFAULT_CAMERA = "exterior_image_1_left"
    _DEFAULT_ACTION_KEY = "action"
    _ACTION_DIM = 7

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        camera: Optional[str] = None,
        action_stats_path: Optional[str] = None,
        action_type: str = "absolute",
        val_ratio: float = 0.1,
        seed: int = 42,
        transforms: Optional[ModalityTransform] = None,
    ):
        self.action_type = action_type
        super().__init__(
            dataset_dir=dataset_dir,
            num_frames=num_frames,
            height=height,
            width=width,
            split=split,
            camera=camera,
            action_stats_path=action_stats_path,
            val_ratio=val_ratio,
            seed=seed,
            transforms=transforms,
        )
