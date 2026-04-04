"""Bridge V2 dataset adapter for WAM training.

BridgeData V2 (Berkeley):
- 60k trajectories (50k teleoperated + 10k rollouts)
- 7-DoF EEF delta actions (relative pose changes + gripper)
- 4 cameras: 1 RGBD over-shoulder + 2 RGB randomized + 1 wrist
- 24 environments, 13 skills
- Low-cost publicly available robot

Data format: LeRobot v2/v3 (HuggingFace datasets with parquet + mp4).
Download: ``huggingface-cli download lerobot/bridge --repo-type dataset``
"""

from typing import Optional

from open_wam.data.lerobot_base import LeRobotBaseDataset
from open_wam.data.transforms.base import ModalityTransform


class BridgeV2Dataset(LeRobotBaseDataset):
    """Bridge V2 dataset for WAM training.

    Thin subclass of :class:`LeRobotBaseDataset` with Bridge V2 defaults.

    Args:
        dataset_dir: Root directory of the Bridge V2 dataset.
        **kwargs: Passed to LeRobotBaseDataset.
    """

    _DEFAULT_CAMERA = "image_0"
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
        val_ratio: float = 0.1,
        seed: int = 42,
        transforms: Optional[ModalityTransform] = None,
    ):
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
