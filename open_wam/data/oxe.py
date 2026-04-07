"""Open X-Embodiment (OXE) dataset adapter for WAM training.

OXE is a large-scale multi-robot dataset aggregating demonstrations from
dozens of embodiments. Datasets are distributed via HuggingFace in
LeRobot v2/v3 format (parquet + mp4).

This adapter uses ActionSpaceAdapter to unify different embodiments into
the canonical 7D/14D action space.
"""

import logging
from typing import Dict, Optional

import numpy as np

from open_wam.data.embodiment import CANONICAL_SINGLE_ARM_DIM, ActionSpaceAdapter
from open_wam.data.lerobot_base import LeRobotBaseDataset
from open_wam.data.transforms.base import ModalityTransform

logger = logging.getLogger(__name__)


# Known OXE dataset configurations
# Maps dataset_name -> (default_camera, default_action_key, default_embodiment, action_dim)
OXE_DATASET_REGISTRY: Dict[str, dict] = {
    "fractal": {"camera": "image", "action_key": "action", "embodiment": "google_robot", "action_dim": 7},
    "bridge": {"camera": "image_0", "action_key": "action", "embodiment": "widowx", "action_dim": 7},
    "kuka": {"camera": "image", "action_key": "action", "embodiment": "kuka", "action_dim": 7},
    "toto": {"camera": "image", "action_key": "action", "embodiment": "sawyer", "action_dim": 7},
    "jaco_play": {"camera": "image", "action_key": "action", "embodiment": "jaco", "action_dim": 7},
    "austin_buds": {"camera": "image", "action_key": "action", "embodiment": "franka", "action_dim": 7},
    "austin_sailor": {"camera": "image", "action_key": "action", "embodiment": "franka", "action_dim": 7},
    "austin_sirius": {"camera": "image", "action_key": "action", "embodiment": "franka", "action_dim": 7},
    "berkeley_autolab_ur5": {"camera": "image", "action_key": "action", "embodiment": "ur5", "action_dim": 7},
    "roboturk": {"camera": "image", "action_key": "action", "embodiment": "sawyer", "action_dim": 7},
    "stanford_hydra": {"camera": "image", "action_key": "action", "embodiment": "franka", "action_dim": 7},
    "ucsd_kitchen": {"camera": "image", "action_key": "action", "embodiment": "xarm", "action_dim": 7},
}


class OXEDataset(LeRobotBaseDataset):
    """Open X-Embodiment dataset with cross-embodiment action conversion.

    Extends :class:`LeRobotBaseDataset` with:
    - OXE_DATASET_REGISTRY for auto-detecting camera/action_key/embodiment
    - ActionSpaceAdapter for converting native actions to canonical format

    Args:
        dataset_dir: Root directory of the OXE subset.
        dataset_name: OXE subset name (e.g., "fractal", "bridge", "kuka").
        embodiment: Robot embodiment name override.
        canonical_action_dim: Target canonical action dimension.
        **kwargs: Passed to LeRobotBaseDataset.
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str = "fractal",
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        camera: Optional[str] = None,
        action_key: Optional[str] = None,
        embodiment: Optional[str] = None,
        canonical_action_dim: int = CANONICAL_SINGLE_ARM_DIM,
        action_stats_path: Optional[str] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        transforms: Optional[ModalityTransform] = None,
    ):
        self.dataset_name = dataset_name
        self.canonical_action_dim = canonical_action_dim

        # Resolve dataset-specific defaults from registry
        defaults = OXE_DATASET_REGISTRY.get(dataset_name, {})
        resolved_camera = camera or defaults.get("camera", "image")
        resolved_action_key = action_key or defaults.get("action_key", "action")
        embodiment_name = embodiment or defaults.get("embodiment", "franka")

        # Action space adapter for cross-embodiment normalization
        self._adapter = ActionSpaceAdapter(
            embodiment=embodiment_name,
            target_dim=canonical_action_dim,
        )

        super().__init__(
            dataset_dir=dataset_dir,
            num_frames=num_frames,
            height=height,
            width=width,
            split=split,
            camera=resolved_camera,
            action_key=resolved_action_key,
            action_stats_path=action_stats_path,
            val_ratio=val_ratio,
            seed=seed,
            transforms=transforms,
        )

        logger.info(
            "OXEDataset '%s': %d episodes (%s split), embodiment=%s, camera=%s",
            dataset_name,
            len(self._indices),
            split,
            embodiment_name,
            resolved_camera,
        )

    def _post_load_actions(self, actions: np.ndarray) -> np.ndarray:
        """Convert native robot actions to canonical via embodiment adapter."""
        return self._adapter.native_to_canonical(actions)

    @property
    def action_dim(self) -> int:
        return self.canonical_action_dim
