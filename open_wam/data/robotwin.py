"""RoboTwin dataset wrappers conforming to the BaseActionDataset interface.

Thin wrappers around the legacy ``video_action_dataset.py`` implementations.
The legacy code is imported via sys.path manipulation and all behavior is
delegated unchanged — this guarantees functional equivalence.
"""

import sys
from pathlib import Path
from typing import Optional

from open_wam.data.base import BaseActionDataset

# Make legacy module importable
_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from video_action_dataset import (  # noqa: E402
    RoboTwinDataset as _LegacyRoboTwinDataset,
    MultiTaskRoboTwinDataset as _LegacyMultiTaskRoboTwinDataset,
    # Constants
    ROBOTWIN_TRAIN_TASKS,
    ROBOTWIN_HOLDOUT_TASKS,
    ROBOTWIN_ALL_TASKS,
    MULTIVIEW_LAYOUT,
    MULTIVIEW_CAMERAS,
    BACKBONE_SUPPORTED_RESOLUTIONS,
    # Helpers
    discover_robotwin_roots,
    assemble_multiview_grid,
    extract_quadrant,
    _crop_and_resize,
    _pad_and_resize,
    _resize_frame,
)

__all__ = [
    "RoboTwinActionDataset",
    "MultiTaskRoboTwinActionDataset",
    "ROBOTWIN_TRAIN_TASKS",
    "ROBOTWIN_HOLDOUT_TASKS",
    "ROBOTWIN_ALL_TASKS",
    "MULTIVIEW_LAYOUT",
    "MULTIVIEW_CAMERAS",
    "BACKBONE_SUPPORTED_RESOLUTIONS",
    "discover_robotwin_roots",
    "assemble_multiview_grid",
    "extract_quadrant",
]


class RoboTwinActionDataset(BaseActionDataset):
    """RoboTwin 2.0 single-task dataset (wraps legacy RoboTwinDataset).

    All parameters are forwarded to the legacy implementation unchanged.
    ``__getitem__`` returns the legacy dict with ``action_trajectory`` aliased
    as ``action`` for BaseActionDataset compatibility.
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        val_ratio: float = 0.1,
        repeat: int = 1,
        task_name: Optional[str] = None,
        seed: int = 42,
        action_stats_path: Optional[str] = None,
        num_val_samples: int = 4,
        target_camera: str = "head_camera",
        window_stride: int = 1,
        multiview: bool = False,
        robot: Optional[str] = None,
        variant: str = "clean_50",
        backbone: Optional[str] = None,
    ):
        self._legacy = _LegacyRoboTwinDataset(
            data_root=data_root,
            num_frames=num_frames,
            height=height,
            width=width,
            split=split,
            val_ratio=val_ratio,
            repeat=repeat,
            task_name=task_name,
            seed=seed,
            action_stats_path=action_stats_path,
            num_val_samples=num_val_samples,
            target_camera=target_camera,
            window_stride=window_stride,
            multiview=multiview,
            robot=robot,
            variant=variant,
            backbone=backbone,
        )

    def __getitem__(self, idx: int) -> dict:
        sample = self._legacy[idx]
        # Alias for BaseActionDataset compatibility
        sample["action"] = sample["action_trajectory"]
        return sample

    def __len__(self) -> int:
        return len(self._legacy)

    @property
    def action_dim(self) -> int:
        return self._legacy.action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._legacy.action_stats

    def denormalize_action(self, action):
        return self._legacy.denormalize_action(action)


class MultiTaskRoboTwinActionDataset(BaseActionDataset):
    """RoboTwin 2.0 multi-task dataset (wraps legacy MultiTaskRoboTwinDataset).

    All parameters are forwarded to the legacy implementation unchanged.
    """

    def __init__(
        self,
        dataset_dir: str,
        robot: str,
        variant: str = "clean_50",
        tasks: Optional[list] = None,
        action_stats_path: Optional[str] = None,
        **kwargs,
    ):
        self._legacy = _LegacyMultiTaskRoboTwinDataset(
            dataset_dir=dataset_dir,
            robot=robot,
            variant=variant,
            tasks=tasks,
            action_stats_path=action_stats_path,
            **kwargs,
        )

    def __getitem__(self, idx: int) -> dict:
        sample = self._legacy[idx]
        sample["action"] = sample["action_trajectory"]
        return sample

    def __len__(self) -> int:
        return len(self._legacy)

    @property
    def action_dim(self) -> int:
        return self._legacy.action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._legacy.action_stats

    def denormalize_action(self, action):
        return self._legacy.denormalize_action(action)
