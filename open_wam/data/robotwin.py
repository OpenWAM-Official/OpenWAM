"""RoboTwin dataset classes implementing BaseActionDataset.

The dataset implementations live in ``_robotwin_impl`` and directly
implement the ``BaseActionDataset`` interface. This module re-exports
them under the public API names along with constants and helpers.
"""

from open_wam.data._robotwin_impl import (
    RoboTwinDataset as RoboTwinActionDataset,
    MultiTaskRoboTwinDataset as MultiTaskRoboTwinActionDataset,
    ROBOTWIN_TRAIN_TASKS,
    ROBOTWIN_HOLDOUT_TASKS,
    ROBOTWIN_ALL_TASKS,
    MULTIVIEW_LAYOUT,
    MULTIVIEW_CAMERAS,
    BACKBONE_SUPPORTED_RESOLUTIONS,
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
