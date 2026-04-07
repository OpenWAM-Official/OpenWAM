"""RoboTwin dataset classes implementing BaseActionDataset.

The dataset implementations live in ``_robotwin_impl`` and directly
implement the ``BaseActionDataset`` interface. This module re-exports
them under the public API names along with constants and helpers.
"""

from open_wam.data._robotwin_impl import (
    BACKBONE_SUPPORTED_RESOLUTIONS,
    MULTIVIEW_CAMERAS,
    MULTIVIEW_LAYOUT,
    ROBOTWIN_ALL_TASKS,
    ROBOTWIN_HOLDOUT_TASKS,
    ROBOTWIN_TRAIN_TASKS,
    _crop_and_resize,  # noqa: F401 — re-exported for robotwin_policy
    _pad_and_resize,  # noqa: F401 — re-exported for robotwin_policy
    _resize_frame,  # noqa: F401 — re-exported for robotwin_policy
    assemble_multiview_grid,
    discover_robotwin_roots,
    extract_quadrant,
)
from open_wam.data._robotwin_impl import (
    MultiTaskRoboTwinDataset as MultiTaskRoboTwinActionDataset,
)
from open_wam.data._robotwin_impl import (
    RoboTwinDataset as RoboTwinActionDataset,
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
