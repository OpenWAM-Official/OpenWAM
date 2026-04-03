"""Single boundary module for imports from examples/wanvideo/wam/.

This is the ONLY file in open_wam/ allowed to use sys.path manipulation.
All other modules that need legacy symbols must import them from here.

This boundary exists because examples/wanvideo/wam/ is a collection of
loose scripts (no __init__.py) that cannot be imported as a proper package.
It will be removed when the legacy code is fully decoupled (plan.md Phase 6).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WAM_DIR = str(_PROJECT_ROOT / "examples" / "wanvideo" / "wam")

if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

# --- Training symbols ---
from train_video_action import (  # noqa: E402
    FlowMatchVideoActionSFTLoss,
    VideoActionTrainingModule,
)

# --- Dataset symbols ---
from video_action_dataset import (  # noqa: E402
    RoboTwinDataset as LegacyRoboTwinDataset,
    MultiTaskRoboTwinDataset as LegacyMultiTaskRoboTwinDataset,
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

# --- Action stats symbols ---
from compute_action_stats import (  # noqa: E402
    compute_action_stats,
    compute_multitask_robotwin_stats,
    parse_tasks_file,
)

__all__ = [
    # Training
    "FlowMatchVideoActionSFTLoss",
    "VideoActionTrainingModule",
    # Dataset classes
    "LegacyRoboTwinDataset",
    "LegacyMultiTaskRoboTwinDataset",
    # Dataset constants
    "ROBOTWIN_TRAIN_TASKS",
    "ROBOTWIN_HOLDOUT_TASKS",
    "ROBOTWIN_ALL_TASKS",
    "MULTIVIEW_LAYOUT",
    "MULTIVIEW_CAMERAS",
    "BACKBONE_SUPPORTED_RESOLUTIONS",
    # Dataset helpers
    "discover_robotwin_roots",
    "assemble_multiview_grid",
    "extract_quadrant",
    "_crop_and_resize",
    "_pad_and_resize",
    "_resize_frame",
    # Action stats
    "compute_action_stats",
    "compute_multitask_robotwin_stats",
    "parse_tasks_file",
]
