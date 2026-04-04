"""Utilities for loading robot manipulation datasets in LeRobot / HuggingFace format.

LeRobot v2/v3 datasets use HuggingFace ``datasets`` library with:
- Parquet metadata files (episode indices, timestamps, actions)
- MP4 video files (camera observations)

This module provides common functions for loading episodes as
(video_frames, action_trajectory) pairs compatible with BaseActionDataset.
"""

import os
import glob
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from PIL import Image


def load_episode_actions(
    parquet_path: str,
    episode_id: int,
    action_key: str = "action",
) -> np.ndarray:
    """Load action trajectory for a single episode from parquet metadata.

    Args:
        parquet_path: Path to the parquet file containing episode data.
        episode_id: Episode index to extract.
        action_key: Column name for action data.

    Returns:
        (T, action_dim) numpy array of actions.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        episode_data = df[df["episode_index"] == episode_id]
        actions = np.stack(episode_data[action_key].values)
        return actions.astype(np.float32)
    except ImportError:
        raise ImportError(
            "pandas and pyarrow are required for LeRobot format. "
            "Install with: pip install pandas pyarrow"
        )


def load_episode_video_frames(
    video_path: str,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    height: int = 480,
    width: int = 832,
) -> List[Image.Image]:
    """Load video frames from an MP4 file.

    Uses the multi-backend video reader (decord → opencv → imageio).

    Args:
        video_path: Path to the MP4 video file.
        start_frame: First frame to extract.
        num_frames: Number of frames to extract (None = all).
        height: Target resize height.
        width: Target resize width.

    Returns:
        List of PIL Images.
    """
    from open_wam.data.video_reader import read_video_frames

    end = start_frame + num_frames if num_frames is not None else None
    return read_video_frames(
        video_path, start=start_frame, end=end,
        height=height, width=width,
    )


def compute_dataset_action_stats(
    all_actions: List[np.ndarray],
) -> dict:
    """Compute extended action statistics across all episodes.

    Computes mean, std, min, max, q01, q99 for all normalization modes.
    Backward compatible — old code that reads only ``mean``/``std`` still works.

    Args:
        all_actions: List of (T_i, action_dim) arrays.

    Returns:
        dict with float32 arrays: mean, std, min, max, q01, q99.
    """
    concatenated = np.concatenate(all_actions, axis=0).astype(np.float64)
    mean = concatenated.mean(axis=0)
    std = np.maximum(concatenated.std(axis=0), 1e-3)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min": concatenated.min(axis=0).astype(np.float32),
        "max": concatenated.max(axis=0).astype(np.float32),
        "q01": np.percentile(concatenated, 1, axis=0).astype(np.float32),
        "q99": np.percentile(concatenated, 99, axis=0).astype(np.float32),
    }


def normalize_actions(
    actions: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """Normalize actions using precomputed statistics."""
    return (actions - stats["mean"]) / stats["std"]


def discover_episodes(data_dir: str, pattern: str = "*.parquet") -> List[str]:
    """Discover all parquet files in a dataset directory."""
    return sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
