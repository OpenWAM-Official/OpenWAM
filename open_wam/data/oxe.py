"""Open X-Embodiment (OXE) dataset adapter for WAM training.

OXE is a large-scale multi-robot dataset aggregating demonstrations from
dozens of embodiments and environments. Datasets are distributed via
HuggingFace in LeRobot v2/v3 format (parquet + mp4).

OXE subsets include:
- fractal (Google Robot): 87k episodes, table-top manipulation
- bridge (WidowX): 60k episodes, diverse tasks
- kuka (KUKA iiwa): bin picking, stacking
- toto (Sawyer): various manipulation
- jaco_play (Kinova Jaco): tabletop play

Data format: LeRobot v2/v3 (HuggingFace datasets with parquet + mp4).
Download: ``huggingface-cli download <org>/<dataset_name> --repo-type dataset``

This adapter uses ActionSpaceAdapter to unify different embodiments into
the canonical 7D/14D action space.
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, List, Dict
from pathlib import Path
from PIL import Image

from open_wam.data.base import BaseActionDataset
from open_wam.data.lerobot_utils import (
    compute_dataset_action_stats,
    normalize_actions,
)
from open_wam.data.embodiment import ActionSpaceAdapter, CANONICAL_SINGLE_ARM_DIM

logger = logging.getLogger(__name__)


# Known OXE dataset configurations
# Maps dataset_name -> (default_camera, default_action_key, default_embodiment)
OXE_DATASET_REGISTRY: Dict[str, dict] = {
    "fractal": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "google_robot",
        "action_dim": 7,
    },
    "bridge": {
        "camera": "image_0",
        "action_key": "action",
        "embodiment": "widowx",
        "action_dim": 7,
    },
    "kuka": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "kuka",
        "action_dim": 7,
    },
    "toto": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "sawyer",
        "action_dim": 7,
    },
    "jaco_play": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "jaco",
        "action_dim": 7,
    },
    "austin_buds": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "franka",
        "action_dim": 7,
    },
    "austin_sailor": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "franka",
        "action_dim": 7,
    },
    "austin_sirius": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "franka",
        "action_dim": 7,
    },
    "berkeley_autolab_ur5": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "ur5",
        "action_dim": 7,
    },
    "roboturk": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "sawyer",
        "action_dim": 7,
    },
    "stanford_hydra": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "franka",
        "action_dim": 7,
    },
    "ucsd_kitchen": {
        "camera": "image",
        "action_key": "action",
        "embodiment": "xarm",
        "action_dim": 7,
    },
}


class OXEDataset(BaseActionDataset):
    """Open X-Embodiment dataset for WAM training.

    Loads OXE subsets from LeRobot format, with automatic embodiment-aware
    action space conversion to a canonical format.

    Args:
        dataset_dir: Root directory of the OXE subset.
        dataset_name: OXE subset name (e.g., "fractal", "bridge", "kuka").
            Used to look up default camera, action key, and embodiment.
        num_frames: Number of video frames per sample.
        height: Target video height.
        width: Target video width.
        split: "train" or "val".
        camera: Camera name override (defaults to dataset-specific default).
        action_key: Action column name override.
        embodiment: Robot embodiment name override (for ActionSpaceAdapter).
        canonical_action_dim: Target canonical action dimension (default 7).
        action_stats_path: Path to precomputed action stats (.npy).
        val_ratio: Fraction of episodes reserved for validation.
        seed: Random seed for train/val split.
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
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.split = split
        self.canonical_action_dim = canonical_action_dim

        # Resolve dataset-specific defaults
        defaults = OXE_DATASET_REGISTRY.get(dataset_name, {})
        self.camera = camera or defaults.get("camera", "image")
        self.action_key = action_key or defaults.get("action_key", "action")
        embodiment_name = embodiment or defaults.get("embodiment", "franka")

        # Action space adapter for cross-embodiment normalization
        self._adapter = ActionSpaceAdapter(
            embodiment=embodiment_name,
            target_dim=canonical_action_dim,
        )
        self._native_action_dim = defaults.get("action_dim", 7)

        # Discover episodes
        self._episodes = self._discover_episodes(dataset_dir)
        if not self._episodes:
            raise FileNotFoundError(
                f"No episodes found in {dataset_dir} for OXE subset '{dataset_name}'."
            )

        # Train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self._episodes))
        n_val = max(1, int(len(indices) * val_ratio))
        if split == "val":
            self._indices = sorted(indices[:n_val])
        else:
            self._indices = sorted(indices[n_val:])

        # Action stats
        self._action_stats_cache = None
        if action_stats_path and os.path.exists(action_stats_path):
            stats = np.load(action_stats_path, allow_pickle=True).item()
            stats["std"] = np.maximum(stats["std"].astype(np.float32), 1e-3)
            stats["mean"] = stats["mean"].astype(np.float32)
            self._action_stats_cache = stats

        logger.info(
            "OXEDataset '%s': %d episodes (%s split), embodiment=%s, camera=%s",
            dataset_name, len(self._indices), split, embodiment_name, self.camera,
        )

    def _discover_episodes(self, dataset_dir: str) -> List[dict]:
        """Discover available episodes in LeRobot format."""
        episodes = []
        parquet_dir = os.path.join(dataset_dir, "data")
        video_dir = os.path.join(dataset_dir, "videos")

        if os.path.isdir(parquet_dir):
            import glob as globmod
            parquet_files = sorted(
                globmod.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True)
            )
            for pf in parquet_files:
                episodes.append({
                    "parquet": pf,
                    "video_dir": video_dir if os.path.isdir(video_dir) else None,
                })
        return episodes

    def __getitem__(self, idx: int) -> dict:
        ep_idx = self._indices[idx]
        episode = self._episodes[ep_idx]
        return self._load_episode(episode, ep_idx)

    def _load_episode(self, episode: dict, episode_id: int) -> dict:
        """Load a single episode from LeRobot parquet + mp4 format."""
        import pandas as pd

        df = pd.read_parquet(episode["parquet"])
        unique_eps = sorted(df["episode_index"].unique())
        actual_ep_id = unique_eps[episode_id % len(unique_eps)]
        ep_data = df[df["episode_index"] == actual_ep_id]

        # Load actions
        actions = np.stack(ep_data[self.action_key].values).astype(np.float32)
        T = len(actions)
        start = 0
        if T > self.num_frames:
            start = np.random.randint(0, T - self.num_frames)
        end = min(start + self.num_frames, T)
        actions = actions[start:end]

        # Convert native actions to canonical via embodiment adapter
        actions = self._adapter.native_to_canonical(actions)

        # Load video frames
        frames = self._load_video_frames(episode, episode_id, start, end)

        # Pad if needed
        actions, action_mask = self._pad_actions(actions)
        frames = self._pad_frames(frames)

        # Normalize actions
        if self._action_stats_cache is not None:
            actions = normalize_actions(actions, self._action_stats_cache)

        return {
            "video": frames,
            "action": torch.from_numpy(actions),
            "action_trajectory": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
            "prompt": f"robot manipulation task",
            "vace_video": None,
            "vace_reference_image": [frames[0]] if frames else None,
            "episode_index": episode_id,
        }

    def _load_video_frames(
        self, episode: dict, episode_id: int, start: int, end: int,
    ) -> List[Image.Image]:
        """Load video frames from MP4 or return placeholder frames."""
        video_dir = episode.get("video_dir")
        if video_dir is None:
            return [Image.new("RGB", (self.width, self.height)) for _ in range(end - start)]

        video_path = os.path.join(
            video_dir, self.camera, f"episode_{episode_id:06d}.mp4"
        )
        if not os.path.exists(video_path):
            logger.warning(
                "Video not found: %s — using placeholder frames", video_path
            )
            return [Image.new("RGB", (self.width, self.height)) for _ in range(end - start)]

        import imageio
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            if i < start:
                continue
            if i >= end:
                break
            img = Image.fromarray(frame).resize((self.width, self.height), Image.LANCZOS)
            frames.append(img)
        reader.close()
        return frames

    def _pad_actions(self, actions: np.ndarray):
        """Pad actions to num_frames, return (padded_actions, mask)."""
        T = len(actions)
        if T >= self.num_frames:
            return actions[:self.num_frames], np.ones(self.num_frames, dtype=bool)
        pad = np.zeros((self.num_frames - T, actions.shape[1]), dtype=np.float32)
        padded = np.concatenate([actions, pad], axis=0)
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(self.num_frames - T, dtype=bool)])
        return padded, mask

    def _pad_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Pad frame list to num_frames."""
        if len(frames) >= self.num_frames:
            return frames[:self.num_frames]
        last = frames[-1] if frames else Image.new("RGB", (self.width, self.height))
        return frames + [last] * (self.num_frames - len(frames))

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def action_dim(self) -> int:
        return self.canonical_action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._action_stats_cache
