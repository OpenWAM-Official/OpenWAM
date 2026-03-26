"""DROID dataset adapter for WAM training.

DROID (DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset):
- 76k demonstrations, 350 hours of interaction
- 7-DoF EEF actions (absolute pose + gripper)
- 3 cameras: 2 exterior Zed 2 + 1 wrist Zed Mini
- Control frequency: 15 Hz
- Robot: Franka Panda

Data format: LeRobot v2/v3 (HuggingFace datasets with parquet + mp4).
Download: ``huggingface-cli download lerobot/droid --repo-type dataset``
or manually to ``/path/to/vla_data/droid/``
"""

import os
import numpy as np
import torch
from typing import Optional, List
from pathlib import Path
from PIL import Image

from open_wam.data.base import BaseActionDataset
from open_wam.data.lerobot_utils import (
    compute_dataset_action_stats,
    normalize_actions,
)


class DROIDDataset(BaseActionDataset):
    """DROID dataset for WAM training.

    Loads DROID episodes from LeRobot format (parquet + mp4) or RLDS format,
    returning video frame sequences and 7-DoF action trajectories.

    Args:
        dataset_dir: Root directory of the DROID dataset.
        num_frames: Number of video frames per sample.
        height: Target video height.
        width: Target video width.
        split: "train" or "val".
        camera: Camera to use ("exterior_image_1_left", "wrist_image_left", etc.).
        action_stats_path: Path to precomputed action stats (.npy).
        action_type: "absolute" (raw EEF pose) or "delta" (relative changes).
    """

    # DROID uses 7-DoF EEF: (x, y, z, roll, pitch, yaw, gripper)
    _ACTION_DIM = 7

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        camera: str = "exterior_image_1_left",
        action_stats_path: Optional[str] = None,
        action_type: str = "absolute",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.split = split
        self.camera = camera
        self.action_type = action_type

        # Discover episodes
        self._episodes = self._discover_episodes(dataset_dir)

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

    def _discover_episodes(self, dataset_dir: str) -> List[dict]:
        """Discover available episodes in the dataset directory.

        Supports multiple directory layouts:
        - LeRobot format: parquet metadata + mp4 videos
        - HDF5 format: episode*.hdf5 files
        """
        episodes = []

        # Try LeRobot format (parquet + videos/)
        parquet_dir = os.path.join(dataset_dir, "data")
        video_dir = os.path.join(dataset_dir, "videos")

        if os.path.isdir(parquet_dir) and os.path.isdir(video_dir):
            import glob
            parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True))
            for pf in parquet_files:
                episodes.append({"type": "lerobot", "parquet": pf, "video_dir": video_dir})
        else:
            # Fallback: list directories as episodes
            for entry in sorted(os.listdir(dataset_dir)):
                ep_dir = os.path.join(dataset_dir, entry)
                if os.path.isdir(ep_dir):
                    episodes.append({"type": "directory", "path": ep_dir})

        if not episodes:
            raise FileNotFoundError(
                f"No episodes found in {dataset_dir}. "
                f"Download DROID with: huggingface-cli download lerobot/droid --repo-type dataset"
            )
        return episodes

    def __getitem__(self, idx: int) -> dict:
        ep_idx = self._indices[idx]
        episode = self._episodes[ep_idx]

        if episode["type"] == "lerobot":
            return self._load_lerobot_episode(episode, ep_idx)
        else:
            return self._load_directory_episode(episode)

    def _load_lerobot_episode(self, episode: dict, episode_id: int) -> dict:
        """Load from LeRobot parquet + mp4 format."""
        import pandas as pd

        df = pd.read_parquet(episode["parquet"])
        ep_data = df[df["episode_index"] == episode_id % len(df["episode_index"].unique())]

        # Load actions
        actions = np.stack(ep_data["action"].values).astype(np.float32)
        T = len(actions)
        start = 0
        if T > self.num_frames:
            start = np.random.randint(0, T - self.num_frames)
        end = min(start + self.num_frames, T)
        actions = actions[start:end]

        # Load video frames
        video_path = os.path.join(
            episode["video_dir"], self.camera,
            f"episode_{episode_id:06d}.mp4"
        )
        frames = self._load_video_frames(video_path, start, end)

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
            "prompt": "robot manipulation task",
            "vace_video": None,
            "vace_reference_image": [frames[0]] if frames else None,
            "episode_index": episode_id,
        }

    def _load_directory_episode(self, episode: dict) -> dict:
        """Load from raw directory format."""
        raise NotImplementedError(
            "Raw DROID directory loading not yet implemented. "
            "Please convert to LeRobot format first."
        )

    def _load_video_frames(self, video_path: str, start: int, end: int) -> List[Image.Image]:
        """Load and resize video frames from MP4."""
        if not os.path.exists(video_path):
            import logging
            logging.getLogger(__name__).warning("Video file not found: %s — using black placeholder frames", video_path)
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
        return self._ACTION_DIM

    @property
    def action_stats(self) -> Optional[dict]:
        return self._action_stats_cache
