"""Bridge V2 dataset adapter for WAM training.

BridgeData V2 (Berkeley):
- 60k trajectories (50k teleoperated + 10k rollouts)
- 7-DoF EEF delta actions (relative pose changes + gripper)
- 4 cameras: 1 RGBD over-shoulder + 2 RGB randomized + 1 wrist
- 24 environments, 13 skills
- Low-cost publicly available robot

Data format: LeRobot v2/v3 (HuggingFace datasets with parquet + mp4).
Download: ``huggingface-cli download lerobot/bridge --repo-type dataset``
or manually to ``/path/to/vla_data/bridge_v2/``
"""

import os
import numpy as np
import torch
from typing import Optional, List
from PIL import Image

from open_wam.data.base import BaseActionDataset
from open_wam.data.lerobot_utils import normalize_actions


class BridgeV2Dataset(BaseActionDataset):
    """Bridge V2 dataset for WAM training.

    Loads Bridge V2 episodes from LeRobot format, returning video frame
    sequences and 7-DoF delta EEF action trajectories.

    Args:
        dataset_dir: Root directory of the Bridge V2 dataset.
        num_frames: Number of video frames per sample.
        height: Target video height.
        width: Target video width.
        split: "train" or "val".
        camera: Camera to use ("image_0" primary, "image_1", etc.).
        action_stats_path: Path to precomputed action stats (.npy).
    """

    # Bridge V2: 7-DoF EEF delta (dx, dy, dz, droll, dpitch, dyaw, gripper)
    _ACTION_DIM = 7

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        camera: str = "image_0",
        action_stats_path: Optional[str] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.split = split
        self.camera = camera

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
        """Discover available episodes."""
        episodes = []

        parquet_dir = os.path.join(dataset_dir, "data")
        video_dir = os.path.join(dataset_dir, "videos")

        if os.path.isdir(parquet_dir) and os.path.isdir(video_dir):
            import glob
            parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True))
            for pf in parquet_files:
                episodes.append({"type": "lerobot", "parquet": pf, "video_dir": video_dir})
        else:
            for entry in sorted(os.listdir(dataset_dir)):
                ep_dir = os.path.join(dataset_dir, entry)
                if os.path.isdir(ep_dir):
                    episodes.append({"type": "directory", "path": ep_dir})

        if not episodes:
            raise FileNotFoundError(
                f"No episodes found in {dataset_dir}. "
                f"Download Bridge V2 with: huggingface-cli download lerobot/bridge --repo-type dataset"
            )
        return episodes

    def __getitem__(self, idx: int) -> dict:
        ep_idx = self._indices[idx]
        episode = self._episodes[ep_idx]

        if episode["type"] == "lerobot":
            return self._load_lerobot_episode(episode, ep_idx)
        else:
            raise NotImplementedError(
                "Raw Bridge V2 directory loading not yet implemented. "
                "Please convert to LeRobot format first."
            )

    def _load_lerobot_episode(self, episode: dict, episode_id: int) -> dict:
        """Load from LeRobot parquet + mp4 format."""
        import pandas as pd

        df = pd.read_parquet(episode["parquet"])
        unique_eps = sorted(df["episode_index"].unique())
        ep_id = unique_eps[episode_id % len(unique_eps)]
        ep_data = df[df["episode_index"] == ep_id]

        # Load actions (delta EEF)
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
            f"episode_{ep_id:06d}.mp4"
        )
        frames = self._load_video_frames(video_path, start, end)

        # Pad
        actions, action_mask = self._pad_actions(actions)
        frames = self._pad_frames(frames)

        # Normalize
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

    def _load_video_frames(self, video_path: str, start: int, end: int) -> List[Image.Image]:
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
        T = len(actions)
        if T >= self.num_frames:
            return actions[:self.num_frames], np.ones(self.num_frames, dtype=bool)
        pad = np.zeros((self.num_frames - T, actions.shape[1]), dtype=np.float32)
        padded = np.concatenate([actions, pad], axis=0)
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(self.num_frames - T, dtype=bool)])
        return padded, mask

    def _pad_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
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
