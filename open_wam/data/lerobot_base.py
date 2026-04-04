"""Base class for all LeRobot-format (parquet + MP4) datasets.

Eliminates code duplication across DROID, Bridge V2, and OXE datasets
by extracting shared logic: episode discovery, video loading, action
padding, normalization, and train/val splitting.

Subclasses only need to set class attributes for defaults and optionally
override ``_post_load_actions()`` for dataset-specific processing
(e.g., OXE's embodiment adapter).
"""

import os
import logging
from typing import Optional, List

import numpy as np
import torch
from PIL import Image

from open_wam.data.base import BaseActionDataset
from open_wam.data.transforms.base import ComposedTransform, ModalityTransform

logger = logging.getLogger(__name__)


class LeRobotBaseDataset(BaseActionDataset):
    """Base for LeRobot-format datasets (parquet metadata + MP4 videos).

    Provides shared implementation for episode discovery, frame loading,
    action padding, normalization, and train/val splitting.

    Subclasses should set:
        _DEFAULT_CAMERA:    Default camera name (e.g., "exterior_image_1_left")
        _DEFAULT_ACTION_KEY: Default parquet column for actions (e.g., "action")
        _ACTION_DIM:        Native action dimensionality

    Args:
        dataset_dir: Root directory of the dataset.
        num_frames: Number of video frames per sample.
        height: Target video height.
        width: Target video width.
        split: "train" or "val".
        camera: Camera name (None → use _DEFAULT_CAMERA).
        action_key: Action column name (None → use _DEFAULT_ACTION_KEY).
        action_stats_path: Path to precomputed action stats (.npy).
        val_ratio: Fraction of episodes reserved for validation.
        seed: Random seed for train/val split.
        transforms: Optional transform pipeline applied to each sample.
    """

    _DEFAULT_CAMERA: str = "image"
    _DEFAULT_ACTION_KEY: str = "action"
    _ACTION_DIM: int = 7

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        camera: Optional[str] = None,
        action_key: Optional[str] = None,
        action_stats_path: Optional[str] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        transforms: Optional[ModalityTransform] = None,
    ):
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.split = split
        self.camera = camera or self._DEFAULT_CAMERA
        self.action_key = action_key or self._DEFAULT_ACTION_KEY
        self.transforms = transforms

        # Discover episodes
        self._episodes = self._discover_episodes(dataset_dir)
        if not self._episodes:
            raise FileNotFoundError(
                f"No episodes found in {dataset_dir}. "
                f"Expected LeRobot format: data/*.parquet + videos/*.mp4"
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
        else:
            # Fallback: list directories as episodes
            for entry in sorted(os.listdir(dataset_dir)):
                ep_dir = os.path.join(dataset_dir, entry)
                if os.path.isdir(ep_dir):
                    episodes.append({"type": "directory", "path": ep_dir})

        return episodes

    def __getitem__(self, idx: int) -> dict:
        ep_idx = self._indices[idx]
        episode = self._episodes[ep_idx]

        if "parquet" in episode:
            data = self._load_lerobot_episode(episode, ep_idx)
        else:
            raise NotImplementedError(
                "Raw directory loading not implemented. Convert to LeRobot format."
            )

        # Apply transforms if configured
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def _load_lerobot_episode(self, episode: dict, episode_id: int) -> dict:
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

        # Hook for subclass processing (e.g., embodiment conversion)
        actions = self._post_load_actions(actions)

        # Load video frames
        frames = self._load_video_frames(episode, episode_id, start, end)

        # Pad
        actions, action_mask = self._pad_actions(actions)
        frames = self._pad_frames(frames)

        # Normalize actions (legacy z-score path; prefer transforms pipeline)
        if self.transforms is None and self._action_stats_cache is not None:
            actions = (actions - self._action_stats_cache["mean"]) / self._action_stats_cache["std"]

        return {
            "video": frames,
            "action": torch.from_numpy(actions),
            "action_trajectory": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
            "prompt": "robot manipulation task",
            "episode_index": episode_id,
        }

    def _post_load_actions(self, actions: np.ndarray) -> np.ndarray:
        """Hook for subclass-specific action processing.

        Override to apply embodiment conversion, action type transforms, etc.
        Default: identity (no-op).
        """
        return actions

    def _load_video_frames(
        self, episode: dict, episode_id: int, start: int, end: int,
    ) -> List[Image.Image]:
        """Load video frames from MP4 using the best available backend.

        Backend priority: decord → opencv → imageio.
        Override via ``OPENWAM_VIDEO_BACKEND`` env var or
        ``open_wam.data.video_reader.set_video_backend()``.
        """
        video_dir = episode.get("video_dir")
        if video_dir is None:
            return [Image.new("RGB", (self.width, self.height)) for _ in range(end - start)]

        video_path = os.path.join(
            video_dir, self.camera, f"episode_{episode_id:06d}.mp4"
        )
        if not os.path.exists(video_path):
            logger.warning("Video not found: %s — using placeholder frames", video_path)
            return [Image.new("RGB", (self.width, self.height)) for _ in range(end - start)]

        from open_wam.data.video_reader import read_video_frames
        return read_video_frames(
            video_path, start=start, end=end,
            height=self.height, width=self.width,
        )

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
