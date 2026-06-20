"""Video -> tensor normalization helpers for the LAPA-DINOv3 provider."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def video_to_tensor(video: Any) -> torch.Tensor:
    """Convert one sample video to ``(T, 3, H, W)`` float in ``[0, 1]``."""
    if isinstance(video, torch.Tensor):
        x = video.detach()
        if x.ndim != 4:
            raise ValueError(f"video tensor must be 4D, got {tuple(x.shape)}")
        if x.shape[0] == 3 and x.shape[1] == 3:
            raise ValueError(
                "Ambiguous video tensor layout with both C and T equal to 3; "
                "pass channels-last (T,H,W,3) or a list of frames."
            )
        if x.shape[1] == 3:
            # (T, C, H, W)
            pass
        elif x.shape[-1] == 3:
            # (T, H, W, C)
            x = x.permute(0, 3, 1, 2)
        elif x.shape[0] == 3:
            # (C, T, H, W)
            x = x.permute(1, 0, 2, 3)
        else:
            raise ValueError(f"cannot infer channel axis for video tensor shape {tuple(x.shape)}")
        x = x.float()
        if x.numel() and float(x.max()) > 2.0:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    frames = list(video)
    if len(frames) < 2:
        raise ValueError("latent action requires at least 2 video frames")
    tensors: list[torch.Tensor] = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            t = frame.detach()
            if t.ndim != 3:
                raise ValueError(f"video frame tensor must be 3D, got {tuple(t.shape)}")
            if t.shape[0] == 3:
                pass
            elif t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            else:
                raise ValueError(f"cannot infer channel axis for frame shape {tuple(t.shape)}")
            t = t.float()
            if t.numel() and float(t.max()) > 2.0:
                t = t / 255.0
        elif isinstance(frame, Image.Image):
            arr = np.asarray(frame.convert("RGB"))
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        else:
            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"video frame array must be HWC RGB, got shape {arr.shape}")
            t = torch.from_numpy(arr).permute(2, 0, 1).float()
            if t.numel() and float(t.max()) > 2.0:
                t = t / 255.0
        tensors.append(t.clamp(0.0, 1.0))
    return torch.stack(tensors, dim=0)


def stack_videos(videos: list[Any]) -> torch.Tensor:
    tensors = [video_to_tensor(v) for v in videos]
    lengths = {int(t.shape[0]) for t in tensors}
    if len(lengths) != 1:
        raise ValueError(f"LAPA online provider requires uniform T_video in a batch, got {sorted(lengths)}")
    if next(iter(lengths)) < 2:
        raise ValueError("LAPA online provider requires at least 2 video frames")
    return torch.stack(tensors, dim=0)
