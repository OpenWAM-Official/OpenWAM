"""Multi-backend video frame reader.

Supports multiple video backends with automatic fallback:
  - decord:   Fastest, GPU-accelerable (preferred)
  - opencv:   Widely available, reliable
  - imageio:  Universal fallback

Backend selection order: decord → opencv → imageio.
Override with ``set_video_backend("opencv")`` or env var ``OPENWAM_VIDEO_BACKEND``.

Usage::

    frames = read_video_frames("episode_000001.mp4", start=10, end=30)
    frames = read_video_frames(path, start=10, end=30, height=480, width=832)
"""

import logging
import os
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Global backend preference
_BACKEND: Optional[str] = None

# Backend availability cache
_AVAILABLE: dict = {}


def set_video_backend(backend: str):
    """Set the preferred video backend.

    Args:
        backend: One of "decord", "opencv", "imageio", or "auto".
    """
    global _BACKEND
    if backend == "auto":
        _BACKEND = None
    else:
        _BACKEND = backend


def get_video_backend() -> str:
    """Return the active video backend name."""
    return _resolve_backend()


def _check_available(name: str) -> bool:
    """Check if a backend is importable (cached)."""
    if name not in _AVAILABLE:
        try:
            if name == "decord":
                import decord  # noqa: F401

                _AVAILABLE[name] = True
            elif name == "opencv":
                import cv2  # noqa: F401

                _AVAILABLE[name] = True
            elif name == "imageio":
                import imageio  # noqa: F401

                _AVAILABLE[name] = True
            else:
                _AVAILABLE[name] = False
        except ImportError:
            _AVAILABLE[name] = False
    return _AVAILABLE[name]


def _resolve_backend() -> str:
    """Resolve which backend to use."""
    # Explicit override
    if _BACKEND is not None:
        if _check_available(_BACKEND):
            return _BACKEND
        logger.warning("Requested video backend '%s' not available, falling back", _BACKEND)

    # Environment variable
    env_backend = os.environ.get("OPENWAM_VIDEO_BACKEND")
    if env_backend and _check_available(env_backend):
        return env_backend

    # Auto-detect priority order
    for name in ("decord", "opencv", "imageio"):
        if _check_available(name):
            return name

    raise ImportError(
        "No video backend available. Install one of: "
        "pip install decord / pip install opencv-python / pip install imageio[ffmpeg]"
    )


def read_video_frames(
    video_path: str,
    start: int = 0,
    end: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> List[Image.Image]:
    """Read video frames from an MP4 file using the best available backend.

    Args:
        video_path: Path to the video file.
        start: First frame index to read.
        end: Last frame index (exclusive). None = read to end.
        height: Target resize height. None = no resize.
        width: Target resize width. None = no resize.

    Returns:
        List of PIL Images.
    """
    backend = _resolve_backend()

    if backend == "decord":
        frames = _read_decord(video_path, start, end)
    elif backend == "opencv":
        frames = _read_opencv(video_path, start, end)
    else:
        frames = _read_imageio(video_path, start, end)

    # Resize if requested
    if height is not None and width is not None:
        frames = [f.resize((width, height), Image.LANCZOS) for f in frames]

    return frames


def _read_decord(path: str, start: int, end: Optional[int]) -> List[Image.Image]:
    """Read frames using decord (fastest backend)."""
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path)
    total = len(vr)

    if end is None:
        end = total
    end = min(end, total)

    if start >= end:
        return []

    indices = list(range(start, end))
    frames_np = vr.get_batch(indices).asnumpy()  # (N, H, W, C)

    return [Image.fromarray(frames_np[i]) for i in range(len(indices))]


def _read_opencv(path: str, start: int, end: Optional[int]) -> List[Image.Image]:
    """Read frames using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    end = min(end, total)

    frames = []
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for i in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR → convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def _read_imageio(path: str, start: int, end: Optional[int]) -> List[Image.Image]:
    """Read frames using imageio (universal fallback)."""
    import imageio

    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        if i < start:
            continue
        if end is not None and i >= end:
            break
        frames.append(Image.fromarray(frame))
    reader.close()
    return frames
