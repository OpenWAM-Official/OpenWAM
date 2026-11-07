"""Shared PyAV video-frame decoding helper.

Hosts ``decode_video_frames`` — the seek-to-keyframe PyAV decode path used by
every LeRobot v3 reader (the ``LeRobotV3Reader`` family). Kept as a leaf utility
module so the shared base reader and the concrete readers depend on it instead
of reaching into a sibling reader.

This module is dependency-light (PIL + PyAV only, no project imports) so any
reader can import it without triggering heavy module side-effects or import cycles.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# Silence libav stderr spam ("Unknown OBU type 0", per-frame hwaccel probe
# warnings). Bypasses Python logging — must use av.logging directly.
try:
    import av as _av

    _av.logging.set_level(_av.logging.FATAL)
except Exception:
    pass


# libdav1d (FFmpeg's AV1 decoder, used for the AV1 clips of LeRobot v3 datasets) starts a thread pool sized to the
# host for every clip PyAV opens: FFmpeg passes its automatic thread count (about one thread per CPU core). With
# DataLoader workers x ranks decoding concurrently, these pools oversubscribe the host and the GPUs wait for data
# (LIBERO on 7x H200: load ~150 on 160 cores). A pool of 2 decodes identical frames with about half the CPU time.
# OPENWAM_VIDEO_DECODE_THREADS sets the pool size; 0 keeps FFmpeg's default. Other decoders (H.264, HEVC, ...) are
# left at PyAV's defaults.
_DECODE_THREADS_ENV = "OPENWAM_VIDEO_DECODE_THREADS"
_DEFAULT_DECODE_THREADS = 2
_MAX_DECODE_THREADS = 256  # libdav1d's DAV1D_MAX_THREADS


def _decode_thread_count() -> int:
    """libdav1d threads per opened clip from ``OPENWAM_VIDEO_DECODE_THREADS`` (default 2; 0 = FFmpeg default)."""
    raw = os.environ.get(_DECODE_THREADS_ENV, "").strip()
    if not raw:
        return _DEFAULT_DECODE_THREADS
    count = int(raw) if raw.isascii() and raw.isdigit() else -1
    if not 0 <= count <= _MAX_DECODE_THREADS:
        raise ValueError(f"{_DECODE_THREADS_ENV} must be an integer in [0, {_MAX_DECODE_THREADS}], got {raw!r}")
    return count


def _limit_decoder_threads(stream) -> None:
    """Cap the libdav1d thread pool of an opened AV1 stream; no-op for other decoders or a count of 0."""
    count = _decode_thread_count()
    if count and stream.codec_context.name == "libdav1d":
        stream.thread_count = count


_seek_fallback_count = 0
_seek_fallback_log_threshold = 20  # log first N events verbosely, then every 100th


def _warn_seek_fallback(video_path: str, min_idx: int, max_idx: int, reason: str) -> None:
    """Record + log a seek-fast-path fallback to seek(0) sequential decode.

    Why: seek-to-keyframe is a perf optimization; a fallback means we paid
    decode cost from frame 0 (slow). If this fires often it's a real signal
    (bad pts metadata, broken keyframe interval) — make it visible.
    """
    global _seek_fallback_count
    _seek_fallback_count += 1
    n = _seek_fallback_count
    if n <= _seek_fallback_log_threshold or n % 100 == 0:
        logger.warning(
            "video_io seek-fallback #%d: %s frames=[%d..%d] reason=%s",
            n,
            video_path,
            min_idx,
            max_idx,
            reason,
        )


def decode_video_frames(video_path: str, frame_indices: List[int], height: int, width: int) -> List[Image.Image]:
    """Decode requested frames via PyAV with seek-to-keyframe optimization.

    Repacked file-NNN.mp4 can hold many episodes, so decoding sequentially from
    frame zero can dominate worker time. Seek to the keyframe immediately
    before ``min(frame_indices)`` and decode forward, bounding the usual work to
    roughly one GOP rather than the full prefix.

    Falls back to seek(0) + sequential when PTS rounding misses a target.

    No fallback chain (decord / cv2): mp4s are byte-exact stream copy from
    validated upstream encodes; pyav failures are real bugs we want to see.
    """
    if not frame_indices:
        return []
    import av

    target = set(frame_indices)
    min_idx = min(frame_indices)
    max_idx = max(frame_indices)
    container = av.open(video_path, options={"hwaccel": "none"})
    try:
        stream = container.streams.video[0]
        _limit_decoder_threads(stream)

        pts_per_frame = None
        if stream.frames and stream.duration and stream.frames > 0:
            pts_per_frame = stream.duration / stream.frames

        seeked = False
        seek_err: Optional[BaseException] = None
        if pts_per_frame and min_idx > 0:
            try:
                container.seek(
                    max(0, int((min_idx - 2) * pts_per_frame)),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
                seeked = True
            except av.FFmpegError as e:  # av.AVError was removed in PyAV 14
                seek_err = e

        idx_map: Dict[int, Image.Image] = {}
        if seeked and pts_per_frame:
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                abs_idx = int(round(frame.pts / pts_per_frame))
                if abs_idx in target:
                    idx_map[abs_idx] = frame.to_image()
                if abs_idx >= max_idx:
                    break
            # PTS rounding can drift ±1; if any target missed, fall back to
            # seek(0) + sequential. Rare (<1% legacy observation).
            if not all(i in idx_map for i in target):
                _warn_seek_fallback(
                    video_path,
                    min_idx,
                    max_idx,
                    reason=f"missed_targets (got {sorted(idx_map.keys())} of {sorted(target)})",
                )
                idx_map.clear()
                container.seek(0, stream=stream, backward=True, any_frame=False)
                seeked = False
        elif seek_err is not None:
            _warn_seek_fallback(
                video_path,
                min_idx,
                max_idx,
                reason=f"seek_raised: {type(seek_err).__name__}: {seek_err}",
            )
        if not seeked:
            for i, frame in enumerate(container.decode(stream)):
                if i in target:
                    idx_map[i] = frame.to_image()
                if i >= max_idx:
                    break
    finally:
        container.close()
    missing = set(frame_indices) - idx_map.keys()
    if missing:
        raise RuntimeError(f"missing frames {missing} in {video_path}")
    return [idx_map[i].resize((width, height), Image.LANCZOS) for i in frame_indices]


__all__ = ["decode_video_frames"]
