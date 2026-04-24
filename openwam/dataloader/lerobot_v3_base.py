"""Base infrastructure for LeRobot v3 format datasets.

Provides three building blocks consumed by sub-dataset adapters:

  ActionComposer           – extracts/concatenates action columns from a DataFrame.
  LeRobot3Dataset          – single-task sliding-window dataset over one data_root.
  MultiTaskLeRobot3Dataset – concatenates LeRobot3Dataset instances; O(log n) dispatch.

Sub-dataset adapters (agibot.py, galaxea.py) supply presets, task discovery, and
register dataset types via from_config classmethods.
"""

from __future__ import annotations

import bisect
import functools
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from openwam.dataloader.base_dataset import BaseActionDataset

logger = logging.getLogger(__name__)

_GETITEM_MAX_RETRIES = 64

# Suppress C-level libav log spam via PyAV's av.logging interface.
# These messages bypass Python logging and go directly to stderr, so
# logging.getLogger("libav.*") has no effect — must use av.logging.set_level.
#
# Two sources:
#   libdav1d: "Unknown OBU type 0" — harmless non-standard metadata OBUs in
#             some AgiBotWorld AV1 streams; frames decode correctly.
#   av1 codec: "doesn't support hardware accelerated AV1 decoding" — emitted
#              per-frame on CPU-only servers when hw probe (av1_cuvid/av1_qsv)
#              fails; harmless, PyAV falls back to software decode automatically.
try:
    import av as _av
    _av.logging.set_level(_av.logging.FATAL)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _decode_video_frames(
    video_path: str,
    frame_indices: List[int],
    height: int,
    width: int,
) -> List[Image.Image]:
    """Decode specific frames from an MP4 file.

    Fallback chain: PyAV (hwaccel=none) → PyAV (default) → decord → cv2.

    Two PyAV attempts are made:
      - First with ``hwaccel=none`` passed at the format level.  This was
        originally intended to suppress per-frame AV1 hw-probe spam
        (av1_cuvid / av1_qsv) on CPU-only servers.  Note: ``hwaccel`` is
        actually an AVCodecContext option, not an AVFormatContext option, so
        passing it here is silently ignored by ffmpeg — it does NOT reliably
        disable hw acceleration.  C-level log spam is instead suppressed via
        ``av.logging.set_level`` at module load time (see module top).
        Despite not working as intended, this path succeeds for the vast
        majority of files and is kept as the fast path.
      - Second with no extra options, letting PyAV perform its natural
        hw → sw fallback.  Some files fail the first path because
        ``codec_context.options = {"threads": "auto"}`` replaces all codec
        defaults rather than merging, which can break sw fallback for files
        whose AV1 stream triggers hw-probe failures mid-decode.

    Returns:
        List of PIL Images resized to ``(width, height)``.
    """
    if not frame_indices:
        return []

    frames = None

    def _pyav_decode(container, set_threads: bool = True) -> List[Image.Image]:
        stream = container.streams.video[0]
        # set_threads=False in 1b: assigning codec_context.options replaces ALL
        # codec defaults (not a merge), which breaks hw→sw fallback for some files.
        # 1b relies on PyAV's natural defaults to recover from hw-probe failures.
        if set_threads:
            stream.codec_context.options = {"threads": "auto"}
        target_set = set(frame_indices)
        min_idx = min(frame_indices)
        max_idx = max(frame_indices)
        idx_map: Dict[int, Image.Image] = {}

        # Seek to near min_idx to skip decoding frames before the target window.
        # Only attempted when frame-count + duration metadata are available.
        # Falls back to sequential decode from 0 if seek fails or PTS-based
        # frame indexing misses any target frame (guards against rounding errors).
        # Corrupted-file exceptions in container.decode() are NOT caught here —
        # they propagate to the 1a/1b try-except in _decode_video_frames so the
        # existing fallback chain (decord → cv2) handles them unchanged.
        pts_per_frame: Optional[float] = None
        seeked = False
        if min_idx > 0 and stream.frames and stream.duration and stream.frames > 0:
            pts_per_frame = stream.duration / stream.frames
            try:
                container.seek(max(0, int((min_idx - 2) * pts_per_frame)), stream=stream)
                seeked = True
            except Exception:
                pass

        if seeked:
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_idx = round(frame.pts / pts_per_frame)
                if frame_idx in target_set:
                    idx_map[frame_idx] = frame.to_image()
                if frame_idx >= max_idx:
                    break
            # PTS rounding can misplace a frame by ±1; fall back to sequential
            # from 0 if any target frame is missing.
            if not all(i in idx_map for i in frame_indices):
                idx_map.clear()
                container.seek(0, stream=stream)
                seeked = False

        if not seeked:
            for i, frame in enumerate(container.decode(stream)):
                if i in target_set:
                    idx_map[i] = frame.to_image()
                if i >= max_idx:
                    break

        container.close()
        result = [idx_map.get(i) for i in frame_indices]
        return result if not any(f is None for f in result) else None

    # 1a. PyAV — hwaccel=none (format-level, fast path for most files)
    try:
        import av
        frames = _pyav_decode(av.open(video_path, options={"hwaccel": "none"}), set_threads=True)
    except Exception as e:
        logger.debug("PyAV (hwaccel=none) failed for %s: %s", video_path, e)

    # 1b. PyAV — no codec options, full natural hw→sw fallback
    # set_threads=False so codec defaults are not replaced; this is the path
    # that recovers files where 1a fails due to hw-probe mid-decode.
    if frames is None:
        try:
            import av
            frames = _pyav_decode(av.open(video_path), set_threads=False)
        except Exception as e:
            logger.debug("PyAV (default) failed for %s: %s", video_path, e)

    # 2. decord
    if frames is None:
        try:
            import decord

            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path)
            valid = [i for i in frame_indices if 0 <= i < len(vr)]
            if len(valid) < len(frame_indices):
                logger.warning(
                    "%s: clamping %d indices to valid range [0, %d)",
                    video_path, len(frame_indices) - len(valid), len(vr),
                )
            batch = vr.get_batch(valid).asnumpy()
            idx_map = {fi: Image.fromarray(batch[j]) for j, fi in enumerate(valid)}
            result = [idx_map.get(i) for i in frame_indices]
            if any(f is None for f in result):
                raise RuntimeError(f"decord: missing frames for indices {[i for i in frame_indices if idx_map.get(i) is None]}")
            frames = result
        except Exception:
            pass

    # 3. cv2
    if frames is None:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            idx_map = {}
            for fi in sorted(set(frame_indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    idx_map[fi] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            result = [idx_map.get(i) for i in frame_indices]
            if not any(f is None for f in result):
                frames = result
        except Exception:
            pass

    # Raise instead of returning black frames: silent black frames corrupt training
    # (loss spikes without any visible error). Let the DataLoader skip this sample.
    if frames is None:
        raise RuntimeError(f"All video backends failed for {video_path}")

    result = []
    for f in frames:
        if f is None:
            # A None entry means _pyav_decode couldn't find that frame index.
            raise RuntimeError(f"Failed to decode a frame from {video_path}")
        result.append(f.resize((width, height), Image.LANCZOS))
    return result


def _flatten_layout(camera_layout) -> List[str]:
    """Flatten a camera layout (flat list or 2D list-of-lists) to a single list."""
    if not camera_layout:
        return []
    if isinstance(camera_layout[0], str):
        return list(camera_layout)
    return [c for row in camera_layout for c in row]


def _stretch_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Direct BILINEAR resize without aspect-ratio preservation."""
    return image.resize((target_width, target_height), Image.BILINEAR)


def assemble_multiview_layout(
    frames_by_camera: Dict[str, Image.Image],
    camera_layout: List[str],
    out_h: int,
    out_w: int,
    top_height_ratio: float = 2.0 / 3.0,
) -> Image.Image:
    """3-camera L-shape composition (FastWAM / RoboTwin compatible).

    Layout:
        top    → (out_h * 2/3, out_w)      full width
        bot-L  → (out_h * 1/3, out_w // 2) half width
        bot-R  → (out_h * 1/3, out_w // 2) half width

    Args:
        frames_by_camera: {camera_name: PIL.Image}. Missing key → black region (caller should pre-pad short cameras).
        camera_layout:    ordered flat list of 3 camera names [top, bot-left, bot-right].
        out_h, out_w:     final canvas size in pixels.
        top_height_ratio: fraction of height for top camera (default 2/3).
    """
    if len(camera_layout) != 3:
        raise ValueError(f"L-shape layout expects 3 cameras, got {len(camera_layout)}: {camera_layout}")

    top_h = int(round(out_h * top_height_ratio))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    right_w = out_w - half_w

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))

    top_frame = frames_by_camera.get(camera_layout[0])
    if top_frame is not None:
        canvas.paste(_stretch_resize(top_frame, top_h, out_w), (0, 0))

    bl_frame = frames_by_camera.get(camera_layout[1])
    if bl_frame is not None:
        canvas.paste(_stretch_resize(bl_frame, bottom_h, half_w), (0, top_h))

    br_frame = frames_by_camera.get(camera_layout[2])
    if br_frame is not None:
        canvas.paste(_stretch_resize(br_frame, bottom_h, right_w), (half_w, top_h))

    return canvas


def _assemble_grid(
    frames_by_cam: Dict[str, Image.Image],
    camera_layout: List[List[str]],
    tile_h: int,
    tile_w: int,
) -> Image.Image:
    """Assemble per-camera frames into an equal-tile grid (2×2 compat path)."""
    n_rows = len(camera_layout)
    n_cols = max(len(row) for row in camera_layout)
    grid = Image.new("RGB", (tile_w * n_cols, tile_h * n_rows))
    for r, row_cams in enumerate(camera_layout):
        for c, cam in enumerate(row_cams):
            f = frames_by_cam.get(cam)
            tile = f.resize((tile_w, tile_h), Image.LANCZOS) if f else Image.new("RGB", (tile_w, tile_h))
            grid.paste(tile, (c * tile_w, r * tile_h))
    return grid


# ---------------------------------------------------------------------------
# Video frame map (global frame index → mp4 file + local offset)
# ---------------------------------------------------------------------------

# Module-level cache: (video_dir, camera) → [(file_path, global_start, global_end), ...]
_VIDEO_FRAME_MAP: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = {}


def _probe_frame_count(path: str) -> int:
    """Return frame count of an mp4 via container metadata (no decoding).

    Attempts PyAV first (fast path), then cv2. If both succeed but disagree
    by more than 1 frame, the larger value is used and a WARNING is emitted —
    a mismatch here means the global frame offset map will be wrong for all
    subsequent files in this camera's file list, causing video-action misalignment.
    """
    n_av: int = 0
    n_cv: int = 0
    try:
        import av
        with av.open(path) as c:
            n_av = c.streams.video[0].frames
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        n_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception:
        pass

    if n_av > 0 and n_cv > 0 and abs(n_av - n_cv) > 1:
        n = max(n_av, n_cv)
        logger.warning(
            "Frame count mismatch for %s: PyAV=%d cv2=%d — using %d. "
            "This may cause video-action misalignment for subsequent files.",
            path, n_av, n_cv, n,
        )
        return n
    n = n_av or n_cv
    if n > 0:
        return n
    logger.warning("Could not probe frame count for %s", path)
    return 0


def _get_video_frame_map(video_dir: str, camera: str) -> List[Tuple[str, int, int]]:
    """Build (and cache) global-frame-index → file mapping for one camera.

    Scans ``video_dir/{camera}/**/*.mp4`` in sorted order, probes each file's
    frame count, and returns a list of ``(file_path, global_start, global_end)``
    tuples (global_end is exclusive).
    """
    key = (video_dir, camera)
    if key in _VIDEO_FRAME_MAP:
        return _VIDEO_FRAME_MAP[key]

    camera_dir = os.path.join(video_dir, camera)
    mp4_files = sorted(Path(camera_dir).rglob("*.mp4"))
    result: List[Tuple[str, int, int]] = []
    cumulative = 0
    for mp4_path in mp4_files:
        n = _probe_frame_count(str(mp4_path))
        if n > 0:
            result.append((str(mp4_path), cumulative, cumulative + n))
            cumulative += n
        else:
            logger.warning("Frame map: skipping %s (frame count unknown)", mp4_path)
    _VIDEO_FRAME_MAP[key] = result
    return result


# ---------------------------------------------------------------------------
# ActionComposer
# ---------------------------------------------------------------------------


class ActionComposer:
    """Extracts and concatenates action fields from a Parquet-sourced DataFrame.

    Handles both single-column actions (AgiBotWorld: ``action`` → 40-D) and
    split-column actions (Galaxea: ``action.left_arm`` + … → 14-D+).
    """

    def __init__(self, action_fields: List[str], info_features: dict):
        self.action_fields = action_fields
        self.field_dims: List[int] = []
        for field in action_fields:
            feat = info_features.get(field)
            if feat is None:
                raise KeyError(f"Action field '{field}' not found in info.json features")
            shape = feat.get("shape", [1])
            self.field_dims.append(shape[0] if shape else 1)
        self._action_dim = sum(self.field_dims)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def extract(self, df_slice) -> np.ndarray:
        """Return ``(T, action_dim)`` float32 array from a DataFrame slice."""
        parts = []
        for field, dim in zip(self.action_fields, self.field_dims):
            col = df_slice[field].values
            arr = np.array(col, dtype=np.float64).reshape(-1, 1) if dim == 1 else np.stack(col)
            parts.append(arr.astype(np.float32))
        return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# LeRobot3Dataset — single task
# ---------------------------------------------------------------------------


class LeRobot3Dataset(BaseActionDataset):
    """Single-task dataset for LeRobot v3 format (chunked parquet + chunked MP4).

    Supports sliding-window sampling, multiview grid assembly, and three
    action normalisation modes (none / zscore / minmax).

    Sub-dataset adapters (agibot.py, galaxea.py) instantiate this class
    directly with their presets; no sub-classing is needed.

    Args:
        data_root:        Directory containing ``data/``, ``meta/``, ``videos/``.
        action_fields:    Parquet column names to concatenate as the action vector.
        target_camera:    Camera name used in single-view mode.
        cameras:          All camera names available for multiview mode.
        camera_layout:    2-D list of camera names for grid assembly (rows × cols).
                          Required when ``multiview=True``.
        fps:              Dataset frame rate (stored for reference; not enforced).
        num_frames:       Sliding-window length.
        height / width:   Output image dimensions.
        split:            ``"train"`` or ``"val"``.
        val_ratio:        Fraction of episodes reserved for validation.
        seed:             RNG seed for the train/val split.
        task_name:        Human-readable task label (used in prompt strings).
        multiview:        If True, assemble ``camera_layout`` into a grid image.
        action_norm_mode: ``"none"`` | ``"zscore"`` | ``"minmax"``.
        window_stride:    Step between consecutive sliding windows.
        video_stride:     Temporal stride applied after loading frames.
        repeat:           Window-index repeat multiplier (cheap data augmentation).
        num_val_samples:  Cap on validation windows (0 = uncapped).
        action_stats_path: Override path to stats.json (default: ``meta/stats.json``).
        action_dim_slice: If set, keep only the first N dimensions of the action vector.
                          Stats are truncated to match.
        action_transform: Optional callable ``(np.ndarray) -> np.ndarray`` applied after
                          extraction. Takes priority over ``action_dim_slice``.
                          Use for semantic remapping (e.g. quat → rotation6d).
        action_out_dim:   Output action dimension when ``action_transform`` is set.
                          Required for the ``action_dim`` property to return the correct value.
    """

    def __init__(
        self,
        data_root: str,
        action_fields: List[str],
        target_camera: str,
        cameras: List[str],
        camera_layout: Optional[List[List[str]]] = None,
        fps: int = 30,
        num_frames: int = 33,
        height: int = 480,
        width: int = 640,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        task_name: Optional[str] = None,
        multiview: bool = False,
        normalize_mode: str = "none",
        window_stride: int = 1,
        video_stride: int = 4,
        repeat: int = 1,
        num_val_samples: int = 0,
        action_stats_path: Optional[str] = None,
        action_dim_slice: Optional[int] = None,
        action_transform: Optional[Callable] = None,
        action_out_dim: Optional[int] = None,
    ):
        import pandas as pd

        self.data_root = data_root
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.split = split
        self.multiview = multiview
        self.video_stride = video_stride
        self.window_stride = window_stride
        self.repeat = repeat
        # Normalise and canonicalise normalize_mode value
        _nm = normalize_mode
        if isinstance(_nm, str):
            if _nm.lower() in ("none", "null", ""):
                _nm = None
            elif _nm.lower() == "minmax":
                logger.warning("normalize_mode='minmax' is deprecated; use 'min-max'")
                _nm = "min-max"
            elif _nm.lower() == "zscore":
                logger.warning("normalize_mode='zscore' is deprecated; use 'z-score'")
                _nm = "z-score"
        self.normalize_mode = _nm
        self.fps = fps
        self.action_dim_slice = action_dim_slice
        self.action_transform = action_transform
        self._action_out_dim = action_out_dim

        # Camera config
        self.target_camera = target_camera
        if multiview:
            self.cameras = cameras
            self.camera_layout = camera_layout or [[c] for c in cameras]
        else:
            self.cameras = [target_camera]
            self.camera_layout = None

        # Load info.json
        info_path = os.path.join(data_root, "meta", "info.json")
        with open(info_path) as f:
            self._info = json.load(f)

        # Action composer
        self._action_composer = ActionComposer(action_fields, self._info["features"])

        # StateComposer: prefer observation.state for proprio over action[0:1].
        # Only used when state_dim == action_dim (same space); otherwise falls back.
        # Logged once at init so __getitem__ stays silent.
        self._state_composer: Optional[ActionComposer] = None
        if "observation.state" in self._info.get("features", {}):
            try:
                sc = ActionComposer(["observation.state"], self._info["features"])
                action_dim = self._action_composer.action_dim
                if action_out_dim is not None:
                    action_dim = action_out_dim
                elif action_dim_slice is not None:
                    action_dim = action_dim_slice
                if sc.action_dim == action_dim:
                    self._state_composer = sc
                else:
                    logger.debug(
                        "%s: observation.state dim=%d != action_dim=%d "
                        "(state space differs from action space); proprio will use action[0:1]",
                        task_name or os.path.basename(data_root), sc.action_dim, action_dim,
                    )
            except KeyError:
                pass
        if self._state_composer is None and "observation.state" not in self._info.get("features", {}):
            logger.debug(
                "%s: observation.state not in meta/info.json features; "
                "proprio will use action[0:1] as fallback",
                task_name or os.path.basename(data_root),
            )

        # Episode metadata
        episodes_dir = os.path.join(data_root, "meta", "episodes")
        ep_parquets = sorted(Path(episodes_dir).rglob("*.parquet"))
        ep_dfs = [pd.read_parquet(p) for p in ep_parquets]
        self._episodes_meta = (
            pd.concat(ep_dfs, ignore_index=True)
            .sort_values("episode_index")
            .reset_index(drop=True)
        )

        self.task_name = task_name or os.path.basename(data_root)
        self._task_descriptions = self._load_task_descriptions()

        # Action normalisation stats
        self._action_stats: Optional[dict] = None
        native_stats_path = os.path.join(data_root, "meta", "stats.json")
        eef_stats_path = os.path.join(data_root, "meta", "eef_stats.json")
        if action_transform is None:
            # Native stats.json covers the raw action space — safe to use directly.
            _sp = action_stats_path or native_stats_path
            if os.path.exists(_sp):
                self._load_action_stats(_sp)
        elif action_stats_path is not None:
            # User provided explicit stats for the post-transform space.
            if os.path.exists(action_stats_path):
                self._load_action_stats(action_stats_path)
        else:
            # action_transform set, no explicit path — load from
            # meta/eef_stats.json written by lerobot_v3_stats_computation.py.
            #
            # Auto-compute is intentionally NOT supported here: in DDP training
            # all ranks execute __init__ concurrently. Without rank-0 gating and
            # a file lock, every rank would race to write the same file, producing
            # corrupted or partially-computed stats. Pre-compute once before
            # training with:
            #   python -m openwam.dataloader.lerobot_v3_stats_computation \
            #       --type agibot --dataset_dir <dir> [--quat_convention xyzw]
            if os.path.exists(eef_stats_path):
                self._load_action_stats(eef_stats_path)

        # Index data parquet files: (chunk_idx, file_idx) → path
        self._data_file_paths: Dict[Tuple[int, int], str] = {}
        for pf in sorted(Path(os.path.join(data_root, "data")).rglob("*.parquet")):
            chunk_parts = [p for p in pf.parts if p.startswith("chunk-")]
            if chunk_parts:
                chunk_idx = int(chunk_parts[-1].split("-")[1])
                file_idx = int(pf.stem.split("-")[1])
                self._data_file_paths[(chunk_idx, file_idx)] = str(pf)

        # Train / val split
        n_ep = len(self._episodes_meta)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ep)
        n_val = max(1, int(n_ep * val_ratio))
        if split == "val":
            ep_indices = sorted(perm[:n_val].tolist())
            if num_val_samples > 0:
                ep_indices = ep_indices[:num_val_samples]
        else:
            ep_indices = sorted(perm[n_val:].tolist())
        self._episode_indices = ep_indices

        # Build sliding-window index: (local_episode_idx, start_frame)
        self._window_index: List[Tuple[int, int]] = []
        for local_idx, ep_idx in enumerate(self._episode_indices):
            ep_len = int(self._episodes_meta.iloc[ep_idx]["length"])
            for s in range(0, max(0, ep_len - num_frames) + 1, window_stride):
                self._window_index.append((local_idx, s))
        if not self._window_index:
            for local_idx in range(len(self._episode_indices)):
                self._window_index.append((local_idx, 0))
        if repeat > 1:
            self._window_index = self._window_index * repeat

        # Startup guards
        _valid_modes = (None, "min-max", "z-score")
        if self.normalize_mode not in _valid_modes:
            raise ValueError(
                f"normalize_mode must be one of {_valid_modes}, got {self.normalize_mode!r}"
            )
        if self.normalize_mode is not None and self._action_stats is None:
            raise RuntimeError(
                f"normalize_mode={self.normalize_mode!r} but no action_stats available for "
                f"{task_name or os.path.basename(data_root)!r}. "
                f"When action_transform is set (e.g. action_format=eef), meta/stats.json covers "
                f"the pre-transform space and cannot be used. Pre-compute EEF stats once before "
                f"training (safe for DDP — no race condition):\n"
                f"  python -m openwam.dataloader.lerobot_v3_stats_computation \\\n"
                f"      --type agibot --dataset_dir <dir> [--quat_convention xyzw]\n"
                f"Or set normalize_mode=null to skip normalization."
            )

        logger.info(
            "LeRobot3Dataset [%s] %s: %d episodes, %d windows, action_dim=%d",
            split, self.task_name,
            len(self._episode_indices), len(self._window_index),
            self._action_composer.action_dim,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_task_descriptions(self) -> Dict[int, str]:
        descriptions: Dict[int, str] = {}
        for _, row in self._episodes_meta.iterrows():
            ep_idx = int(row["episode_index"])
            tasks = row.get("tasks")
            if tasks is None or not len(tasks):
                continue
            for t in tasks:
                t_str = str(t)
                if t_str in ("qualified", "unqualified"):
                    continue
                if "@" in t_str:
                    descriptions[ep_idx] = t_str.split("@", 1)[1]
                    break
                if t_str.isascii():
                    descriptions[ep_idx] = t_str
                    break
            if ep_idx not in descriptions:
                descriptions[ep_idx] = str(tasks[0])
        return descriptions

    def _load_action_stats(self, stats_path: str) -> None:
        with open(stats_path) as f:
            raw = json.load(f)
        means, stds, mins, maxs = [], [], [], []
        for field, dim in zip(self._action_composer.action_fields, self._action_composer.field_dims):
            s = raw.get(field, {})
            means.append(np.array(s.get("mean", [0.0] * dim), dtype=np.float32).flatten())
            stds.append(np.maximum(np.array(s.get("std", [1.0] * dim), dtype=np.float32).flatten(), 1e-3))
            mins.append(np.array(s.get("min", [-1.0] * dim), dtype=np.float32).flatten())
            maxs.append(np.array(s.get("max", [1.0] * dim), dtype=np.float32).flatten())
        s = self.action_dim_slice
        self._action_stats = {
            "mean": np.concatenate(means)[:s],
            "std": np.concatenate(stds)[:s],
            "min": np.concatenate(mins)[:s],
            "max": np.concatenate(maxs)[:s],
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=4)
    def _load_data_file(self, chunk_idx: int, file_idx: int):
        import pandas as pd

        path = self._data_file_paths.get((chunk_idx, file_idx))
        if path is None:
            raise FileNotFoundError(f"Data file not found: chunk={chunk_idx}, file={file_idx}")
        return pd.read_parquet(path)

    def _get_episode_data(self, ep_row, start_frame: int, end_frame: int):
        chunk_idx = int(ep_row["data/chunk_index"])
        file_idx = int(ep_row["data/file_index"])
        ep_idx = int(ep_row["episode_index"])

        df = self._load_data_file(chunk_idx, file_idx)
        ep_data = df[df["episode_index"] == ep_idx].reset_index(drop=True)

        if len(ep_data) == 0:
            for alt_file_idx in sorted(
                fi for (ci, fi) in self._data_file_paths if ci == chunk_idx and fi != file_idx
            ):
                alt_df = self._load_data_file(chunk_idx, alt_file_idx)
                alt_data = alt_df[alt_df["episode_index"] == ep_idx].reset_index(drop=True)
                if len(alt_data) > 0:
                    logger.warning(
                        "ep_idx=%d: metadata says file-%03d but data in file-%03d — using correct file",
                        ep_idx, file_idx, alt_file_idx,
                    )
                    ep_data = alt_data
                    break

        return ep_data.iloc[start_frame:end_frame]

    def _get_video_frames(self, ep_data, camera: str) -> List[Image.Image]:
        """Load video frames for a camera using the parquet global frame index.

        Uses the ``index`` column from ``ep_data`` as the global frame position
        across all mp4 files for that camera, then uses _get_video_frame_map to
        locate the correct file(s) and local offsets. Handles multi-file cameras
        (video splits by frame count, not episode count).
        """
        tile_h = self.height // 2 if self.multiview else self.height
        tile_w = self.width // 2 if self.multiview else self.width
        n = len(ep_data)

        if n == 0:
            return []

        global_indices = ep_data["index"].values
        frame_start = int(global_indices[0])
        frame_end = int(global_indices[-1]) + 1

        video_dir = os.path.join(self.data_root, "videos")
        frame_map = _get_video_frame_map(video_dir, camera)
        if not frame_map:
            raise RuntimeError(f"No video files found for camera '{camera}' in {video_dir}")

        frames: List[Image.Image] = []
        for file_path, file_start, file_end in frame_map:
            overlap_start = max(frame_start, file_start)
            overlap_end = min(frame_end, file_end)
            if overlap_start >= overlap_end:
                continue
            local_indices = list(range(overlap_start - file_start, overlap_end - file_start))
            frames.extend(_decode_video_frames(file_path, local_indices, tile_h, tile_w))

        # Raise if zero frames were decoded: frame_map exists but global range falls
        # outside all files, which indicates a broken frame offset map.
        if not frames:
            raise RuntimeError(
                f"Camera '{camera}': no frames decoded for global range [{frame_start}, {frame_end})"
            )

        # Level-2 alignment check: actual decoded count vs expected from parquet index.
        # A mismatch means _probe_frame_count returned a wrong value for at least one
        # file, shifting all subsequent global offsets and misaligning video with action.
        # Kept as WARNING (not raised) because the shortfall is recovered by last-frame
        # repeat below — data is degraded but not completely invalid.
        expected = frame_end - frame_start
        if len(frames) != expected:
            logger.warning(
                "Camera '%s': decoded %d frames but expected %d (global [%d, %d)). "
                "Likely a frame count metadata error in _probe_frame_count — "
                "check the WARNING above for the relevant mp4 file.",
                camera, len(frames), expected, frame_start, frame_end,
            )

        # Pad short clips with the last frame (same strategy as RoboTwin).
        if len(frames) < n:
            frames = frames + [frames[-1]] * (n - len(frames))
        return frames[:n]

    # ------------------------------------------------------------------
    # Padding / normalisation
    # ------------------------------------------------------------------

    def _pad_actions(self, actions: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
        T = len(actions)
        if T >= target:
            return actions[:target], np.ones(target, dtype=bool)
        last = actions[-1:] if T > 0 else np.zeros((1, actions.shape[1]), dtype=np.float32)
        padded = np.concatenate([actions, np.repeat(last, target - T, axis=0)])
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(target - T, dtype=bool)])
        return padded, mask

    def _pad_video(self, frames: List[Image.Image], target: int) -> Tuple[List[Image.Image], np.ndarray]:
        T = len(frames)
        if T >= target:
            return frames[:target], np.ones(target, dtype=bool)
        last = frames[-1] if frames else Image.new("RGB", (self.width, self.height))
        padded = frames + [last] * (target - T)
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(target - T, dtype=bool)])
        return padded, mask

    def _normalize(self, actions: np.ndarray) -> np.ndarray:
        if self._action_stats is None or self.normalize_mode is None:
            return actions
        if self.normalize_mode == "z-score":
            return (actions - self._action_stats["mean"]) / self._action_stats["std"]
        if self.normalize_mode == "min-max":
            mn, mx = self._action_stats["min"], self._action_stats["max"]
            return 2.0 * (actions - mn) / np.maximum(mx - mn, 1e-6) - 1.0
        return actions

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self._action_stats is None or self.normalize_mode is None:
            return action
        if self.normalize_mode == "z-score":
            return action * self._action_stats["std"] + self._action_stats["mean"]
        if self.normalize_mode == "min-max":
            mn, mx = self._action_stats["min"], self._action_stats["max"]
            return (action + 1.0) * np.maximum(mx - mn, 1e-6) / 2.0 + mn
        return action

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, idx: int) -> dict:
        for _attempt in range(_GETITEM_MAX_RETRIES):
            local_idx, start_frame = self._window_index[idx]
            ep_global_idx = self._episode_indices[local_idx]
            ep_row = self._episodes_meta.iloc[ep_global_idx]
            ep_len = int(ep_row["length"])
            ep_idx = int(ep_row["episode_index"])
            end_frame = min(start_frame + self.num_frames, ep_len)

            # Actions
            ep_data = self._get_episode_data(ep_row, start_frame, end_frame)
            if len(ep_data) == 0:
                chunk_idx = int(ep_row["data/chunk_index"])
                file_idx = int(ep_row["data/file_index"])
                file_path = self._data_file_paths.get((chunk_idx, file_idx), "unknown")
                logger.warning(
                    "Empty episode data ep_idx=%d start=%d, file=%s",
                    ep_idx, start_frame, file_path,
                )
                idx = (idx + 1) % len(self)
                continue
            actions = self._action_composer.extract(ep_data)
            if self.action_transform is not None:
                actions = self.action_transform(actions)
            elif self.action_dim_slice is not None:
                actions = actions[:, :self.action_dim_slice]

            # Video
            try:
                if self.multiview:
                    all_frames = {cam: self._get_video_frames(ep_data, cam) for cam in self.cameras}
                    T = len(ep_data)
                    # Warn once if any camera is shorter than expected; pad with last frame.
                    for cam, frames in all_frames.items():
                        if len(frames) < T:
                            logger.warning(
                                "ep_idx=%d cam='%s': got %d frames, expected %d — "
                                "padding tail with last frame (likely mp4 frame-count mismatch).",
                                ep_idx, cam, len(frames), T,
                            )
                            all_frames[cam] = frames + [frames[-1]] * (T - len(frames))
                    flat_layout = _flatten_layout(self.camera_layout)
                    video_frames = [
                        assemble_multiview_layout(
                            {cam: all_frames[cam][t] for cam in self.cameras},
                            flat_layout, out_h=self.height, out_w=self.width,
                        )
                        for t in range(T)
                    ]
                else:
                    video_frames = self._get_video_frames(ep_data, self.target_camera)
            except RuntimeError as e:
                logger.warning("Skipping idx=%d ep_idx=%d: %s", idx, ep_idx, e)
                idx = (idx + 1) % len(self)
                continue

            # Stride → pad
            video_strided = video_frames[:: self.video_stride]
            video_target = (self.num_frames + self.video_stride - 1) // self.video_stride
            video_target += (1 - video_target % 4) % 4  # VACE requires num_frames % 4 == 1
            video_strided, video_mask = self._pad_video(video_strided, video_target)

            # Normalize before splitting so proprio and action share the same space
            # (mirrors RoboTwin: _action_normalizer.normalize applied before the 0:1 / 1:N cut).
            actions = self._normalize(actions)

            # Window split: frame 0 = proprio anchor, frames 1..N = action trajectory.
            # Prefer observation.state (native sensor reading); fall back to action[0:1].
            if self._state_composer is not None:
                raw_state = self._state_composer.extract(ep_data)
                proprio_np = raw_state[0:1]    # (1, state_dim) — native, separate space
                proprio_source = "native"
            else:
                proprio_np = actions[0:1].astype(np.float32)   # already normalized
                proprio_source = "action_fallback"
            action_np = actions[1:self.num_frames].astype(np.float32)  # (T-1, D)
            action_np, action_mask = self._pad_actions(action_np, self.num_frames - 1)

            chunk_idx = int(ep_row["data/chunk_index"])
            file_idx = int(ep_row["data/file_index"])
            episode_path = self._data_file_paths.get((chunk_idx, file_idx), "")

            prompt = f"The robot is performing: {self._task_descriptions.get(ep_idx, self.task_name)}"
            if self.multiview and self.camera_layout:
                cam_names = [c.split(".")[-1] for c in _flatten_layout(self.camera_layout)]
                prompt += f" Cameras: {', '.join(cam_names)}."

            return {
                "video": video_strided,
                "vace_video": None,
                "first_frame_image": [video_strided[0]] if video_strided else [],
                "action_trajectory": torch.from_numpy(action_np),
                "action_mask": torch.from_numpy(action_mask),
                "video_mask": torch.from_numpy(video_mask),
                "proprio": torch.from_numpy(proprio_np),
                "proprio_mask": torch.ones(1, dtype=torch.bool),
                "prompt": prompt,
                "episode_index": ep_idx,
                "episode_path": episode_path,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "episode_length": ep_len,
                "task_name": self.task_name,
                "proprio_source": proprio_source,
            }
        raise RuntimeError(
            f"No valid sample found after {_GETITEM_MAX_RETRIES} retries "
            f"starting at idx={idx} in {self.task_name!r}"
        )

    @property
    def action_dim(self) -> int:
        if self._action_out_dim is not None:
            return self._action_out_dim
        if self.action_dim_slice is not None:
            return self.action_dim_slice
        return self._action_composer.action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._action_stats


# ---------------------------------------------------------------------------
# MultiTaskLeRobot3Dataset — multi-task wrapper
# ---------------------------------------------------------------------------


class MultiTaskLeRobot3Dataset(BaseActionDataset):
    """Concatenates a list of LeRobot3Dataset instances with O(log n) dispatch.

    Sub-dataset adapters (AgibotDataset, GalaxeaDataset) build the dataset
    list and pass it to ``super().__init__(datasets)``.
    """

    def __init__(self, datasets: List[LeRobot3Dataset]):
        if not datasets:
            raise ValueError("datasets list must not be empty")
        self._datasets = datasets
        total = 0
        self._cumulative_lengths: List[int] = []
        for ds in datasets:
            total += len(ds)
            self._cumulative_lengths.append(total)
        logger.info(
            "%s: %d tasks, %d total windows, action_dim=%d",
            self.__class__.__name__, len(datasets), total, self.action_dim,
        )

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict:
        ds_idx = bisect.bisect_right(self._cumulative_lengths, idx)
        if ds_idx > 0:
            idx -= self._cumulative_lengths[ds_idx - 1]
        return self._datasets[ds_idx][idx]

    @property
    def action_dim(self) -> int:
        return self._datasets[0].action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._datasets[0].action_stats

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return self._datasets[0].denormalize_action(action)
