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

# dav1d emits "Unknown OBU type 0" ERRORs for non-standard metadata OBUs
# embedded by some AgiBotWorld encoders. These are harmless — frames decode
# correctly — but flood the log. Silence at CRITICAL to suppress the spam.
logging.getLogger("libav.libdav1d").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _decode_video_frames(
    video_path: str,
    frame_indices: List[int],
    height: int,
    width: int,
) -> List[Image.Image]:
    """
    Decode the specified frame indices from an MP4 file into resized RGB PIL images.

    Attempts decoding with three backends in order: PyAV (software decoding, with hardware acceleration disabled), decord, then OpenCV; if all backends fail, returns black images for each requested index. If `frame_indices` is empty, returns an empty list. The returned images are resized to (width, height) using a high-quality resampling filter, and any individual missing frames are replaced by black images of that size.

    Parameters:
        video_path (str): Path to the MP4 file.
        frame_indices (List[int]): Global frame indices to decode (may be non-contiguous or unsorted).
        height (int): Output image height in pixels.
        width (int): Output image width in pixels.

    Returns:
        List[Image.Image]: A list of PIL RGB images corresponding to `frame_indices`, each sized (width, height). If decoding of a frame fails, a black image is returned in its place.
    """
    if not frame_indices:
        return []

    # Accumulate decoded frames across backends. Each backend only fills slots
    # that are still None, so a single bad/out-of-range index does not discard
    # frames that an earlier backend already decoded successfully.
    frames: List[Optional[Image.Image]] = [None] * len(frame_indices)

    def _fill_missing(idx_to_image: Dict[int, Image.Image]) -> None:
        for pos, fi in enumerate(frame_indices):
            if frames[pos] is None and fi in idx_to_image:
                frames[pos] = idx_to_image[fi]

    def _missing_indices() -> List[int]:
        return [fi for pos, fi in enumerate(frame_indices) if frames[pos] is None]

    # 1. PyAV — software decoding, no hardware acceleration
    try:
        import av

        container = av.open(video_path, options={"hwaccel": "none"})
        stream = container.streams.video[0]
        stream.codec_context.options = {"threads": "auto"}
        target_set = set(frame_indices)
        max_idx = max(frame_indices)
        idx_map: Dict[int, Image.Image] = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in target_set:
                idx_map[i] = frame.to_image()
            if i >= max_idx:
                break
        container.close()
        _fill_missing(idx_map)
    except Exception as e:
        logger.debug("PyAV failed for %s: %s", video_path, e)

    # 2. decord — only fill slots still missing after PyAV
    if any(f is None for f in frames):
        try:
            import decord

            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path)
            missing = _missing_indices()
            valid = [i for i in missing if 0 <= i < len(vr)]
            if len(valid) < len(missing):
                logger.warning(
                    "%s: clamping %d indices to valid range [0, %d)",
                    video_path, len(missing) - len(valid), len(vr),
                )
            if valid:
                batch = vr.get_batch(valid).asnumpy()
                if len(batch) != len(valid):
                    raise RuntimeError(
                        f"decord returned {len(batch)} frames for {len(valid)} indices"
                    )
                idx_to_image = {
                    src_idx: Image.fromarray(batch[j])
                    for j, src_idx in enumerate(valid)
                }
            else:
                idx_to_image = {}
            _fill_missing(idx_to_image)
        except Exception:
            pass

    # 3. cv2 — only fill slots still missing after PyAV/decord
    if any(f is None for f in frames):
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            idx_map = {}
            for fi in sorted(set(_missing_indices())):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    idx_map[fi] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            _fill_missing(idx_map)
        except Exception:
            pass

    # Fallback: if nothing was decoded at all, warn once; otherwise only
    # positions that remain None will be filled with black tiles below.
    if all(f is None for f in frames):
        logger.warning("All video backends failed for %s — using black frames", video_path)

    return [
        f.resize((width, height), Image.LANCZOS) if f is not None else Image.new("RGB", (width, height))
        for f in frames
    ]


def _assemble_grid(
    frames_by_cam: Dict[str, Image.Image],
    camera_layout: List[List[str]],
    tile_h: int,
    tile_w: int,
) -> Image.Image:
    """
    Assemble per-camera frames into a tiled grid image according to a 2D camera layout.

    Each entry in `camera_layout` is a row (list) of camera names; the layout defines the grid shape
    and ordering of tiles. Frames for named cameras are resized to `(tile_w, tile_h)`; cameras not
    present in `frames_by_cam` are replaced with black tiles.

    Parameters:
        frames_by_cam (Dict[str, PIL.Image.Image]): Mapping from camera name to its frame image.
        camera_layout (List[List[str]]): 2D list where each inner list is a row of camera names.
        tile_h (int): Height in pixels for each tile in the output grid.
        tile_w (int): Width in pixels for each tile in the output grid.

    Returns:
        PIL.Image.Image: RGB image containing the assembled grid of tiles.
    """
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
    """
    Probe MP4 container metadata to determine the number of video frames.

    Tries to read container metadata (first via PyAV, then via OpenCV). If neither backend can determine a positive frame count, returns 0.

    Parameters:
        path (str): File path to the MP4 video.

    Returns:
        int: Number of frames in the video, or 0 if the count could not be determined.
    """
    try:
        import av
        with av.open(path) as c:
            n = c.streams.video[0].frames
            if n > 0:
                return n
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass
    logger.warning("Could not probe frame count for %s", path)
    return 0


def _get_video_frame_map(video_dir: str, camera: str) -> List[Tuple[str, int, int]]:
    """
    Builds and caches a mapping from a camera's global frame index space to its MP4 files.

    Scans files under `video_dir/{camera}/**/*.mp4` in lexical sorted order, probes each file's frame
    count, and accumulates ranges so each tuple maps consecutive global frame indices to a source file.
    Files with unknown frame counts are skipped (a warning is logged). The result is cached per
    `(video_dir, camera)` so repeated calls return the cached mapping.

    Parameters:
        video_dir (str): Root directory containing per-camera subdirectories of MP4 files.
        camera (str): Camera subdirectory name under `video_dir` to scan.

    Returns:
        List[Tuple[str, int, int]]: A list of `(file_path, global_start, global_end)` entries where
        `global_start` is inclusive and `global_end` is exclusive, and `file_path` is the MP4 file path.
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
        """
        Initialize the composer by resolving each action field's dimensionality and computing the total action dimension.

        Parameters:
            action_fields (List[str]): Ordered list of feature names to include in the composed action vector.
            info_features (dict): Mapping from feature name to its metadata (expected to contain a `"shape"` entry, e.g. `{"shape": [D]}`).

        Raises:
            KeyError: If any name in `action_fields` is not present in `info_features`.
        """
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
        """
        Total number of scalar elements in the concatenated action vector.

        Returns:
            int: The sum of per-field action dimensions produced by the ActionComposer.
        """
        return self._action_dim

    def extract(self, df_slice) -> np.ndarray:
        """
        Compose a contiguous action matrix by concatenating the dataset's configured action fields from a DataFrame slice.

        Parameters:
            df_slice (pandas.DataFrame): Row slice containing the action columns named in self.action_fields; each column must contain either scalar values or sequence-like vectors matching the shapes declared in the dataset's info.

        Returns:
            action_array (numpy.ndarray): Array shaped (T, action_dim) where T is the number of rows in df_slice and `action_dim` is the total concatenated action dimensionality. Values are returned as dtype `float32`. Scalar action columns are treated as single-dimension features and multi-element columns are stacked along the feature axis.
        """
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
        action_norm_mode: str = "none",
        window_stride: int = 1,
        video_stride: int = 4,
        repeat: int = 1,
        num_val_samples: int = 0,
        action_stats_path: Optional[str] = None,
        action_dim_slice: Optional[int] = None,
        action_transform: Optional[Callable] = None,
        action_out_dim: Optional[int] = None,
        video_num_frames: Optional[int] = None,
        sample_transform: Optional[Callable] = None,
    ):
        """
        Initialize a LeRobot3Dataset configured to load episodes, actions, and video frames from a LeRobot v3-format data root.

        Parameters:
            data_root (str): Path to the dataset root containing `meta/`, `data/`, and `videos/`.
            action_fields (List[str]): Names of action columns to compose into the action vector.
            target_camera (str): Primary camera name used when not in multiview mode.
            cameras (List[str]): List of camera names available for multiview mode.
            camera_layout (Optional[List[List[str]]]): 2D layout mapping camera names to grid positions when multiview is True.
            fps (int): Target frames-per-second for video handling.
            num_frames (int): Number of action timesteps per sample (temporal window length).
            height (int): Output video frame height in pixels.
            width (int): Output video frame width in pixels.
            split (str): Dataset split to build ("train" or "val").
            val_ratio (float): Fraction of episodes reserved for validation when splitting.
            seed (int): RNG seed used for deterministic train/validation permutation.
            task_name (Optional[str]): Optional explicit task name; defaults to the data_root basename.
            multiview (bool): If True, decode and assemble multiview grids from `cameras`; otherwise use `target_camera` only.
            action_norm_mode (str): Action normalization mode ("none", "zscore", "minmax", etc.).
            window_stride (int): Step between sliding-window start positions within an episode.
            video_stride (int): Temporal stride applied to decoded video frames before padding.
            repeat (int): Multiply the sliding-window index by this factor for cheap augmentation.
            num_val_samples (int): Optional cap on number of validation episodes when split == "val".
            action_stats_path (Optional[str]): Path to JSON file with per-action normalization statistics; used only when `action_transform` is None.
            action_dim_slice (Optional[int]): If set, truncate composed action vectors to this many dimensions.
            action_transform (Optional[Callable]): Optional callable to transform actions; when provided, action stats are not loaded.
            action_out_dim (Optional[int]): If set, override the dataset reported action output dimensionality.
            video_num_frames (Optional[int]): If set, decode exactly this many video frames per sample (actions still use `num_frames`).
            sample_transform (Optional[Callable]): Optional callable applied to the final sample dict inside __getitem__ (runs in DataLoader workers).

        Behavior:
            Loads dataset metadata (`meta/info.json`), constructs an ActionComposer from `action_fields` and the info features, loads episode metadata from either `meta/episodes/*.parquet` or `meta/episodes.jsonl`, derives chunk/file indices when absent, indexes `data/**/*.parquet` for fast parquet lookup, optionally loads action normalization statistics (unless `action_transform` is provided), partitions episodes into train/validation according to `split` and `val_ratio`, and builds a sliding-window index of (episode, start_frame) tuples used by __getitem__. Logs a brief dataset summary including episode and window counts and the composed action dimension.
        """
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
        self.action_norm_mode = action_norm_mode
        self.fps = fps
        self.action_dim_slice = action_dim_slice
        self.action_transform = action_transform
        self._action_out_dim = action_out_dim
        # video_num_frames: if set, decode only this many video frames per sample
        # (actions still use num_frames). Default None → decode num_frames
        # (current behaviour; WAM/AgiBot unaffected).
        self.video_num_frames = video_num_frames
        # sample_transform: optional callable applied to the final sample dict
        # inside __getitem__ (runs in DataLoader workers). Used by alternate trainer
        # to push optional external model processor + padding onto worker processes. WAM and
        # AgiBot leave this as None → raw dict returned as before.
        self.sample_transform = sample_transform

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

        # Episode metadata — support both formats:
        #   (a) meta/episodes/*.parquet  (nested parquet metadata, AgiBot-style)
        #   (b) meta/episodes.jsonl      (LeRobot v3 standard, one JSON per line)
        episodes_dir = os.path.join(data_root, "meta", "episodes")
        episodes_jsonl = os.path.join(data_root, "meta", "episodes.jsonl")
        ep_parquets = sorted(Path(episodes_dir).rglob("*.parquet")) if os.path.isdir(episodes_dir) else []
        if ep_parquets:
            ep_dfs = [pd.read_parquet(p) for p in ep_parquets]
            self._episodes_meta = (
                pd.concat(ep_dfs, ignore_index=True)
                .sort_values("episode_index")
                .reset_index(drop=True)
            )
        elif os.path.exists(episodes_jsonl):
            self._episodes_meta = pd.read_json(episodes_jsonl, lines=True).sort_values("episode_index").reset_index(drop=True)
        else:
            raise FileNotFoundError(
                f"Neither {episodes_dir}/*.parquet nor {episodes_jsonl} found"
            )

        # Derive data/chunk_index and data/file_index when absent (LeRobot v3 standard layout)
        chunks_size = int(self._info.get("chunks_size", 1000))
        if "data/chunk_index" not in self._episodes_meta.columns:
            self._episodes_meta["data/chunk_index"] = self._episodes_meta["episode_index"] // chunks_size
        if "data/file_index" not in self._episodes_meta.columns:
            self._episodes_meta["data/file_index"] = self._episodes_meta["episode_index"]

        self.task_name = task_name or os.path.basename(data_root)
        self._task_descriptions = self._load_task_descriptions()

        # Action normalisation stats
        # Skip loading when action_transform is set: stats.json is for the pre-transform
        # action space and cannot be used after semantic remapping (e.g. 40-D → 20-D EEF).
        self._action_stats: Optional[dict] = None
        stats_path = action_stats_path or os.path.join(data_root, "meta", "stats.json")
        if os.path.exists(stats_path) and action_transform is None:
            self._load_action_stats(stats_path)

        # Index data parquet files: (chunk_idx, file_idx) → path
        # Stem may be "file-NNN" (AgiBot style) or "episode_NNNNNN" (LeRobot v3 std).
        self._data_file_paths: Dict[Tuple[int, int], str] = {}
        for pf in sorted(Path(os.path.join(data_root, "data")).rglob("*.parquet")):
            chunk_parts = [p for p in pf.parts if p.startswith("chunk-")]
            if not chunk_parts:
                continue
            chunk_idx = int(chunk_parts[-1].split("-")[1])
            stem = pf.stem
            if "-" in stem:
                file_idx = int(stem.split("-")[1])
            elif "_" in stem:
                file_idx = int(stem.split("_")[1])
            else:
                continue
            self._data_file_paths[(chunk_idx, file_idx)] = str(pf)

        # Train / val split.
        # - Respect val_ratio == 0 (no validation episodes taken).
        # - For n_ep > 1, always keep at least 1 training episode so the
        #   "train" split never becomes empty on small datasets.
        n_ep = len(self._episodes_meta)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ep)
        if val_ratio <= 0 or n_ep == 0:
            n_val = 0
        else:
            n_val = max(1, int(n_ep * val_ratio))
            if n_ep > 1:
                n_val = min(n_val, n_ep - 1)
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
        """
        Derives a mapping from episode index to a human-readable task description using episode metadata.

        Iterates the dataset's episode metadata rows and inspects the `tasks` field for each episode. For an episode with a non-empty `tasks` list, selects the description using the following precedence:
        - skip entries equal to "qualified" or "unqualified";
        - if an entry contains "@", use the substring after the first "@";
        - if an entry is ASCII, use it directly;
        - otherwise fall back to the first task entry.

        Returns:
            dict: Mapping from `episode_index` (int) to the chosen task description (str).
        """
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
        """
        Load action normalization statistics from a JSON file and store concatenated per-dimension arrays on self._action_stats.

        Parameters:
            stats_path (str): Path to a JSON file containing per-action-field statistics; each field may include `mean`, `std`, `min`, and `max` arrays.

        Behavior:
            Reads per-field `mean`, `std`, `min`, and `max` from the JSON for each field in the ActionComposer. Missing entries are filled with sensible defaults (means = 0.0, std = 1.0, min = -1.0, max = 1.0). Per-dimension standard deviations are clamped to a minimum of 1e-3. The per-field arrays are concatenated in the ActionComposer field order, truncated by `self.action_dim_slice` if set, and stored as NumPy arrays under `self._action_stats` with keys `"mean"`, `"std"`, `"min"`, and `"max"`.
        """
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
        """
        Load a parquet data file identified by chunk and file indices.

        Parameters:
            chunk_idx (int): Chunk index part of the on-disk data file key.
            file_idx (int): File index part of the on-disk data file key.

        Returns:
            pandas.DataFrame: Contents of the parquet file.

        Raises:
            FileNotFoundError: If no file path is registered for the given (chunk_idx, file_idx) key.
        """
        import pandas as pd

        path = self._data_file_paths.get((chunk_idx, file_idx))
        if path is None:
            raise FileNotFoundError(f"Data file not found: chunk={chunk_idx}, file={file_idx}")
        return pd.read_parquet(path)

    def _get_episode_data(self, ep_row, start_frame: int, end_frame: int):
        """
        Retrieve the episode's row slice from the appropriate parquet data file using episode metadata.

        This loads the parquet chunk identified by `ep_row["data/chunk_index"]` and `ep_row["data/file_index"]`, filters rows for the episode `ep_row["episode_index"]`, and returns the slice from `start_frame` (inclusive) to `end_frame` (exclusive). If the episode is not found in the expected file, the function searches other files within the same chunk and, if found, logs a warning and uses the recovered file.

        Parameters:
            ep_row (pandas.Series): Metadata row containing at least `data/chunk_index`, `data/file_index`, and `episode_index`.
            start_frame (int): Inclusive start index of the returned slice within the episode.
            end_frame (int): Exclusive end index of the returned slice within the episode.

        Returns:
            pandas.DataFrame: Episode rows for the requested frame range with a reset integer index.
        """
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

    def _get_video_frames(self, ep_row, ep_data, camera: str) -> List[Image.Image]:
        """
        Return a list of RGB PIL images corresponding to the rows in `ep_data` for `camera`, handling two on-disk layouts and padding as needed.

        This will:
        - Use a per-episode MP4 when a file exists at videos/chunk-{chunk}/{camera}/episode_{ep_idx}.mp4, decoding frames using `frame_index` from `ep_data` if present, otherwise using sequential local indices.
        - Otherwise use a global per-camera MP4 layout by mapping global frame indices from `ep_data["index"]` into the set of MP4 files under videos/{camera}/**/*.mp4 and decoding only the overlapping frame ranges.
        - If `ep_data` is empty, returns a list of black placeholder images.
        - If some frames cannot be decoded, returns black placeholders; if fewer frames are decoded than requested, pads by repeating the last decoded frame to reach the requested length.
        - In multiview mode, each returned image is sized (height//2, width//2); otherwise sized (height, width).

        Returns:
            List[PIL.Image.Image]: A list of `len(ep_data)` RGB images sized according to the dataset's height/width (or half-size per tile when multiview), one per row in `ep_data`.
        """
        tile_h = self.height // 2 if self.multiview else self.height
        tile_w = self.width // 2 if self.multiview else self.width
        n = len(ep_data)
        placeholder = [Image.new("RGB", (tile_w, tile_h)) for _ in range(n)]
        if n == 0:
            return placeholder

        # --- Layout 1: LeRobot v3 per-episode MP4 ---
        chunk_col = f"videos/{camera}/chunk_index"
        _file_col = f"videos/{camera}/file_index"  # reserved for future per-file lookup
        if chunk_col in ep_row.index:
            v_chunk = int(ep_row[chunk_col])
        else:
            v_chunk = int(ep_row["data/chunk_index"])
        ep_idx = int(ep_row["episode_index"])
        v3_path = os.path.join(
            self.data_root, "videos",
            f"chunk-{v_chunk:03d}", camera, f"episode_{ep_idx:06d}.mp4",
        )
        if os.path.exists(v3_path):
            if "frame_index" in ep_data.columns:
                local_indices = [int(x) for x in ep_data["frame_index"].values]
            else:
                local_indices = list(range(n))
            return _decode_video_frames(v3_path, local_indices, tile_h, tile_w)

        # --- Layout 2: global-index frame map (AgiBot/Galaxea) ---
        if "index" not in ep_data.columns:
            logger.warning(
                "Video layout unsupported for camera '%s': no v3 mp4 at %s and "
                "no global 'index' column in parquet",
                camera, v3_path,
            )
            return placeholder

        global_indices = ep_data["index"].values
        frame_start = int(global_indices[0])
        frame_end = int(global_indices[-1]) + 1

        video_dir = os.path.join(self.data_root, "videos")
        frame_map = _get_video_frame_map(video_dir, camera)
        if not frame_map:
            logger.warning("No video files found for camera '%s' in %s", camera, video_dir)
            return placeholder

        frames: List[Image.Image] = []
        for file_path, file_start, file_end in frame_map:
            overlap_start = max(frame_start, file_start)
            overlap_end = min(frame_end, file_end)
            if overlap_start >= overlap_end:
                continue
            local_indices = list(range(overlap_start - file_start, overlap_end - file_start))
            frames.extend(_decode_video_frames(file_path, local_indices, tile_h, tile_w))

        if not frames:
            logger.warning("Camera '%s': no frames for global range [%d, %d)", camera, frame_start, frame_end)
            return placeholder

        if len(frames) < n:
            frames = frames + [frames[-1]] * (n - len(frames))
        return frames[:n]

    # ------------------------------------------------------------------
    # Padding / normalisation
    # ------------------------------------------------------------------

    def _pad_actions(self, actions: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or truncate an action sequence to exactly `target` timesteps.

        Parameters:
            actions (np.ndarray): Array of shape (T, D) containing the action sequence.
            target (int): Desired sequence length.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - padded_actions: Array of shape (target, D) containing the original actions truncated or padded.
                - mask: Boolean array of shape (target,) with `True` for timesteps that come from the original sequence and `False` for padded timesteps.

        Notes:
            If `T < target`, padding is produced by repeating the last action; if `T == 0` zeros are used for padding.
        """
        T = len(actions)
        if T >= target:
            return actions[:target], np.ones(target, dtype=bool)
        last = actions[-1:] if T > 0 else np.zeros((1, actions.shape[1]), dtype=np.float32)
        padded = np.concatenate([actions, np.repeat(last, target - T, axis=0)])
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(target - T, dtype=bool)])
        return padded, mask

    def _pad_video(self, frames: List[Image.Image], target: int) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Pad or truncate a sequence of video frames to a fixed target length, repeating the last frame when padding.

        Parameters:
            frames (List[PIL.Image.Image]): Input list of RGB frames.
            target (int): Desired number of frames in the output sequence.

        Returns:
            Tuple[List[PIL.Image.Image], numpy.ndarray]:
                - Padded or truncated list of frames of length `target`.
                - Boolean mask of length `target` with `True` for positions corresponding to original frames and `False` for padded positions.
        """
        T = len(frames)
        if T >= target:
            return frames[:target], np.ones(target, dtype=bool)
        last = frames[-1] if frames else Image.new("RGB", (self.width, self.height))
        padded = frames + [last] * (target - T)
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(target - T, dtype=bool)])
        return padded, mask

    def _normalize(self, actions: np.ndarray) -> np.ndarray:
        """
        Apply the configured action normalization to a batch of action vectors.

        Normalizes according to self.action_norm_mode using statistics in self._action_stats:
        - "zscore": subtracts per-dimension mean and divides by per-dimension std.
        - "minmax": scales each dimension to [-1, 1] using per-dimension min and max.
        If self._action_stats is None or mode is "none", the input is returned unchanged. Unknown modes return the input unchanged.

        Parameters:
            actions (np.ndarray): Array of action vectors with shape (T, D) or compatible broadcasting.

        Returns:
            np.ndarray: Normalized action array with the same shape as `actions`.
        """
        if self._action_stats is None or self.action_norm_mode == "none":
            return actions
        if self.action_norm_mode == "zscore":
            return (actions - self._action_stats["mean"]) / self._action_stats["std"]
        if self.action_norm_mode == "minmax":
            mn, mx = self._action_stats["min"], self._action_stats["max"]
            return 2.0 * (actions - mn) / np.maximum(mx - mn, 1e-6) - 1.0
        return actions

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert a normalized action array back to the original action scale according to the dataset's normalization mode.

        Parameters:
            action (np.ndarray): Normalized action values to be denormalized.

        Returns:
            np.ndarray: Action values mapped back to the original scale with the same shape as `action`.
            - If no statistics are loaded or the mode is `"none"`, returns `action` unchanged.
            - If the mode is `"zscore"`, applies inverse z-score using stored `mean` and `std`.
            - If the mode is `"minmax"`, maps values from [-1, 1] back to the stored `[min, max]` range.
        """
        if self._action_stats is None or self.action_norm_mode == "none":
            return action
        if self.action_norm_mode == "zscore":
            return action * self._action_stats["std"] + self._action_stats["mean"]
        if self.action_norm_mode == "minmax":
            mn, mx = self._action_stats["min"], self._action_stats["max"]
            return (action + 1.0) * np.maximum(mx - mn, 1e-6) / 2.0 + mn
        return action

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of sliding-window samples available in the dataset.

        Returns:
            int: The total count of windows tracked by the dataset's internal index.
        """
        return len(self._window_index)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a dataset sample for the sliding window at the given global index.

        The method resolves the sliding-window entry into an episode and start frame, loads the corresponding action trajectory and video frames (supporting per-episode and global-frame-map layouts and optional multiview grid assembly), applies temporal stride, pads actions and video to fixed lengths, normalizes actions according to the dataset configuration, and assembles a sample dictionary containing model inputs and metadata.

        Parameters:
            idx (int): Global sliding-window index into the dataset.

        Returns:
            dict: A sample dictionary with the following keys:
                - "video": list[PIL.Image.Image] — strided and padded video frames (per-frame images).
                - "action": torch.Tensor — padded action trajectory (shape [T, D]).
                - "action_mask": torch.Tensor — boolean mask for valid (non-padded) action timesteps (shape [T]).
                - "prompt": str — textual prompt describing the task and optional camera names.
                - "episode_index": int — episode identifier for the sample.
                - "info": dict — auxiliary metadata containing:
                    - "action_trajectory": torch.Tensor — same action array as in "action".
                    - "video_mask": torch.Tensor — boolean mask for valid (non-padded) video frames.
                    - "vace_reference_image": list — first strided frame if available, otherwise empty.
                    - "vace_video": None — reserved field.
                    - "start_frame": int — start frame within the episode.
                    - "end_frame": int — end frame within the episode (exclusive).
                    - "episode_length": int — total length of the episode.
                    - "task_name": str — dataset-level task name.
                    - additional columns from the episode parquet rows (when present), such as state, intrinsics, etc.
        """
        # Iteratively skip empty episodes to avoid unbounded recursion when
        # many/all episodes are empty (e.g. corrupted dataset or misconfig).
        # Bound attempts to len(self) so we scan each window at most once.
        n_windows = len(self)
        max_attempts = max(1, n_windows)
        cur_idx = idx
        for _attempt in range(max_attempts):
            local_idx, start_frame = self._window_index[cur_idx]
            ep_global_idx = self._episode_indices[local_idx]
            ep_row = self._episodes_meta.iloc[ep_global_idx]
            ep_len = int(ep_row["length"])
            ep_idx = int(ep_row["episode_index"])
            end_frame = min(start_frame + self.num_frames, ep_len)

            # Actions
            ep_data = self._get_episode_data(ep_row, start_frame, end_frame)
            if len(ep_data) > 0:
                break
            chunk_idx = int(ep_row["data/chunk_index"])
            file_idx = int(ep_row["data/file_index"])
            file_path = self._data_file_paths.get((chunk_idx, file_idx), "unknown")
            logger.warning(
                "Empty episode data ep_idx=%d start=%d, file=%s",
                ep_idx, start_frame, file_path,
            )
            cur_idx = (cur_idx + 1) % n_windows
        else:
            raise RuntimeError(
                f"All {n_windows} windows yielded empty episode data for task "
                f"{self.task_name!r}; check dataset integrity."
            )
        actions = self._action_composer.extract(ep_data)
        if self.action_transform is not None:
            actions = self.action_transform(actions)
        elif self.action_dim_slice is not None:
            actions = actions[:, :self.action_dim_slice]

        # Video — decode only as many frames as needed. single-image
        # consumers set video_num_frames=1 so we avoid decoding 15 unused frames.
        # Slice ep_data to the video horizon so _get_video_frames sees the right
        # range (needed for both per-episode local indices and global frame-map).
        video_frames_count = self.video_num_frames if self.video_num_frames is not None else self.num_frames
        video_rows = min(video_frames_count, len(ep_data))
        ep_data_video = ep_data.iloc[:video_rows] if video_rows < len(ep_data) else ep_data
        if self.multiview:
            tile_h, tile_w = self.height // 2, self.width // 2
            all_frames = {
                cam: self._get_video_frames(ep_row, ep_data_video, cam)
                for cam in self.cameras
            }
            T = len(ep_data_video)
            video_frames = [
                _assemble_grid(
                    {cam: all_frames[cam][t] for cam in self.cameras if t < len(all_frames[cam])},
                    self.camera_layout, tile_h, tile_w,
                )
                for t in range(T)
            ]
        else:
            video_frames = self._get_video_frames(ep_row, ep_data_video, self.target_camera)

        # Stride → pad. Target count derives from video_frames_count so
        # _pad_video doesn't duplicate up to the full action horizon.
        video_strided = video_frames[:: self.video_stride]
        video_target = (video_frames_count + self.video_stride - 1) // self.video_stride
        video_target += (1 - video_target % 4) % 4  # VACE requires num_frames % 4 == 1
        actions, action_mask = self._pad_actions(actions, self.num_frames)
        video_strided, video_mask = self._pad_video(video_strided, video_target)
        actions = self._normalize(actions)

        prompt = f"The robot is performing: {self._task_descriptions.get(ep_idx, self.task_name)}"
        if self.multiview and self.camera_layout:
            cam_names = [c.split(".")[-1] for row in self.camera_layout for c in row]
            prompt += f" Cameras: {', '.join(cam_names)}."

        # Build info dict: model-specific fields + extra parquet columns
        info = {
            "action_trajectory": torch.from_numpy(actions),
            "video_mask": torch.from_numpy(video_mask),
            "vace_reference_image": [video_strided[0]] if video_strided else [],
            "vace_video": None,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "episode_length": ep_len,
            "task_name": self.task_name,
        }

        # Extra parquet columns → info (state, fov, intrinsics, etc.)
        known_cols = {"episode_index", "frame_index", "index", "timestamp",
                      "task_index"} | set(self._action_composer.action_fields)
        for col in ep_data.columns:
            if col not in known_cols and col not in info:
                vals = ep_data[col].values
                try:
                    info[col] = np.stack(vals) if hasattr(vals[0], '__len__') else vals
                except (TypeError, ValueError):
                    pass

        sample = {
            "video": video_strided,
            "action": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
            "prompt": prompt,
            "episode_index": ep_idx,
            "info": info,
        }
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample

    @property
    def action_dim(self) -> int:
        """
        Get the effective action output dimensionality for this dataset.

        Returns:
            int: Effective action dimension — uses the explicit output dimension if set, otherwise the action-dimension slice if provided, otherwise the composed action dimension from the ActionComposer.
        """
        if self._action_out_dim is not None:
            return self._action_out_dim
        if self.action_dim_slice is not None:
            return self.action_dim_slice
        return self._action_composer.action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        """
        Stored action normalization statistics for the dataset.

        Returns:
            A dict mapping statistic names ('mean', 'std', 'min', 'max') to numpy arrays concatenated across action fields, or `None` if statistics were not loaded.
        """
        return self._action_stats

    @classmethod
    def from_config(cls, config, split: str = "train") -> "LeRobot3Dataset":
        """
        Create a LeRobot3Dataset instance from a configuration-like object for the specified split.

        The `config` may be either an object with attributes or a mapping (supports `.get`). Values are taken from an attribute if present and not None, otherwise from `config.get(key, default)`. Recognized keys (and their meanings) include:
        - `dataset_dir` or `data_root`: dataset root directory (preferred key order).
        - `action_fields`: list of action column names (defaults to `["action"]`).
        - `target_camera`, `cameras`: camera selection; if `cameras` is missing but `target_camera` is provided, `cameras` will default to `[target_camera]`.
        - Optional dataset constructor overrides: `camera_layout`, `fps`, `num_frames`, `height`, `width`, `val_ratio`, `seed`, `task_name`, `multiview`, `action_norm_mode`, `window_stride`, `video_stride`, `repeat`, `num_val_samples`, `action_stats_path`, `action_dim_slice`, `action_out_dim`, `video_num_frames`.

        Parameters:
            config: object or mapping containing dataset configuration values.
            split (str): dataset split to construct (e.g., `"train"` or `"val"`).

        Returns:
            LeRobot3Dataset: a new dataset instance configured according to the provided `config` and `split`.
        """
        def _get(k, default=None):
            if hasattr(config, k):
                v = getattr(config, k)
                if v is not None:
                    return v
            if hasattr(config, "get"):
                return config.get(k, default)
            return default

        target_camera = _get("target_camera")
        cameras = _get("cameras") or ([target_camera] if target_camera else [])
        kwargs = dict(
            data_root=_get("dataset_dir") or _get("data_root"),
            action_fields=list(_get("action_fields") or ["action"]),
            target_camera=target_camera,
            cameras=list(cameras),
            split=split,
        )
        for k in (
            "camera_layout", "fps", "num_frames", "height", "width",
            "val_ratio", "seed", "task_name", "multiview", "action_norm_mode",
            "window_stride", "video_stride", "repeat", "num_val_samples",
            "action_stats_path", "action_dim_slice", "action_out_dim",
            "video_num_frames",
        ):
            v = _get(k)
            if v is not None:
                kwargs[k] = v
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# MultiTaskLeRobot3Dataset — multi-task wrapper
# ---------------------------------------------------------------------------


class MultiTaskLeRobot3Dataset(BaseActionDataset):
    """Concatenates a list of LeRobot3Dataset instances with O(log n) dispatch.

    Sub-dataset adapters (AgibotDataset, GalaxeaDataset) build the dataset
    list and pass it to ``super().__init__(datasets)``.
    """

    def __init__(self, datasets: List[LeRobot3Dataset]):
        """
        Initialize the concatenated multi-task dataset and compute cumulative window counts for fast index routing.

        Parameters:
            datasets (List[LeRobot3Dataset]): Non-empty list of per-task datasets to concatenate.

        Raises:
            ValueError: If `datasets` is empty.

        Side effects:
            Stores `datasets` on the instance and builds `self._cumulative_lengths` as a running total of each dataset's length for O(log n) dispatch in __getitem__. Also logs a summary line including task and window counts.
        """
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
        """
        Total number of samples available across all constituent datasets.

        Returns:
            total_length (int): Sum of lengths of all underlying datasets (the final cumulative length).
        """
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict:
        """
        Route a global index to the appropriate sub-dataset and return the corresponding sample.

        Parameters:
            idx (int): Global sample index across the concatenated datasets.

        Returns:
            dict: The sample dictionary produced by the selected sub-dataset for the resolved local index.
        """
        ds_idx = bisect.bisect_right(self._cumulative_lengths, idx)
        if ds_idx > 0:
            idx -= self._cumulative_lengths[ds_idx - 1]
        return self._datasets[ds_idx][idx]

    @property
    def action_dim(self) -> int:
        """
        Expose the action vector dimensionality used by the concatenated sub-datasets.

        Returns:
            int: The action dimension from the first sub-dataset.
        """
        return self._datasets[0].action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        """
        Action normalization statistics for the concatenated datasets.

        Returns:
            Optional[dict]: The action statistics dictionary from the first sub-dataset, or `None` if no stats are available.
        """
        return self._datasets[0].action_stats

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert a normalized action vector back to the dataset's original action scale.

        Returns:
            The denormalized action array in the original action space (unchanged if no normalization was configured).
        """
        return self._datasets[0].denormalize_action(action)
