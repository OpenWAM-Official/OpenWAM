"""
VideoActionDataset base class and RoboTwin dataset implementations.

Reads RoboTwin 2.0 episode HDF5 files directly with lazy loading —
no data is cached in memory. Action stats are loaded from a precomputed file.

Classes:
    RoboTwinDataset       — Single-task training/eval dataset.
    MultiTaskRoboTwinDataset — Multi-task dataset using discover_robotwin_roots().
"""

import io
import json
import os
import glob
import random
import numpy as np
import torch
import h5py
from abc import ABC, abstractmethod
from PIL import Image
from typing import Optional


# ---------------------------------------------------------------------------
# Per-backbone supported resolutions
#
# "vace"  — Wan2.1-VACE-1.3B / 14B: limited to training resolutions tested by Wan team.
# "ti2v"  — Wan2.2-TI2V-5B: only requires height % 32 == 0 and width % 32 == 0.
# None    — unknown/unspecified backbone: falls back to divisibility-by-32 check.
# ---------------------------------------------------------------------------

BACKBONE_SUPPORTED_RESOLUTIONS: dict = {
    "vace": {(480, 832), (720, 1280)},
    "ti2v": None,   # any (h%32==0, w%32==0) is valid
}

# ---------------------------------------------------------------------------
# RoboTwin 2.0 task split (42 train / 8 holdout)
#
# Holdout tasks cover diverse skill types so each category retains training
# coverage.  The selection aligns with 6 of the 8 tasks evaluated in the
# RoboTwin 2.0 paper (Table 3, Section 4.3).
# ---------------------------------------------------------------------------

ROBOTWIN_HOLDOUT_TASKS = [
    "handover_block",       # handover  (handover_mic remains in train)
    "move_can_pot",         # place/move (20 other place/move tasks in train)
    "open_laptop",          # open      (open_microwave remains in train)
    "pick_dual_bottles",    # pick      (adjust_bottle, grab_roller, pick_diverse_bottles in train)
    "place_object_basket",  # place     (16 other place_* tasks in train)
    "press_stapler",        # press     (click_alarmclock, click_bell in train)
    "stack_bowls_two",      # stack     (stack_blocks_two/three, stack_bowls_three in train)
    "turn_switch",          # rotate    (rotate_qrcode, scan_object, shake_bottle* in train)
]

ROBOTWIN_ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
    "blocks_ranking_size", "click_alarmclock", "click_bell",
    "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away",
    "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left",
    "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox",
    "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket",
    "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three",
    "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
    "stamp_seal", "turn_switch",
]

ROBOTWIN_TRAIN_TASKS = sorted(
    t for t in ROBOTWIN_ALL_TASKS if t not in ROBOTWIN_HOLDOUT_TASKS
)


def discover_robotwin_roots(
    dataset_dir: str,
    robot: str,
    variant: str = "clean_50",
    tasks: Optional[list] = None,
) -> list:
    """Auto-discover per-task data roots from the RoboTwin dataset directory.

    Args:
        dataset_dir: Top-level directory (e.g. ``/path/to/robotwin_2_0/dataset``).
        robot: Robot name (e.g. ``"aloha-agilex"``).
        variant: ``"clean_50"`` or ``"randomized_500"``.
        tasks: List of task names.  Defaults to ``ROBOTWIN_TRAIN_TASKS``.

    Returns:
        List of ``(task_name, data_root)`` tuples for tasks that exist on disk.
    """
    if tasks is None:
        tasks = ROBOTWIN_TRAIN_TASKS

    roots = []
    for task in tasks:
        data_root = os.path.join(dataset_dir, task, f"{robot}_{variant}", "data")
        if os.path.isdir(data_root):
            roots.append((task, data_root))
    return roots


# ---------------------------------------------------------------------------
# Multi-view 2x2 grid layout (DreamZero / DreamGen style)
#
# Tiles 4 camera views into a spatial grid at the original training
# resolution.  Each quadrant is (height//2, width//2).
#
#   +------------------+------------------+
#   |   head_camera    | third_view_rgb   |
#   +------------------+------------------+
#   |  left_camera     |  right_camera    |
#   +------------------+------------------+
#
# third_view_rgb is a fixed external camera shared across arx-x5, franka,
# and ur5.  It provides a front-facing overview of the full workspace.
# ---------------------------------------------------------------------------

MULTIVIEW_LAYOUT = [["head_camera", "third_view_rgb"], ["left_camera", "right_camera"]]
MULTIVIEW_CAMERAS = ["head_camera", "third_view_rgb", "left_camera", "right_camera"]


def _crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Center-crop and resize image to target size (preserves aspect ratio).

    Scales the image so the shorter side matches the target, then
    center-crops to the exact target resolution. Avoids aspect ratio distortion.
    """
    img_w, img_h = image.size
    scale = max(target_width / img_w, target_height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


def assemble_multiview_grid(
    frames_by_camera: dict,
    camera_layout: list,
    quadrant_h: int,
    quadrant_w: int,
) -> Image.Image:
    """Assemble per-camera frames into a 2x2 spatial grid.

    Args:
        frames_by_camera: ``{camera_name: PIL.Image}`` for each camera.
            Missing cameras are rendered as black.
        camera_layout: 2D list of camera names (or ``None`` for black).
        quadrant_h: Height of each quadrant in pixels.
        quadrant_w: Width of each quadrant in pixels.

    Returns:
        Assembled PIL Image of size ``(cols * quadrant_w, rows * quadrant_h)``.
    """
    rows = len(camera_layout)
    cols = max(len(row) for row in camera_layout)
    canvas = Image.new("RGB", (cols * quadrant_w, rows * quadrant_h), (0, 0, 0))

    for r, row in enumerate(camera_layout):
        for c, cam_name in enumerate(row):
            if cam_name is None:
                continue  # black quadrant
            frame = frames_by_camera.get(cam_name)
            if frame is None:
                continue  # missing camera → black
            # Crop-and-resize to quadrant size
            frame = _crop_and_resize(frame, quadrant_h, quadrant_w)
            canvas.paste(frame, (c * quadrant_w, r * quadrant_h))

    return canvas


def extract_quadrant(
    grid_image: Image.Image,
    row: int,
    col: int,
    quadrant_h: int,
    quadrant_w: int,
) -> Image.Image:
    """Crop a single quadrant from a grid image (for eval visualization).

    Args:
        grid_image: Full grid PIL Image.
        row: Row index (0-based).
        col: Column index (0-based).
        quadrant_h: Height of each quadrant.
        quadrant_w: Width of each quadrant.

    Returns:
        Cropped PIL Image of size ``(quadrant_w, quadrant_h)``.
    """
    left = col * quadrant_w
    top = row * quadrant_h
    return grid_image.crop((left, top, left + quadrant_w, top + quadrant_h))


def _pad_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Resize preserving aspect ratio, then center-pad with black to target size.

    Scales the image so the longer side fits within the target, then pads
    the shorter side with black pixels to reach the exact target resolution.
    """
    img_w, img_h = image.size
    scale = min(target_width / img_w, target_height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    left = (target_width - new_w) // 2
    top = (target_height - new_h) // 2
    canvas.paste(image, (left, top))
    return canvas


def _resize_frame(
    image: Image.Image,
    target_height: int,
    target_width: int,
    resize_mode: str = "pad",
) -> Image.Image:
    """Resize a single frame according to *resize_mode*.

    Modes:
        ``crop``    – scale preserving aspect ratio so the image covers the
                      target area (short-side fit), then center-crop the
                      overflowing dimension.
        ``pad``     – scale preserving aspect ratio so the image fits inside
                      the target area (long-side fit), then center-pad the
                      shorter dimension with black pixels.
        ``stretch`` – directly resize to target size, ignoring aspect ratio.
    """
    if resize_mode == "stretch":
        return image.resize((target_width, target_height), Image.LANCZOS)
    if resize_mode == "pad":
        return _pad_and_resize(image, target_height, target_width)
    return _crop_and_resize(image, target_height, target_width)


class VideoActionDataset(torch.utils.data.Dataset, ABC):
    """Abstract base for video-action datasets.

    Subclasses must implement __getitem__ returning a dict with keys:
        "video":                List[PIL.Image]   # target video (robot)
        "vace_video":           List[PIL.Image] | None  # control video (optional)
        "vace_reference_image": List[PIL.Image]   # [first frame of target]
        "action_trajectory":    torch.Tensor       # (T, action_dim) normalized
        "prompt":               str

    All temporal dimensions must be aligned 1:1.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def action_stats(self) -> dict:
        """Return {"mean": np.ndarray(action_dim,), "std": np.ndarray(action_dim,)}"""
        ...

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        stats = self.action_stats
        return action * stats["std"] + stats["mean"]

class RoboTwinDataset(VideoActionDataset):
    """RoboTwin 2.0 HDF5 dataset for bimanual robot video-action training.

    Reads episode HDF5 files with JPEG-encoded camera observations and
    14/16-DoF joint-space actions (qpos). Supports all 5 RoboTwin embodiments.

    Epoch strategy:
        Training enumerates all valid ``(episode, start_frame)`` windows
        exhaustively with configurable stride (``window_stride``), so one
        epoch = one pass through every window. ``repeat`` multiplies the
        window list for additional passes.

    Action = joint_action/vector (T, 14|16): [left_arm, left_gripper, right_arm, right_gripper]
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        split: str = "train",
        val_ratio: float = 0.1,
        repeat: int = 1,
        task_name: Optional[str] = None,
        seed: int = 42,
        action_stats_path: Optional[str] = None,
        num_val_samples: int = 4,
        target_camera: str = "head_camera",
        window_stride: int = 1,
        multiview: bool = False,
        robot: Optional[str] = None,
        variant: str = "clean_50",
        backbone: Optional[str] = None,
    ):
        super().__init__()
        self.robot = robot
        self.variant = variant

        # Validate resolution against backbone constraints.
        _supported = BACKBONE_SUPPORTED_RESOLUTIONS.get(backbone, None) if backbone else None
        if _supported is not None:
            if (height, width) not in _supported:
                supported_str = ", ".join(f"{h}x{w}" for h, w in sorted(_supported))
                raise ValueError(
                    f"backbone='{backbone}' only supports resolutions: {supported_str}. "
                    f"Got {height}x{width}."
                )
        elif height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                f"Resolution {height}x{width} must be divisible by 32 "
                f"(VAE downsamples by 16, patch size 2)."
            )

        self.data_root = data_root
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.repeat = repeat
        self.split = split
        self.task_name = task_name or "manipulation"
        self.target_camera = target_camera
        self.window_stride = max(1, window_stride)
        self.multiview = multiview
        self.cameras = MULTIVIEW_CAMERAS if multiview else [target_camera]
        self.camera_layout = MULTIVIEW_LAYOUT if multiview else None
        self.quadrant_h = height // 2 if multiview else height
        self.quadrant_w = width // 2 if multiview else width

        # ---- Discover and sort target episode files ----
        pattern = os.path.join(data_root, "episode*.hdf5")
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f"No episode*.hdf5 files found in {data_root}")

        # Train/val split (deterministic)
        rng = random.Random(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        if val_ratio <= 0.0:
            n_val = 0  # All episodes for training
        elif val_ratio >= 1.0:
            n_val = len(all_files)  # All episodes for validation
        else:
            n_val = max(1, int(len(all_files) * val_ratio))
        if split == "val":
            selected = sorted(indices[:n_val])
        else:
            selected = sorted(indices[n_val:])

        self._episode_files = [all_files[i] for i in selected]
        if not self._episode_files:
            raise ValueError(
                f"No episodes selected for split='{split}' with val_ratio={val_ratio} "
                f"({len(all_files)} total episodes in {data_root})"
            )
        print(f"RoboTwinDataset: {len(self._episode_files)} episodes ({split})")

        # Probe episodes for length and action dim
        self._episode_lengths = []
        self._action_dim_detected = None

        for path in self._episode_files:
            with h5py.File(path, "r") as f:
                T = f[f"observation/{target_camera}/rgb"].shape[0]
                self._episode_lengths.append(T)

                if self._action_dim_detected is None:
                    self._action_dim_detected = f["joint_action/vector"].shape[1]

        print(f"  Episode lengths: min={min(self._episode_lengths)}, "
              f"max={max(self._episode_lengths)}, "
              f"action_dim={self._action_dim_detected}")

        # ---- Validate multiview cameras ----
        if self.multiview:
            with h5py.File(self._episode_files[0], "r") as f:
                obs_keys = list(f["observation"].keys()) if "observation" in f else []
                for cam in self.cameras:
                    if cam == "third_view_rgb":
                        if "third_view_rgb" not in f:
                            print(f"  WARNING: '{cam}' not found at top level in "
                                  f"{self._episode_files[0]}. "
                                  f"Will use black frames.")
                    else:
                        cam_key = f"observation/{cam}/rgb"
                        if cam_key not in f:
                            print(f"  WARNING: multiview camera '{cam}' not found in "
                                  f"{self._episode_files[0]}. Available: {obs_keys}. "
                                  f"Will use black frames for missing cameras.")
            print(f"  Multiview mode: layout={self.camera_layout}, "
                  f"quadrant={self.quadrant_h}x{self.quadrant_w}")

        # ---- Exhaustive window enumeration ----
        self._window_index = []  # List of (episode_idx, start_frame)
        for ep_idx, ep_len in enumerate(self._episode_lengths):
            max_start = max(0, ep_len - self.num_frames)
            for start in range(0, max_start + 1, self.window_stride):
                self._window_index.append((ep_idx, start))
        if repeat > 1:
            self._window_index = self._window_index * repeat
        print(f"  Exhaustive windows: {len(self._window_index)} "
              f"(stride={self.window_stride}, repeat={repeat})")

        # ---- Load scene_info for active arm detection ----
        self._scene_info = {}
        scene_info_path = os.path.join(data_root, "..", "scene_info.json")
        if not os.path.exists(scene_info_path):
            scene_info_path = os.path.join(data_root, "scene_info.json")
        if os.path.exists(scene_info_path):
            with open(scene_info_path) as _f:
                self._scene_info = json.load(_f)
            print(f"  Scene info loaded from {scene_info_path} "
                  f"({len(self._scene_info)} entries)")
        else:
            print(f"  No scene_info.json found, active_arm will default to 'both'")

        # ---- Load action stats ----
        self._action_dim_value = self._action_dim_detected or 14
        self._action_stats = None
        stats_path = action_stats_path or os.path.join(data_root, "action_stats.npy")
        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True).item()
            mean = stats["mean"].astype(np.float32)
            std = np.maximum(stats["std"].astype(np.float32), 1e-3)
            if self._action_dim_detected is not None and len(mean) != self._action_dim_detected:
                raise ValueError(
                    f"Action stats dimension ({len(mean)}) does not match "
                    f"detected action_dim ({self._action_dim_detected}) from data. "
                    f"Check that {stats_path} was computed for this robot/dataset."
                )
            self._action_stats = {"mean": mean, "std": std}
            self._action_dim_value = len(mean)
            print(f"  Action stats loaded from {stats_path}")
            print(f"    Mean: {mean}")
            print(f"    Std:  {std}")
        else:
            print(f"  WARNING: No action stats found at {stats_path}, "
                  "actions will NOT be normalized")

        # ---- Try loading instruction prompts ----
        self._instructions = {}
        instr_dir = os.path.join(os.path.dirname(data_root), "instructions")
        if os.path.isdir(instr_dir):
            for fname in os.listdir(instr_dir):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(instr_dir, fname), "r") as jf:
                            self._instructions[fname] = json.load(jf)
                    except Exception:
                        pass
            if self._instructions:
                print(f"  Loaded {len(self._instructions)} instruction files")

        # ---- Validation mode ----
        self._val_samples = None
        if split == "val" and num_val_samples > 0:
            val_rng = random.Random(seed + 1)
            self._val_samples = []
            for _ in range(num_val_samples):
                ep_idx = val_rng.randint(0, len(self._episode_files) - 1)
                ep_len = self._episode_lengths[ep_idx]
                max_start = max(0, ep_len - self.num_frames)
                start_idx = val_rng.randint(0, max_start)
                self._val_samples.append((ep_idx, start_idx))
            print(f"  Val: {len(self._val_samples)} fixed samples")
        elif split == "val":
            print(f"  Val: exhaustive windows ({len(self._window_index)} samples)")

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def action_stats(self) -> dict:
        return self._action_stats

    def __len__(self):
        if self._val_samples is not None:
            return len(self._val_samples)
        return len(self._window_index)

    def _decode_jpeg(self, jpeg_bytes) -> Image.Image:
        """Decode JPEG bytes from HDF5 to a PIL RGB image."""
        return Image.open(io.BytesIO(bytes(jpeg_bytes))).convert("RGB")

    def _read_camera_frames(self, f, camera_key: str, start: int, end: int):
        """Read and decode JPEG frames from an observation camera or third_view_rgb.

        Args:
            f: Open HDF5 file handle.
            camera_key: Camera identifier. For observation cameras this is just
                the camera name (e.g. "head_camera") and data lives at
                ``observation/{camera_key}/rgb``.  The special value
                ``"third_view_rgb"`` reads directly from the top-level dataset.
            start: Start frame index (inclusive).
            end: End frame index (exclusive).

        Returns:
            List of PIL.Image.Image in RGB.
        """
        if camera_key == "third_view_rgb":
            raw = f["third_view_rgb"][start:end]
        else:
            raw = f[f"observation/{camera_key}/rgb"][start:end]
        return [self._decode_jpeg(raw[i]) for i in range(len(raw))]

    def _read_multiview_frames(self, f, cameras, start, end):
        """Read frames from multiple cameras and assemble into 2x2 grids.

        Args:
            f: Open HDF5 file handle.
            cameras: List of camera names to read.
            start: Start frame index (inclusive).
            end: End frame index (exclusive).

        Returns:
            List of grid PIL Images (already at full resolution).
        """
        # Read frames per camera, falling back to black on KeyError
        per_camera = {}
        for cam in cameras:
            try:
                per_camera[cam] = self._read_camera_frames(f, cam, start, end)
            except KeyError:
                n = end - start
                per_camera[cam] = [
                    Image.new("RGB", (self.quadrant_w, self.quadrant_h), (0, 0, 0))
                    for _ in range(n)
                ]

        # Assemble per-timestep grids
        n = end - start
        grids = []
        for t in range(n):
            frames_t = {cam: per_camera[cam][t] for cam in cameras}
            grid = assemble_multiview_grid(
                frames_t, self.camera_layout, self.quadrant_h, self.quadrant_w
            )
            grids.append(grid)
        return grids

    def _get_prompt(self, ep_idx: int) -> str:
        """Get text prompt for episode, from instructions or task_name.

        Instruction files use RoboTwin format: ``{"seen": [...], "unseen": [...]}``.
        During training, a random instruction is sampled from "seen".
        During validation, the first "seen" instruction is used for reproducibility.
        """
        ep_file = os.path.basename(self._episode_files[ep_idx])
        ep_num = ep_file.replace("episode", "").replace(".hdf5", "")
        instr_key = f"episode{ep_num}.json"

        base_prompt = None
        if instr_key in self._instructions:
            instr = self._instructions[instr_key]
            if isinstance(instr, dict):
                # RoboTwin format: {"seen": [...], "unseen": [...]}
                pool = instr.get("seen") or instr.get("unseen") or []
                if pool:
                    if self.split == "train":
                        base_prompt = random.choice(pool)
                    else:
                        base_prompt = pool[0]
                elif "instruction" in instr:
                    base_prompt = instr["instruction"]
            elif isinstance(instr, str):
                base_prompt = instr
            elif isinstance(instr, list) and len(instr) > 0:
                base_prompt = instr[0] if isinstance(instr[0], str) else str(instr[0])

        if base_prompt is None:
            base_prompt = f"The bimanual robot is performing a {self.task_name} task."

        if self.multiview:
            return (
                f"A multi-view video shows that {base_prompt} "
                f"The video is split into four views: "
                f"head camera (top-left), third-person view (top-right), "
                f"left camera (bottom-left), right camera (bottom-right)."
            )

        return base_prompt

    def __getitem__(self, idx):
        # ---- Determine episode index and start frame ----
        if self._val_samples is not None:
            ep_idx, start = self._val_samples[idx]
        else:
            ep_idx, start = self._window_index[idx]

        end = start + self.num_frames
        path = self._episode_files[ep_idx]
        ep_len = self._episode_lengths[ep_idx]
        actual_end = min(end, ep_len)

        # ---- Read target frames + actions ----
        with h5py.File(path, "r") as f:
            if self.multiview:
                target_frames = self._read_multiview_frames(
                    f, self.cameras, start, actual_end
                )
            else:
                target_frames = self._read_camera_frames(
                    f, self.target_camera, start, actual_end
                )
            actions = f["joint_action/vector"][start:actual_end].astype(np.float32)

        actual_len = len(target_frames)

        # ---- Pad target + actions if shorter than num_frames ----
        valid_len = actual_len
        if actual_len < self.num_frames:
            pad_len = self.num_frames - actual_len
            target_frames = target_frames + [target_frames[-1]] * pad_len
            actions = np.concatenate([
                actions,
                np.repeat(actions[-1:], pad_len, axis=0),
            ], axis=0)

        # Action mask: True for real frames, False for padded frames
        action_mask = torch.ones(self.num_frames, dtype=torch.bool)
        action_mask[valid_len:] = False

        # ---- Normalize actions ----
        if self._action_stats is not None:
            actions = (actions - self._action_stats["mean"]) / self._action_stats["std"]

        # ---- Crop and resize frames to target resolution ----
        if self.multiview:
            video = target_frames
        else:
            video = [
                _crop_and_resize(frame, self.height, self.width)
                for frame in target_frames
            ]
        vace_reference_image = [video[0]]

        prompt = self._get_prompt(ep_idx)

        # ---- Extract active arm from scene_info ----
        ep_key = f"episode_{ep_idx}"
        ep_scene = self._scene_info.get(ep_key, {})
        ep_info = ep_scene.get("info", ep_scene)
        active_arm = ep_info.get("{a}", ep_info.get("active_arm", "both"))

        result = {
            "video": video,
            "vace_video": None,
            "vace_reference_image": vace_reference_image,
            "action_trajectory": torch.from_numpy(actions),
            "action_mask": action_mask,
            "prompt": prompt,
            # Metadata
            "episode_index": ep_idx,
            "episode_path": path,
            "start_frame": start,
            "end_frame": min(end, ep_len),
            "episode_length": ep_len,
            "task_name": self.task_name,
            "active_arm": active_arm,
        }
        return result


class MultiTaskRoboTwinDataset(VideoActionDataset):
    """Multi-task wrapper over multiple RoboTwinDatasets.

    Concatenates per-task RoboTwinDatasets so one epoch covers all tasks.
    Action stats are shared across tasks (loaded from a single file).

    Uses ``discover_robotwin_roots()`` to auto-discover per-task data
    directories and creates one ``RoboTwinDataset`` per task.

    Args:
        dataset_dir: Top-level RoboTwin dataset directory.
        robot: Target robot name (e.g. ``"aloha-agilex"``).
        variant: Data variant (``"clean_50"`` or ``"randomized_500"``).
        tasks: List of task names.  Defaults to ``ROBOTWIN_TRAIN_TASKS``.
        action_stats_path: Path to shared action stats (.npy).
        **kwargs: Forwarded to each ``RoboTwinDataset`` (num_frames, height,
            width, split, val_ratio, repeat, seed, target_camera,
            window_stride, num_val_samples, backbone, ...).
    """

    def __init__(
        self,
        dataset_dir: str,
        robot: str,
        variant: str = "clean_50",
        tasks: Optional[list] = None,
        action_stats_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        roots = discover_robotwin_roots(dataset_dir, robot, variant, tasks)
        if not roots:
            task_list = tasks or ROBOTWIN_TRAIN_TASKS
            raise FileNotFoundError(
                f"No task data found in {dataset_dir} for robot={robot}, "
                f"variant={variant}. Checked {len(task_list)} tasks."
            )

        self._sub_datasets = []
        self._cumulative_lengths = []
        cumulative = 0

        print(f"MultiTaskRoboTwinDataset: {len(roots)} tasks, "
              f"robot={robot}, variant={variant}")

        for task_name, data_root in roots:
            ds = RoboTwinDataset(
                data_root=data_root,
                task_name=task_name.replace("_", " "),
                action_stats_path=action_stats_path,
                robot=robot,
                variant=variant,
                **kwargs,
            )
            self._sub_datasets.append(ds)
            cumulative += len(ds)
            self._cumulative_lengths.append(cumulative)

        self._total_length = cumulative
        self._action_stats_shared = (
            self._sub_datasets[0].action_stats if self._sub_datasets else None
        )
        self._action_dim_value = (
            self._sub_datasets[0].action_dim if self._sub_datasets else 14
        )

        print(f"  Total samples: {self._total_length} "
              f"(across {len(self._sub_datasets)} tasks)")

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def action_stats(self) -> dict:
        return self._action_stats_shared

    def __len__(self):
        return self._total_length

    def __getitem__(self, idx):
        # Binary search for the sub-dataset containing this index
        lo, hi = 0, len(self._cumulative_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative_lengths[mid]:
                hi = mid
            else:
                lo = mid + 1
        ds_idx = lo
        local_idx = idx if ds_idx == 0 else idx - self._cumulative_lengths[ds_idx - 1]
        return self._sub_datasets[ds_idx][local_idx]
