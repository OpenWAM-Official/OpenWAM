"""
VideoActionDataset base class and RoboTwin dataset implementations.

Reads RoboTwin 2.0 episode HDF5 files directly with lazy loading —
no data is cached in memory. Action stats are loaded from a precomputed file.

Classes:
    RoboTwinDataset       — Single-task training/eval dataset.
    MultiTaskRoboTwinDataset — Multi-task dataset using discover_robotwin_roots().
"""

import glob
import json
import os
import random
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.base_dataset import BaseActionDataset
from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d

# ---------------------------------------------------------------------------
# Per-backbone supported resolutions
#
# "vace"  — Wan2.1-VACE-1.3B / 14B: limited to training resolutions tested by Wan team.
# "ti2v"  — Wan2.2-TI2V-5B: only requires height % 32 == 0 and width % 32 == 0.
# None    — unknown/unspecified backbone: falls back to divisibility-by-32 check.
# ---------------------------------------------------------------------------

BACKBONE_SUPPORTED_RESOLUTIONS: dict = {
    "vace": {(480, 832), (720, 1280)},
    "ti2v": None,  # any (h%32==0, w%32==0) is valid
}

# ---------------------------------------------------------------------------
# RoboTwin 2.0 task split (42 train / 8 holdout)
#
# Holdout tasks cover diverse skill types so each category retains training
# coverage.  The selection aligns with 6 of the 8 tasks evaluated in the
# RoboTwin 2.0 paper (Table 3, Section 4.3).
# ---------------------------------------------------------------------------

# All 50 tasks participate in training by default.
# Uncomment below to hold out 8 tasks for OOD evaluation:
# ROBOTWIN_HOLDOUT_TASKS = [
#     "handover_block",       # handover
#     "move_can_pot",         # place/move
#     "open_laptop",          # open
#     "pick_dual_bottles",    # pick
#     "place_object_basket",  # place
#     "press_stapler",        # press
#     "stack_bowls_two",      # stack
#     "turn_switch",          # rotate
# ]
ROBOTWIN_HOLDOUT_TASKS = []

ROBOTWIN_ALL_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]

ROBOTWIN_TRAIN_TASKS = sorted(t for t in ROBOTWIN_ALL_TASKS if t not in ROBOTWIN_HOLDOUT_TASKS)

# ---------------------------------------------------------------------------
# Action mode constants
# ---------------------------------------------------------------------------

EEF_ACTION_DIM = 20  # [xyz(3) + rot6d(6) + gripper(1)] × 2 arms
EEF_GRIPPER_INDICES = [9, 19]  # gripper positions in 20D EEF vector
JOINT_GRIPPER_INDICES = [6, 13]  # gripper positions in 14D joint vector


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
#   |   head_camera    |  front_camera    |
#   +------------------+------------------+
#   |  left_camera     |  right_camera    |
#   +------------------+------------------+
#
# front_camera is a fixed external camera providing a front-facing overview
# of the full workspace.  (Note: arx-x5 data uses "third_view_rgb" at the
# top level instead of "observation/front_camera/rgb" — the _read_camera_frames
# method handles this transparently.)
# ---------------------------------------------------------------------------

MULTIVIEW_LAYOUT = [["head_camera", "front_camera"], ["left_camera", "right_camera"]]
MULTIVIEW_CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]


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


class RoboTwinDataset(BaseActionDataset):
    """RoboTwin 2.0 HDF5 dataset for bimanual robot video-action training.

    Reads episode HDF5 files with JPEG-encoded camera observations and
    either joint-space or end-effector actions. Supports all 5 RoboTwin
    embodiments.

    Action modes:
        ``joint`` — reads ``joint_action/vector`` (14/16D qpos).
            Normalisation: min-max → [-1, 1] for joints, binary {0, 1}
            for grippers (1 = closed, 0 = open).
        ``eef`` — reads ``endpose/`` keys and assembles 20D EEF vector:
            ``[xyz(3) + rot6d(6) + gripper(1)] × 2 arms``.
            No normalisation applied; gripper inverted so 1 = closed.

    Epoch strategy:
        Training enumerates all valid ``(episode, start_frame)`` windows
        exhaustively with configurable stride (``window_stride``), so one
        epoch = one pass through every window.  ``repeat`` multiplies the
        window list for additional passes.
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
        video_stride: int = 4,
        multiview: bool = False,
        robot: Optional[str] = None,
        variant: str = "clean_50",
        backbone: Optional[str] = None,
        action_mode: str = "joint",
    ):
        super().__init__()
        self.robot = robot
        self.variant = variant
        self.action_mode = action_mode

        if action_mode not in ("joint", "eef"):
            raise ValueError(f"action_mode must be 'joint' or 'eef', got '{action_mode}'")

        # Validate resolution against backbone constraints.
        _supported = BACKBONE_SUPPORTED_RESOLUTIONS.get(backbone, None) if backbone else None
        if _supported is not None:
            if (height, width) not in _supported:
                supported_str = ", ".join(f"{h}x{w}" for h, w in sorted(_supported))
                raise ValueError(
                    f"backbone='{backbone}' only supports resolutions: {supported_str}. Got {height}x{width}."
                )
        elif height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                f"Resolution {height}x{width} must be divisible by 32 (VAE downsamples by 16, patch size 2)."
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
        self.video_stride = max(1, video_stride)
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
        print(f"RoboTwinDataset: {len(self._episode_files)} episodes ({split}, action_mode={action_mode})")

        # Probe episodes for length and action dim
        self._episode_lengths = []
        self._action_dim_detected = None

        for path in self._episode_files:
            with h5py.File(path, "r") as f:
                T = f[f"observation/{target_camera}/rgb"].shape[0]
                self._episode_lengths.append(T)

                if self._action_dim_detected is None:
                    if action_mode == "eef":
                        # Validate endpose keys exist
                        if "endpose/left_endpose" not in f:
                            raise KeyError(
                                f"action_mode='eef' requires 'endpose/left_endpose' in HDF5. Not found in {path}"
                            )
                        self._action_dim_detected = EEF_ACTION_DIM
                    else:
                        self._action_dim_detected = f["joint_action/vector"].shape[1]

        print(
            f"  Episode lengths: min={min(self._episode_lengths)}, "
            f"max={max(self._episode_lengths)}, "
            f"action_dim={self._action_dim_detected}"
        )

        # ---- Validate multiview cameras ----
        if self.multiview:
            with h5py.File(self._episode_files[0], "r") as f:
                obs_keys = list(f["observation"].keys()) if "observation" in f else []
                for cam in self.cameras:
                    obs_path = f"observation/{cam}/rgb"
                    # front_camera may live at top-level "third_view_rgb" in arx-x5 data
                    found = obs_path in f or (cam == "front_camera" and "third_view_rgb" in f)
                    if not found:
                        print(
                            f"  WARNING: multiview camera '{cam}' not found in "
                            f"{self._episode_files[0]}. Available obs: {obs_keys}. "
                            f"Will use black frames for missing cameras."
                        )
            print(f"  Multiview mode: layout={self.camera_layout}, quadrant={self.quadrant_h}x{self.quadrant_w}")

        # ---- Exhaustive window enumeration ----
        self._window_index = []  # List of (episode_idx, start_frame)
        for ep_idx, ep_len in enumerate(self._episode_lengths):
            max_start = max(0, ep_len - self.num_frames)
            for start in range(0, max_start + 1, self.window_stride):
                self._window_index.append((ep_idx, start))
        if repeat > 1:
            self._window_index = self._window_index * repeat
        num_video_frames = len(list(range(0, self.num_frames, self.video_stride)))
        print(
            f"  Exhaustive windows: {len(self._window_index)} "
            f"(stride={self.window_stride}, repeat={repeat}, "
            f"video_stride={self.video_stride} → {num_video_frames} video frames, "
            f"{self.num_frames} action steps)"
        )

        # ---- Load scene_info for active arm detection ----
        self._scene_info = {}
        scene_info_path = os.path.join(data_root, "..", "scene_info.json")
        if not os.path.exists(scene_info_path):
            scene_info_path = os.path.join(data_root, "scene_info.json")
        if os.path.exists(scene_info_path):
            with open(scene_info_path) as _f:
                self._scene_info = json.load(_f)
            print(f"  Scene info loaded from {scene_info_path} ({len(self._scene_info)} entries)")
        else:
            print("  No scene_info.json found, active_arm will default to 'both'")

        # ---- Load action stats (joint mode: min-max; eef mode: none) ----
        self._action_dim_value = self._action_dim_detected or 14
        self._action_stats = None
        self._norm_min = None
        self._norm_max = None
        self._norm_range = None

        if action_mode == "eef":
            self._action_dim_value = EEF_ACTION_DIM
            print("  EEF mode: no action normalization applied")
        else:
            stats_path = action_stats_path or os.path.join(data_root, "action_stats.npy")
            if os.path.exists(stats_path):
                stats = np.load(stats_path, allow_pickle=True).item()
                if "min" not in stats or "max" not in stats:
                    raise ValueError(
                        f"Joint mode requires 'min' and 'max' keys in action stats. "
                        f"Found keys: {list(stats.keys())}. Re-run action stats computation."
                    )
                self._norm_min = stats["min"].astype(np.float32)
                self._norm_max = stats["max"].astype(np.float32)
                if self._action_dim_detected is not None and len(self._norm_min) != self._action_dim_detected:
                    raise ValueError(
                        f"Action stats dimension ({len(self._norm_min)}) does not match "
                        f"detected action_dim ({self._action_dim_detected}) from data. "
                        f"Check that {stats_path} was computed for this robot/dataset."
                    )
                self._norm_range = np.maximum(self._norm_max - self._norm_min, 1e-6)
                self._action_stats = {"min": self._norm_min, "max": self._norm_max}
                self._action_dim_value = len(self._norm_min)
                print(f"  Action stats loaded from {stats_path} (min-max normalization)")
                print(f"    Min: {self._norm_min}")
                print(f"    Max: {self._norm_max}")
            else:
                print(f"  WARNING: No action stats found at {stats_path}, joint actions will NOT be normalized")

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
        if self._action_stats is None:
            return None
        if self.action_mode == "eef":
            return None
        # Return min/max plus equivalent mean/std for model buffer compatibility.
        # min-max [-1,1] ↔ z-score with mean=(min+max)/2, std=(max-min)/2.
        stats = dict(self._action_stats)
        equiv_mean = (stats["min"] + stats["max"]) / 2.0
        equiv_std = np.maximum(stats["max"] - stats["min"], 1e-6) / 2.0
        for gi in JOINT_GRIPPER_INDICES:
            if gi < len(equiv_mean):
                equiv_mean[gi] = 0.0
                equiv_std[gi] = 1.0
        stats["mean"] = equiv_mean
        stats["std"] = equiv_std
        return stats

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized actions back to the robot-native format.

        Joint mode: min-max inverse for joint dims; gripper dims are binary
            {0=open, 1=closed} and are re-inverted to raw RoboTwin convention
            (1=open, 0=closed).
        EEF mode: gripper is binary {0=open, 1=closed}, re-inverted to raw.
        """
        if self.action_mode == "eef":
            result = action.copy() if isinstance(action, np.ndarray) else np.array(action)
            for gi in EEF_GRIPPER_INDICES:
                if gi < result.shape[-1]:
                    # Binary 1=closed → raw 1=open
                    result[..., gi] = 1.0 - (result[..., gi] > 0.5).astype(np.float32)
            return result
        # Joint mode
        if self._action_stats is None:
            result = action.copy() if isinstance(action, np.ndarray) else np.array(action)
        else:
            stats = self._action_stats
            result = 0.5 * (action + 1.0) * (stats["max"] - stats["min"]) + stats["min"]
        # Gripper: binary 1=closed → raw 1=open
        for gi in JOINT_GRIPPER_INDICES:
            if gi < result.shape[-1]:
                result[..., gi] = 1.0 - (action[..., gi] > 0.5).astype(np.float32)
        return result

    def __len__(self):
        if self._val_samples is not None:
            return len(self._val_samples)
        return len(self._window_index)

    def _decode_jpeg(self, jpeg_bytes) -> Image.Image:
        """Decode JPEG bytes from HDF5 to a PIL RGB image.

        RoboTwin encodes frames by passing RGB arrays directly to
        ``cv2.imencode`` (which expects BGR), so R and B channels are
        swapped inside the JPEG. Using ``cv2.imdecode`` reverses this
        swap, giving back the original RGB order — no further
        conversion needed.
        """
        arr = cv2.imdecode(np.frombuffer(bytes(jpeg_bytes), np.uint8), cv2.IMREAD_COLOR)
        return Image.fromarray(arr)

    def _read_camera_frames(self, f, camera_key: str, start: int, end: int):
        """Read and decode JPEG frames from an observation camera.

        Handles two HDF5 layouts transparently:
        - ``observation/{camera_key}/rgb`` — standard per-camera path
          (aloha-agilex, franka, ur5, tiangong)
        - ``third_view_rgb`` — top-level key used by arx-x5 for the
          external fixed camera (equivalent to front_camera)

        When *camera_key* is ``"front_camera"``, the method first tries
        ``observation/front_camera/rgb`` and falls back to the top-level
        ``third_view_rgb`` key for backward compatibility with arx-x5 data.

        Args:
            f: Open HDF5 file handle.
            camera_key: Camera name (e.g. ``"head_camera"``, ``"front_camera"``).
            start: Start frame index (inclusive).
            end: End frame index (exclusive).

        Returns:
            List of PIL.Image.Image in RGB.
        """
        obs_path = f"observation/{camera_key}/rgb"
        if obs_path in f:
            raw = f[obs_path][start:end]
        elif camera_key == "front_camera" and "third_view_rgb" in f:
            # arx-x5 stores the front/third-person camera at top level
            raw = f["third_view_rgb"][start:end]
        else:
            raise KeyError(f"Camera '{camera_key}' not found. Tried '{obs_path}' and top-level 'third_view_rgb'.")
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
                per_camera[cam] = [Image.new("RGB", (self.quadrant_w, self.quadrant_h), (0, 0, 0)) for _ in range(n)]

        # Assemble per-timestep grids
        n = end - start
        grids = []
        for t in range(n):
            frames_t = {cam: per_camera[cam][t] for cam in cameras}
            grid = assemble_multiview_grid(frames_t, self.camera_layout, self.quadrant_h, self.quadrant_w)
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
            # Ensure base_prompt ends with punctuation before appending view description
            if base_prompt and base_prompt[-1] not in ".!?":
                base_prompt += "."
            return (
                f"A multi-view video shows that {base_prompt} "
                f"The video is split into four views: "
                f"head camera (top-left), front camera (top-right), "
                f"left camera (bottom-left), right camera (bottom-right)."
            )

        return base_prompt

    def _read_eef_actions(self, f, start: int, end: int) -> np.ndarray:
        """Read endpose keys and assemble 20D EEF action vector.

        Layout: [left_xyz(3), left_rot6d(6), left_grip(1),
                 right_xyz(3), right_rot6d(6), right_grip(1)]
        Gripper convention: 1 = closed, 0 = open (inverted from raw HDF5).
        """
        left_ep = f["endpose/left_endpose"][start:end]  # (T, 7): xyz + quat_xyzw
        right_ep = f["endpose/right_endpose"][start:end]
        # Binarize + invert: raw 1=open → (raw > 0.5) gives True=open
        # → invert to 1=closed, 0=open (matching X-VLA / starVLA convention)
        left_grip = 1.0 - (f["endpose/left_gripper"][start:end] > 0.5).astype(np.float64)
        right_grip = 1.0 - (f["endpose/right_gripper"][start:end] > 0.5).astype(np.float64)

        left = np.concatenate(
            [
                left_ep[:, :3],
                quat_xyzw_to_rotation_6d(left_ep[:, 3:]),
                left_grip[:, None] if left_grip.ndim == 1 else left_grip,
            ],
            axis=-1,
        )  # (T, 10)

        right = np.concatenate(
            [
                right_ep[:, :3],
                quat_xyzw_to_rotation_6d(right_ep[:, 3:]),
                right_grip[:, None] if right_grip.ndim == 1 else right_grip,
            ],
            axis=-1,
        )  # (T, 10)

        return np.concatenate([left, right], axis=-1).astype(np.float32)  # (T, 20)

    def _normalize_joint_actions(self, actions: np.ndarray) -> np.ndarray:
        """Min-max normalize joint dims to [-1,1], preserve binary gripper dims.

        Gripper channels are already binarized (1=closed, 0=open) before this
        method is called, so they are copied through unchanged.
        """
        normalized = 2.0 * (actions - self._norm_min) / self._norm_range - 1.0
        # Restore pre-binarized gripper values (skip min-max for gripper dims)
        for gi in JOINT_GRIPPER_INDICES:
            if gi < actions.shape[1]:
                normalized[:, gi] = actions[:, gi]
        return normalized

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
                target_frames = self._read_multiview_frames(f, self.cameras, start, actual_end)
            else:
                target_frames = self._read_camera_frames(f, self.target_camera, start, actual_end)
            if self.action_mode == "eef":
                actions = self._read_eef_actions(f, start, actual_end)
            else:
                actions = f["joint_action/vector"][start:actual_end].astype(np.float32)

        actual_len = len(target_frames)

        # ---- Pad target + actions if shorter than num_frames ----
        valid_len = actual_len
        if actual_len < self.num_frames:
            pad_len = self.num_frames - actual_len
            target_frames = target_frames + [target_frames[-1]] * pad_len
            actions = np.concatenate(
                [
                    actions,
                    np.repeat(actions[-1:], pad_len, axis=0),
                ],
                axis=0,
            )

        # Action mask: True for real frames, False for padded frames
        action_mask = torch.ones(self.num_frames, dtype=torch.bool)
        action_mask[valid_len:] = False

        # ---- Normalize actions ----
        if self.action_mode == "joint":
            # Binarize gripper channels (raw: 1=open, 0=closed → 1=closed, 0=open)
            for gi in JOINT_GRIPPER_INDICES:
                if gi < actions.shape[1]:
                    actions[:, gi] = 1.0 - (actions[:, gi] > 0.5).astype(np.float32)
            # Min-max normalize joint dims if stats are available
            if self._norm_min is not None:
                actions = self._normalize_joint_actions(actions)

        # ---- Crop and resize frames to target resolution ----
        if self.multiview:
            video = target_frames
        else:
            video = [_crop_and_resize(frame, self.height, self.width) for frame in target_frames]

        # ---- Subsample video frames (actions stay at full resolution) ----
        if self.video_stride > 1:
            video_indices = list(range(0, len(video), self.video_stride))
            video = [video[i] for i in video_indices]

        vace_reference_image = [video[0]]

        prompt = self._get_prompt(ep_idx)

        # ---- Extract active arm from scene_info ----
        ep_key = f"episode_{ep_idx}"
        ep_scene = self._scene_info.get(ep_key, {})
        ep_info = ep_scene.get("info", ep_scene)
        active_arm = ep_info.get("{a}", ep_info.get("active_arm", "both"))

        action_tensor = torch.from_numpy(actions)
        result = {
            "video": video,
            "vace_video": None,
            "vace_reference_image": vace_reference_image,
            "action_trajectory": action_tensor,
            "action": action_tensor,  # BaseActionDataset compat
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


class MultiTaskRoboTwinDataset(BaseActionDataset):
    """Multi-task wrapper over multiple RoboTwinDatasets.

    Concatenates per-task RoboTwinDatasets so one epoch covers all tasks.
    Action stats are shared across tasks (loaded from a single file).

    Uses ``discover_robotwin_roots()`` to auto-discover per-task data
    directories and creates one ``RoboTwinDataset`` per task.

    Args:
        dataset_dir: Top-level RoboTwin dataset directory.
        robot: Target robot name (e.g. ``"aloha-agilex"``).
        variant: ``"clean_50"``, ``"randomized_500"``, or ``"both"``
            (merges clean_50 + randomized_500 into a single dataset).
        tasks: List of task names.  Defaults to ``ROBOTWIN_TRAIN_TASKS``.
        action_stats_path: Path to shared action stats (.npy).
        action_mode: ``"joint"`` (14D) or ``"eef"`` (20D).
        **kwargs: Forwarded to each ``RoboTwinDataset`` (num_frames, height,
            width, split, val_ratio, repeat, seed, target_camera,
            window_stride, num_val_samples, backbone, ...).
    """

    _BOTH_VARIANTS = ["clean_50", "randomized_500"]

    @classmethod
    def from_config(cls, config, split: str = "train"):
        """Build from Hydra DictConfig or dict, handling task resolution.

        Resolves ``task_name`` / ``train_tasks`` / ``holdout_tasks`` from config
        into a concrete task list, then constructs the dataset.  This is the
        preferred entry point when building via the dataset registry.
        """

        def _get(key, default=None):
            if hasattr(config, key):
                val = getattr(config, key)
                return default if val is None else val
            if hasattr(config, "get"):
                val = config.get(key, default)
                return default if val is None else val
            return default

        # ---- Task resolution ----
        task_name = _get("task_name", None)
        if task_name:
            tasks = [task_name]
        else:
            train_tasks = _get("train_tasks", None)
            holdout_tasks = _get("holdout_tasks", None)
            if train_tasks:
                tasks = list(train_tasks)
            elif holdout_tasks:
                tasks = sorted(t for t in ROBOTWIN_ALL_TASKS if t not in holdout_tasks)
            else:
                tasks = ROBOTWIN_TRAIN_TASKS

        return cls(
            dataset_dir=_get("dataset_dir"),
            robot=_get("robot", "aloha-agilex"),
            variant=_get("variant", "both"),
            tasks=tasks,
            action_stats_path=_get("action_stats_path", None),
            action_mode=_get("action_mode", "eef"),
            num_frames=int(_get("num_frames", 33)),
            height=int(_get("height", 480)),
            width=int(_get("width", 640)),
            split=split,
            val_ratio=float(_get("val_ratio", 0.0)),
            repeat=int(_get("repeat", 1)),
            target_camera=_get("target_camera", "head_camera"),
            window_stride=int(_get("window_stride", 1)),
            video_stride=int(_get("video_stride", 4)),
            multiview=bool(_get("multiview", True)),
            backbone=_get("backbone", None),
        )

    def __init__(
        self,
        dataset_dir: str,
        robot: str,
        variant: str = "clean_50",
        tasks: Optional[list] = None,
        action_stats_path: Optional[str] = None,
        action_mode: str = "joint",
        **kwargs,
    ):
        super().__init__()
        self.action_mode = action_mode

        # ---- Resolve variant(s) ----
        if variant == "both":
            variant_list = self._BOTH_VARIANTS
        else:
            variant_list = [variant]

        # ---- Discover per-task roots ----
        all_roots = []  # list of (display_name, data_root, variant_name)
        for v in variant_list:
            roots = discover_robotwin_roots(dataset_dir, robot, v, tasks)
            for task_name, data_root in roots:
                display = f"{task_name}/{v}" if len(variant_list) > 1 else task_name
                all_roots.append((display, data_root, v))

        if not all_roots:
            task_list = tasks or ROBOTWIN_TRAIN_TASKS
            raise FileNotFoundError(
                f"No task data found in {dataset_dir} for robot={robot}, "
                f"variant={variant}. Checked {len(task_list)} tasks."
            )

        print(
            f"MultiTaskRoboTwinDataset: {len(all_roots)} task-variant pairs, "
            f"robot={robot}, variant={variant}, action_mode={action_mode}"
        )

        self._sub_datasets = []
        self._cumulative_lengths = []
        cumulative = 0

        try:
            from tqdm import tqdm

            task_iter = tqdm(all_roots, desc="Loading tasks", unit="task")
            _use_tqdm = True
        except ImportError:
            task_iter = all_roots
            _use_tqdm = False

        import os as _os
        import sys

        for display_name, data_root, v in task_iter:
            if _use_tqdm:
                task_iter.set_postfix_str(display_name)
                # Suppress per-task prints to keep progress bar clean
                _old_stdout = sys.stdout
                sys.stdout = open(_os.devnull, "w")
            try:
                ds = RoboTwinDataset(
                    data_root=data_root,
                    task_name=display_name.split("/")[0].replace("_", " "),
                    action_stats_path=action_stats_path,
                    robot=robot,
                    variant=v,
                    action_mode=action_mode,
                    **kwargs,
                )
            finally:
                if _use_tqdm:
                    sys.stdout.close()
                    sys.stdout = _old_stdout
            self._sub_datasets.append(ds)
            cumulative += len(ds)
            self._cumulative_lengths.append(cumulative)

        self._total_length = cumulative
        self._action_stats_shared = self._sub_datasets[0].action_stats if self._sub_datasets else None
        self._action_dim_value = self._sub_datasets[0].action_dim if self._sub_datasets else 14

        print(f"  Total samples: {self._total_length} (across {len(self._sub_datasets)} sub-datasets)")

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def action_stats(self) -> dict:
        return self._action_stats_shared

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if not self._sub_datasets:
            return action
        return self._sub_datasets[0].denormalize_action(action)

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
