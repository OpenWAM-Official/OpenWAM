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
import time
from typing import Any, Optional

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import (
    DEFAULT_MULTIVIEW_CAMERA_LAYOUT,
    assemble_multiview_layout,
    crop_and_resize,
    format_prompt_for_inference,
)
from openwam.dataloader.transforms.normalize import (
    YAML_TO_NORM_MODE,
    Normalizer,
    load_mode_stats,
)
from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

_JOINT_ACTION_DIM = 14  # aloha-agilex qpos vector


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

# Known RoboTwin 2.0 task catalog. Dataset loading discovers the tasks that are
# actually present on disk instead of using this list as a train/val selector.
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

# ---------------------------------------------------------------------------
# Action mode constants
# ---------------------------------------------------------------------------

EEF_ACTION_DIM = 20  # [xyz(3) + rot6d(6) + gripper(1)] × 2 arms
EEF_GRIPPER_INDICES = [9, 19]  # gripper positions in 20D EEF vector
JOINT_GRIPPER_INDICES = [6, 13]  # gripper positions in 14D joint vector


def discover_robotwin_roots(
    dataset_dir: str,
    embodiment: str,
    variant: str = "clean_50",
    tasks: Optional[list] = None,
) -> list:
    """Auto-discover per-task data roots from the RoboTwin dataset directory.

    Args:
        dataset_dir: Top-level directory (e.g. ``/path/to/robotwin_2_0/dataset``).
        embodiment: Robot embodiment name (e.g. ``"aloha-agilex"``).
        variant: ``"clean_50"`` or ``"randomized_500"``.
        tasks: Optional internal task restriction.  When omitted, every task
            directory containing the requested embodiment/variant is discovered.

    Returns:
        List of ``(task_name, data_root)`` tuples for tasks that exist on disk.
    """
    if tasks is None:
        if not os.path.isdir(dataset_dir):
            return []
        tasks = sorted(
            entry
            for entry in os.listdir(dataset_dir)
            if os.path.isdir(
                os.path.join(
                    dataset_dir,
                    entry,
                    f"{embodiment}_{variant}",
                    "data",
                )
            )
        )

    roots = []
    for task in tasks:
        data_root = os.path.join(dataset_dir, task, f"{embodiment}_{variant}", "data")
        if os.path.isdir(data_root):
            roots.append((task, data_root))
    return roots


# ---------------------------------------------------------------------------
# 3-camera L-shape multi-view layout
#
#   +---------------------------+
#   |       camera_layout[0]    |   top, full width, ~2/3 height
#   +-------------+-------------+
#   |   cam[1]    |   cam[2]    |   bottom halves
#   +-------------+-------------+
# ---------------------------------------------------------------------------


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


def _resolve_prompt(
    instructions: dict,
    ep_file: str,
    split: str,
    task_name: str,
) -> str:
    """Pure function: build the training-time prompt from raw state.

    Extracted from :meth:`RoboTwinDataset._get_prompt` so tests and
    downstream adapters can reproduce training-time prompts without having
    to instantiate a full dataset (no ``__new__`` + private-attr injection).

    Must stay byte-compatible with what training sees — deployment uses the
    same ``format_prompt_for_inference`` helper.

    Args:
        instructions: Mapping from ``episode<N>.json`` to a RoboTwin
            instruction dict (``{"seen": [...], "unseen": [...]}``) or a
            plain string / list.
        ep_file: Episode file name or full path (``episode<N>.hdf5``).
        split: ``"train"`` samples a random instruction from ``seen``;
            anything else deterministically picks the first entry.
        task_name: Falls back to ``f"... performing a {task_name} task."``
            when instructions don't supply a prompt for this episode.

    Returns:
        The final wrapped prompt string that the model sees.
    """
    ep_basename = os.path.basename(ep_file)
    ep_num = ep_basename.replace("episode", "").replace(".hdf5", "")
    instr_key = f"episode{ep_num}.json"

    base_prompt = None
    if instr_key in instructions:
        instr = instructions[instr_key]
        if isinstance(instr, dict):
            # RoboTwin format: {"seen": [...], "unseen": [...]}
            pool = instr.get("seen") or instr.get("unseen") or []
            if pool:
                if split == "train":
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
        base_prompt = f"The bimanual robot is performing a {task_name} task."

    return format_prompt_for_inference(base_prompt)


class RoboTwinDataset(BaseDataset):
    """RoboTwin 2.0 HDF5 dataset for bimanual robot video-action training.

    Reads episode HDF5 files with JPEG-encoded camera observations and
    either joint-space or end-effector actions. Supports all 5 RoboTwin
    embodiments.

    Action modes:
        ``joint`` — reads ``joint_action/vector`` (14/16D qpos).
            Normalisation: min-max → [-1, 1] for all dims including gripper.
        ``eef`` — reads ``endpose/`` keys and assembles 20D EEF vector:
            ``[xyz(3) + rot6d(6) + gripper(1)] × 2 arms``.
            No normalisation applied; gripper uses raw continuous values.

    Epoch strategy:
        Training enumerates all valid ``(episode, start_frame)`` windows
        exhaustively with configurable stride (``window_stride``), so one
        epoch = one pass through every window.
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 33,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        task_name: Optional[str] = None,
        normalization_stats_path: Optional[str] = None,
        normalize_mode: Optional[str] = "min-max",
        target_camera: str = "head_camera",
        window_stride: int = 1,
        video_stride: int = 4,
        multiview: bool = False,
        camera_layout: Optional[list] = None,
        embodiment: Optional[str] = None,
        variant: str = "clean_50",
        backbone: Optional[str] = None,
        action_mode: str = "joint",
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        unify_state_map: Optional[Any] = None,
        # Optional load-time video color jitter, applied consistently across a
        # clip's frames and ONLY on the train split. None / False / {} → disabled
        # (default). Truthy → enabled; a dict overrides
        # the per-channel strengths {brightness, contrast, saturation, hue}.
        color_jitter: Optional[Any] = None,
    ):
        super().__init__()
        self.embodiment = embodiment
        self.variant = variant
        self.action_mode = action_mode
        self._unify_action = bool(unify_action)
        self._unify_action_map = unify_action_map
        if unify_state_map is not None and list(unify_state_map) != list(self._unify_action_map or ()):
            raise ValueError(
                "unify_state_map must be null or equal to unify_action_map here: "
                "this reader's state shares the action's raw layout"
            )
        self.normalize_mode = normalize_mode if normalize_mode not in ("", "none", "null") else None

        # ── load-time video augmentation ──────────────────────────────────
        # Color jitter is applied in __getitem__ to the decoded clip (same
        # random factors across all frames, via VideoColorJitter). Built only
        # for the train split; val / disabled keeps video byte-identical.
        self._color_jitter = None
        if color_jitter_enabled(color_jitter) and split == "train":
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.2)),
                contrast=float(cj_get("contrast", 0.2)),
                saturation=float(cj_get("saturation", 0.2)),
                hue=float(cj_get("hue", 0.0)),
            )

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
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        self.num_frames = int(num_frames)
        self.num_action_steps = self.num_frames - 1
        self.height = height
        self.width = width
        self.split = split
        self.task_name = task_name or "manipulation"
        self.target_camera = target_camera
        self.window_stride = max(1, window_stride)
        self.video_stride = max(1, video_stride)
        # video_stride sub-samples frames within each window:
        # range(0, num_frames, video_stride). num_video_frames is whatever that
        # yields. For clean encoder temporal downsampling it should match the
        # encoder's contract (Wan VAE causal: (num_video_frames - 1) % 4 == 0;
        # non-causal: num_video_frames % tc == 0) — NOT enforced here; a mismatch
        # surfaces downstream at encode time.
        self._raw_window_len = self.num_frames
        self._video_sample_indices = list(range(0, self.num_frames, self.video_stride))
        self.num_video_frames = len(self._video_sample_indices)
        self.multiview = bool(multiview)
        if self.multiview:
            if camera_layout is None:
                camera_layout = list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT)
            if len(camera_layout) != 3:
                raise ValueError(
                    f"multiview requires exactly 3 cameras [top, bot-left, bot-right], "
                    f"got {len(camera_layout)}: {camera_layout}"
                )
            self.cameras = list(camera_layout)
            self.camera_layout = list(camera_layout)
        else:
            self.cameras = [target_camera]
            self.camera_layout = None

        # ---- Discover and sort target episode files ----
        pattern = os.path.join(data_root, "episode*.hdf5")
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f"No episode*.hdf5 files found in {data_root}")

        self._episode_files = list(all_files)
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

        # ---- Probe original observation image size (first camera, first episode) ----
        self._obs_image_size = None  # (H, W) of the raw JPEG frames in HDF5
        with h5py.File(self._episode_files[0], "r") as f:
            probe_cam = self.cameras[0]
            obs_path = f"observation/{probe_cam}/rgb"
            if obs_path in f and f[obs_path].shape[0] > 0:
                probe_jpeg = f[obs_path][0]
                probe_img = self._decode_jpeg(probe_jpeg)
                self._obs_image_size = (probe_img.height, probe_img.width)
                print(
                    f"  Raw observation image size: {self._obs_image_size[0]}x{self._obs_image_size[1]} "
                    f"(probed from camera '{probe_cam}')"
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
            print(f"  Multiview mode: 3-cam L-shape, cameras={self.cameras}, output={self.height}x{self.width}")

        # ---- Exhaustive window enumeration ----
        self._window_index = []  # List of (episode_idx, start_frame)
        for ep_idx, ep_len in enumerate(self._episode_lengths):
            if ep_len < 2:
                continue
            if self.split == "val":
                # Keep validation loss comparable to the historical full-window
                # distribution.
                max_start = max(0, ep_len - self._raw_window_len)
            else:
                # FastWAM-aligned tail semantics, constrained to starts with at
                # least one valid future action label under OpenWAM's t+1 action
                # alignment. Tail windows are padded and masked out in the loss.
                max_start = max(0, ep_len - 2)
            for start in range(0, max_start + 1, self.window_stride):
                self._window_index.append((ep_idx, start))
        if not self._window_index:
            raise ValueError(
                "No valid RoboTwin windows with at least one action label were selected "
                f"for split='{split}'. Check episode lengths."
            )
        print(
            f"  Exhaustive windows: {len(self._window_index)} "
            f"(window_stride={self.window_stride}, "
            f"raw_window_len={self._raw_window_len}, video_stride={self.video_stride}, "
            f"→ {self.num_video_frames} video frames, "
            f"{self.num_action_steps} action steps + 1 proprio)"
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

        # ---- Action normalization (unified for joint & eef via Normalizer) ----
        # _raw_action_dim_value = the width this reader's HDF5 path produces and
        # the Normalizer operates on (14 joint / 20 eef). When unify is on, the
        # PUBLIC _action_dim_value (model action head + finalized payloads)
        # becomes UNIFY_DIM, with raw dims scattered via _unify_dst_index.
        self._raw_action_dim_value = (
            EEF_ACTION_DIM if self.action_mode == "eef" else (self._action_dim_detected or _JOINT_ACTION_DIM)
        )
        self._unify_dst_index: Optional[np.ndarray] = None
        if self._unify_action:
            if self._unify_action_map is None:
                # No spec → identity map: raw dims 0..raw-1 in order (flat
                # single-list form, NOT [[...]] which parses as one src->dst pair).
                spec = list(range(self._raw_action_dim_value))
            else:
                spec = self._unify_action_map
            self._unify_dst_index = parse_unify_spec(spec, UNIFY_DIM)
            if self._unify_dst_index.shape[0] != self._raw_action_dim_value:
                raise ValueError(
                    f"RoboTwin unify_action_map maps {self._unify_dst_index.shape[0]} source dims "
                    f"but action_mode={self.action_mode!r} produces {self._raw_action_dim_value}-D. "
                    f"They must match."
                )
            self._action_dim_value = UNIFY_DIM
        else:
            self._action_dim_value = self._raw_action_dim_value
        self._normalizer = None  # Normalizer or None if disabled / stats missing
        self._mode_stats: Optional[dict] = None  # raw stats dict for the active mode
        self.normalization_stats_path: Optional[str] = None  # resolved path to the stats .npy file

        if self.normalize_mode is not None:
            if self.normalize_mode not in YAML_TO_NORM_MODE:
                raise ValueError(
                    f"normalize_mode must be one of {list(YAML_TO_NORM_MODE)} or null, got '{self.normalize_mode}'"
                )
            print(
                f"  [normalizer] Loading action normalizer "
                f"(normalize_mode={self.normalize_mode}, action_mode={self.action_mode})"
            )
            # Resolve stats path: explicit > default under data_root
            if normalization_stats_path is not None:
                stats_path = normalization_stats_path
                if os.path.exists(stats_path):
                    print(f"  [normalizer] Using explicit stats file: {stats_path} (exists ✓)")
                else:
                    print(
                        f"  [normalizer] WARNING: explicit normalization_stats_path does not exist: {stats_path}\n"
                        f"  [normalizer]          '{self.normalize_mode}' normalization DISABLED."
                    )
                    stats_path = None
            else:
                # Stats resolution is owned by the multi-task reader, which
                # auto-builds <dataset_dir>/meta/robotwin_<variant>_normalization_stats.npy
                # (rank 0 computes, other ranks wait) and forwards the path to
                # every sub-dataset. A directly-constructed single-task reader
                # must receive normalization_stats_path explicitly.
                stats_path = None
                print(
                    "  [normalizer] No explicit normalization_stats_path for a single-task reader; "
                    "normalization DISABLED. Construct via the multi-task reader for auto-built "
                    "variant-level stats, or pass normalization_stats_path."
                )

            if stats_path is not None:
                mode_stats = load_mode_stats(stats_path, self.action_mode)
                if mode_stats is None:
                    print(
                        f"  [normalizer] WARNING: stats file {stats_path} has no '{self.action_mode}' entry; "
                        f"normalization DISABLED."
                    )
                else:
                    # Stats live in RAW action space (the Normalizer runs before
                    # the unify scatter), so validate against _raw_action_dim_value
                    # — NOT _action_dim_value, which is UNIFY_DIM when unify is on.
                    expected_dim = self._raw_action_dim_value
                    got_dim = len(mode_stats["mean"])
                    if got_dim != expected_dim:
                        raise ValueError(
                            f"Stats dim mismatch for action_mode='{self.action_mode}': "
                            f"expected {expected_dim}, got {got_dim} from {stats_path}."
                        )
                    self._mode_stats = mode_stats
                    self._normalizer = Normalizer(
                        mode=YAML_TO_NORM_MODE[self.normalize_mode],
                        stats=mode_stats,
                    )
                    self.normalization_stats_path = stats_path
                    print(
                        f"  [normalizer] Active: mode={self.normalize_mode}, "
                        f"action_mode={self.action_mode}, dim={got_dim}, stats={stats_path}"
                    )
        else:
            print("  [normalizer] Action normalization DISABLED (normalize_mode=None)")

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

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self) -> Optional[dict]:
        """Return the raw stats dict for the active action_mode (or None)."""
        return dict(self._mode_stats) if self._mode_stats is not None else None

    def denormalize_action(self, action) -> np.ndarray:
        """Invert the train-time action transform for inference / deployment.

        Mirrors the forward path in reverse: the model emits unified-space
        actions, so we **un-unify first** (gather the UNIFY_DIM vector back to
        the raw 14/20-D layout) and **then unnormalize** — the exact inverse of
        ``normalize -> map_to_unify`` in ``_build_sample``.

        If no normalizer is active (no stats / normalize_mode=None), only the
        un-unify step applies (and is a no-op when unify is off).
        """
        arr = np.asarray(action) if not isinstance(action, np.ndarray) else action
        if self._unify_dst_index is not None:
            arr = unmap_from_unify(arr, self._unify_dst_index)  # (..., UNIFY_DIM) -> (..., raw)
        if self._normalizer is None:
            return np.asarray(arr).copy()
        return self._normalizer.unnormalize(arr)

    def __len__(self):
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
        """Read 3 cameras and assemble into L-shape composition at (height, width).

        Missing cameras are black-padded.
        """
        n = end - start
        per_camera = {}
        for cam in cameras:
            try:
                per_camera[cam] = self._read_camera_frames(f, cam, start, end)
            except KeyError:
                ph, pw = self._obs_image_size or (self.height, self.width)
                per_camera[cam] = [Image.new("RGB", (pw, ph), (0, 0, 0)) for _ in range(n)]

        images = []
        for t in range(n):
            frames_t = {cam: per_camera[cam][t] for cam in cameras}
            images.append(
                assemble_multiview_layout(
                    frames_t,
                    self.camera_layout,
                    self.height,
                    self.width,
                )
            )
        return images

    def _get_prompt(self, ep_idx: int) -> str:
        """Get the wrapped text prompt for episode *ep_idx*.

        Thin wrapper over :func:`_resolve_prompt` — all logic lives in the
        pure helper so tests and downstream adapters don't need a fully
        constructed ``RoboTwinDataset`` to reproduce training-time prompts.
        """
        return _resolve_prompt(
            instructions=self._instructions,
            ep_file=self._episode_files[ep_idx],
            split=self.split,
            task_name=self.task_name,
        )

    def _read_eef_actions(self, f, start: int, end: int) -> np.ndarray:
        """Read endpose keys and assemble 20D EEF action vector.

        Layout: [left_xyz(3), left_rot6d(6), left_grip(1),
                 right_xyz(3), right_rot6d(6), right_grip(1)]
        Gripper values are raw continuous values from HDF5 (1=open, 0=closed).
        """
        left_ep = f["endpose/left_endpose"][start:end]  # (T, 7): xyz + quat_xyzw
        right_ep = f["endpose/right_endpose"][start:end]
        left_grip = f["endpose/left_gripper"][start:end].astype(np.float64)
        right_grip = f["endpose/right_gripper"][start:end].astype(np.float64)

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

    def _read_raw_actions(self, f, start: int, end: int) -> np.ndarray:
        """Read raw action array (``joint`` or ``eef``) in the given [start, end) range."""
        if self.action_mode == "eef":
            return self._read_eef_actions(f, start, end)
        return f["joint_action/vector"][start:end].astype(np.float32)

    def _build_sample(self, ep_idx: int, start: int) -> dict:
        """Assemble a single sample at (ep_idx, start).

        - ``num_frames``: raw HDF5 window length (state/action rate).
        - Video is subsampled by ``video_stride`` to ``num_video_frames``.
        - State/action stay at raw rate.
        - ``proprio = raw_actions[0:1]`` (time dim kept).
        - ``action = raw_actions[1:num_frames]`` (length ``num_frames-1``).
        """
        path = self._episode_files[ep_idx]
        ep_len = self._episode_lengths[ep_idx]

        raw_end = start + self._raw_window_len
        actual_raw_end = min(raw_end, ep_len)
        actual_raw_len = max(0, actual_raw_end - start)

        if actual_raw_len <= 0:
            raise IndexError(f"Window [{start}, {raw_end}) has no frames (ep_len={ep_len}).")
        if actual_raw_len < 2:
            raise IndexError(
                f"Window [{start}, {raw_end}) has no valid action label "
                f"(actual_raw_len={actual_raw_len}, ep_len={ep_len})."
            )

        with h5py.File(path, "r") as f:
            if self.multiview:
                raw_frames = self._read_multiview_frames(f, self.cameras, start, actual_raw_end)
            else:
                raw_frames = self._read_camera_frames(f, self.target_camera, start, actual_raw_end)
            raw_actions = self._read_raw_actions(f, start, actual_raw_end)

        # Pad to full window if the episode ended early
        if actual_raw_len < self._raw_window_len:
            pad_len = self._raw_window_len - actual_raw_len
            raw_frames = raw_frames + [raw_frames[-1]] * pad_len
            raw_actions = np.concatenate(
                [raw_actions, np.repeat(raw_actions[-1:], pad_len, axis=0)],
                axis=0,
            )

        # Normalize before splitting so proprio and action share the same space
        if self._normalizer is not None:
            raw_actions = self._normalizer.normalize(raw_actions)

        # Unify: scatter normalized (T, raw) -> (T, UNIFY_DIM) AFTER normalization
        # (so unify operates in normalized space; the inverse un-unify in
        # denormalize_action runs BEFORE unnormalize). _unify_dim_mask marks the
        # mapped slots; unmapped slots stay 0 and are masked out below.
        unify_dim_mask = None
        if self._unify_dst_index is not None:
            raw_actions, unify_dim_mask = map_to_unify(raw_actions.astype(np.float32), self._unify_dst_index, UNIFY_DIM)

        # Video: subsampled. State/action: raw rate.
        sampled_video = [raw_frames[i] for i in self._video_sample_indices]
        if not self.multiview:
            sampled_video = [crop_and_resize(frame, self.height, self.width) for frame in sampled_video]

        proprio_np = raw_actions[0:1].astype(np.float32)
        action_np = raw_actions[1 : self.num_frames].astype(np.float32)

        video_mask = torch.tensor(
            [idx < actual_raw_len for idx in self._video_sample_indices],
            dtype=torch.bool,
        )
        # 2-D action_mask (num_action_steps, action_dim): time × dim validity.
        # Without unify, RoboTwin is bimanual / single-tensor — every valid
        # timestep has all dims real, so dim validity is all-True. With unify,
        # dim validity = the scattered slots (unify_dim_mask); unmapped slots are
        # masked out so they never enter the loss.
        time_validity = torch.tensor(
            [(t + 1) < actual_raw_len for t in range(self.num_action_steps)],
            dtype=torch.bool,
        )
        action_mask = time_validity.unsqueeze(-1).expand(-1, self._action_dim_value).contiguous()
        # 2-D proprio_mask (1, action_dim): all True when proprio is present.
        proprio_mask = torch.full(
            (1, self._action_dim_value),
            fill_value=bool(0 < actual_raw_len),
            dtype=torch.bool,
        )
        if unify_dim_mask is not None:
            dim_valid = torch.from_numpy(unify_dim_mask)  # (UNIFY_DIM,) bool
            action_mask = action_mask & dim_valid.unsqueeze(0)
            proprio_mask = proprio_mask & dim_valid.unsqueeze(0)

        prompt = self._get_prompt(ep_idx)

        ep_key = f"episode_{ep_idx}"
        ep_scene = self._scene_info.get(ep_key, {})
        ep_info = ep_scene.get("info", ep_scene)
        active_arm = ep_info.get("active_arm", "both")

        action_tensor = torch.from_numpy(action_np)
        proprio_tensor = torch.from_numpy(proprio_np)

        return {
            "video": sampled_video,
            "vace_video": None,
            "first_frame_image": [sampled_video[0]],
            "action": action_tensor,
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": proprio_tensor,
            "proprio_mask": proprio_mask,
            "prompt": prompt,
            "episode_index": ep_idx,
            "episode_path": path,
            "start_frame": start,
            "end_frame": min(raw_end, ep_len),
            "episode_length": ep_len,
            "task_name": self.task_name,
            "active_arm": active_arm,
        }

    def __getitem__(self, idx):
        ep_idx, start = self._window_index[idx]

        sample = self._build_sample(ep_idx, start)
        if self._color_jitter is not None:
            # Same jitter factors across the whole clip (temporal consistency).
            sample["video"] = self._color_jitter.apply({"video": sample["video"]})["video"]
            # Keep the first-frame conditioning image in sync with the jittered clip.
            sample["first_frame_image"] = [sample["video"][0]]
        return sample


class MultiTaskRoboTwinDataset(BaseDataset):
    """Multi-task wrapper over multiple RoboTwinDatasets.

    Concatenates per-task RoboTwinDatasets so one epoch covers all tasks.
    Action stats are shared across tasks (loaded from a single file).

    Uses ``discover_robotwin_roots()`` to auto-discover per-task data
    directories and creates one ``RoboTwinDataset`` per task.

    Args:
        dataset_dir: Top-level RoboTwin dataset directory.
        embodiment: Robot embodiment name (e.g. ``"aloha-agilex"``).
        variant: ``"clean_50"``, ``"randomized_500"``, or ``"both"``
            (merges clean_50 + randomized_500 into a single dataset).
        tasks: Optional internal task restriction. Defaults to every task
            discovered on disk.
        normalization_stats_path: Path to shared action stats (.npy).
        action_mode: ``"joint"`` (14D) or ``"eef"`` (20D).
        **kwargs: Forwarded to each ``RoboTwinDataset`` (num_frames, height,
            width, split, target_camera, window_stride, backbone, ...).
    """

    _BOTH_VARIANTS = ["clean_50", "randomized_500"]

    @classmethod
    def from_config(cls, config, split: str = "train"):
        """Build from Hydra DictConfig or dict and discover all tasks on disk."""

        def _get(key, default=None):
            if hasattr(config, key):
                val = getattr(config, key)
                return default if val is None else val
            if hasattr(config, "get"):
                val = config.get(key, default)
                return default if val is None else val
            return default

        # Optional camera_layout may be a ListConfig; normalize to a plain list
        _cam_layout = _get("camera_layout", None)
        if _cam_layout is not None:
            _cam_layout = list(_cam_layout)

        # normalize_mode can be null/None to disable
        _norm_mode = _get("normalize_mode", "min-max")
        if isinstance(_norm_mode, str) and _norm_mode.lower() in ("none", "null", ""):
            _norm_mode = None

        return cls(
            dataset_dir=_get("dataset_dir"),
            embodiment=_get("embodiment", "aloha-agilex"),
            variant=_get("variant", "both"),
            normalization_stats_path=_get("normalization_stats_path", None),
            normalize_mode=_norm_mode,
            action_mode=_get("action_mode", "eef"),
            num_frames=int(_get("num_frames", 33)),
            height=int(_get("height", 384)),
            width=int(_get("width", 320)),
            split=split,
            target_camera=_get("target_camera", "head_camera"),
            window_stride=int(_get("window_stride", 1)),
            video_stride=int(_get("video_stride", 4)),
            multiview=bool(_get("multiview", True)),
            camera_layout=_cam_layout,
            backbone=_get("backbone", None),
            unify_action=bool(_get("unify_action", False)),
            unify_action_map=_get("unify_action_map", None),
            unify_state_map=_get("unify_state_map", None),
            color_jitter=_get("color_jitter", None),
        )

    def __init__(
        self,
        dataset_dir: str,
        embodiment: str,
        variant: str = "clean_50",
        tasks: Optional[list] = None,
        normalization_stats_path: Optional[str] = None,
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
            roots = discover_robotwin_roots(dataset_dir, embodiment, v, tasks)
            for _t, data_root in roots:
                display = f"{_t}/{v}" if len(variant_list) > 1 else _t
                all_roots.append((display, data_root, v))

        if not all_roots:
            raise FileNotFoundError(
                f"No task data found in {dataset_dir} for embodiment={embodiment}, variant={variant}."
            )

        print(
            f"MultiTaskRoboTwinDataset: {len(all_roots)} task-variant pairs, "
            f"embodiment={embodiment}, variant={variant}, action_mode={action_mode}"
        )

        # ---- Resolve shared action-stats path (auto-compute if missing) ----
        _norm_mode_kw = kwargs.get("normalize_mode", "min-max")
        if isinstance(_norm_mode_kw, str) and _norm_mode_kw.lower() in ("none", "null", ""):
            _norm_mode_kw = None

        _default_stats_name = f"robotwin_{variant}_normalization_stats.npy"
        _scope_label = "multi-task"

        if _norm_mode_kw is None:
            print(
                f"[normalizer] {_scope_label} normalization DISABLED (normalize_mode=None); "
                f"sub-datasets will run with raw action values."
            )
        else:
            print(
                f"[normalizer] Resolving {_scope_label} shared stats "
                f"(normalize_mode={_norm_mode_kw}, action_mode={action_mode}, "
                f"embodiment={embodiment}, variant={variant})"
            )
            if normalization_stats_path is not None:
                if os.path.exists(normalization_stats_path):
                    print(
                        f"[normalizer] Using explicit shared stats file: {normalization_stats_path} "
                        f"(exists ✓, will be forwarded to every sub-dataset)"
                    )
                else:
                    print(
                        f"[normalizer] WARNING: explicit normalization_stats_path does not exist: "
                        f"{normalization_stats_path}\n"
                        f"[normalizer]          sub-datasets will fall back to their own auto-resolution."
                    )
            else:
                normalization_stats_path = os.path.join(dataset_dir, "meta", _default_stats_name)
                if os.path.exists(normalization_stats_path):
                    print(
                        f"[normalizer] Found pre-computed {_scope_label} stats file: {normalization_stats_path} "
                        f"(exists ✓, will load)"
                    )
                else:
                    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import (
                        atomic_save_stats_npy,
                        cleanup_partial_stats_checkpoint,
                        compute_multitask_robotwin_stats,
                    )

                    try:
                        import torch.distributed as dist
                    except Exception:
                        dist = None

                    dist_ready = dist is not None and dist.is_available() and dist.is_initialized()
                    rank = dist.get_rank() if dist_ready else 0
                    is_rank0 = rank == 0

                    # Poll-based cross-rank synchronization: rank 0 owns the
                    # full computation (which can run for hours on large
                    # datasets) and other ranks wait on the resulting file.
                    # Going through ``dist.barrier()`` on a single CPU rank
                    # would either block the NCCL collective or trip its
                    # internal timeout; explicit polling keeps the wait
                    # transport-agnostic.
                    #
                    # Defaults are tuned for "real" multi-task RoboTwin runs
                    # (~hours of stats compute) but are overridable via env
                    # so CI / smoke runs can tighten the wait and oncall can
                    # extend it on bigger datasets without a code change.
                    def _env_positive_float(name: str, default: float) -> float:
                        raw = os.environ.get(name)
                        if not raw:
                            return default
                        try:
                            value = float(raw)
                        except ValueError:
                            print(f"[normalizer] WARNING: invalid {name}={raw!r}, falling back to default {default}")
                            return default
                        if value <= 0:
                            print(
                                f"[normalizer] WARNING: non-positive {name}={raw!r}, falling back to default {default}"
                            )
                            return default
                        return value

                    wait_timeout_s = _env_positive_float("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60)
                    poll_interval_s = _env_positive_float("OPENWAM_STATS_POLL_INTERVAL_S", 10)

                    print(
                        f"[normalizer] No pre-computed {_scope_label} stats at default location: "
                        f"{normalization_stats_path}\n"
                        f"[normalizer]   → computing now across {len(all_roots)} task-variant pairs "
                        f"and will save to: {normalization_stats_path}\n"
                        f"[normalizer]   (this may take a while for large datasets)"
                    )
                    if is_rank0:
                        os.makedirs(os.path.dirname(normalization_stats_path), exist_ok=True)
                        stats = compute_multitask_robotwin_stats(
                            dataset_dir=dataset_dir,
                            embodiment=embodiment,
                            variant=variant,
                            tasks=tasks,
                            checkpoint_path=normalization_stats_path,
                        )
                        atomic_save_stats_npy(normalization_stats_path, stats)
                        cleanup_partial_stats_checkpoint(normalization_stats_path)
                        print(f"[normalizer] Saved newly-computed {_scope_label} stats → {normalization_stats_path}")
                    elif dist_ready:
                        print(
                            f"[normalizer] Rank {rank} waiting for rank 0 to finish shared stats at {normalization_stats_path}"
                        )
                        deadline = time.monotonic() + wait_timeout_s
                        while not os.path.exists(normalization_stats_path):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    f"Timed out while waiting for rank 0 to produce shared stats: {normalization_stats_path}"
                                )
                            time.sleep(poll_interval_s)

                    if not os.path.exists(normalization_stats_path):
                        raise FileNotFoundError(
                            f"Expected shared stats file to exist after computation, but it was not found: "
                            f"{normalization_stats_path}"
                        )
        self.normalization_stats_path: Optional[str] = normalization_stats_path

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

        import contextlib
        import os as _os

        for display_name, data_root, v in task_iter:
            # Suppress per-task prints to keep the tqdm progress bar clean.
            # ExitStack guarantees both the devnull file and the stdout
            # redirection are torn down even if RoboTwinDataset.__init__ raises
            # — the old ``sys.stdout = open(...)`` idiom leaked stdout on error.
            with contextlib.ExitStack() as _stack:
                if _use_tqdm:
                    task_iter.set_postfix_str(display_name)
                    _devnull = _stack.enter_context(open(_os.devnull, "w"))
                    _stack.enter_context(contextlib.redirect_stdout(_devnull))
                ds = RoboTwinDataset(
                    data_root=data_root,
                    task_name=display_name.split("/")[0].replace("_", " "),
                    normalization_stats_path=normalization_stats_path,
                    embodiment=embodiment,
                    variant=v,
                    action_mode=action_mode,
                    **kwargs,
                )
            self._sub_datasets.append(ds)
            cumulative += len(ds)
            self._cumulative_lengths.append(cumulative)

        self._total_length = cumulative
        self._normalization_stats_shared = self._sub_datasets[0].normalization_stats if self._sub_datasets else None
        self._action_dim_value = self._sub_datasets[0].action_dim if self._sub_datasets else 14

        print(f"  Total samples: {self._total_length} (across {len(self._sub_datasets)} sub-datasets)")

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self) -> dict:
        return self._normalization_stats_shared

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
