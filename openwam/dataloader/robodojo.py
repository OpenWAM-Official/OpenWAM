"""Formal RoboDojo simulation and RoboDojo_real HDF5 dataloaders.

RoboDojo simulation records achieved end-effector poses as
environment-origin-relative positions with world-frame wxyz orientations.
RoboDojo_real records native per-arm robot-base poses and must not receive the
simulation base transform. Dual-X5 constants and the env-origin → robot-base
/ EEF20 helpers live in
``openwam.dataloader.robodojo_contract`` and
``openwam.dataloader.utils.poses``.  The Isaac eval runtime
(XPolicyLab) keeps pinned copies of these contracts and must not import
this package.
This reader applies those transforms, packs raw EEF20, normalizes in that raw
space, and only then scatters into OpenWAM's shared 80-D action space.

Gripper channels are the official ``state/*_ee_joint_states`` values in
``[0, 1]``: ``0`` is closed and ``1`` is open.  That is the same raw
direction the pretraining mixture uses before normalization.  Closed-gripper
float noise around ``-3e-17`` is clipped to ``0``.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.robodojo_contract import (
    EEF20_DIM,
    GRIPPER_CONVENTION,
    ROBODOJO_CONTRACT_ID,
    ROBODOJO_EMBODIMENT,
    ROBODOJO_REAL_CONTRACT_ID,
    ROBODOJO_REAL_GRIPPER_SENSOR_ATOL,
    ROBODOJO_REAL_SOURCE_FRAME,
    ROBODOJO_REAL_VARIANT,
    ROBODOJO_SIM_SOURCE_FRAME,
    ROBODOJO_SIM_VARIANT,
    discover_episodes,
    resolve_robodojo_calibration,
    robodojo_real_frame_contract,
    validate_calibration,
    validate_dataset_variant,
    validate_embodiment,
)
from openwam.dataloader.transforms.multiview import (
    assemble_multiview_layout,
    crop_and_resize,
    format_prompt_for_inference,
)
from openwam.dataloader.transforms.normalize import (
    YAML_TO_NORM_MODE,
    Normalizer,
)
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, STAT_KEYS
from openwam.dataloader.utils.poses import (
    arms_to_eef20,
    env_relative_world_to_robot_base,
)
from openwam.dataloader.utils.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

DEPLOY_ACTION_MODE = "eef"
DEFAULT_ROBODOJO_CAMERA_LAYOUT = (
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist",
)
# Compatibility alias for the original simulation-only reader API.
ROBODOJO_SOURCE_FRAME = ROBODOJO_SIM_SOURCE_FRAME

_POSE_KEYS = (
    "state/left_ee_poses",
    "state/right_ee_poses",
)
_GRIPPER_KEYS = (
    "state/left_ee_joint_states",
    "state/right_ee_joint_states",
)
_CAMERA_DATASETS = {camera: f"vision/{camera}/colors" for camera in DEFAULT_ROBODOJO_CAMERA_LAYOUT}
_REQUIRED_DATASETS = (*_POSE_KEYS, *_GRIPPER_KEYS, *_CAMERA_DATASETS.values(), "instruction")
_QUATERNION_ATOL = 1e-6
# Official HDF5 occasionally stores a closed gripper as ~-3e-17.
_GRIPPER_ATOL = 1e-8
_CANONICAL_UNIFY_DST_INDEX = np.asarray(
    [*range(10), *range(34, 44)],
    dtype=np.int64,
)


def _config_get(config, key: str, default=None):
    if isinstance(config, Mapping):
        return config[key] if key in config else default
    elif hasattr(config, key):
        return getattr(config, key)
    elif hasattr(config, "get"):
        return config.get(key, default)
    return default


def _normalize_mode(value: Any) -> str | None:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode in {"", "none", "null"}:
        return None
    if mode not in YAML_TO_NORM_MODE:
        raise ValueError(f"normalize_mode must be one of {sorted(YAML_TO_NORM_MODE)} or null, got {value!r}")
    return mode


def _reject_calibration_path(calibration_path: str | Path | None) -> None:
    if calibration_path is not None:
        raise ValueError("RoboDojo uses the built-in dual-X5 base constants; calibration_path is not accepted")


def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def calibration_fingerprint(calibration: Mapping[str, Any]) -> str:
    """Return SHA-256 of the validated canonical sim calibration JSON."""
    return _mapping_fingerprint(validate_calibration(calibration))


def real_frame_contract_fingerprint(embodiment: str) -> str:
    """Fingerprint the native real-data frame and preprocessing contract."""
    return _mapping_fingerprint(robodojo_real_frame_contract(embodiment))


def _arm_pose_in_base(
    pose: np.ndarray,
    calibration: Mapping[str, Any],
    arm: str,
) -> np.ndarray:
    arm_calibration = calibration["arms"][arm]
    return env_relative_world_to_robot_base(
        pose,
        arm_calibration["base_pos_relative_to_env_origin"],
        arm_calibration["base_quat_wxyz"],
    )


def _sanitize_gripper(
    value: Any,
    *,
    source: str,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> np.ndarray:
    gripper = np.asarray(value)
    if gripper.dtype.kind not in "fiu" or not np.all(np.isfinite(gripper)):
        raise ValueError(f"{source} must contain only finite gripper values")
    validate_dataset_variant(variant)
    atol = ROBODOJO_REAL_GRIPPER_SENSOR_ATOL if variant == ROBODOJO_REAL_VARIANT else _GRIPPER_ATOL
    if np.any((gripper < -atol) | (gripper > 1.0 + atol)):
        raise ValueError(f"{source} gripper values must be within [0, 1] up to the {variant} sensor tolerance {atol:g}")
    return np.clip(gripper, 0.0, 1.0)


def read_calibrated_eef20(
    episode: h5py.File | h5py.Group | str | Path,
    calibration: Mapping[str, Any] | None,
    start: int | None = None,
    end: int | None = None,
    *,
    variant: str = ROBODOJO_SIM_VARIANT,
    embodiment: str = ROBODOJO_EMBODIMENT,
) -> np.ndarray:
    """Read achieved states through the selected release's EEF20 path.

    The stats computation imports and calls this function directly, guaranteeing
    reader/stats conversion parity without a second frame or rotation
    implementation.
    """
    validate_embodiment(embodiment, variant=variant)
    if isinstance(episode, (str, Path)):
        with h5py.File(episode, "r") as handle:
            return read_calibrated_eef20(
                handle,
                calibration,
                start,
                end,
                variant=variant,
                embodiment=embodiment,
            )

    selection = slice(start, end)
    left_pose = np.asarray(episode[_POSE_KEYS[0]][selection])
    right_pose = np.asarray(episode[_POSE_KEYS[1]][selection])
    left_gripper = _sanitize_gripper(
        episode[_GRIPPER_KEYS[0]][selection],
        source=_GRIPPER_KEYS[0],
        variant=variant,
    )
    right_gripper = _sanitize_gripper(
        episode[_GRIPPER_KEYS[1]][selection],
        source=_GRIPPER_KEYS[1],
        variant=variant,
    )
    if variant == ROBODOJO_REAL_VARIANT:
        if calibration is not None:
            raise ValueError("RoboDojo_real uses native per-arm base poses; calibration must be None")
        left_base = left_pose
        right_base = right_pose
    else:
        if calibration is None:
            raise ValueError("RoboDojo sim requires the dual-X5 calibration")
        left_base = _arm_pose_in_base(left_pose, calibration, "left")
        right_base = _arm_pose_in_base(right_pose, calibration, "right")
    return arms_to_eef20(
        left_base,
        left_gripper,
        right_base,
        right_gripper,
    ).astype(np.float32, copy=False)


def _decode_instruction(value: Any, *, source: str) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{source}: instruction must be valid UTF-8") from error
    if not isinstance(value, str):
        raise ValueError(f"{source}: instruction must be a scalar string or bytes")
    instruction = value.strip()
    if not instruction:
        raise ValueError(f"{source}: instruction must not be empty")
    return instruction


def validate_robodojo_episode(
    path: str | Path,
    *,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> dict[str, Any]:
    """Validate every binding HDF5 field and return index-time metadata."""
    validate_dataset_variant(variant)
    episode_path = Path(path)
    with h5py.File(episode_path, "r") as handle:
        for key in _REQUIRED_DATASETS:
            if key not in handle:
                raise KeyError(f"{episode_path}: missing required dataset {key!r}")

        for key in _POSE_KEYS:
            shape = handle[key].shape
            if len(shape) != 2 or shape[1] != 7:
                raise ValueError(f"{episode_path}:{key} must have exact shape (T, 7), got {shape}")
        for key in _GRIPPER_KEYS:
            shape = handle[key].shape
            if len(shape) != 2 or shape[1] != 1:
                raise ValueError(f"{episode_path}:{key} must have exact shape (T, 1), got {shape}")
        for key in _CAMERA_DATASETS.values():
            dataset = handle[key]
            shape = dataset.shape
            is_padded_uint8 = len(shape) == 2 and shape[1] > 0 and dataset.dtype == np.dtype(np.uint8)
            if len(shape) != 1 and not is_padded_uint8:
                raise ValueError(
                    f"{episode_path}:{key} must have shape (T,) for fixed/vlen "
                    f"JPEG entries or (T, max_jpeg_bytes) for padded uint8, got {shape}"
                )
            variable_dtype = h5py.check_dtype(vlen=dataset.dtype)
            is_fixed_bytes = dataset.dtype.kind == "S"
            is_vlen_uint8 = variable_dtype is not None and np.dtype(variable_dtype) == np.dtype(np.uint8)
            if not is_fixed_bytes and not is_vlen_uint8 and not is_padded_uint8:
                raise ValueError(
                    f"{episode_path}:{key} must use fixed byte-string dtype or "
                    f"HDF5 vlen/padded uint8 storage, got dtype {dataset.dtype}"
                )

        lengths = {key: int(handle[key].shape[0]) for key in (*_POSE_KEYS, *_GRIPPER_KEYS, *_CAMERA_DATASETS.values())}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1 or 0 in unique_lengths:
            raise ValueError(f"{episode_path}: all state and camera datasets must have equal non-zero T; got {lengths}")
        length = next(iter(unique_lengths))
        if length < 2:
            raise ValueError(f"{episode_path}: RoboDojo episodes need at least two frames, got T={length}")

        for key in _POSE_KEYS:
            poses = np.asarray(handle[key][()])
            if poses.dtype.kind not in "fiu" or not np.all(np.isfinite(poses)):
                raise ValueError(f"{episode_path}:{key} must contain only finite poses")
            norms = np.linalg.norm(poses[:, 3:7].astype(np.float64), axis=-1)
            if not np.all(np.isclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_ATOL)):
                bad_index = int(
                    np.flatnonzero(
                        ~np.isclose(
                            norms,
                            1.0,
                            rtol=0.0,
                            atol=_QUATERNION_ATOL,
                        )
                    )[0]
                )
                raise ValueError(
                    f"{episode_path}:{key} must contain unit wxyz quaternions; "
                    f"row {bad_index} has norm {norms[bad_index]:.8g}"
                )

        for key in _GRIPPER_KEYS:
            _sanitize_gripper(
                handle[key][()],
                source=f"{episode_path}:{key}",
                variant=variant,
            )

        instruction_dataset = handle["instruction"]
        if instruction_dataset.shape != ():
            raise ValueError(f"{episode_path}:instruction must be scalar, got shape {instruction_dataset.shape}")
        instruction = _decode_instruction(
            instruction_dataset[()],
            source=str(episode_path),
        )

    return {"length": length, "instruction": instruction}


def _jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(np.uint8) or value.ndim != 1:
            raise ValueError("RoboDojo vlen/padded JPEG entries must be one-dimensional uint8")
        return value.tobytes()
    if isinstance(value, np.bytes_):
        return value.tobytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ValueError(
        f"unsupported RoboDojo JPEG entry type {type(value).__name__}; expected fixed bytes or vlen/padded uint8"
    )


def _decode_jpeg(value: Any, *, source: str) -> Image.Image:
    encoded = _jpeg_bytes(value)
    if not encoded:
        raise ValueError(f"{source}: JPEG entry is empty")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB").copy()
    except Exception as error:
        raise ValueError(f"{source}: could not decode JPEG") from error


def discover_robodojo_tasks(
    dataset_dir: str | Path,
    *,
    embodiment: str = ROBODOJO_EMBODIMENT,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> list[str]:
    """Discover sorted task directories under the formal RoboDojo root."""
    validate_embodiment(embodiment, variant=variant)
    root = Path(dataset_dir)
    flat_data = root / embodiment / "data"
    if flat_data.is_dir() and any(flat_data.glob("episode_*.hdf5")):
        raise ValueError(
            "flat RoboDojo demo layout '<dataset_root>/arx_x5/data' is not supported; "
            "use '<dataset_root>/<task>/arx_x5/data'"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"RoboDojo dataset root does not exist: {root}")
    return sorted(child.name for child in root.iterdir() if child.is_dir() and (child / embodiment / "data").is_dir())


def _requested_tasks_exist(available: Sequence[str], requested: Sequence[str]) -> None:
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(
            "requested RoboDojo task(s) are missing from the formal dataset root: " + ", ".join(missing)
        )


def resolve_robodojo_tasks(
    dataset_dir: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    embodiment: str = ROBODOJO_EMBODIMENT,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> list[str]:
    """Discover all formal tasks, optionally restricting an internal scan."""
    available = discover_robodojo_tasks(
        dataset_dir,
        embodiment=embodiment,
        variant=variant,
    )
    if not available:
        raise FileNotFoundError(f"no formal RoboDojo task directories found in {dataset_dir}")
    if tasks is None:
        return available
    requested = sorted(set(str(task) for task in tasks))
    if not requested:
        raise ValueError("RoboDojo task scan cannot be empty")
    _requested_tasks_exist(available, requested)
    return requested


def default_robodojo_stats_path(
    dataset_dir: str | Path,
    *,
    variant: str,
    embodiment: str,
) -> Path:
    """Return the canonical in-dataset stats path.

    Stats are always pooled over EVERY task the corpus holds for the variant
    (task selections share the same file). The real corpus mixes embodiments
    under one root, so its filename keeps the embodiment; sim is arx_x5-only.
    """
    validate_embodiment(embodiment, variant=variant)
    if variant == ROBODOJO_REAL_VARIANT:
        filename = f"robodojo_real_{embodiment}_normalization_stats.npy"
    else:
        filename = "robodojo_normalization_stats.npy"
    return Path(dataset_dir) / "meta" / filename


def ensure_robodojo_stats(
    dataset_dir: str | Path,
    *,
    variant: str,
    embodiment: str,
) -> Path:
    """Find or auto-build release/embodiment-matched pooled stats.

    Always computed over every task in the corpus, regardless of the training
    task selection. Rank 0 scans the corpus and writes atomically (tmp +
    rename), so readers never observe a partially written NumPy payload;
    other torchrun ranks poll until the file appears.
    """
    stats_path = default_robodojo_stats_path(
        dataset_dir,
        variant=variant,
        embodiment=embodiment,
    )
    if stats_path.is_file():
        return stats_path

    try:
        import torch.distributed as dist

        dist_ready = dist.is_available() and dist.is_initialized()
    except Exception:
        dist_ready = False
    if dist_ready:
        rank = dist.get_rank()
    else:
        # torchrun sets RANK before init_process_group; honor it so
        # pre-init constructions still elect a single builder.
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        print(
            f"[RoboDojo] normalization stats not found; computing {stats_path}",
            flush=True,
        )
        # Imported lazily to avoid a module cycle: the stats implementation
        # deliberately imports this reader's canonical conversion function.
        from openwam.dataloader.utils.stats_computation.robodojo_stats_computation import (
            build_and_save_robodojo_stats,
        )

        build_and_save_robodojo_stats(
            dataset_dir=dataset_dir,
            output=stats_path,
            tasks=discover_robodojo_tasks(
                dataset_dir,
                embodiment=embodiment,
                variant=variant,
            ),
            embodiment=embodiment,
            variant=variant,
            action_mode=DEPLOY_ACTION_MODE,
        )
        print(
            f"[RoboDojo] wrote normalization stats to {stats_path}",
            flush=True,
        )
    else:
        deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
        poll_interval_s = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
        while not stats_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for rank 0 to build RoboDojo stats: {stats_path}")
            time.sleep(poll_interval_s)
    return stats_path


def _load_validated_stats(
    path: str | Path,
    calibration: Mapping[str, Any] | None,
    *,
    variant: str,
    embodiment: str,
    expected_tasks: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    stats_path = Path(path)
    try:
        payload = np.load(stats_path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"could not read RoboDojo normalization stats from {stats_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{stats_path}: normalization stats payload must be a mapping")
    mode_stats = payload.get(DEPLOY_ACTION_MODE)
    if not isinstance(mode_stats, Mapping):
        raise KeyError(f"{stats_path}: normalization stats must contain nested key {DEPLOY_ACTION_MODE!r}")

    validated: dict[str, np.ndarray] = {}
    for key in STAT_KEYS:
        if key not in mode_stats:
            raise KeyError(f"{stats_path}:eef is missing stats vector {key!r}")
        value = np.asarray(mode_stats[key])
        if value.shape != (EEF20_DIM,):
            raise ValueError(f"{stats_path}:eef.{key} must have exact shape ({EEF20_DIM},), got {value.shape}")
        if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
            raise ValueError(f"{stats_path}:eef.{key} must contain only finite numeric values")
        validated[key] = value.astype(np.float32, copy=False)

    rot6d_dims = np.asarray(ROT6D_DIMS_EEF20, dtype=np.int64)
    rot6d_identity = {
        "mean": 0.0,
        "std": 1.0,
        "min": -1.0,
        "max": 1.0,
        "q01": -1.0,
        "q99": 1.0,
    }
    stale_rot6d = [
        key
        for key, expected in rot6d_identity.items()
        if not np.array_equal(
            validated[key][rot6d_dims],
            np.full(len(rot6d_dims), expected, dtype=np.float32),
        )
    ]
    if stale_rot6d:
        raise ValueError(
            f"{stats_path}:eef has non-identity rot6d statistics for "
            f"{', '.join(stale_rot6d)} on dimensions 3..8 and 13..18; "
            "regenerate the RoboDojo stats file with "
            "robodojo_stats_computation"
        )

    if np.any(validated["std"] <= 0):
        raise ValueError(f"{stats_path}:eef.std values must all be positive")
    if np.any(validated["max"] < validated["min"]):
        raise ValueError(f"{stats_path}:eef.max must be >= eef.min")
    if np.any(validated["q99"] < validated["q01"]):
        raise ValueError(f"{stats_path}:eef.q99 must be >= eef.q01")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{stats_path}:metadata must be a mapping")
    if variant == ROBODOJO_REAL_VARIANT:
        fingerprint_key = "frame_contract_fingerprint"
        fingerprint_label = "frame contract fingerprint"
        expected_fingerprint = real_frame_contract_fingerprint(embodiment)
        expected_contract = ROBODOJO_REAL_CONTRACT_ID
        expected_source_frame = ROBODOJO_REAL_SOURCE_FRAME
    else:
        fingerprint_key = "calibration_fingerprint"
        fingerprint_label = "calibration fingerprint"
        if calibration is None:
            raise ValueError("RoboDojo sim stats validation requires calibration")
        expected_fingerprint = calibration_fingerprint(calibration)
        expected_contract = ROBODOJO_CONTRACT_ID
        expected_source_frame = ROBODOJO_SIM_SOURCE_FRAME
    saved_fingerprint = metadata.get(
        fingerprint_key,
        mode_stats.get(fingerprint_key),
    )
    if not isinstance(saved_fingerprint, str):
        raise ValueError(f"{stats_path}: metadata.{fingerprint_key} is required")
    if saved_fingerprint != expected_fingerprint:
        raise ValueError(
            f"RoboDojo normalization stats {fingerprint_label} mismatch: "
            f"stats={saved_fingerprint}, expected={expected_fingerprint}"
        )
    recorded_variant = metadata.get("variant", ROBODOJO_SIM_VARIANT)
    if recorded_variant != variant:
        raise ValueError(f"{stats_path}: metadata.variant must be {variant!r}, got {recorded_variant!r}")
    if metadata.get("embodiment") != embodiment:
        raise ValueError(
            f"{stats_path}: metadata.embodiment must be {embodiment!r}, got {metadata.get('embodiment')!r}"
        )
    if metadata.get("source_frame") != expected_source_frame:
        raise ValueError(
            f"{stats_path}: metadata.source_frame must be "
            f"{expected_source_frame!r}, got {metadata.get('source_frame')!r}"
        )
    if expected_tasks is not None:
        recorded_tasks = metadata.get("tasks")
        expected_task_list = sorted(set(str(task) for task in expected_tasks))
        if not isinstance(recorded_tasks, Sequence) or isinstance(
            recorded_tasks,
            (str, bytes),
        ):
            raise ValueError(f"{stats_path}: metadata.tasks is required")
        if sorted(set(str(task) for task in recorded_tasks)) != expected_task_list:
            raise ValueError(
                f"{stats_path}: metadata.tasks does not match the selected "
                f"corpus; stats={list(recorded_tasks)}, "
                f"selected={expected_task_list}"
            )
    recorded_convention = metadata.get("gripper_convention")
    if recorded_convention != GRIPPER_CONVENTION:
        raise ValueError(
            f"{stats_path}: metadata.gripper_convention must be "
            f"{GRIPPER_CONVENTION!r} (0=closed, 1=open), got "
            f"{recorded_convention!r}; regenerate the RoboDojo stats file"
        )
    recorded_contract = metadata.get("contract_id")
    if recorded_contract != expected_contract:
        raise ValueError(
            f"{stats_path}: metadata.contract_id must be "
            f"{expected_contract!r}, got {recorded_contract!r}; "
            "regenerate the RoboDojo stats file"
        )
    return validated


class RoboDojoDataset(BaseDataset):
    """One formal RoboDojo task rooted at ``<task>/<embodiment>/data``."""

    DEPLOY_ACTION_MODE = DEPLOY_ACTION_MODE

    def __init__(
        self,
        data_root: str | Path,
        *,
        dataset_root: str | Path | None = None,
        task_name: str | None = None,
        calibration: Mapping[str, Any] | None = None,
        calibration_path: str | Path | None = None,
        num_frames: int = 33,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        action_mode: str = DEPLOY_ACTION_MODE,
        embodiment: str = ROBODOJO_EMBODIMENT,
        variant: str = ROBODOJO_SIM_VARIANT,
        robot: str | None = None,
        normalization_stats_path: str | Path | None = None,
        normalization_stats_tasks: Sequence[str] | None = None,
        normalize_mode: str | None = "min-max",
        window_stride: int = 1,
        video_stride: int = 4,
        multiview: bool = True,
        camera_layout: Sequence[str] | None = None,
        target_camera: str = "cam_head",
        unify_action: bool = True,
        unify_action_map: Any = None,
        unify_state_map: Any = None,
        color_jitter: Any = None,
    ):
        super().__init__()
        if action_mode != DEPLOY_ACTION_MODE:
            raise ValueError(f"RoboDojo supports only action_mode='eef', got {action_mode!r}")
        if robot is not None and robot != embodiment:
            raise ValueError(f"RoboDojo robot and embodiment disagree: {robot!r} != {embodiment!r}")
        validate_embodiment(embodiment, variant=variant)
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        if int(window_stride) < 1:
            raise ValueError(f"window_stride must be >= 1, got {window_stride}")
        if int(video_stride) < 1:
            raise ValueError(f"video_stride must be >= 1, got {video_stride}")
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {height}x{width}")

        self.action_mode = DEPLOY_ACTION_MODE
        self.embodiment = embodiment
        self.variant = variant
        self.num_frames = int(num_frames)
        self.num_action_steps = self.num_frames - 1
        self.height = int(height)
        self.width = int(width)
        self.split = split
        self.window_stride = int(window_stride)
        self.video_stride = int(video_stride)
        self._video_sample_indices = list(range(0, self.num_frames, self.video_stride))
        self.num_video_frames = len(self._video_sample_indices)
        self.multiview = bool(multiview)
        self.target_camera = str(target_camera)

        # Apply one sampled set of color factors to the assembled clip.  The
        # transform is intentionally train-only so validation/deployment input
        # remains unchanged.
        self._color_jitter = None
        if color_jitter_enabled(color_jitter) and split == "train":
            jitter_get = color_jitter.get if hasattr(color_jitter, "get") else lambda key, default: default
            self._color_jitter = VideoColorJitter(
                brightness=float(jitter_get("brightness", 0.2)),
                contrast=float(jitter_get("contrast", 0.2)),
                saturation=float(jitter_get("saturation", 0.2)),
                hue=float(jitter_get("hue", 0.0)),
            )

        if camera_layout is None:
            camera_layout = DEFAULT_ROBODOJO_CAMERA_LAYOUT
        self.camera_layout = tuple(str(camera) for camera in camera_layout)
        if self.multiview:
            if len(self.camera_layout) != 3 or set(self.camera_layout) != set(DEFAULT_ROBODOJO_CAMERA_LAYOUT):
                raise ValueError(
                    "RoboDojo multiview camera_layout must contain exactly "
                    f"{list(DEFAULT_ROBODOJO_CAMERA_LAYOUT)}, got "
                    f"{list(self.camera_layout)}"
                )
        elif self.target_camera not in _CAMERA_DATASETS:
            raise ValueError(f"target_camera must be one of {list(_CAMERA_DATASETS)}, got {self.target_camera!r}")

        _reject_calibration_path(calibration_path)
        if variant == ROBODOJO_REAL_VARIANT:
            if calibration is not None:
                raise ValueError("RoboDojo_real uses native per-arm base poses; calibration must be None")
            self.calibration = None
            self.frame_contract = robodojo_real_frame_contract(embodiment)
            self.frame_contract_fingerprint = real_frame_contract_fingerprint(embodiment)
            # Alias so callers can read one calibration fingerprint field
            # across sim and real embodiments.
            self.calibration_fingerprint = self.frame_contract_fingerprint
            self.source_frame = ROBODOJO_REAL_SOURCE_FRAME
            self.contract_id = ROBODOJO_REAL_CONTRACT_ID
        else:
            self.calibration = resolve_robodojo_calibration(calibration)
            self.calibration_fingerprint = calibration_fingerprint(self.calibration)
            self.frame_contract = None
            self.frame_contract_fingerprint = self.calibration_fingerprint
            self.source_frame = ROBODOJO_SIM_SOURCE_FRAME
            self.contract_id = ROBODOJO_CONTRACT_ID

        supplied_root = Path(data_root)
        if supplied_root.name == "data":
            if dataset_root is None or not task_name:
                raise ValueError(
                    "formal RoboDojo provenance requires explicit dataset_root "
                    "and task_name when data_root points to a task data directory"
                )
            formal_dataset_root = Path(dataset_root)
            self.task_name = str(task_name)
            expected_data_root = formal_dataset_root / self.task_name / embodiment / "data"
            if supplied_root.resolve() != expected_data_root.resolve():
                raise ValueError(
                    "formal RoboDojo provenance mismatch: data_root must equal "
                    "dataset_root/task_name/embodiment/data; "
                    f"got data_root={supplied_root}, "
                    f"dataset_root={formal_dataset_root}, "
                    f"task_name={self.task_name!r}"
                )
            episode_paths = discover_episodes(
                formal_dataset_root,
                self.task_name,
                embodiment=embodiment,
                variant=variant,
            )
            self.data_root = str(supplied_root)
        else:
            if not task_name:
                raise ValueError("task_name is required when data_root is the RoboDojo dataset root")
            formal_dataset_root = supplied_root
            if dataset_root is not None and Path(dataset_root).resolve() != formal_dataset_root.resolve():
                raise ValueError(
                    "formal RoboDojo provenance mismatch: dataset_root must "
                    "equal data_root when data_root is the dataset root"
                )
            self.task_name = str(task_name)
            episode_paths = discover_episodes(
                formal_dataset_root,
                self.task_name,
                embodiment=embodiment,
                variant=variant,
            )
            self.data_root = str(episode_paths[0].parent)
        self.dataset_root = str(formal_dataset_root)

        # Validate the complete discovered corpus before split/index selection.
        all_metadata = [validate_robodojo_episode(path, variant=variant) for path in episode_paths]
        selected_indices = list(range(len(episode_paths)))
        if not selected_indices:
            raise ValueError(f"no RoboDojo episodes selected for split={split!r}")

        self._episode_files = [str(episode_paths[index]) for index in selected_indices]
        self._episode_lengths = [int(all_metadata[index]["length"]) for index in selected_indices]
        self._instructions = [str(all_metadata[index]["instruction"]) for index in selected_indices]

        self._window_index: list[tuple[int, int]] = []
        for episode_index, episode_length in enumerate(self._episode_lengths):
            for start in range(
                0,
                episode_length - 1,
                self.window_stride,
            ):
                self._window_index.append((episode_index, start))
        if not self._window_index:
            raise ValueError("no RoboDojo windows contain a real next-state action target")

        self._raw_action_dim_value = EEF20_DIM
        self._unify_action = bool(unify_action)
        if unify_state_map is not None and list(unify_state_map) != ["0-9", "34-43"]:
            raise ValueError(
                'unify_state_map must be null or the canonical ["0-9", "34-43"]: '
                "RoboDojo state shares the action's raw EEF20 layout"
            )
        self._unify_dst_index: np.ndarray | None = None
        if self._unify_action:
            if unify_action_map is None:
                raise ValueError(
                    'RoboDojo unify_action=true requires an explicit unify_action_map; use ["0-9", "34-43"]'
                )
            self._unify_dst_index = parse_unify_spec(
                unify_action_map,
                UNIFY_DIM,
            )
            if self._unify_dst_index.shape != (EEF20_DIM,):
                raise ValueError(
                    "RoboDojo unify_action_map must map exactly 20 raw EEF "
                    f"dimensions, got {self._unify_dst_index.shape[0]}"
                )
            if not np.array_equal(
                self._unify_dst_index,
                _CANONICAL_UNIFY_DST_INDEX,
            ):
                raise ValueError('RoboDojo requires the canonical unify_action_map ["0-9", "34-43"] exactly')
            self._action_dim_value = UNIFY_DIM
        else:
            self._action_dim_value = EEF20_DIM

        self.normalize_mode = _normalize_mode(normalize_mode)
        self._mode_stats: dict[str, np.ndarray] | None = None
        self._normalizer: Normalizer | None = None
        self.normalization_stats_path: str | None = None
        if self.normalize_mode is not None:
            if normalization_stats_path is None:
                stats_path = ensure_robodojo_stats(
                    self.dataset_root,
                    variant=self.variant,
                    embodiment=self.embodiment,
                )
            else:
                stats_path = Path(normalization_stats_path)
            if not stats_path.is_file():
                raise FileNotFoundError(f"RoboDojo normalization stats do not exist: {stats_path}")
            self._mode_stats = _load_validated_stats(
                stats_path,
                self.calibration,
                variant=self.variant,
                embodiment=self.embodiment,
                expected_tasks=normalization_stats_tasks,
            )
            self._normalizer = Normalizer(
                mode=YAML_TO_NORM_MODE[self.normalize_mode],
                stats=self._mode_stats,
            )
            self.normalization_stats_path = str(stats_path)

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self) -> dict[str, np.ndarray] | None:
        if self._mode_stats is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._mode_stats.items()}

    def denormalize_action(self, action) -> np.ndarray:
        """Gather unified output to EEF20, then invert raw-space normalization."""
        array = np.asarray(action)
        if self._unify_dst_index is not None:
            array = unmap_from_unify(array, self._unify_dst_index)
        if self._normalizer is not None:
            array = self._normalizer.unnormalize(array)
        return np.asarray(array).copy()

    def __len__(self) -> int:
        return len(self._window_index)

    def _frame(
        self,
        handle: h5py.File,
        camera: str,
        frame_index: int,
    ) -> Image.Image:
        dataset_key = _CAMERA_DATASETS[camera]
        return _decode_jpeg(
            handle[dataset_key][frame_index],
            source=f"{handle.filename}:{dataset_key}[{frame_index}]",
        )

    def _video_at(
        self,
        handle: h5py.File,
        frame_index: int,
    ) -> Image.Image:
        if self.multiview:
            frames = {camera: self._frame(handle, camera, frame_index) for camera in self.camera_layout}
            return assemble_multiview_layout(
                frames,
                list(self.camera_layout),
                self.height,
                self.width,
            )
        return crop_and_resize(
            self._frame(handle, self.target_camera, frame_index),
            self.height,
            self.width,
        )

    def _build_sample(self, episode_index: int, start: int) -> dict[str, Any]:
        path = self._episode_files[episode_index]
        episode_length = self._episode_lengths[episode_index]
        actual_length = min(self.num_frames, episode_length - start)
        if actual_length < 2:
            raise IndexError(
                f"RoboDojo window start={start} has no real next-state target for episode length {episode_length}"
            )

        with h5py.File(path, "r") as handle:
            states = read_calibrated_eef20(
                handle,
                self.calibration,
                start,
                start + actual_length,
                variant=self.variant,
                embodiment=self.embodiment,
            )
            sampled_video = []
            for relative_index in self._video_sample_indices:
                source_index = start + relative_index if relative_index < actual_length else episode_length - 1
                sampled_video.append(self._video_at(handle, source_index))

        if actual_length < self.num_frames:
            states = np.concatenate(
                [
                    states,
                    np.repeat(
                        states[-1:],
                        self.num_frames - actual_length,
                        axis=0,
                    ),
                ],
                axis=0,
            )

        if self._color_jitter is not None:
            sampled_video = self._color_jitter.apply({"video": sampled_video})["video"]

        # Binding order: raw EEF20 -> normalize -> unified 80-D scatter.
        transformed = states
        if self._normalizer is not None:
            transformed = self._normalizer.normalize(transformed)
        dim_mask = np.ones(EEF20_DIM, dtype=bool)
        if self._unify_dst_index is not None:
            transformed, dim_mask = map_to_unify(
                np.asarray(transformed, dtype=np.float32),
                self._unify_dst_index,
                UNIFY_DIM,
            )

        transformed = np.asarray(transformed, dtype=np.float32)
        proprio = torch.from_numpy(transformed[0:1])
        action = torch.from_numpy(transformed[1 : self.num_frames])
        time_validity = torch.tensor(
            [action_index + 1 < actual_length for action_index in range(self.num_action_steps)],
            dtype=torch.bool,
        )
        dimension_validity = torch.from_numpy(dim_mask)
        action_mask = (time_validity[:, None] & dimension_validity[None, :]).contiguous()
        proprio_mask = dimension_validity[None, :].clone()
        video_mask = torch.tensor(
            [relative_index < actual_length for relative_index in self._video_sample_indices],
            dtype=torch.bool,
        )

        return {
            "video": sampled_video,
            "vace_video": None,
            "first_frame_image": [sampled_video[0]],
            "action": action,
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "prompt": format_prompt_for_inference(self._instructions[episode_index]),
            "episode_index": episode_index,
            "episode_path": path,
            "start_frame": start,
            "end_frame": min(start + self.num_frames, episode_length),
            "episode_length": episode_length,
            "task_name": self.task_name,
            "variant": self.variant,
            "embodiment": self.embodiment,
            "source_frame": self.source_frame,
            "active_arm": "both",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index, start = self._window_index[index]
        return self._build_sample(episode_index, start)


class MultiTaskRoboDojoDataset(BaseDataset):
    """Concatenate formal RoboDojo tasks with shared calibration and stats."""

    DEPLOY_ACTION_MODE = DEPLOY_ACTION_MODE

    @classmethod
    def from_config(cls, config, split: str = "train"):
        camera_layout = _config_get(config, "camera_layout", None)
        if camera_layout is not None:
            camera_layout = list(camera_layout)
        return cls(
            dataset_dir=_config_get(
                config,
                "dataset_dir",
                _config_get(config, "dataset_root", None),
            ),
            split=split,
            embodiment=_config_get(config, "embodiment", ROBODOJO_EMBODIMENT),
            variant=_config_get(
                config,
                "variant",
                ROBODOJO_SIM_VARIANT,
            ),
            robot=_config_get(config, "robot", None),
            action_mode=_config_get(config, "action_mode", DEPLOY_ACTION_MODE),
            normalization_stats_path=_config_get(
                config,
                "normalization_stats_path",
                None,
            ),
            normalize_mode=_config_get(config, "normalize_mode", "min-max"),
            num_frames=int(_config_get(config, "num_frames", 33)),
            height=int(_config_get(config, "height", 384)),
            width=int(_config_get(config, "width", 320)),
            window_stride=int(_config_get(config, "window_stride", 1)),
            video_stride=int(_config_get(config, "video_stride", 4)),
            multiview=bool(_config_get(config, "multiview", True)),
            camera_layout=camera_layout,
            target_camera=_config_get(config, "target_camera", "cam_head"),
            unify_action=bool(_config_get(config, "unify_action", True)),
            unify_action_map=_config_get(config, "unify_action_map", None),
            unify_state_map=_config_get(config, "unify_state_map", None),
            color_jitter=_config_get(config, "color_jitter", None),
        )

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        calibration: Mapping[str, Any] | None = None,
        split: str = "train",
        embodiment: str = ROBODOJO_EMBODIMENT,
        variant: str = ROBODOJO_SIM_VARIANT,
        normalization_stats_path: str | Path | None = None,
        action_mode: str = DEPLOY_ACTION_MODE,
        robot: str | None = None,
        **dataset_kwargs,
    ):
        super().__init__()
        if dataset_dir is None:
            raise ValueError("RoboDojo dataset_dir is required")
        if "calibration_path" in dataset_kwargs:
            _reject_calibration_path(dataset_kwargs.pop("calibration_path"))

        validate_embodiment(embodiment, variant=variant)
        if variant == ROBODOJO_REAL_VARIANT and calibration is not None:
            raise ValueError("RoboDojo_real uses native per-arm base poses; calibration must be None")

        selected_tasks = resolve_robodojo_tasks(
            dataset_dir,
            embodiment=embodiment,
            variant=variant,
        )
        self.dataset_dir = str(Path(dataset_dir))
        self.tasks = selected_tasks
        self.split = split
        self.action_mode = action_mode
        self.embodiment = embodiment
        self.variant = variant
        if variant == ROBODOJO_REAL_VARIANT:
            self.calibration = None
            self.frame_contract = robodojo_real_frame_contract(embodiment)
            self.frame_contract_fingerprint = real_frame_contract_fingerprint(embodiment)
            self.calibration_fingerprint = self.frame_contract_fingerprint
        else:
            self.calibration = resolve_robodojo_calibration(calibration)
            self.frame_contract = None
            self.frame_contract_fingerprint = calibration_fingerprint(self.calibration)
            self.calibration_fingerprint = self.frame_contract_fingerprint
        requested_normalize_mode = _normalize_mode(dataset_kwargs.get("normalize_mode", "min-max"))
        normalization_stats_tasks = None
        if requested_normalize_mode is not None and normalization_stats_path is None:
            normalization_stats_path = ensure_robodojo_stats(
                dataset_dir,
                variant=variant,
                embodiment=embodiment,
            )
        self.normalization_stats_path = (
            str(Path(normalization_stats_path)) if normalization_stats_path is not None else None
        )

        self._sub_datasets: list[RoboDojoDataset] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0
        for task in selected_tasks:
            episodes = discover_episodes(
                dataset_dir,
                task,
                embodiment=embodiment,
                variant=variant,
            )
            dataset = RoboDojoDataset(
                data_root=episodes[0].parent,
                dataset_root=dataset_dir,
                calibration=calibration,
                task_name=task,
                split=split,
                embodiment=embodiment,
                variant=variant,
                robot=robot,
                action_mode=action_mode,
                normalization_stats_path=normalization_stats_path,
                normalization_stats_tasks=normalization_stats_tasks,
                **dataset_kwargs,
            )
            self._sub_datasets.append(dataset)
            cumulative += len(dataset)
            self._cumulative_lengths.append(cumulative)
        if not self._sub_datasets or cumulative == 0:
            raise ValueError("RoboDojo task discovery produced an empty dataset")

        self._total_length = cumulative
        first = self._sub_datasets[0]
        self._action_dim_value = first.action_dim
        self.calibration = first.calibration
        self.calibration_fingerprint = first.calibration_fingerprint
        self.frame_contract = first.frame_contract
        self.frame_contract_fingerprint = first.frame_contract_fingerprint
        self.source_frame = first.source_frame
        self.contract_id = first.contract_id
        self._normalization_stats_shared = first.normalization_stats
        # Keep the resolved active path, which is None when normalization is off.
        self.normalization_stats_path = first.normalization_stats_path

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self) -> dict[str, np.ndarray] | None:
        if self._normalization_stats_shared is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._normalization_stats_shared.items()}

    def denormalize_action(self, action) -> np.ndarray:
        return self._sub_datasets[0].denormalize_action(action)

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self._cumulative_lengths, index)
        previous = 0 if dataset_index == 0 else self._cumulative_lengths[dataset_index - 1]
        return self._sub_datasets[dataset_index][index - previous]


__all__ = [
    "DEFAULT_ROBODOJO_CAMERA_LAYOUT",
    "DEPLOY_ACTION_MODE",
    "GRIPPER_CONVENTION",
    "MultiTaskRoboDojoDataset",
    "ROBODOJO_CONTRACT_ID",
    "ROBODOJO_REAL_CONTRACT_ID",
    "ROBODOJO_REAL_SOURCE_FRAME",
    "ROBODOJO_REAL_VARIANT",
    "ROBODOJO_SIM_VARIANT",
    "ROBODOJO_SOURCE_FRAME",
    "RoboDojoDataset",
    "calibration_fingerprint",
    "default_robodojo_stats_path",
    "discover_robodojo_tasks",
    "ensure_robodojo_stats",
    "read_calibrated_eef20",
    "real_frame_contract_fingerprint",
    "resolve_robodojo_tasks",
    "validate_robodojo_episode",
]
