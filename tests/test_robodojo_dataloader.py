"""Contract tests for the formal multi-task RoboDojo HDF5 reader."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.robodojo import (
    DEFAULT_ROBODOJO_CAMERA_LAYOUT,
    DEPLOY_ACTION_MODE,
    GRIPPER_CONVENTION,
    MultiTaskRoboDojoDataset,
    RoboDojoDataset,
    calibration_fingerprint,
    read_calibrated_eef20,
)
from openwam.dataloader.robodojo_contract import save_calibration
from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.poses import (
    arms_to_eef20,
    env_relative_world_to_robot_base,
)


def valid_calibration() -> dict:
    half_sqrt = 2**-0.5
    return {
        "schema_version": 1,
        "embodiment": "arx_x5",
        "endpoint": {
            "link_name": "link6",
            "pose_frame_contract": "rigid_terminal_arm_frame_independent_of_gripper_motion",
        },
        "arms": {
            "left": {
                "base_pos_relative_to_env_origin": [-0.3, -0.45, 0.765],
                "base_quat_wxyz": [half_sqrt, 0.0, 0.0, half_sqrt],
            },
            "right": {
                "base_pos_relative_to_env_origin": [0.3, -0.45, 0.765],
                "base_quat_wxyz": [half_sqrt, 0.0, 0.0, half_sqrt],
            },
        },
    }


def write_calibration(root: Path) -> Path:
    path = root / "calibration.json"
    save_calibration(valid_calibration(), path)
    return path


def encode_jpeg(color: tuple[int, int, int], height: int = 18, width: int = 20) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


def source_arrays(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_sqrt = 2**-0.5
    quaternions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
            [half_sqrt, half_sqrt, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    left = np.empty((T, 7), dtype=np.float64)
    right = np.empty((T, 7), dtype=np.float64)
    for index in range(T):
        left[index, :3] = [-0.1 + 0.05 * index, -0.2 + 0.02 * index, 0.9 + 0.01 * index]
        right[index, :3] = [0.2 - 0.03 * index, -0.1 + 0.04 * index, 1.0 - 0.02 * index]
        left[index, 3:] = quaternions[index % len(quaternions)]
        right[index, 3:] = quaternions[(index + 1) % len(quaternions)]
    left_grip = np.linspace(0.05, 0.95, T, dtype=np.float64)[:, None]
    right_grip = np.linspace(0.9, 0.1, T, dtype=np.float64)[:, None]
    return left, left_grip, right, right_grip


def formal_data_dir(
    root: Path,
    task: str,
    embodiment: str = "arx_x5",
) -> Path:
    data = root / task / embodiment / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data


def write_episode(
    root: Path,
    task: str = "pick_mug",
    *,
    episode: int = 0,
    T: int = 5,
    jpeg_storage: str = "vlen",
    instruction: str | bytes = "Pick up the mug.",
    embodiment: str = "arx_x5",
    mutate=None,
) -> Path:
    data = formal_data_dir(root, task, embodiment)
    path = data / f"episode_{episode:04d}.hdf5"
    left, left_grip, right, right_grip = source_arrays(T)
    arrays = {
        "state/left_ee_poses": left,
        "state/right_ee_poses": right,
        "state/left_ee_joint_states": left_grip,
        "state/right_ee_joint_states": right_grip,
    }
    if mutate is not None:
        mutate(arrays)

    camera_bytes = {
        "vision/cam_head/colors": [encode_jpeg((240 - i, 10 + i, 20)) for i in range(T)],
        "vision/cam_left_wrist/colors": [encode_jpeg((10, 230 - i, 20 + i)) for i in range(T)],
        "vision/cam_right_wrist/colors": [encode_jpeg((10 + i, 20, 220 - i)) for i in range(T)],
    }
    with h5py.File(path, "w") as handle:
        for key, value in arrays.items():
            handle.create_dataset(key, data=value)
        for key, values in camera_bytes.items():
            if jpeg_storage == "vlen":
                dataset = handle.create_dataset(
                    key,
                    shape=(T,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                for index, value in enumerate(values):
                    dataset[index] = np.frombuffer(value, dtype=np.uint8)
            elif jpeg_storage == "fixed":
                width = max(len(value) for value in values)
                handle.create_dataset(key, data=np.asarray(values, dtype=f"S{width}"))
            elif jpeg_storage == "padded":
                width = max(len(value) for value in values)
                padded = np.zeros((T, width), dtype=np.uint8)
                for index, value in enumerate(values):
                    encoded = np.frombuffer(value, dtype=np.uint8)
                    padded[index, : encoded.size] = encoded
                handle.create_dataset(key, data=padded)
            else:
                raise ValueError(jpeg_storage)
        handle.create_dataset("instruction", data=instruction)
        # This source exists in observed data but must not be used for EEF labels.
        handle.create_dataset("action/left_arm_joint_states", data=np.full((T, 6), 999.0))
        handle.create_dataset("action/right_arm_joint_states", data=np.full((T, 6), -999.0))
    return path


def expected_raw_eef20(T: int, calibration: dict | None = None) -> np.ndarray:
    calibration = calibration or valid_calibration()
    left, left_grip, right, right_grip = source_arrays(T)
    left_cal = calibration["arms"]["left"]
    right_cal = calibration["arms"]["right"]
    left_base = env_relative_world_to_robot_base(
        left,
        left_cal["base_pos_relative_to_env_origin"],
        left_cal["base_quat_wxyz"],
    )
    right_base = env_relative_world_to_robot_base(
        right,
        right_cal["base_pos_relative_to_env_origin"],
        right_cal["base_quat_wxyz"],
    )
    return arms_to_eef20(left_base, left_grip, right_base, right_grip).astype(np.float32)


def flat_stats() -> dict:
    stats = {
        "mean": np.linspace(-0.25, 0.25, 20, dtype=np.float32),
        "std": np.linspace(0.5, 1.5, 20, dtype=np.float32),
        "min": np.linspace(-2.0, -1.0, 20, dtype=np.float32),
        "max": np.linspace(1.0, 2.0, 20, dtype=np.float32),
        "q01": np.linspace(-1.8, -0.8, 20, dtype=np.float32),
        "q99": np.linspace(0.8, 1.8, 20, dtype=np.float32),
    }
    rot = np.asarray(ROT6D_DIMS_EEF20)
    stats["mean"][rot] = 0.0
    stats["std"][rot] = 1.0
    stats["min"][rot] = -1.0
    stats["max"][rot] = 1.0
    stats["q01"][rot] = -1.0
    stats["q99"][rot] = 1.0
    return stats


def write_stats(path: Path, calibration: dict | None = None) -> Path:
    calibration = calibration or valid_calibration()
    np.save(
        path,
        {
            "eef": flat_stats(),
            "metadata": {
                "pool": "action_state",
                "action_rows": 4,
                "state_rows": 5,
                "source_frame": "env_origin_relative_position_world_orientation_wxyz",
                "endpoint": "link6",
                "embodiment": "arx_x5",
                "calibration_fingerprint": calibration_fingerprint(calibration),
                "contract_id": "robodojo-eef20-v1",
                "gripper_convention": GRIPPER_CONVENTION,
            },
        },
    )
    return path


def build_single(
    root: Path,
    *,
    task: str = "pick_mug",
    normalize_mode=None,
    normalization_stats_path: Path | None = None,
    **kwargs,
) -> RoboDojoDataset:
    embodiment = kwargs.pop("embodiment", "arx_x5")
    return RoboDojoDataset(
        data_root=formal_data_dir(root, task, embodiment),
        dataset_root=root,
        task_name=task,
        embodiment=embodiment,
        num_frames=kwargs.pop("num_frames", 5),
        height=kwargs.pop("height", 48),
        width=kwargs.pop("width", 64),
        video_stride=kwargs.pop("video_stride", 1),
        normalize_mode=normalize_mode,
        normalization_stats_path=normalization_stats_path,
        unify_action=kwargs.pop("unify_action", False),
        **kwargs,
    )


def test_formal_reader_applies_live_calibration_and_state_t_plus_one_targets(
    tmp_path: Path,
):
    episode = write_episode(tmp_path, T=5, jpeg_storage="fixed")
    dataset = build_single(tmp_path)

    with h5py.File(episode, "r") as handle:
        shared_raw = read_calibrated_eef20(handle, dataset.calibration)
    expected = expected_raw_eef20(5)
    np.testing.assert_array_equal(shared_raw, expected)

    sample = dataset[0]
    assert DEPLOY_ACTION_MODE == "eef"
    assert dataset.DEPLOY_ACTION_MODE == "eef"
    assert dataset.action_mode == "eef"
    assert dataset.action_dim == 20
    np.testing.assert_array_equal(sample["proprio"].numpy(), expected[0:1])
    np.testing.assert_array_equal(sample["action"].numpy(), expected[1:5])
    assert sample["action"].shape == (4, 20)
    assert sample["proprio"].shape == (1, 20)
    assert sample["action_mask"].shape == (4, 20)
    assert sample["proprio_mask"].shape == (1, 20)
    assert sample["action_mask"].all()
    assert sample["proprio_mask"].all()
    assert sample["episode_path"] == str(episode)
    assert sample["task_name"] == "pick_mug"
    assert sample["active_arm"] == "both"


@pytest.mark.parametrize("embodiment", ["arx_x5", "piper", "piper_x"])
def test_real_reader_preserves_native_pose_and_clips_only_gripper_sensor_noise(
    tmp_path: Path,
    embodiment: str,
):
    noise = -0.04 if embodiment != "arx_x5" else -0.01
    episode = write_episode(
        tmp_path,
        T=4,
        embodiment=embodiment,
        mutate=lambda arrays: arrays["state/left_ee_joint_states"].__setitem__((0, 0), noise),
    )
    dataset = build_single(
        tmp_path,
        num_frames=4,
        variant="real",
        embodiment=embodiment,
    )

    left, left_grip, right, right_grip = source_arrays(4)
    left_grip[0, 0] = 0.0
    expected = arms_to_eef20(left, left_grip, right, right_grip).astype(np.float32)
    with h5py.File(episode, "r") as handle:
        raw = read_calibrated_eef20(
            handle,
            None,
            variant="real",
            embodiment=embodiment,
        )

    assert dataset.calibration is None
    assert dataset.variant == "real"
    assert dataset.source_frame == ("per_arm_robot_base_position_and_orientation_wxyz")
    np.testing.assert_array_equal(raw, expected)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["proprio"].numpy(), expected[0:1])
    np.testing.assert_array_equal(sample["action"].numpy(), expected[1:])
    assert sample["variant"] == "real"
    assert sample["embodiment"] == embodiment


@pytest.mark.parametrize("jpeg_storage", ["fixed", "vlen", "padded"])
def test_instruction_and_three_camera_l_shape_support(tmp_path: Path, jpeg_storage: str):
    write_episode(
        tmp_path,
        T=3,
        jpeg_storage=jpeg_storage,
        instruction=b"Move the blue block.",
    )
    dataset = build_single(tmp_path, num_frames=3, video_stride=1)
    sample = dataset[0]

    assert DEFAULT_ROBODOJO_CAMERA_LAYOUT == (
        "cam_head",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert sample["prompt"] == format_prompt_for_inference("Move the blue block.")
    assert len(sample["video"]) == 3
    assert sample["video_mask"].tolist() == [True, True, True]
    assert all(frame.size == (64, 48) for frame in sample["video"])
    assert sample["first_frame_image"][0] is sample["video"][0]
    # Verify the helper's top / bottom-left / bottom-right placement.
    frame = np.asarray(sample["video"][0])
    assert frame[5, 32, 0] > 200
    assert frame[42, 8, 1] > 200
    assert frame[42, 56, 2] > 190


def test_color_jitter_is_configured_for_train_only_and_updates_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_episode(tmp_path, T=3)
    jitter_config = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.0,
    }
    clean = build_single(tmp_path, num_frames=3, color_jitter=None)
    train = build_single(tmp_path, num_frames=3, color_jitter=jitter_config)
    validation = build_single(
        tmp_path,
        num_frames=3,
        color_jitter=jitter_config,
        split="val",
    )

    assert isinstance(train._color_jitter, VideoColorJitter)
    assert train._color_jitter.brightness == 0.2
    assert clean._color_jitter is None
    assert validation._color_jitter is None

    # Fixed factors make the pixel-level assertion deterministic while still
    # exercising the production transform on the complete assembled clip.
    monkeypatch.setattr("random.uniform", lambda lower, upper: upper)
    clean_sample = clean[0]
    jittered_sample = train[0]
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(clean_sample["video"], jittered_sample["video"])
    )
    assert jittered_sample["first_frame_image"][0] is jittered_sample["video"][0]


def test_from_config_threads_color_jitter_to_all_tasks(tmp_path: Path):
    for task in ("task_a", "task_b"):
        write_episode(tmp_path, task=task, T=3)
    config = {
        "dataset_dir": str(tmp_path),
        "normalize_mode": None,
        "num_frames": 3,
        "height": 48,
        "width": 64,
        "video_stride": 1,
        "unify_action": False,
        "color_jitter": {
            "brightness": 0.1,
            "contrast": 0.2,
            "saturation": 0.3,
            "hue": 0.0,
        },
    }

    dataset = MultiTaskRoboDojoDataset.from_config(config, split="train")
    assert all(isinstance(task_dataset._color_jitter, VideoColorJitter) for task_dataset in dataset._sub_datasets)
    assert all(task_dataset._color_jitter.saturation == 0.3 for task_dataset in dataset._sub_datasets)


def test_short_and_tail_windows_repeat_last_values_and_mask_padding(tmp_path: Path):
    write_episode(tmp_path, T=5)
    dataset = build_single(tmp_path, num_frames=7, video_stride=2)

    assert len(dataset) == 4
    assert dataset._window_index == [(0, 0), (0, 1), (0, 2), (0, 3)]
    first = dataset[0]
    assert first["action"].shape == (6, 20)
    assert first["video_mask"].tolist() == [True, True, True, False]
    assert first["action_mask"].any(dim=-1).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    np.testing.assert_array_equal(
        first["action"][4:].numpy(),
        np.repeat(first["action"][3:4].numpy(), 2, axis=0),
    )

    tail = dataset[len(dataset) - 1]
    assert tail["start_frame"] == 3
    assert tail["action_mask"].any(dim=-1).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert tail["video_mask"].tolist() == [True, False, False, False]
    np.testing.assert_array_equal(
        tail["action"].numpy(),
        np.repeat(tail["action"][0:1].numpy(), 6, axis=0),
    )


def test_normalize_raw20_then_scatter_to_80_and_inverse_denormalize(
    tmp_path: Path,
):
    write_episode(tmp_path, T=4)
    stats_path = write_stats(tmp_path / "stats.npy")
    dataset = RoboDojoDataset(
        data_root=formal_data_dir(tmp_path, "pick_mug"),
        dataset_root=tmp_path,
        task_name="pick_mug",
        normalization_stats_path=stats_path,
        normalize_mode="z-score",
        num_frames=4,
        height=48,
        width=64,
        video_stride=1,
        unify_action=True,
        unify_action_map=["0-9", "34-43"],
    )
    sample = dataset[0]
    raw = expected_raw_eef20(4)
    stats = flat_stats()
    normalized = ((raw - stats["mean"]) / stats["std"]).astype(np.float32)
    mapped = np.r_[0:10, 34:44]
    unmapped = np.setdiff1d(np.arange(80), mapped)

    assert dataset.action_dim == 80
    assert dataset.normalization_stats_path == str(stats_path)
    assert dataset.normalization_stats is not None
    np.testing.assert_allclose(sample["proprio"].numpy()[0, mapped], normalized[0], atol=1e-6)
    np.testing.assert_allclose(sample["action"].numpy()[:, mapped], normalized[1:], atol=1e-6)
    np.testing.assert_array_equal(sample["action"].numpy()[:, unmapped], 0.0)
    assert sample["action_mask"][:, mapped].all()
    assert not sample["action_mask"][:, unmapped].any()
    assert sample["proprio_mask"][:, mapped].all()
    assert not sample["proprio_mask"][:, unmapped].any()
    np.testing.assert_allclose(
        dataset.denormalize_action(sample["action"].numpy()),
        raw[1:],
        atol=2e-6,
    )


def test_unification_requires_explicit_complete_map(tmp_path: Path):
    write_episode(tmp_path, T=3)
    common = {
        "data_root": formal_data_dir(tmp_path, "pick_mug"),
        "dataset_root": tmp_path,
        "task_name": "pick_mug",
        "normalize_mode": None,
        "num_frames": 3,
        "height": 48,
        "width": 64,
    }
    with pytest.raises(ValueError, match="explicit unify_action_map"):
        RoboDojoDataset(**common, unify_action=True)
    with pytest.raises(ValueError, match="20"):
        RoboDojoDataset(
            **common,
            unify_action=True,
            unify_action_map=["0-9"],
        )
    with pytest.raises(ValueError, match=r"canonical.*0-9.*34-43"):
        RoboDojoDataset(
            **common,
            unify_action=True,
            unify_action_map=["1-20"],
        )


def test_normalization_auto_generates_or_validates_explicit_stats(
    tmp_path: Path,
):
    write_episode(tmp_path, T=3)
    common = {
        "data_root": formal_data_dir(tmp_path, "pick_mug"),
        "dataset_root": tmp_path,
        "task_name": "pick_mug",
        "normalize_mode": "min-max",
        "num_frames": 3,
        "height": 48,
        "width": 64,
        "unify_action": False,
    }
    automatic = RoboDojoDataset(**common)
    expected_automatic = tmp_path / "meta" / "robodojo_normalization_stats.npy"
    assert automatic.normalization_stats_path == str(expected_automatic)
    assert expected_automatic.is_file()
    # A second construction discovers the complete file instead of recomputing.
    assert RoboDojoDataset(**common).normalization_stats_path == str(expected_automatic)
    with pytest.raises(FileNotFoundError, match="normalization"):
        RoboDojoDataset(**common, normalization_stats_path=tmp_path / "missing.npy")

    malformed = write_stats(tmp_path / "malformed.npy")
    payload = np.load(malformed, allow_pickle=True).item()
    payload["eef"]["mean"] = np.zeros(19)
    np.save(malformed, payload)
    with pytest.raises(ValueError, match=r"mean.*\(20,\)"):
        RoboDojoDataset(**common, normalization_stats_path=malformed)

    other_calibration = valid_calibration()
    other_calibration["arms"]["left"]["base_pos_relative_to_env_origin"][0] += 0.1
    mismatch = write_stats(tmp_path / "mismatch.npy", other_calibration)
    with pytest.raises(ValueError, match="calibration fingerprint mismatch"):
        RoboDojoDataset(**common, normalization_stats_path=mismatch)

    stale_rot6d = write_stats(tmp_path / "stale_rot6d.npy")
    payload = np.load(stale_rot6d, allow_pickle=True).item()
    payload["eef"]["mean"][3] = 0.25
    np.save(stale_rot6d, payload)
    with pytest.raises(ValueError, match=r"rot6d.*regenerate"):
        RoboDojoDataset(**common, normalization_stats_path=stale_rot6d)

    flipped = write_stats(tmp_path / "flipped.npy")
    payload = np.load(flipped, allow_pickle=True).item()
    payload["metadata"]["gripper_convention"] = "one_closed_zero_open"
    np.save(flipped, payload)
    with pytest.raises(ValueError, match="gripper_convention"):
        RoboDojoDataset(**common, normalization_stats_path=flipped)

    missing_convention = write_stats(tmp_path / "missing_convention.npy")
    payload = np.load(missing_convention, allow_pickle=True).item()
    del payload["metadata"]["gripper_convention"]
    np.save(missing_convention, payload)
    with pytest.raises(ValueError, match="gripper_convention"):
        RoboDojoDataset(**common, normalization_stats_path=missing_convention)

    wrong_contract = write_stats(tmp_path / "wrong_contract.npy")
    payload = np.load(wrong_contract, allow_pickle=True).item()
    payload["metadata"]["contract_id"] = "robodojo-eef20-v0"
    np.save(wrong_contract, payload)
    with pytest.raises(ValueError, match="contract_id"):
        RoboDojoDataset(**common, normalization_stats_path=wrong_contract)

    missing_contract = write_stats(tmp_path / "missing_contract.npy")
    payload = np.load(missing_contract, allow_pickle=True).item()
    del payload["metadata"]["contract_id"]
    np.save(missing_contract, payload)
    with pytest.raises(ValueError, match="contract_id"):
        RoboDojoDataset(**common, normalization_stats_path=missing_contract)


def test_calibration_path_is_rejected(tmp_path: Path):
    write_episode(tmp_path, T=3)
    with pytest.raises(ValueError, match="calibration_path is not accepted"):
        RoboDojoDataset(
            data_root=formal_data_dir(tmp_path, "pick_mug"),
            dataset_root=tmp_path,
            task_name="pick_mug",
            calibration_path=tmp_path / "calibration.json",
            normalize_mode=None,
            unify_action=False,
            num_frames=3,
            height=48,
            width=64,
        )
    with pytest.raises(ValueError, match="calibration_path is not accepted"):
        MultiTaskRoboDojoDataset(
            dataset_dir=tmp_path,
            calibration_path=tmp_path / "calibration.json",
            normalize_mode=None,
            unify_action=False,
            num_frames=3,
            height=48,
            width=64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda arrays: arrays.pop("state/right_ee_poses"),
            "state/right_ee_poses",
        ),
        (
            lambda arrays: arrays.__setitem__("state/left_ee_poses", arrays["state/left_ee_poses"][:, :6]),
            r"left_ee_poses.*\(T, 7\)",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "state/right_ee_joint_states",
                arrays["state/right_ee_joint_states"][:-1],
            ),
            "equal non-zero T",
        ),
        (
            lambda arrays: arrays["state/left_ee_poses"].__setitem__((0, 0), np.nan),
            "finite",
        ),
        (
            lambda arrays: arrays["state/right_ee_poses"].__setitem__((0, slice(3, 7)), [2.0, 0.0, 0.0, 0.0]),
            "unit wxyz",
        ),
        (
            lambda arrays: arrays["state/left_ee_joint_states"].__setitem__((0, 0), 1.1),
            r"\[0, 1\]",
        ),
    ],
)
def test_episode_schema_is_validated_before_indexing(tmp_path: Path, mutation, message: str):
    write_episode(tmp_path, T=3, mutate=mutation)
    with pytest.raises((KeyError, ValueError), match=message):
        build_single(tmp_path, num_frames=3)


def test_official_closed_gripper_float_noise_is_accepted_and_clipped(tmp_path: Path):
    noise = -3.21e-17
    path = write_episode(
        tmp_path,
        T=3,
        mutate=lambda arrays: arrays["state/left_ee_joint_states"].__setitem__((0, 0), noise),
    )
    dataset = build_single(tmp_path, num_frames=3)
    raw = read_calibrated_eef20(path, valid_calibration())
    assert raw[0, 9] == 0.0
    sample = dataset[0]
    assert float(sample["proprio"].numpy()[0, 9]) == 0.0


def test_episode_schema_rejects_single_frame_empty_instruction_and_camera_mismatch(
    tmp_path: Path,
):
    write_episode(tmp_path / "single", T=1)
    with pytest.raises(ValueError, match="at least two frames"):
        build_single(tmp_path / "single", num_frames=2)

    write_episode(tmp_path / "empty", T=3, instruction=b"   ")
    with pytest.raises(ValueError, match="instruction.*empty"):
        build_single(tmp_path / "empty", num_frames=3)

    path = write_episode(tmp_path / "camera", T=3)
    with h5py.File(path, "a") as handle:
        del handle["vision/cam_head/colors"]
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        dataset = handle.create_dataset("vision/cam_head/colors", shape=(2,), dtype=dtype)
        jpeg = np.frombuffer(encode_jpeg((1, 2, 3)), dtype=np.uint8)
        dataset[0] = jpeg
        dataset[1] = jpeg
    with pytest.raises(ValueError, match="equal non-zero T"):
        build_single(tmp_path / "camera", num_frames=3)


def test_episode_schema_rejects_non_jpeg_camera_storage_dtype(tmp_path: Path):
    path = write_episode(tmp_path, T=3)
    with h5py.File(path, "a") as handle:
        del handle["vision/cam_head/colors"]
        handle.create_dataset(
            "vision/cam_head/colors",
            data=np.arange(3, dtype=np.uint8),
        )
    with pytest.raises(ValueError, match=r"fixed byte-string.*vlen/padded uint8"):
        build_single(tmp_path, num_frames=3)


def test_episode_schema_rejects_two_dimensional_non_uint8_camera_storage(
    tmp_path: Path,
):
    path = write_episode(tmp_path, T=3)
    with h5py.File(path, "a") as handle:
        del handle["vision/cam_head/colors"]
        handle.create_dataset(
            "vision/cam_head/colors",
            data=np.zeros((3, 32), dtype=np.int16),
        )
    with pytest.raises(ValueError, match=r"fixed/vlen JPEG entries.*padded uint8"):
        build_single(tmp_path, num_frames=3)


def test_direct_reader_requires_verified_formal_dataset_root(tmp_path: Path):
    write_episode(tmp_path, T=3)
    formal = RoboDojoDataset(
        data_root=formal_data_dir(tmp_path, "pick_mug"),
        dataset_root=tmp_path,
        task_name="pick_mug",
        normalize_mode=None,
        unify_action=False,
        num_frames=3,
        height=48,
        width=64,
    )
    assert len(formal) == 2

    flat_root = tmp_path / "flat"
    flat_data = flat_root / "arx_x5" / "data"
    flat_data.mkdir(parents=True)
    with pytest.raises(ValueError, match=r"formal.*dataset_root.*task_name"):
        RoboDojoDataset(
            data_root=flat_data,
            dataset_root=flat_root,
            task_name="pick_mug",
            normalize_mode=None,
            unify_action=False,
            num_frames=3,
            height=48,
            width=64,
        )


def test_only_eef_supported_embodiment_num_frames_and_formal_layout_are_accepted(
    tmp_path: Path,
):
    write_episode(tmp_path, T=3)
    data_root = formal_data_dir(tmp_path, "pick_mug")
    common = {
        "data_root": data_root,
        "dataset_root": tmp_path,
        "task_name": "pick_mug",
        "normalize_mode": None,
        "unify_action": False,
        "height": 48,
        "width": 64,
    }
    with pytest.raises(ValueError, match="action_mode='eef'"):
        RoboDojoDataset(**common, action_mode="joint")
    with pytest.raises(ValueError, match="arx_x5"):
        RoboDojoDataset(**common, embodiment="franka")
    with pytest.raises(ValueError, match="num_frames.*>= 2"):
        RoboDojoDataset(**common, num_frames=1)

    flat = tmp_path / "flat"
    flat_data = flat / "arx_x5" / "data"
    flat_data.mkdir(parents=True)
    (flat_data / "episode_0000.hdf5").write_bytes(b"not-hdf5")
    with pytest.raises(ValueError, match="flat.*not supported"):
        MultiTaskRoboDojoDataset(
            dataset_dir=flat,
            normalize_mode=None,
            unify_action=False,
            num_frames=3,
            height=48,
            width=64,
        )


def test_multitask_discovers_all_tasks_sorted_and_shares_files(tmp_path: Path):
    for index, task in enumerate(("task_b", "task_a", "task_holdout")):
        write_episode(tmp_path, task=task, episode=index, T=3)
    stats = write_stats(tmp_path / "stats.npy")
    common = {
        "dataset_dir": tmp_path,
        "normalization_stats_path": stats,
        "normalize_mode": "min-max",
        "num_frames": 3,
        "height": 48,
        "width": 64,
        "video_stride": 1,
        "unify_action": True,
        "unify_action_map": ["0-9", "34-43"],
    }

    train = MultiTaskRoboDojoDataset(
        **common,
        split="train",
    )
    assert [dataset.task_name for dataset in train._sub_datasets] == [
        "task_a",
        "task_b",
        "task_holdout",
    ]
    assert all(dataset.normalization_stats_path == str(stats) for dataset in train._sub_datasets)
    assert all(dataset.calibration_fingerprint == train.calibration_fingerprint for dataset in train._sub_datasets)
    assert train.action_dim == 80
    np.testing.assert_allclose(
        train.denormalize_action(train[0]["action"].numpy()),
        expected_raw_eef20(3)[1:],
        atol=2e-6,
    )


def test_multitask_auto_stats_use_meta_and_cover_the_discovered_corpus(
    tmp_path: Path,
):
    for task in ("task_b", "task_a"):
        write_episode(tmp_path, task=task, T=3)

    dataset = MultiTaskRoboDojoDataset(
        dataset_dir=tmp_path,
        normalize_mode="min-max",
        num_frames=3,
        height=48,
        width=64,
        video_stride=1,
        unify_action=False,
    )
    expected = tmp_path / "meta" / "robodojo_normalization_stats.npy"
    assert dataset.normalization_stats_path == str(expected)
    assert all(sub_dataset.normalization_stats_path == str(expected) for sub_dataset in dataset._sub_datasets)
    payload = np.load(expected, allow_pickle=True).item()
    assert payload["metadata"]["tasks"] == ["task_a", "task_b"]


def test_registry_constructs_robodojo_and_exports_classes(tmp_path: Path):
    write_episode(tmp_path, T=3)
    config = {
        "type": "robodojo",
        "dataset_dir": str(tmp_path),
        "embodiment": "arx_x5",
        "action_mode": "eef",
        "normalization_stats_path": None,
        "normalize_mode": None,
        "num_frames": 3,
        "height": 48,
        "width": 64,
        "video_stride": 1,
        "multiview": True,
        "camera_layout": [
            "cam_head",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
        "unify_action": True,
        "unify_action_map": ["0-9", "34-43"],
    }
    dataset = build_dataset(config, split="train")
    assert isinstance(dataset, MultiTaskRoboDojoDataset)
    assert isinstance(dataset._sub_datasets[0], RoboDojoDataset)
    assert dataset[0]["action"].shape == (2, 80)
