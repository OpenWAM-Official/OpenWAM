"""Tests for bounded pooled raw-EEF20 RoboDojo normalization statistics."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from openwam.dataloader.robodojo import (
    GRIPPER_CONVENTION,
    RoboDojoDataset,
    calibration_fingerprint,
    read_calibrated_eef20,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.stats_computation.robodojo_stats_computation import (
    atomic_save_stats_npy,
    build_and_save_robodojo_stats,
    compute_robodojo_stats,
    iter_episode_eef20,
    main,
)
from openwam.dataloader.robodojo_contract import arx_x5_calibration
from tests.test_robodojo_dataloader import (
    expected_raw_eef20,
    formal_data_dir,
    valid_calibration,
    write_episode,
)

STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def test_reader_and_stats_use_bit_identical_calibrated_raw_eef20(tmp_path: Path):
    episode = write_episode(tmp_path, T=6, jpeg_storage="vlen")
    calibration = arx_x5_calibration()
    dataset = RoboDojoDataset(
        data_root=formal_data_dir(tmp_path, "pick_mug"),
        dataset_root=tmp_path,
        task_name="pick_mug",
        normalize_mode=None,
        unify_action=False,
        num_frames=4,
        height=48,
        width=64,
    )

    with h5py.File(episode, "r") as handle:
        reader_rows = read_calibrated_eef20(handle, dataset.calibration)
    stats_rows = list(iter_episode_eef20([episode], calibration))

    assert len(stats_rows) == 1
    np.testing.assert_array_equal(stats_rows[0], reader_rows)
    np.testing.assert_array_equal(reader_rows, expected_raw_eef20(6))


def test_pools_all_states_and_only_real_next_state_targets_without_crossing_episodes(
    tmp_path: Path,
):
    write_episode(tmp_path, task="task_a", episode=0, T=4)
    write_episode(tmp_path, task="task_a", episode=1, T=3)

    payload = compute_robodojo_stats(
        dataset_dir=tmp_path,
        task_name="task_a",
        reservoir_cap=100,
    )

    eef = payload["eef"]
    metadata = payload["metadata"]
    assert metadata["pool"] == "action_state"
    assert metadata["state_rows"] == 7
    assert metadata["action_rows"] == 5
    assert metadata["num_timesteps"] == 12
    assert metadata["tasks"] == ["task_a"]
    assert metadata["calibration_fingerprint"] == calibration_fingerprint(
        valid_calibration()
    )
    assert metadata["source_frame"] == (
        "env_origin_relative_position_world_orientation_wxyz"
    )
    assert metadata["endpoint"] == "link6"
    assert metadata["embodiment"] == "arx_x5"
    assert metadata["contract_id"] == "robodojo-eef20-v1"
    assert metadata["gripper_convention"] == GRIPPER_CONVENTION

    episode_0 = expected_raw_eef20(4)
    episode_1 = expected_raw_eef20(3)
    expected_pool = np.concatenate(
        [episode_0, episode_0[1:], episode_1, episode_1[1:]], axis=0
    )
    ordinary_dims = np.setdiff1d(np.arange(20), ROT6D_DIMS_EEF20)
    np.testing.assert_allclose(
        np.asarray(eef["mean"])[ordinary_dims],
        expected_pool.mean(axis=0)[ordinary_dims],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(eef["min"])[ordinary_dims],
        expected_pool.min(axis=0)[ordinary_dims],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(eef["max"])[ordinary_dims],
        expected_pool.max(axis=0)[ordinary_dims],
        atol=1e-6,
    )
    for key in STAT_KEYS:
        assert np.asarray(eef[key]).shape == (20,)
        assert np.isfinite(eef[key]).all()


def test_rot6d_stats_are_pinned_to_identity_and_reservoir_is_bounded(
    tmp_path: Path,
):
    write_episode(tmp_path, T=20)
    payload = compute_robodojo_stats(
        dataset_dir=tmp_path,
        task_name="pick_mug",
        reservoir_cap=3,
    )
    eef = payload["eef"]
    rot = np.asarray(ROT6D_DIMS_EEF20)
    np.testing.assert_array_equal(np.asarray(eef["mean"])[rot], 0.0)
    np.testing.assert_array_equal(np.asarray(eef["std"])[rot], 1.0)
    np.testing.assert_array_equal(np.asarray(eef["min"])[rot], -1.0)
    np.testing.assert_array_equal(np.asarray(eef["max"])[rot], 1.0)
    np.testing.assert_array_equal(np.asarray(eef["q01"])[rot], -1.0)
    np.testing.assert_array_equal(np.asarray(eef["q99"])[rot], 1.0)
    assert payload["metadata"]["reservoir_cap"] == 3
    assert payload["metadata"]["reservoir_rows"] == 3


def test_stats_task_selection_matches_reader_train_and_holdout_rules(
    tmp_path: Path,
):
    for task in ("task_b", "task_a", "task_holdout"):
        write_episode(tmp_path, task=task, T=3)

    train = compute_robodojo_stats(
        dataset_dir=tmp_path,
        train_tasks=["task_b", "task_a", "task_holdout"],
        holdout_tasks=["task_holdout"],
        split="train",
        reservoir_cap=100,
    )
    assert train["metadata"]["tasks"] == ["task_a", "task_b"]
    assert train["metadata"]["state_rows"] == 6
    assert train["metadata"]["action_rows"] == 4

    validation = compute_robodojo_stats(
        dataset_dir=tmp_path,
        holdout_tasks=["task_holdout"],
        split="val",
        reservoir_cap=100,
    )
    assert validation["metadata"]["tasks"] == ["task_holdout"]
    assert validation["metadata"]["state_rows"] == 3
    assert validation["metadata"]["action_rows"] == 2

    with pytest.raises(FileNotFoundError, match="missing"):
        compute_robodojo_stats(
            dataset_dir=tmp_path,
            train_tasks=["task_a", "missing"],
        )


def test_atomic_build_writes_deploy_payload_and_leaves_no_partial_file(
    tmp_path: Path,
):
    write_episode(tmp_path, T=4)
    output = tmp_path / "stats.npy"
    np.save(output, {"stale": True})

    result = build_and_save_robodojo_stats(
        dataset_dir=tmp_path,
        output=output,
        task_name="pick_mug",
        reservoir_cap=100,
    )

    assert result == output
    loaded = np.load(output, allow_pickle=True).item()
    assert set(STAT_KEYS).issubset(loaded["eef"])
    assert loaded["metadata"]["action_rows"] == 3
    assert loaded["metadata"]["state_rows"] == 4
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert not list(tmp_path.glob(f"{output.name}.tmp*"))

    direct = tmp_path / "direct.npy"
    atomic_save_stats_npy(direct, loaded)
    assert np.load(direct, allow_pickle=True).item()["metadata"] == loaded[
        "metadata"
    ]


def test_generated_stats_are_compatible_with_generic_deploy_normalizer(
    tmp_path: Path,
):
    from openwam.deploy.model_loader import (
        _build_normalizer,
        _UnifyAwareNormalizer,
    )

    write_episode(tmp_path, T=4)
    output = tmp_path / "normalization_stats.npy"
    build_and_save_robodojo_stats(
        dataset_dir=tmp_path,
        output=output,
        task_name="pick_mug",
        reservoir_cap=100,
    )
    config = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef",
                "unify_action": True,
                "unify_action_map": ["0-9", "34-43"],
            }
        }
    )

    normalizer = _build_normalizer(config, str(tmp_path))
    assert isinstance(normalizer, _UnifyAwareNormalizer)
    assert normalizer._dst_index.tolist() == [
        *range(10),
        *range(34, 44),
    ]
    raw = expected_raw_eef20(4)
    unified = normalizer.normalize(raw)
    assert unified.shape == (4, 80)
    np.testing.assert_allclose(
        normalizer.unnormalize(unified),
        raw,
        atol=2e-6,
    )


def test_stats_reject_empty_data_and_non_npy_output(tmp_path: Path):
    with pytest.raises((FileNotFoundError, ValueError), match="RoboDojo|empty"):
        compute_robodojo_stats(
            dataset_dir=tmp_path,
            task_name="missing",
        )

    write_episode(tmp_path, T=3)
    with pytest.raises(ValueError, match=r"\.npy"):
        build_and_save_robodojo_stats(
            dataset_dir=tmp_path,
            output=tmp_path / "stats.json",
            task_name="pick_mug",
        )
    with pytest.raises(ValueError, match="calibration_path is not accepted"):
        compute_robodojo_stats(
            dataset_dir=tmp_path,
            calibration_path=tmp_path / "calibration.json",
            task_name="pick_mug",
        )


def test_cli_requires_npy_and_succeeds_with_normalization_disabled_for_scan(
    tmp_path: Path,
):
    write_episode(tmp_path, task="task_a", T=4, jpeg_storage="fixed")
    config = tmp_path / "robodojo.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "type": "robodojo",
                "dataset_dir": str(tmp_path),
                "task_name": "task_a",
                "train_tasks": None,
                "holdout_tasks": None,
                "split": "train",
                "embodiment": "arx_x5",
                "action_mode": "eef",
                # Deliberately nonexistent: a stats scan must not try to load it.
                "normalization_stats_path": str(tmp_path / "does-not-exist.npy"),
                "normalize_mode": "min-max",
                "unify_action": True,
                "unify_action_map": ["0-9", "34-43"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\.npy"):
        main(
            [
                "--config",
                str(config),
                "--output",
                str(tmp_path / "stats.json"),
            ]
        )

    output = tmp_path / "cli_stats.npy"
    assert (
        main(
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--reservoir-cap",
                "7",
            ]
        )
        == 0
    )
    loaded = np.load(output, allow_pickle=True).item()
    assert loaded["metadata"]["state_rows"] == 4
    assert loaded["metadata"]["action_rows"] == 3
    assert loaded["metadata"]["reservoir_cap"] == 7
