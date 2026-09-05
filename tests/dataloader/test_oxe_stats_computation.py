from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from openwam.dataloader.oxe_droid import (
    DROID_DATA_POPULATION_DIGEST_KEY,
    DROID_EEF_STATS_CONTRACT,
    DROID_EEF_STATS_CONTRACT_KEY,
    DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
    DROID_PROMPT_INPUTS_DIGEST_KEY,
    OxeDroidDataset,
    compute_droid_prompt_inputs_digest,
)
from openwam.dataloader.utils.lerobotv3 import (
    digest_lerobot_v3_data_population,
    resolve_lerobot_v3_data_population,
)
from openwam.dataloader.utils.stats_computation.oxe_stats_computation import compute_dataset_stats


def _write_manifest(root, episode_lengths):
    (root / "meta" / "episodes").mkdir()
    total_frames = sum(episode_lengths)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "total_frames": total_frames,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )
    starts = np.cumsum([0, *episode_lengths[:-1]]).tolist()
    pd.DataFrame(
        {
            "episode_index": list(range(len(episode_lengths))),
            "length": episode_lengths,
            "dataset_from_index": starts,
            "data/chunk_index": [0] * len(episode_lengths),
            "data/file_index": [0] * len(episode_lengths),
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000.parquet")


def _write_exclusions(root, canonical, prompt_owned=None, independently_owned=None):
    prompt_owned = prompt_owned or []
    independently_owned = independently_owned if independently_owned is not None else canonical
    fallback_chain = list(OxeDroidDataset.PROMPT_FALLBACK_COLS)
    total_frames = resolve_lerobot_v3_data_population(root).total_rows
    (root / "meta" / "excluded_episodes.json").write_text(
        json.dumps(
            {
                "episode_indices": canonical,
                "droid_prompt_exclusions": {
                    "schema_version": DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
                    "fallback_chain": fallback_chain,
                    "episode_indices": prompt_owned,
                    "independently_owned_episode_indices": independently_owned,
                    "latest_scan": {
                        "episode_indices": prompt_owned,
                        "stats": {
                            "rows_scanned": total_frames,
                            "unresolved_rows": 0,
                            "episodes_all_unresolved": len(prompt_owned),
                            "episodes_partially_unresolved": 0,
                            "task_index_missing_from_tasks_parquet": 0,
                            "fallback_chain": fallback_chain,
                        },
                        DROID_PROMPT_INPUTS_DIGEST_KEY: compute_droid_prompt_inputs_digest(root),
                    },
                },
            }
        )
    )


def test_droid_stats_exclude_reader_blacklist_population(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    n_rows = 60
    kept_state = np.zeros((n_rows, 7), dtype=np.float32)
    kept_state[:, :6] = 1_000.0  # Old task-TCP state pose: must not enter stats.
    kept_pose = np.zeros((n_rows, 6), dtype=np.float32)
    kept_pose[:, 0] = 1.0
    kept_action = np.zeros((n_rows, 7), dtype=np.float32)
    kept_action[:, 0] = 1.0
    kept_tcp_action = np.zeros((n_rows, 7), dtype=np.float32)
    kept_tcp_action[:, :6] = 2_000.0  # Old task-TCP action: must not enter stats.
    # Deliberately malformed 2-D poses: filtering must happen in Arrow before
    # numpy conversion, matching the reader's episode-level exclusion behavior.
    excluded_pose = np.zeros((n_rows, 2), dtype=np.float32)
    excluded_pose[:, 0] = 100.0
    excluded_state = np.zeros((n_rows, 7), dtype=np.float32)
    pd.DataFrame(
        {
            "episode_index": [0] * n_rows + [1] * n_rows,
            "task_index": [0] * (n_rows * 2),
            "state": list(kept_state) + list(excluded_state),
            "other_information.observation_gripper_pose6d": list(kept_pose) + list(excluded_pose),
            "other_information.action_wrist_pose": list(kept_action) + list(excluded_pose),
            "other_information.action_tcp_pose": list(kept_tcp_action) + list(excluded_pose),
            **{column: [""] * (n_rows * 2) for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [n_rows, n_rows])
    _write_exclusions(root, canonical=[1])

    stats, n_state, n_action = compute_dataset_stats(root, "DROID", rot6d_identity=False)

    assert n_state == n_rows
    assert n_action == n_rows
    assert stats["n_samples"] == n_rows * 2
    assert stats["max"][0] == 1.0
    assert stats["excluded_episode_indices"] == [1]
    assert stats[DROID_EEF_STATS_CONTRACT_KEY] == DROID_EEF_STATS_CONTRACT
    assert stats[DROID_DATA_POPULATION_DIGEST_KEY] == digest_lerobot_v3_data_population(
        resolve_lerobot_v3_data_population(root)
    )


def test_droid_stats_aggregate_canonical_openness_not_raw_closedness(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    state_values = np.zeros((4, 7), dtype=np.float32)
    state_values[:, :6] = 1_000.0  # Legacy task-TCP state pose sentinel.
    state_values[:, 6] = [0.1, 0.2, 0.3, 0.4]
    observation_pose_values = np.zeros((4, 6), dtype=np.float32)
    action_values = np.zeros((4, 7), dtype=np.float32)
    action_values[:, 6] = [0.15, 0.25, 0.35, 0.45]
    tcp_action_values = np.full((4, 7), 2_000.0, dtype=np.float32)
    pd.DataFrame(
        {
            "episode_index": [0] * 4,
            "task_index": [0] * 4,
            "state": list(state_values),
            "other_information.observation_gripper_pose6d": list(observation_pose_values),
            "other_information.action_wrist_pose": list(action_values),
            "other_information.action_tcp_pose": list(tcp_action_values),
            **{column: [""] * 4 for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [4])
    _write_exclusions(root, canonical=[])

    stats, _, _ = compute_dataset_stats(root, "DROID", rot6d_identity=False)

    assert stats["min"][0] == 0.0
    assert stats["max"][0] == 0.0
    assert stats["min"][9] == pytest.approx(0.55)
    assert stats["max"][9] == pytest.approx(0.9)
    assert stats["mean"][9] == pytest.approx(0.725)
    assert stats["q01"][9] == pytest.approx(0.5535)
    assert stats["q99"][9] == pytest.approx(0.8965)
    assert stats[DROID_EEF_STATS_CONTRACT_KEY] == DROID_EEF_STATS_CONTRACT


def test_droid_stats_fail_clearly_when_every_row_is_excluded(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    state_values = list(np.zeros((120, 7), dtype=np.float32))
    pose_values = list(np.zeros((120, 6), dtype=np.float32))
    action_values = list(np.zeros((120, 7), dtype=np.float32))
    pd.DataFrame(
        {
            "episode_index": [0] * 120,
            "task_index": [0] * 120,
            "state": state_values,
            "other_information.observation_gripper_pose6d": pose_values,
            "other_information.action_wrist_pose": action_values,
            **{column: [""] * 120 for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [120])
    _write_exclusions(root, canonical=[0])

    with pytest.raises(ValueError, match="every parquet row is excluded"):
        compute_dataset_stats(root, "DROID", rot6d_identity=False)


def test_droid_stats_ignore_unreferenced_backup_parquet(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    state_values = list(np.zeros((2, 7), dtype=np.float32))
    pose_values = list(np.zeros((2, 6), dtype=np.float32))
    action_values = list(np.zeros((2, 7), dtype=np.float32))
    frame = pd.DataFrame(
        {
            "episode_index": [0, 0],
            "task_index": [0, 0],
            "state": state_values,
            "other_information.observation_gripper_pose6d": pose_values,
            "other_information.action_wrist_pose": action_values,
            **{column: [""] * 2 for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    )
    frame.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    backup = frame.copy()
    backup["episode_index"] = 999
    backup.to_parquet(root / "data" / "chunk-000" / "backup.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [2])
    _write_exclusions(root, canonical=[])

    stats, n_state, n_action = compute_dataset_stats(root, "DROID", rot6d_identity=False)

    assert n_state == 2
    assert n_action == 2
    assert stats["n_samples"] == 4


def test_droid_stats_ignore_unaddressed_tail_rows_in_referenced_shard(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    valid_state = np.zeros(7, dtype=np.float32)
    valid_pose = np.zeros(6, dtype=np.float32)
    valid_action = np.zeros(7, dtype=np.float32)
    malformed_tail = np.zeros(2, dtype=np.float32)
    pd.DataFrame(
        {
            "episode_index": [0, 999],
            "task_index": [0, 0],
            "state": [valid_state, malformed_tail],
            "other_information.observation_gripper_pose6d": [valid_pose, malformed_tail],
            "other_information.action_wrist_pose": [valid_action, malformed_tail],
            **{column: ["", ""] for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [1])
    _write_exclusions(root, canonical=[])

    stats, n_state, n_action = compute_dataset_stats(root, "DROID", rot6d_identity=False)

    assert n_state == 1
    assert n_action == 1
    assert stats["n_samples"] == 2


def test_droid_stats_do_not_substitute_backup_for_missing_manifest_file(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    state_values = list(np.zeros((2, 7), dtype=np.float32))
    pose_values = list(np.zeros((2, 6), dtype=np.float32))
    action_values = list(np.zeros((2, 7), dtype=np.float32))
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "task_index": [0, 0],
            "state": state_values,
            "other_information.observation_gripper_pose6d": pose_values,
            "other_information.action_wrist_pose": action_values,
            **{column: [""] * 2 for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [2])
    _write_exclusions(root, canonical=[])
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.rename(data_path.with_name("backup.parquet"))

    with pytest.raises((FileNotFoundError, ValueError), match="file-000.parquet"):
        compute_dataset_stats(root, "DROID", rot6d_identity=False)


def test_droid_stats_reject_manifest_episode_mapping_mismatch(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    state_values = list(np.zeros((2, 7), dtype=np.float32))
    pose_values = list(np.zeros((2, 6), dtype=np.float32))
    action_values = list(np.zeros((2, 7), dtype=np.float32))
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "task_index": [0, 0],
            "state": state_values,
            "other_information.observation_gripper_pose6d": pose_values,
            "other_information.action_wrist_pose": action_values,
            **{column: [""] * 2 for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_manifest(root, [1, 1])
    _write_exclusions(root, canonical=[])
    manifest_path = root / "meta" / "episodes" / "chunk-000.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest["episode_index"] = [1, 0]
    manifest.to_parquet(manifest_path)

    with pytest.raises(ValueError, match="prompt source population"):
        compute_dataset_stats(root, "DROID", rot6d_identity=False)
