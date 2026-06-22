"""Tests for robomind_stats_computation.

The Accumulator (mean/std/min/max + reservoir q01/q99) is RoboCOIN's and is
already covered by test_robocoin_compute_stats.py — this file only exercises
the RoboMIND-specific pieces:

  * ``discover_datasets_by_robot_type`` — groups buckets by robot_type, resolves
    each group's eef_kind from info.json cameras, and raises on a kind clash;
  * ``compute_stats_for_robot_type`` — pools action+state through the SAME
    ``robomind_raw_to_20d`` the reader uses, emitting the 20-D ``eef`` schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.robomind import robomind_raw_to_20d
from openwam.dataloader.utils.stats_computation import robomind_stats_computation as _mod

P = "observation.images."
# (robot_type, raw_dim, camera keys, expected eef_kind)
EMB = {
    "single_euler": ("franka_panda_3rgb", 7, [P + "camera_top"]),
    "single_quat": ("franka_panda_sim", 8, [P + "camera_front_external", P + "camera_handeye"]),
    "dual_euler": (
        "agilex_cobot_magic_v2",
        14,
        [P + "camera_front", P + "camera_left_wrist", P + "camera_right_wrist"],
    ),
}


def _make_stats_bucket(root: Path, kind: str, suffix: str, n_rows: int = 64) -> None:
    """Minimal on-disk bucket: meta/info.json + one data parquet (no videos —
    the stats script only reads observation.state + action)."""
    robot_type, raw_dim, cams = EMB[kind]
    bucket = root / f"{robot_type}_{suffix}"
    (bucket / "meta").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(abs(hash(suffix)) % (2**31))
    state = rng.uniform(-1, 1, size=(n_rows, raw_dim)).astype(np.float32)
    if kind == "single_quat":
        q = rng.uniform(-1, 1, size=(n_rows, 4)).astype(np.float32)
        state[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    action = rng.uniform(-1, 1, size=(n_rows, raw_dim)).astype(np.float32)
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"observation.state": list(state), "action": list(action)})
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")
    features = {c: {"dtype": "video"} for c in cams}
    features.update({"action": {"shape": [raw_dim]}, "observation.state": {"shape": [raw_dim]}})
    (bucket / "meta" / "info.json").write_text(json.dumps({"robot_type": robot_type, "features": features}))


class TestDiscover:
    def test_groups_by_robot_type_and_resolves_kind(self, tmp_path):
        # two benchmarks of the same robot_type pool into one group
        _make_stats_bucket(tmp_path, "single_euler", "b10")
        _make_stats_bucket(tmp_path, "single_euler", "b11")
        _make_stats_bucket(tmp_path, "dual_euler", "b10")
        groups = _mod.discover_datasets_by_robot_type(str(tmp_path))
        assert set(groups) == {"franka_panda_3rgb", "agilex_cobot_magic_v2"}
        assert len(groups["franka_panda_3rgb"]["dirs"]) == 2
        assert groups["franka_panda_3rgb"]["eef_kind"] == "single_euler"
        assert groups["agilex_cobot_magic_v2"]["eef_kind"] == "dual_euler"

    def test_inconsistent_kind_for_same_robot_type_raises(self, tmp_path):
        import pytest

        _make_stats_bucket(tmp_path, "single_euler", "b10")
        # Corrupt the second bucket: same robot_type, but agilex wrist cameras.
        bad = tmp_path / "franka_panda_3rgb_bad" / "meta"
        bad.mkdir(parents=True)
        (bad / "info.json").write_text(
            json.dumps(
                {
                    "robot_type": "franka_panda_3rgb",
                    "features": {P + "camera_left_wrist": {}, P + "camera_right_wrist": {}, P + "camera_front": {}},
                }
            )
        )
        with pytest.raises(ValueError, match="inconsistent eef_kind"):
            _mod.discover_datasets_by_robot_type(str(tmp_path))


class TestComputeStats:
    def test_schema_and_parity_with_reader_converter(self, tmp_path):
        _make_stats_bucket(tmp_path, "single_euler", "b10", n_rows=64)
        groups = _mod.discover_datasets_by_robot_type(str(tmp_path))
        g = groups["franka_panda_3rgb"]
        stats = _mod.compute_stats_for_robot_type("franka_panda_3rgb", g["eef_kind"], g["dirs"])

        # complete 20-D schema + robomind-specific metadata
        for k in ("mean", "std", "min", "max", "q01", "q99"):
            assert len(stats[k]) == 20
        assert stats["robot_type"] == "franka_panda_3rgb"
        assert stats["eef_kind"] == "single_euler"
        assert stats["pool"] == "action+state"
        assert stats["num_timesteps"] == 64 * 2  # action + state pooled

        # single-arm right half [10:20] is zero-range across the pool
        assert np.allclose(stats["min"][10:], 0.0) and np.allclose(stats["max"][10:], 0.0)

        # parity: recompute the pooled 20-D directly with the reader's converter
        df = pd.read_parquet(g["dirs"][0] + "/data/chunk-000/file-000.parquet")
        action = robomind_raw_to_20d(np.stack(df["action"].values).astype(np.float32), "single_euler")
        state = robomind_raw_to_20d(np.stack(df["observation.state"].values).astype(np.float32), "single_euler")
        pooled = np.concatenate([action, state], axis=0)
        np.testing.assert_allclose(stats["min"], pooled.min(0), atol=1e-5)
        np.testing.assert_allclose(stats["max"], pooled.max(0), atol=1e-5)
        np.testing.assert_allclose(stats["mean"], pooled.mean(0), atol=1e-3)

    def test_dual_arm_fills_all_20_dims(self, tmp_path):
        _make_stats_bucket(tmp_path, "dual_euler", "b10", n_rows=64)
        groups = _mod.discover_datasets_by_robot_type(str(tmp_path))
        g = groups["agilex_cobot_magic_v2"]
        stats = _mod.compute_stats_for_robot_type("agilex_cobot_magic_v2", g["eef_kind"], g["dirs"])
        # right half is NOT degenerate for the bimanual embodiment
        assert not np.allclose(np.array(stats["max"][10:]) - np.array(stats["min"][10:]), 0.0)
