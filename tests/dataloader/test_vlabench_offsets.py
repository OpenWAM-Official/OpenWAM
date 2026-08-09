"""VLABench episode -> data-shard offset repair.

The upstream ``VLABench/vlabench_primitive_ft_lerobot_video`` metadata writes
``dataset_from_index = length * episode_index`` and mis-assigns
``data/chunk_index`` / ``data/file_index`` in step with it. Reading those
columns pairs each video clip with a different episode's actions and prompt —
silent corruption, or an ``IndexError`` on an empty slice. ``VLABenchDataset``
rebuilds the mapping from real parquet row counts instead.

These tests drive ``_add_data_offsets`` directly against synthetic shards, so
they need neither the 13 GB dataset nor a video decoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openwam.dataloader.vlabench import VLABenchDataset


def _make_reader(tmp_path):
    """A VLABenchDataset shell with only what _add_data_offsets touches."""
    reader = object.__new__(VLABenchDataset)
    reader._dataset_dir = tmp_path
    return reader


def _write_shards(tmp_path, shard_row_counts, chunk=0):
    """Write data/chunk-000/file-NNN.parquet with the given row counts."""
    data_dir = tmp_path / "data" / f"chunk-{chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(shard_row_counts):
        table = pa.table({"task_index": pa.array(np.zeros(n, dtype=np.int64))})
        pq.write_table(table, data_dir / f"file-{i:03d}.parquet")


def _corrupt_eps(lengths):
    """Episode metadata carrying the upstream bug, as published."""
    lengths = np.asarray(lengths, dtype=np.int64)
    episode_index = np.arange(len(lengths), dtype=np.int64)
    return pd.DataFrame(
        {
            "episode_index": episode_index,
            "length": lengths,
            # The upstream bug, verbatim.
            "dataset_from_index": lengths * episode_index,
            "dataset_to_index": lengths * (episode_index + 1),
            # Wrong in step with it: everything claims shard 0.
            "data/chunk_index": np.zeros(len(lengths), dtype=np.int64),
            "data/file_index": np.zeros(len(lengths), dtype=np.int64),
        }
    )


class TestAddDataOffsets:
    def test_repairs_shard_assignment_and_offsets(self, tmp_path):
        """Episodes 0-1 live in shard 0, 2-3 in shard 1 — as the row counts say."""
        lengths = [93, 74, 78, 105]
        _write_shards(tmp_path, [93 + 74, 78 + 105])
        eps = _corrupt_eps(lengths)

        _make_reader(tmp_path)._add_data_offsets(eps)

        np.testing.assert_array_equal(eps["data/file_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(eps["data/chunk_index"], [0, 0, 0, 0])
        np.testing.assert_array_equal(eps["_data_row_offset"], [0, 93, 0, 78])

    def test_ignores_the_corrupt_index_columns(self, tmp_path):
        """The repair must not read dataset_from_index — poison it and re-check."""
        lengths = [93, 74, 78, 105]
        _write_shards(tmp_path, [93 + 74, 78 + 105])
        eps = _corrupt_eps(lengths)
        eps["dataset_from_index"] = -12345
        eps["dataset_to_index"] = -12345

        _make_reader(tmp_path)._add_data_offsets(eps)

        np.testing.assert_array_equal(eps["data/file_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(eps["_data_row_offset"], [0, 93, 0, 78])

    def test_row_order_independent(self, tmp_path):
        """Packing follows episode_index, not the DataFrame's row order."""
        lengths = [93, 74, 78, 105]
        _write_shards(tmp_path, [93 + 74, 78 + 105])
        eps = _corrupt_eps(lengths).iloc[::-1].reset_index(drop=True)

        _make_reader(tmp_path)._add_data_offsets(eps)

        by_ep = eps.set_index("episode_index")
        np.testing.assert_array_equal(by_ep.loc[[0, 1, 2, 3], "data/file_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(by_ep.loc[[0, 1, 2, 3], "_data_row_offset"], [0, 93, 0, 78])

    def test_single_episode_per_shard(self, tmp_path):
        lengths = [10, 20, 30]
        _write_shards(tmp_path, [10, 20, 30])

        eps = _corrupt_eps(lengths)
        _make_reader(tmp_path)._add_data_offsets(eps)

        np.testing.assert_array_equal(eps["data/file_index"], [0, 1, 2])
        np.testing.assert_array_equal(eps["_data_row_offset"], [0, 0, 0])

    def test_rejects_total_row_mismatch(self, tmp_path):
        """A dataset that cannot be explained by repacking must fail loudly."""
        _write_shards(tmp_path, [100])  # 100 rows on disk
        eps = _corrupt_eps([93, 74])  # 167 rows claimed

        with pytest.raises(ValueError, match="cannot be repaired by repacking"):
            _make_reader(tmp_path)._add_data_offsets(eps)

    def test_rejects_episode_spanning_two_shards(self, tmp_path):
        """A straddling episode would be silently truncated by the window slice."""
        # Episode 0 (93 rows) fits in shard 0, but episode 1 starts at row 93 and
        # shard 0 holds only 100 rows, so its 74 rows run over into shard 1.
        _write_shards(tmp_path, [100, 67])
        eps = _corrupt_eps([93, 74])

        with pytest.raises(ValueError, match="spans a data shard boundary"):
            _make_reader(tmp_path)._add_data_offsets(eps)

    def test_missing_data_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No data parquet files"):
            _make_reader(tmp_path)._add_data_offsets(_corrupt_eps([10]))


class TestDeployContract:
    """Keys the deploy server reads out of a VLABench checkpoint's saved config."""

    @staticmethod
    def _config():
        from pathlib import Path

        import yaml

        repo_root = Path(__file__).resolve().parents[2]
        with open(repo_root / "configs" / "dataloader" / "vlabench.yaml") as f:
            return yaml.safe_load(f)

    def test_action_mode_matches_deploy_action_mode(self):
        """yaml `action_mode` (deploy READS) must equal DEPLOY_ACTION_MODE (reader WRITES).

        The reader writes normalization_stats.npy keyed by DEPLOY_ACTION_MODE;
        the policy server looks up `dataloader.action_mode` from the saved
        config and defaults to "joint" when the key is missing. A mismatch — or
        an omission — aborts the deploy with a missing-stats KeyError only after
        the full model load.
        """
        assert self._config().get("action_mode") == VLABenchDataset.DEPLOY_ACTION_MODE

    def test_unify_map_width_matches_action_dim(self):
        """`unify_action_map` must scatter exactly ACTION_DIM raw dims."""
        from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec

        cfg = self._config()
        assert cfg.get("unify_action") is True
        dst_index = parse_unify_spec(cfg["unify_action_map"], UNIFY_DIM)
        assert len(dst_index) == VLABenchDataset.ACTION_DIM
        # Single arm -> the pretrained left-arm slots.
        np.testing.assert_array_equal(dst_index, np.arange(10))
