"""Tests for RoboCOIN stats accumulation and trim-population provenance.

mean/std/min/max are streamed exactly; q01/q99 come from a bounded reservoir
sample. Synthetic bucket tests additionally pin trim-aware population selection
and the on-disk provenance contract consumed by ``RoboCOINDataset``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from openwam.dataloader.robocoin import (
    ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS,
    RoboCOINDataset,
    robocoin_bucket_exclusions_provenance,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.stats_computation import robocoin_stats_computation as stats_module
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import (
    Accumulator,
    compute_stats_for_robot_type,
    discover_datasets_by_robot_type,
)
from tests.dataloader.test_robocoin_trim import (
    _make_bucket,
    _set_info_splits,
    _trim_row,
    _write_trim_csv,
)


def _make_encoded_stats_bucket(tmp_path):
    root = tmp_path / "root"
    bucket = _make_bucket(root / "bucket", [10])
    data_path = bucket / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(data_path)
    source_values = np.array([100, 100, 2, 3, 4, 5, 6, 7, 100, 100], dtype=np.float32)
    for column in ("eef_sim_pose_action", "eef_sim_pose_state"):
        eef = np.stack(df[column].values).astype(np.float32)
        eef[:, 0] = source_values
        df[column] = list(eef)
    df.to_parquet(data_path, index=False)
    return root, bucket


def _make_trim_csv(tmp_path, *, episode_index=0, total_frames=10):
    return _write_trim_csv(
        tmp_path / "trim.csv",
        [
            _trim_row(
                dataset="bucket",
                episode_index=episode_index,
                total_frames=total_frames,
                trim_head_to=2,
                trim_tail_from=8,
            )
        ],
    )


def _write_stats(root, payload):
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    path = meta / "stats_test_robot.json"
    path.write_text(json.dumps(payload))
    return path


def _write_exclusions(bucket, episode_indices):
    path = bucket / "meta" / "excluded_episodes.json"
    path.write_text(json.dumps({"episode_indices": episode_indices}))
    return path


def _set_robot_type(bucket, robot_type):
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = robot_type
    info_path.write_text(json.dumps(info))


@pytest.fixture
def trimmed_stats_case(tmp_path):
    root, bucket = _make_encoded_stats_bucket(tmp_path)
    trim_csv = _make_trim_csv(tmp_path)
    result = compute_stats_for_robot_type(
        "test_robot",
        [str(bucket)],
        rot6d_identity=False,
        trim_csv=str(trim_csv),
    )
    return root, bucket, trim_csv, result


class TestAccumulator:
    def test_exact_stats_and_complete_schema(self):
        rng = np.random.RandomState(0)
        data = rng.uniform(-3.0, 3.0, size=(50000, 4)).astype(np.float32)
        acc = Accumulator(dim=4, reservoir_cap=10000, seed=1)
        for i in range(0, len(data), 1000):
            acc.update_batch(data[i : i + 1000])
        out = acc.finalize()
        # mean/std/min/max are streamed over every row → exact.
        np.testing.assert_allclose(out["mean"], data.mean(0), atol=1e-3)
        np.testing.assert_allclose(out["std"], data.std(0), atol=1e-3)
        np.testing.assert_allclose(out["min"], data.min(0), atol=1e-5)
        np.testing.assert_allclose(out["max"], data.max(0), atol=1e-5)
        # schema carries all six fields the readers may need.
        assert {"mean", "std", "min", "max", "q01", "q99"}.issubset(out)
        assert len(out["q01"]) == 4 and len(out["q99"]) == 4

    def test_reservoir_quantiles_approximate_truth(self):
        rng = np.random.RandomState(0)
        data = rng.uniform(-3.0, 3.0, size=(50000, 4)).astype(np.float32)
        acc = Accumulator(dim=4, reservoir_cap=10000, seed=1)
        for i in range(0, len(data), 1000):
            acc.update_batch(data[i : i + 1000])
        out = acc.finalize()
        # cap (10k) < N (50k) → reservoir is a uniform subsample; q01/q99 are
        # close to the true quantiles within sampling error.
        np.testing.assert_allclose(out["q01"], np.quantile(data, 0.01, axis=0), atol=0.15)
        np.testing.assert_allclose(out["q99"], np.quantile(data, 0.99, axis=0), atol=0.15)

    def test_reservoir_holds_all_when_under_cap_gives_exact_quantiles(self):
        rng = np.random.RandomState(2)
        data = rng.uniform(0.0, 1.0, size=(500, 3)).astype(np.float32)
        acc = Accumulator(dim=3, reservoir_cap=10000, seed=0)
        acc.update_batch(data)
        out = acc.finalize()
        # N (500) < cap → reservoir holds every row → quantiles are exact.
        np.testing.assert_allclose(out["q01"], np.quantile(data, 0.01, axis=0), atol=1e-5)
        np.testing.assert_allclose(out["q99"], np.quantile(data, 0.99, axis=0), atol=1e-5)


class TestWholeBucketExclusion:
    def test_discovery_and_direct_compute_share_the_same_exclusion(self, tmp_path):
        root = tmp_path / "root"
        kept = _make_bucket(root / "kept-airbot", [10])
        excluded_name = next(iter(ROBOCOIN_WHOLE_BUCKET_EXCLUSIONS))
        excluded = _make_bucket(root / excluded_name, [10])
        _set_robot_type(kept, "airbot_mmk2")
        _set_robot_type(excluded, "airbot_mmk2")

        groups = discover_datasets_by_robot_type(str(root))
        assert groups == {"airbot_mmk2": [str(kept)]}

        result = compute_stats_for_robot_type(
            "airbot_mmk2",
            [str(kept), str(excluded)],
            rot6d_identity=False,
        )
        assert result["eef"]["num_datasets"] == 1
        assert result["eef"]["num_timesteps"] == 20  # action + state
        assert result["whole_bucket_exclusions_provenance"] == (robocoin_bucket_exclusions_provenance("airbot_mmk2"))

    def test_affected_robot_type_rejects_stats_without_exclusion_provenance(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(root / "kept-airbot", [40])
        _set_robot_type(bucket, "airbot_mmk2")
        payload = compute_stats_for_robot_type("airbot_mmk2", [str(bucket)], rot6d_identity=False)
        payload.pop("whole_bucket_exclusions_provenance")
        meta = root / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "stats_airbot_mmk2.json").write_text(json.dumps(payload))

        with pytest.raises(DataContractError, match="whole_bucket_exclusions_provenance"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="quantile",
                num_frames=5,
                video_stride=1,
            )


class TestTrimmedPopulationStats:
    def test_stats_use_only_kept_rows_and_serialize_trim_provenance(self, trimmed_stats_case):
        _, _, trim_csv, result = trimmed_stats_case
        eef = result["eef"]

        # Six kept frames in [2, 8), pooled once from action and once from state.
        assert eef["num_timesteps"] == 12
        assert eef["min"][0] == pytest.approx(2.0)
        assert eef["max"][0] == pytest.approx(7.0)

        # The complete output must survive the CLI's JSON serialization without
        # losing either the exact input digest or the trim-policy parameters.
        round_tripped = json.loads(json.dumps(result))
        provenance = round_tripped["trim_provenance"]
        assert provenance == result["trim_provenance"]
        assert provenance["schema_version"] == 1
        assert provenance["sha256"] == hashlib.sha256(trim_csv.read_bytes()).hexdigest()
        assert provenance["min_len"] == 1
        assert provenance["zero_span_policy"] == "drop"
        assert round_tripped["excluded_episodes_provenance"] == {
            "schema_version": 1,
            "policy": "drop_matching_episode_index_before_trim",
            "datasets": {"bucket": {"episode_indices": []}},
        }
        population = round_tripped["population"]
        assert {key: population[key] for key in ("schema_version", "split", "policy")} == {
            "schema_version": 2,
            "split": "train",
            "policy": "info_split_then_exclusion_then_trim",
        }
        assert population["datasets"]["bucket"]["num_episodes"] == 1
        assert population["datasets"]["bucket"]["num_rows"] == 6
        assert len(population["datasets"]["bucket"]["effective_population_digest"]) == 64

    def test_stats_apply_train_split_before_exclusion_and_trim(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(root / "bucket", [10, 10])
        _set_info_splits(bucket, {"train": "0:1", "val": "1:2"})
        data_path = bucket / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(data_path)
        encoded = np.array([100, 100, 2, 3, 4, 5, 6, 7, 100, 100] + [999] * 10, dtype=np.float32)
        for column in ("eef_sim_pose_action", "eef_sim_pose_state"):
            values = np.stack(df[column].values).astype(np.float32)
            values[:, 0] = encoded
            df[column] = list(values)
        df.to_parquet(data_path, index=False)

        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket)],
            rot6d_identity=False,
            trim_csv=str(_make_trim_csv(tmp_path)),
            split="train",
        )

        # Six kept train rows, pooled from action and state.  The val episode's
        # sentinel value must not enter any statistic.
        assert result["eef"]["num_timesteps"] == 12
        assert result["eef"]["min"][0] == pytest.approx(2.0)
        assert result["eef"]["max"][0] == pytest.approx(7.0)

    def test_stats_exclude_blacklisted_episode_and_serialize_population(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(root / "bucket", [10, 10])
        data_path = bucket / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(data_path)
        source_values = np.array(
            [100, 100, 2, 3, 4, 5, 6, 7, 100, 100] + [999] * 10,
            dtype=np.float32,
        )
        for column in ("eef_sim_pose_action", "eef_sim_pose_state"):
            eef = np.stack(df[column].values).astype(np.float32)
            eef[:, 0] = source_values
            df[column] = list(eef)
        df.to_parquet(data_path, index=False)
        _write_exclusions(bucket, [1])

        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket)],
            rot6d_identity=False,
            trim_csv=str(_make_trim_csv(tmp_path)),
        )

        assert result["eef"]["num_timesteps"] == 12
        assert result["eef"]["min"][0] == pytest.approx(2.0)
        assert result["eef"]["max"][0] == pytest.approx(7.0)
        assert result["excluded_episodes_provenance"]["datasets"] == {"bucket": {"episode_indices": [1]}}

    @pytest.mark.parametrize(
        ("episode_index", "total_frames"),
        [(0, 9), (99, 10)],
        ids=["stale-length", "unknown-episode"],
    )
    def test_stale_or_unknown_trim_spec_fails_closed(self, tmp_path, episode_index, total_frames):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        trim_csv = _make_trim_csv(
            tmp_path,
            episode_index=episode_index,
            total_frames=total_frames,
        )

        with pytest.raises(DataContractError):
            compute_stats_for_robot_type(
                "test_robot",
                [str(bucket)],
                rot6d_identity=False,
                trim_csv=str(trim_csv),
            )

    def test_trim_span_crossing_parquet_boundary_uses_physical_global_rows(self, tmp_path):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        first_path = bucket / "data" / "chunk-000" / "file-000.parquet"
        second_path = bucket / "data" / "chunk-000" / "file-001.parquet"
        df = pd.read_parquet(first_path)
        df.iloc[:5].to_parquet(first_path, index=False)
        df.iloc[5:].to_parquet(second_path, index=False)
        trim_csv = _make_trim_csv(tmp_path)

        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket)],
            rot6d_identity=False,
            trim_csv=str(trim_csv),
        )
        assert result["eef"]["num_timesteps"] == 12
        assert result["eef"]["min"][0] == pytest.approx(2.0)
        assert result["eef"]["max"][0] == pytest.approx(7.0)

    def test_stats_ignore_sorting_earlier_non_numeric_backup_shard(self, tmp_path):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        real_path = bucket / "data" / "chunk-000" / "file-000.parquet"
        backup_path = bucket / "data" / "chunk--backup" / "file-old.parquet"
        backup_path.parent.mkdir()
        backup = pd.read_parquet(real_path)
        for column in ("eef_sim_pose_action", "eef_sim_pose_state"):
            values = np.stack(backup[column].values).astype(np.float32)
            values[:, 0] = 999
            backup[column] = list(values)
        backup.to_parquet(backup_path, index=False)

        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket)],
            rot6d_identity=False,
            trim_csv=str(_make_trim_csv(tmp_path)),
        )

        assert result["eef"]["num_files"] == 1
        assert result["eef"]["min"][0] == pytest.approx(2.0)
        assert result["eef"]["max"][0] == pytest.approx(7.0)

    def test_stats_reject_duplicate_numeric_shard_coordinates(self, tmp_path):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        canonical = bucket / "data" / "chunk-000" / "file-000.parquet"
        alias = canonical.with_name("file-0.parquet")
        pd.read_parquet(canonical).to_parquet(alias, index=False)

        with pytest.raises(DataContractError, match="duplicate numeric data shard coordinates"):
            compute_stats_for_robot_type(
                "test_robot",
                [str(bucket)],
                rot6d_identity=False,
                trim_csv=str(_make_trim_csv(tmp_path)),
            )

    def test_stats_fail_if_trim_csv_changes_during_scan(self, tmp_path, monkeypatch):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        trim_csv = _make_trim_csv(tmp_path)
        replacement = _write_trim_csv(
            tmp_path / "replacement.csv",
            [_trim_row(trim_head_to=3, trim_tail_from=7)],
        )
        original_read_parquet = stats_module.pd.read_parquet
        replaced = False

        def replace_before_read(*args, **kwargs):
            nonlocal replaced
            if not replaced:
                replacement.replace(trim_csv)
                replaced = True
            return original_read_parquet(*args, **kwargs)

        monkeypatch.setattr(stats_module.pd, "read_parquet", replace_before_read)
        with pytest.raises(DataContractError, match="changed while.*stats scan"):
            compute_stats_for_robot_type(
                "test_robot",
                [str(bucket)],
                rot6d_identity=False,
                trim_csv=str(trim_csv),
            )

    def test_stats_fail_if_exclusions_change_during_scan(self, tmp_path, monkeypatch):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        exclusions = _write_exclusions(bucket, [])
        replacement = bucket / "meta" / "replacement-exclusions.json"
        replacement.write_text(json.dumps({"episode_indices": [0]}))
        original_read_parquet = stats_module.pd.read_parquet
        replaced = False

        def replace_before_read(*args, **kwargs):
            nonlocal replaced
            if not replaced:
                replacement.replace(exclusions)
                replaced = True
            return original_read_parquet(*args, **kwargs)

        monkeypatch.setattr(stats_module.pd, "read_parquet", replace_before_read)
        with pytest.raises(DataContractError, match="changed while.*stats scan"):
            compute_stats_for_robot_type(
                "test_robot",
                [str(bucket)],
                rot6d_identity=False,
                trim_csv=str(_make_trim_csv(tmp_path)),
            )


def test_untrimmed_stats_keep_legacy_nonstandard_parquet_discovery(tmp_path):
    _, bucket = _make_encoded_stats_bucket(tmp_path)
    original = bucket / "data" / "chunk-000" / "file-000.parquet"
    legacy_dir = bucket / "data" / "legacy"
    legacy_dir.mkdir()
    original.rename(legacy_dir / "shard.parquet")

    result = compute_stats_for_robot_type(
        "test_robot",
        [str(bucket)],
        rot6d_identity=False,
        trim_csv=None,
    )
    assert result["eef"]["num_timesteps"] == 20


class TestReaderTrimStatsProvenance:
    def test_matching_provenance_is_accepted(self, trimmed_stats_case):
        root, bucket, trim_csv, result = trimmed_stats_case
        _write_stats(root, result)

        reader = RoboCOINDataset(
            dataset_dir=str(bucket),
            normalize_mode="min-max",
            trim_csv=str(trim_csv),
        )
        assert reader._eps_df["length"].tolist() == [6]
        assert reader._normalization_stats is not None
        dims = list(ROT6D_DIMS_EEF20)
        np.testing.assert_array_equal(reader._normalization_stats["mean"][dims], 0.0)
        np.testing.assert_array_equal(reader._normalization_stats["std"][dims], 1.0)
        np.testing.assert_array_equal(reader._normalization_stats["min"][dims], -1.0)
        np.testing.assert_array_equal(reader._normalization_stats["max"][dims], 1.0)

    def test_missing_exclusion_provenance_is_rejected(self, trimmed_stats_case):
        root, bucket, trim_csv, result = trimmed_stats_case
        payload = json.loads(json.dumps(result))
        payload.pop("excluded_episodes_provenance")
        _write_stats(root, payload)

        with pytest.raises(DataContractError, match="excluded_episodes_provenance"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_non_train_stats_population_is_rejected(self, trimmed_stats_case):
        root, bucket, trim_csv, result = trimmed_stats_case
        result["population"]["split"] = "val"
        _write_stats(root, result)

        with pytest.raises(DataContractError, match="train-derived"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_info_split_drift_after_stats_is_rejected(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(root / "bucket", [10, 10])
        trim_csv = _make_trim_csv(tmp_path)
        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket)],
            rot6d_identity=False,
            trim_csv=str(trim_csv),
            split="train",
        )
        _write_stats(root, result)

        # The generated stats covered both episodes. Narrowing train afterwards
        # must invalidate the stored physical-span digest even though the
        # top-level label still truthfully says "train".
        _set_info_splits(bucket, {"train": "0:1", "val": "1:2"})
        with pytest.raises(DataContractError, match="effective train population"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_sibling_info_split_drift_invalidates_pooled_stats(self, tmp_path):
        root = tmp_path / "root"
        bucket_a = _make_bucket(root / "bucket-a", [10])
        bucket_b = _make_bucket(root / "bucket-b", [10, 10])
        trim_csv = _write_trim_csv(
            tmp_path / "trim.csv",
            [_trim_row(dataset="bucket-a")],
        )
        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket_a), str(bucket_b)],
            rot6d_identity=False,
            trim_csv=str(trim_csv),
            split="train",
        )
        _write_stats(root, result)

        _set_info_splits(bucket_b, {"train": "0:1", "val": "1:2"})
        with pytest.raises(DataContractError, match="contributor 'bucket-b'.*no longer matches"):
            RoboCOINDataset(
                dataset_dir=str(bucket_a),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("schema_version", True),
            ("policy", "keep"),
            ("episode_indices", [True]),
        ],
    )
    def test_mismatched_exclusion_provenance_is_rejected(
        self,
        trimmed_stats_case,
        field,
        bad_value,
    ):
        root, bucket, trim_csv, result = trimmed_stats_case
        payload = json.loads(json.dumps(result))
        if field == "episode_indices":
            payload["excluded_episodes_provenance"]["datasets"]["bucket"][field] = bad_value
        else:
            payload["excluded_episodes_provenance"][field] = bad_value
        _write_stats(root, payload)

        with pytest.raises(DataContractError, match="excluded_episodes_provenance"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_changed_current_bucket_exclusions_are_rejected(self, trimmed_stats_case):
        root, bucket, trim_csv, result = trimmed_stats_case
        _write_stats(root, result)
        _write_exclusions(bucket, [0])

        with pytest.raises(DataContractError, match="current exclusion population"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_changed_sibling_exclusions_in_pooled_stats_are_rejected(self, tmp_path):
        root = tmp_path / "root"
        bucket_a = _make_bucket(root / "bucket-a", [10])
        bucket_b = _make_bucket(root / "bucket-b", [10])
        trim_csv = _write_trim_csv(
            tmp_path / "trim.csv",
            [_trim_row(dataset="bucket-a")],
        )
        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket_a), str(bucket_b)],
            rot6d_identity=False,
            trim_csv=str(trim_csv),
        )
        _write_stats(root, result)
        _write_exclusions(bucket_b, [0])

        with pytest.raises(DataContractError, match="contributor 'bucket-b'.*current exclusion population"):
            RoboCOINDataset(
                dataset_dir=str(bucket_a),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_reader_fails_if_sibling_exclusions_change_during_construction(
        self,
        tmp_path,
        monkeypatch,
    ):
        root = tmp_path / "root"
        bucket_a = _make_bucket(root / "bucket-a", [10])
        bucket_b = _make_bucket(root / "bucket-b", [10])
        trim_csv = _write_trim_csv(
            tmp_path / "trim.csv",
            [_trim_row(dataset="bucket-a")],
        )
        result = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket_a), str(bucket_b)],
            rot6d_identity=False,
            trim_csv=str(trim_csv),
        )
        _write_stats(root, result)
        target = bucket_b / "meta" / "excluded_episodes.json"
        replacement = bucket_b / "meta" / "replacement-exclusions.json"
        replacement.write_text(json.dumps({"episode_indices": [0]}))
        original_post_init = RoboCOINDataset._post_init

        def post_init_then_replace(self, info):
            original_post_init(self, info)
            replacement.replace(target)

        monkeypatch.setattr(RoboCOINDataset, "_post_init", post_init_then_replace)
        with pytest.raises(DataContractError, match="changed while reader construction"):
            RoboCOINDataset(
                dataset_dir=str(bucket_a),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_reader_fails_if_exclusions_change_during_construction(
        self,
        trimmed_stats_case,
        monkeypatch,
    ):
        root, bucket, trim_csv, result = trimmed_stats_case
        _write_stats(root, result)
        target = bucket / "meta" / "excluded_episodes.json"
        replacement = bucket / "meta" / "replacement-exclusions.json"
        replacement.write_text(json.dumps({"episode_indices": [0]}))
        original_load_prompts = RoboCOINDataset._load_prompts

        def load_prompts_then_replace(self):
            original_load_prompts(self)
            replacement.replace(target)

        monkeypatch.setattr(RoboCOINDataset, "_load_prompts", load_prompts_then_replace)
        with pytest.raises(DataContractError, match="changed while reader construction"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_missing_provenance_is_rejected_when_trim_is_enabled(self, trimmed_stats_case):
        root, bucket, trim_csv, result = trimmed_stats_case
        payload = json.loads(json.dumps(result))
        payload.pop("trim_provenance")
        _write_stats(root, payload)

        with pytest.raises(DataContractError):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    @pytest.mark.parametrize(
        "missing_field",
        ["schema_version", "sha256", "min_len", "zero_span_policy"],
    )
    def test_each_missing_provenance_field_is_rejected(self, trimmed_stats_case, missing_field):
        root, bucket, trim_csv, result = trimmed_stats_case
        payload = json.loads(json.dumps(result))
        payload["trim_provenance"].pop(missing_field)
        _write_stats(root, payload)

        with pytest.raises(DataContractError):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("schema_version", 2),
            ("sha256", "0" * 64),
            ("min_len", 2),
            ("zero_span_policy", "keep"),
        ],
    )
    def test_each_mismatched_provenance_field_is_rejected(
        self,
        trimmed_stats_case,
        field,
        bad_value,
    ):
        root, bucket, trim_csv, result = trimmed_stats_case
        payload = json.loads(json.dumps(result))
        payload["trim_provenance"][field] = bad_value
        _write_stats(root, payload)

        with pytest.raises(DataContractError):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_trimmed_stats_are_rejected_when_reader_trim_is_disabled(self, trimmed_stats_case):
        root, bucket, _, result = trimmed_stats_case
        _write_stats(root, result)

        with pytest.raises(DataContractError):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=None,
            )

    def test_legacy_stats_without_provenance_are_accepted_when_trim_is_disabled(
        self,
        trimmed_stats_case,
    ):
        root, bucket, _, result = trimmed_stats_case
        legacy_payload = json.loads(json.dumps(result))
        legacy_payload.pop("trim_provenance")
        legacy_payload.pop("excluded_episodes_provenance")
        _write_stats(root, legacy_payload)

        reader = RoboCOINDataset(
            dataset_dir=str(bucket),
            normalize_mode="min-max",
            trim_csv=None,
        )
        assert reader._eps_df["length"].tolist() == [10]
        assert reader._normalization_stats is not None

    def test_reader_rejects_new_stats_if_csv_changes_after_offsets(self, tmp_path, monkeypatch):
        root, bucket = _make_encoded_stats_bucket(tmp_path)
        trim_csv = _make_trim_csv(tmp_path)

        replacement = _write_trim_csv(
            tmp_path / "replacement-for-b-stats.csv",
            [_trim_row(trim_head_to=3, trim_tail_from=7)],
        )
        replacement.replace(trim_csv)
        stats_b = compute_stats_for_robot_type(
            "test_robot",
            [str(bucket)],
            rot6d_identity=False,
            trim_csv=str(trim_csv),
        )
        _write_stats(root, stats_b)

        restore_a = _write_trim_csv(tmp_path / "restore-a.csv", [_trim_row()])
        restore_a.replace(trim_csv)
        replace_with_b = _write_trim_csv(
            tmp_path / "replace-with-b.csv",
            [_trim_row(trim_head_to=3, trim_tail_from=7)],
        )
        original_add_offsets = RoboCOINDataset._add_data_offsets

        def add_offsets_then_replace(self, eps):
            original_add_offsets(self, eps)
            replace_with_b.replace(trim_csv)

        monkeypatch.setattr(RoboCOINDataset, "_add_data_offsets", add_offsets_then_replace)
        with pytest.raises(DataContractError):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode="min-max",
                trim_csv=str(trim_csv),
            )

    def test_reader_rejects_csv_change_without_normalization(self, tmp_path, monkeypatch):
        _, bucket = _make_encoded_stats_bucket(tmp_path)
        trim_csv = _make_trim_csv(tmp_path)
        replacement = _write_trim_csv(
            tmp_path / "replacement.csv",
            [_trim_row(trim_head_to=3, trim_tail_from=7)],
        )
        original_add_offsets = RoboCOINDataset._add_data_offsets

        def add_offsets_then_replace(self, eps):
            original_add_offsets(self, eps)
            replacement.replace(trim_csv)

        monkeypatch.setattr(RoboCOINDataset, "_add_data_offsets", add_offsets_then_replace)
        with pytest.raises(DataContractError, match="changed while reader construction"):
            RoboCOINDataset(
                dataset_dir=str(bucket),
                normalize_mode=None,
                trim_csv=str(trim_csv),
            )
