"""Tests for openwam/dataloader/utils/lerobotv3.py helpers.

Pure-function tests — no parquet IO needed for apply_info_splits;
compute_file_local_offsets uses a synthetic in-memory DataFrame matching
the LeRobot v3 schema shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openwam.dataloader.utils.lerobotv3 import (
    DataContractError,
    apply_info_splits,
    build_multibucket,
    compute_file_local_offsets,
    subsample_episodes_by_hours,
    water_fill_hours,
)

# ---------------------------------------------------------------------------
# apply_info_splits
# ---------------------------------------------------------------------------


@pytest.fixture
def split_eps_df():
    return pd.DataFrame({"episode_index": list(range(20)), "length": [10] * 20})


class TestApplyInfoSplits:
    def test_train_default_fallback_returns_full(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "train", {})
        assert len(out) == 20
        # Index reset
        assert out.index.tolist() == list(range(20))

    def test_val_default_fallback_returns_empty(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "val", {})
        assert len(out) == 0

    def test_info_split_train_range(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "train", {"train": "0:15"})
        assert len(out) == 15
        assert out["episode_index"].max() == 14

    def test_info_split_val_range(self, split_eps_df):
        out = apply_info_splits(split_eps_df, "val", {"train": "0:15", "val": "15:20"})
        assert len(out) == 5
        assert list(out["episode_index"]) == [15, 16, 17, 18, 19]

    def test_malformed_split_spec_raises(self, split_eps_df):
        with pytest.raises(ValueError, match=r"must be 'start:end'"):
            apply_info_splits(split_eps_df, "train", {"train": "not-a-range"})

    def test_split_outside_info_falls_through(self, split_eps_df):
        # Spec only declares "train", val should fall through to default.
        out = apply_info_splits(split_eps_df, "val", {"train": "0:15"})
        assert len(out) == 0


# ---------------------------------------------------------------------------
# compute_file_local_offsets
# ---------------------------------------------------------------------------


class TestComputeFileLocalOffsets:
    def test_offsets_within_shard_are_exclusive_cumsum(self, tiny_episodes_df):
        out = compute_file_local_offsets(tiny_episodes_df, "data/chunk_index", "data/file_index")
        # First episode in any shard must have offset 0.
        for (chunk, file), grp in tiny_episodes_df.groupby(["data/chunk_index", "data/file_index"]):
            sorted_grp = grp.sort_values("dataset_from_index")
            grp_offsets = out[sorted_grp.index.to_numpy()]
            assert grp_offsets[0] == 0
            # Each successive offset equals previous + previous length.
            for i in range(1, len(sorted_grp)):
                assert grp_offsets[i] == grp_offsets[i - 1] + int(sorted_grp.iloc[i - 1]["length"])

    def test_output_dtype_int64(self, tiny_episodes_df):
        out = compute_file_local_offsets(tiny_episodes_df, "data/chunk_index", "data/file_index")
        assert out.dtype == np.int64

    def test_video_offsets_independent_from_data_offsets(self, tiny_episodes_df):
        data_off = compute_file_local_offsets(tiny_episodes_df, "data/chunk_index", "data/file_index")
        video_off = compute_file_local_offsets(tiny_episodes_df, "videos/cam/chunk_index", "videos/cam/file_index")
        # In the synthetic fixture the (chunk, file) columns are identical
        # between data and video, so offsets must match too.
        assert (data_off == video_off).all()


# ---------------------------------------------------------------------------
# water_fill_hours
# ---------------------------------------------------------------------------


class TestWaterFillHours:
    def test_equal_buckets_under_budget(self):
        # 3 buckets of 10h each, budget 15h → each gets 5h.
        out = water_fill_hours([10.0, 10.0, 10.0], 15.0)
        assert out == [5.0, 5.0, 5.0]

    def test_one_small_bucket_redistributes(self):
        # Small bucket 0.5h consumed fully; remaining 4.5h split between
        # two 10h buckets → 2.25h each.
        out = water_fill_hours([0.5, 10.0, 10.0], 5.0)
        assert out[0] == 0.5
        assert abs(out[1] - 2.25) < 1e-9
        assert abs(out[2] - 2.25) < 1e-9

    def test_over_budget_returns_all(self):
        # Total budget exceeds available — return each bucket's full size.
        out = water_fill_hours([1.0, 1.0, 1.0], 100.0)
        assert out == [1.0, 1.0, 1.0]

    def test_zero_buckets(self):
        assert water_fill_hours([], 10.0) == []

    def test_negative_or_zero_budget(self):
        assert water_fill_hours([5.0, 5.0], 0.0) == [0.0, 0.0]
        assert water_fill_hours([5.0, 5.0], -3.0) == [0.0, 0.0]

    def test_multiple_small_buckets_iterative_convergence(self):
        # Two tiny buckets (0.1h, 0.2h), three large (10h each), budget 5h.
        # Iter 1: fair_share = 5/5 = 1.0 → 0.1 and 0.2 are below → take in full.
        #   remaining = 5 - 0.3 = 4.7, active = 3 large.
        # Iter 2: fair_share = 4.7/3 ≈ 1.567 → none below → all get 1.567.
        out = water_fill_hours([0.1, 0.2, 10.0, 10.0, 10.0], 5.0)
        assert out[0] == 0.1
        assert out[1] == 0.2
        assert all(abs(out[i] - 4.7 / 3) < 1e-9 for i in (2, 3, 4))
        assert abs(sum(out) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# subsample_episodes_by_hours
# ---------------------------------------------------------------------------


def _make_eps_df(n_episodes: int, ep_length: int = 100, with_offsets: bool = False) -> pd.DataFrame:
    """Build a synthetic eps_df with uniform episode lengths."""
    df = pd.DataFrame(
        {
            "episode_index": list(range(n_episodes)),
            "length": [ep_length] * n_episodes,
        }
    )
    if with_offsets:
        df["_data_row_offset"] = [i * ep_length for i in range(n_episodes)]
    return df


class TestSubsampleEpisodesByHours:
    def test_below_budget_returns_full(self):
        # 10 episodes × 100 frames / 30 fps / 3600 = 0.00926h total.
        # Target 1h >> total → return full.
        df = _make_eps_df(10)
        out = subsample_episodes_by_hours(df, target_hours=1.0, fps=30.0, seed=42)
        assert len(out) == len(df)

    def test_seeded_reproducible(self):
        # 1000 episodes × 100 frames / 30 fps / 3600 ≈ 0.93h total.
        # Target 0.1h ≈ 10800 frames ≈ 108 episodes.
        df = _make_eps_df(1000)
        out1 = subsample_episodes_by_hours(df, target_hours=0.1, fps=30.0, seed=42)
        out2 = subsample_episodes_by_hours(df, target_hours=0.1, fps=30.0, seed=42)
        pd.testing.assert_frame_equal(out1, out2)

    def test_different_seed_likely_different(self):
        df = _make_eps_df(1000)
        out1 = subsample_episodes_by_hours(df, target_hours=0.1, fps=30.0, seed=42)
        out2 = subsample_episodes_by_hours(df, target_hours=0.1, fps=30.0, seed=7919)
        # With 1000 episodes and ~108 selected, P(identical subset) is
        # astronomically small; rely on the episode_index set differing.
        assert set(out1["episode_index"]) != set(out2["episode_index"])

    def test_target_hit_bounds(self):
        # Each episode = 100 frames / 30 fps = 3.33s; target 60s = 0.01667h.
        # Greedy stops the first time cum >= target → actual in
        # [target, target + max_ep_length / fps / 3600).
        df = _make_eps_df(1000)
        target = 60.0 / 3600.0
        out = subsample_episodes_by_hours(df, target_hours=target, fps=30.0, seed=42)
        actual_hours = out["length"].sum() / 30.0 / 3600.0
        assert actual_hours >= target
        assert actual_hours < target + (100 / 30.0 / 3600.0) + 1e-9

    def test_effective_valid_range_drives_budget(self):
        """Segment-trimmed episodes are charged only for sampleable frames."""
        df = _make_eps_df(20, ep_length=100)
        df["_valid_start"] = 10
        df["_valid_end"] = 60  # 50 effective frames per episode, not 100
        target = 250.0 / 30.0 / 3600.0
        out = subsample_episodes_by_hours(df, target_hours=target, fps=30.0, seed=42)
        assert len(out) == 5
        effective_frames = (out["_valid_end"] - out["_valid_start"]).sum()
        assert effective_frames == 250

    def test_zero_window_episode_cannot_satisfy_tiny_positive_budget(self):
        df = _make_eps_df(2, ep_length=10)
        explicit = np.array([0, 10], dtype=np.int64)
        out = subsample_episodes_by_hours(
            df,
            target_hours=1e-12,
            fps=30.0,
            seed=1,  # permutation visits zero-weight episode 0 first
            episode_frames=explicit,
        )
        assert out["episode_index"].to_list() == [1]

    def test_all_zero_sampleable_frames_raise(self):
        df = _make_eps_df(2, ep_length=10)
        with pytest.raises(ValueError, match="no positive sampleable frame"):
            subsample_episodes_by_hours(
                df,
                target_hours=0.1,
                fps=30.0,
                seed=42,
                episode_frames=np.zeros(2, dtype=np.int64),
            )

    def test_unpaired_or_out_of_bounds_valid_range_raises(self):
        unpaired = _make_eps_df(2, ep_length=10)
        unpaired["_valid_start"] = 1
        with pytest.raises(ValueError, match="missing '_valid_end'"):
            subsample_episodes_by_hours(unpaired, target_hours=0.1, fps=30.0, seed=42)

        out_of_bounds = _make_eps_df(2, ep_length=10)
        out_of_bounds["_valid_start"] = 0
        out_of_bounds["_valid_end"] = [11, 10]
        with pytest.raises(ValueError, match="start <= end <= length"):
            subsample_episodes_by_hours(out_of_bounds, target_hours=0.1, fps=30.0, seed=42)

    def test_negative_target_raises(self):
        df = _make_eps_df(10)
        with pytest.raises(ValueError, match="target_hours must be > 0"):
            subsample_episodes_by_hours(df, target_hours=-1.0, fps=30.0, seed=42)
        with pytest.raises(ValueError, match="target_hours must be > 0"):
            subsample_episodes_by_hours(df, target_hours=0.0, fps=30.0, seed=42)

    def test_preserves_row_order(self):
        # Selected rows must appear in their original episode_index order
        # so downstream offset columns stay aligned.
        df = _make_eps_df(1000, with_offsets=True)
        out = subsample_episodes_by_hours(df, target_hours=0.05, fps=30.0, seed=42)
        # Episode indices must be monotonically increasing in the output.
        ep_idxs = out["episode_index"].to_list()
        assert ep_idxs == sorted(ep_idxs)
        # Each selected row keeps its _data_row_offset column intact.
        for _, row in out.iterrows():
            assert row["_data_row_offset"] == row["episode_index"] * 100


# ---------------------------------------------------------------------------
# build_multibucket error policy
# ---------------------------------------------------------------------------


def _fake_bucket_classes(failures: dict):
    """Return (reader_cls, wrapper_cls) where ``failures[name]`` is raised."""

    class _Bucket:
        def __init__(self, dataset_dir, **_kw):
            exc = failures.get(Path(dataset_dir).name)
            if exc is not None:
                raise exc
            self.dataset_id = Path(dataset_dir).name

        def __len__(self):
            return 5

    class _Wrapper:
        def __init__(self, buckets):
            self.buckets = list(buckets)

    return _Bucket, _Wrapper


class TestBuildMultibucketErrorPolicy:
    """Which per-bucket failures are tolerated, and which abort the launch.

    The tolerance is deliberate — one truncated shard in a large root must
    not kill a training run — but it is exactly wrong for a failure that proves
    the DATA is broken, because dropping that bucket removes a slice of the
    training set behind a single WARNING.
    """

    SUBS = [Path("/nonexistent/root") / f"b{i}" for i in range(3)]

    def test_environmental_failure_is_skipped(self, caplog):
        reader, wrapper = _fake_bucket_classes({"b1": OSError("stale NFS handle")})
        out = build_multibucket(
            reader, self.SUBS, {"split": "train"}, base_seed=0, total_hours=None, wrapper_cls=wrapper
        )
        assert [b.dataset_id for b in out.buckets] == ["b0", "b2"]

    def test_data_contract_error_aborts_the_build(self):
        reader, wrapper = _fake_bucket_classes({"b1": DataContractError("prompt tables have diverged")})
        with pytest.raises(DataContractError, match="diverged"):
            build_multibucket(reader, self.SUBS, {"split": "train"}, base_seed=0, total_hours=None, wrapper_cls=wrapper)

    def test_data_contract_error_is_not_a_keyerror_or_valueerror(self):
        # Readers must raise DataContractError explicitly; the ordinary exception
        # types stay tolerated so this cannot be triggered by accident.
        reader, wrapper = _fake_bucket_classes({"b1": ValueError("looks fatal but is not typed as such")})
        out = build_multibucket(
            reader, self.SUBS, {"split": "train"}, base_seed=0, total_hours=None, wrapper_cls=wrapper
        )
        assert [b.dataset_id for b in out.buckets] == ["b0", "b2"]
        assert issubclass(DataContractError, RuntimeError)
