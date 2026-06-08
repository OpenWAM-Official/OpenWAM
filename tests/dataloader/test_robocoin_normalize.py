"""Tests for the in-reader normalization logic on RoboCOINDataset.

We don't need a full RoboCOIN dataset on disk to test ``_normalize_array``;
the method is a pure function of (arr, normalize_mode, _normalization_stats). We
construct a thin stub instance by bypassing __init__.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.robocoin import RoboCOINDataset


def _stub_reader(normalize_mode: str | None, stats: dict | None):
    """Build a RoboCOINDataset instance without running its full __init__."""
    r = RoboCOINDataset.__new__(RoboCOINDataset)
    r._normalize_mode = normalize_mode
    r._normalization_stats = stats
    return r


class TestNormalizeArray:
    def test_no_stats_returns_input_untouched(self):
        r = _stub_reader(normalize_mode=None, stats=None)
        arr = np.arange(20, dtype=np.float32).reshape(1, 20)
        out = r._normalize_array(arr)
        assert (out == arr).all()

    def test_min_max_maps_min_to_neg_one(self, known_stats_dict):
        r = _stub_reader(normalize_mode="min-max", stats=known_stats_dict)
        # At the min value, normalized must be -1.
        arr = np.array([[0.0, -1.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_min_max_maps_max_to_pos_one(self, known_stats_dict):
        r = _stub_reader(normalize_mode="min-max", stats=known_stats_dict)
        arr = np.array([[10.0, 1.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_min_max_midpoint_maps_to_zero(self, known_stats_dict):
        r = _stub_reader(normalize_mode="min-max", stats=known_stats_dict)
        # midpoint of [0, 10] = 5; midpoint of [-1, 1] = 0
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_maps_mean_to_zero(self, known_stats_dict):
        r = _stub_reader(normalize_mode="z-score", stats=known_stats_dict)
        arr = np.array([[5.0, 0.0]], dtype=np.float32)  # equals mean
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_one_std_above_mean_maps_to_one(self, known_stats_dict):
        r = _stub_reader(normalize_mode="z-score", stats=known_stats_dict)
        # mean+std: 5+2=7 ; 0+0.5=0.5
        arr = np.array([[7.0, 0.5]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_quantile_maps_q01_q99_to_unit(self, known_stats_dict):
        r = _stub_reader(normalize_mode="quantile", stats=known_stats_dict)
        # q01 → -1, q99 → +1 (dim0 q01/q99 = 1/9; dim1 = -0.8/0.8)
        arr = np.array([[1.0, -0.8], [9.0, 0.8]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-6)

    def test_quantile_midpoint_maps_to_zero(self, known_stats_dict):
        r = _stub_reader(normalize_mode="quantile", stats=known_stats_dict)
        # midpoint of [q01, q99]: dim0 (1,9)→5 ; dim1 (-0.8,0.8)→0
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_quantile_clips_outliers_to_unit(self, known_stats_dict):
        r = _stub_reader(normalize_mode="quantile", stats=known_stats_dict)
        # values beyond q01/q99 are clipped to [-1, 1] (this is quantile's point)
        arr = np.array([[-50.0, -9.0], [50.0, 9.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-6)
        assert (np.abs(out) <= 1.0 + 1e-6).all()

    def test_broadcasts_along_time(self, known_stats_dict):
        # Multi-row input still gets per-column normalization.
        r = _stub_reader(normalize_mode="min-max", stats=known_stats_dict)
        arr = np.array([[0.0, -1.0], [10.0, 1.0], [5.0, 0.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(
            out,
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
            atol=1e-6,
        )

    def test_unknown_mode_passes_through(self, known_stats_dict):
        # Unknown mode shouldn't raise — _normalize_array silently returns input.
        r = _stub_reader(normalize_mode="weird-mode", stats=known_stats_dict)
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = r._normalize_array(arr)
        np.testing.assert_allclose(out, arr, atol=1e-6)


class TestActionStatsContract:
    def test_normalization_stats_returns_none_even_when_internal_stats_set(self, known_stats_dict):
        """Reader is pretraining-only; the public normalization_stats stays None."""
        r = _stub_reader(normalize_mode="min-max", stats=known_stats_dict)
        assert r.normalization_stats is None
