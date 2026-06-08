"""Tests for the extracted normalization helper in ``utils.normalization``.

Mirrors ``test_robocoin_normalize.py`` but exercises the pure-function
form directly, so the helper can be safely reused by the upcoming OXE
readers without going through a RoboCOIN stub.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.normalization import apply_normalization


def _stats():
    return {
        "mean": np.array([5.0, 0.0], dtype=np.float32),
        "std": np.array([2.0, 0.5], dtype=np.float32),
        "min": np.array([0.0, -1.0], dtype=np.float32),
        "max": np.array([10.0, 1.0], dtype=np.float32),
    }


class TestApplyNormalization:
    def test_no_stats_passthrough(self):
        arr = np.arange(20, dtype=np.float32).reshape(1, 20)
        out = apply_normalization(arr, None, "min-max")
        assert (out == arr).all()

    def test_none_mode_passthrough(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        for mode in (None, "none", "null"):
            out = apply_normalization(arr, _stats(), mode)
            np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_min_max_min_maps_to_neg_one(self):
        arr = np.array([[0.0, -1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_min_max_max_maps_to_pos_one(self):
        arr = np.array([[10.0, 1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_min_max_mid_maps_to_zero(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_mean_maps_to_zero(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "z-score")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_one_std_maps_to_one(self):
        arr = np.array([[7.0, 0.5]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "z-score")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_unknown_mode_passes_through(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "weird-mode")
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_broadcast_across_leading_dims(self):
        arr = np.array([[0.0, -1.0], [10.0, 1.0], [5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(
            out,
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
            atol=1e-6,
        )


def _stats_with_quantile():
    return {
        "mean": np.array([0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0], dtype=np.float32),
        "min": np.array([-100.0, -50.0], dtype=np.float32),  # outliers
        "max": np.array([100.0, 50.0], dtype=np.float32),
        "q01": np.array([-1.0, -2.0], dtype=np.float32),
        "q99": np.array([1.0, 2.0], dtype=np.float32),
    }


class TestQuantileNormalization:
    def test_q01_maps_to_neg_one(self):
        arr = np.array([[-1.0, -2.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_q99_maps_to_pos_one(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_midpoint_maps_to_zero(self):
        arr = np.array([[0.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_outlier_clipped_to_pos_one(self):
        # Value 100 is way beyond q99=1 → clipped to 1.0
        arr = np.array([[100.0, 50.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_outlier_clipped_to_neg_one(self):
        arr = np.array([[-100.0, -50.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_quantile_missing_q01_raises(self):
        import pytest

        stats_no_q = {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.ones(2, dtype=np.float32),
            "min": -np.ones(2, dtype=np.float32),
            "max": np.ones(2, dtype=np.float32),
        }
        with pytest.raises(KeyError, match="quantile normalization requires"):
            apply_normalization(np.zeros((1, 2), dtype=np.float32), stats_no_q, "quantile")

    def test_quantile_no_op_for_null_stats(self):
        arr = np.array([[100.0, 100.0]], dtype=np.float32)
        out = apply_normalization(arr, None, "quantile")
        np.testing.assert_allclose(out, arr, atol=1e-7)

    def test_quantile_within_bounds_no_clip(self):
        arr = np.array([[0.5, 1.0], [-0.5, -1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        # No clipping for values within [q01, q99]
        expected = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)
