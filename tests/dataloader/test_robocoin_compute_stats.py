"""Tests for the RoboCOIN stats Accumulator (mean/std/min/max + reservoir q01/q99).

mean/std/min/max are streamed exactly; q01/q99 come from a bounded reservoir
sample, so they are asserted against the true quantiles with a small sampling
tolerance.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


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
