"""Tests for package-native evaluation metrics."""

import numpy as np
from PIL import Image

from open_wam.evaluation.metrics import compute_video_metrics


def test_compute_video_metrics_basic(monkeypatch):
    rng = np.random.default_rng(0)
    frames_a = [Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)) for _ in range(2)]
    frames_b = [Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)) for _ in range(2)]

    monkeypatch.setattr(
        "open_wam.evaluation.metrics._compute_lpips",
        lambda gen_np, gt_np: 0.0,
    )

    metrics = compute_video_metrics(frames_a, frames_b)
    assert set(metrics) == {"video_mse", "video_psnr", "video_ssim", "video_lpips"}
    assert metrics["video_mse"] > 0.0
    assert np.isfinite(metrics["video_psnr"])
    assert metrics["video_lpips"] == 0.0
