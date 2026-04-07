"""Video and action quality metrics for evaluation workflows."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_lpips_model = None


def _to_numpy_rgb(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return frame


def _compute_lpips(gen_np, gt_np):
    """Compute mean LPIPS over frame pairs."""
    global _lpips_model
    import lpips as _lpips_lib

    if _lpips_model is None:
        _lpips_model = _lpips_lib.LPIPS(net="alex").eval()
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()

    device = next(_lpips_model.parameters()).device
    vals = []
    for g, t in zip(gen_np, gt_np):
        g_t = torch.from_numpy(g).permute(2, 0, 1).float() / 127.5 - 1.0
        t_t = torch.from_numpy(t).permute(2, 0, 1).float() / 127.5 - 1.0
        with torch.no_grad():
            d = _lpips_model(g_t.unsqueeze(0).to(device), t_t.unsqueeze(0).to(device))
        vals.append(d.item())
    return float(np.mean(vals))


def compute_video_metrics(gen_frames, gt_frames):
    """Compute MSE, PSNR, SSIM, and LPIPS between generated and GT frames."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    gen_np = [_to_numpy_rgb(frame) for frame in gen_frames]
    gt_np = [_to_numpy_rgb(frame) for frame in gt_frames]
    n = min(len(gen_np), len(gt_np))
    gen_np, gt_np = gen_np[:n], gt_np[:n]

    mse_vals, psnr_vals, ssim_vals = [], [], []
    for g, t in zip(gen_np, gt_np):
        if g.shape != t.shape:
            g = np.array(Image.fromarray(g).resize((t.shape[1], t.shape[0]), Image.LANCZOS))
        mse_vals.append(float(np.mean((g.astype(np.float32) - t.astype(np.float32)) ** 2)))
        psnr_vals.append(peak_signal_noise_ratio(t, g, data_range=255))
        ssim_vals.append(structural_similarity(t, g, data_range=255, channel_axis=2))

    lpips_val = _compute_lpips(gen_np, gt_np)

    return {
        "video_mse": float(np.mean(mse_vals)),
        "video_psnr": float(np.mean(psnr_vals)),
        "video_ssim": float(np.mean(ssim_vals)),
        "video_lpips": float(lpips_val),
    }


__all__ = ["compute_video_metrics"]
