"""Whole-clip Wan2.2 VAE encode (OPENWAM_VAE_FULLCLIP) is the chunked causal encode of VideoVAE38_."""

import pytest
import torch

from openwam.model.kernels import vae_fast
from openwam.model.video_backbone.wan.models.vae import VideoVAE38_


@pytest.mark.parametrize("num_frames", [5, 9])
def test_whole_clip_encode_matches_chunked_encode(monkeypatch, num_frames):
    monkeypatch.setattr(vae_fast, "_COMPILE", False)
    monkeypatch.setattr(vae_fast, "_CL", False)
    torch.manual_seed(0)
    model = VideoVAE38_(dim=8, z_dim=4, dec_dim=8, num_res_blocks=1).double().eval()
    scale = [torch.randn(4, dtype=torch.float64), torch.rand(4, dtype=torch.float64) + 0.5]
    video = torch.randn(1, 3, num_frames, 32, 32, dtype=torch.float64)  # 1 + 4k frames
    with torch.no_grad():
        reference = model.encode(video, scale)
        whole_clip = vae_fast.encode(model, video, scale)
    assert whole_clip.shape == reference.shape and whole_clip.is_contiguous()
    torch.testing.assert_close(whole_clip, reference, rtol=0, atol=1e-12)
