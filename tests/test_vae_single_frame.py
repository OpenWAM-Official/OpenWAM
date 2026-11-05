"""Full-clip dispatch preserves the native VAE's single-frame deploy path."""

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.model.kernels import vae_fast
from openwam.model.video_backbone.wan import conditioning
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.models.vae import VideoVAE38_, WanVideoVAE38


@pytest.fixture
def tiny_vae(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(vae_fast, "_FULL", True)
    monkeypatch.setattr(vae_fast, "_COMPILE", False)
    monkeypatch.setattr(vae_fast, "_CL", False)
    # Keep the production batch_encode dispatcher without constructing a full
    # checkpoint-sized model. The encoder and all its layers are real modules.
    vae = WanVideoVAE38.__new__(WanVideoVAE38)
    torch.nn.Module.__init__(vae)
    vae.model = VideoVAE38_(dim=8, z_dim=4, dec_dim=8, num_res_blocks=1).double().eval().requires_grad_(False)
    vae.scale = [torch.randn(4, dtype=torch.float64), torch.rand(4, dtype=torch.float64) + 0.5]
    vae.z_dim = 4
    vae.upsampling_factor = 16
    return vae


@pytest.mark.parametrize("num_frames", [1, 5, 9])
def test_batch_encode_dispatch_matches_native(tiny_vae, monkeypatch, num_frames):
    video = torch.randn(2, 3, num_frames, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        reference = tiny_vae.model.encode(video, tiny_vae.scale)
        assert vae_fast.supported(tiny_vae.model, video) is (num_frames > 1)

    native_encode = tiny_vae.model.encode
    fast_encode = vae_fast.encode
    native_calls, fast_calls = [], []

    def record_native(x, scale):
        native_calls.append(x.shape[2])
        return native_encode(x, scale)

    def record_fast(model, x, scale):
        fast_calls.append(x.shape[2])
        return fast_encode(model, x, scale)

    monkeypatch.setattr(tiny_vae.model, "encode", record_native)
    monkeypatch.setattr(vae_fast, "encode", record_fast)
    actual = tiny_vae.batch_encode(video, device="cpu")

    assert native_calls == ([1] if num_frames == 1 else [])
    assert fast_calls == ([num_frames] if num_frames > 1 else [])
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0 if num_frames == 1 else 1e-12)


def test_single_frame_does_not_compile(tiny_vae, monkeypatch):
    monkeypatch.setattr(vae_fast, "_COMPILE", True)
    monkeypatch.setattr(vae_fast, "_COMPILED", {})

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Single-frame encoding must not compile or enter the whole-clip encoder")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    video = torch.randn(1, 3, 1, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        reference = tiny_vae.model.encode(video, tiny_vae.scale)
    actual = tiny_vae.batch_encode(video, device="cpu")
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert vae_fast._COMPILED == {}


@pytest.mark.parametrize("image_as_list", [False, True])
def test_ti2v_deploy_first_frame_uses_native_encode(tiny_vae, monkeypatch, image_as_list):
    # Even multi-frame generation encodes the observed first frame separately.
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    pixels = wan_encode.preprocess_video([image], dtype=torch.float64, device="cpu")
    with torch.no_grad():
        reference = tiny_vae.model.encode(pixels, tiny_vae.scale)
    native_encode = tiny_vae.model.encode
    native_calls = []

    def record_native(x, scale):
        native_calls.append(x.shape[2])
        return native_encode(x, scale)

    monkeypatch.setattr(tiny_vae.model, "encode", record_native)
    inputs_shared = {"num_frames": 9}
    conditioning.finalize_ti2v_first_frame_latents(
        inputs_shared,
        [image] if image_as_list else image,
        is_ti2v=True,
        encoder=None,
        vae=tiny_vae,
        dtype=torch.float64,
        device="cpu",
    )
    assert native_calls == [1]
    assert inputs_shared["fuse_vae_embedding_in_latents"] is True
    assert inputs_shared["num_clean_prefix_frames"] == 0
    assert inputs_shared["first_frame_latents"].shape[2] == 1
    torch.testing.assert_close(inputs_shared["first_frame_latents"], reference, rtol=0, atol=0)


def test_disabled_fullclip_preserves_native_multiframe_encode(tiny_vae, monkeypatch):
    monkeypatch.setattr(vae_fast, "_FULL", False)

    def unexpected_fast_encode(*args, **kwargs):
        pytest.fail("The disabled full-clip encoder must not be used")

    monkeypatch.setattr(vae_fast, "encode", unexpected_fast_encode)
    video = torch.randn(1, 3, 9, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        reference = tiny_vae.model.encode(video, tiny_vae.scale)
    actual = tiny_vae.batch_encode(video, device="cpu")
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
