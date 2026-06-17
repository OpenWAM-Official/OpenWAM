"""Native-VACE input convention parity + behavior tests (CPU-only).

Covers the refactor in :pull:`/* this PR */` that switched
``WanVideoBackbone.preprocess_input_for_train`` / ``preprocess_input_for_inference`` from the
old latent-space "manually inject ref frame" path to the native
pixel-space ``{vace_video, vace_video_mask, ref_image=None}`` convention.

Two things to pin:

1. ``WanVideoBackbone._build_vace_context_from_pixels`` is a batched
   reimplementation of the vendored ``WanVideoUnit_VACE.process`` (no
   ``vace_reference_image`` prepend). At B=1 the two paths must produce
   element-equal vace_context — otherwise the refactor introduces a real
   numerical shift instead of just code cleanup.

2. ``WanVideoBackbone._build_vace_pixel_inputs`` builds the canonical
   "know first frame, predict the rest" pixel-space inputs with the
   correct preprocessed-black (-1) padding and ``mask=[0, 1, 1, ...]``.

A minimal real ``WanVideoPipeline`` is sufficient — we plug a small fake VAE
matching the Wan VAE surface (``upsampling_factor``, ``batch_encode``,
``encode``, ``model.z_dim``) so we don't need to load real weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openwam.model.video_backbone.wan import conditioning

# ---------------------------------------------------------------------------
# Fake Wan VAE: just enough surface for the helper + the vendored unit.
# ---------------------------------------------------------------------------


class _FakeWanVAE:
    """Deterministic, parameter-free stand-in for ``WanVideoVAE``.

    The "encode" maps pixel (B, 3, T, H, W) → latent (B, z_dim, T_lat, H/8,
    W/8) by:
      - temporal: causal-style (T - 1) // 4 + 1 grouping, first frame solo.
      - spatial: 8x downsample via avg-pool.
      - channels: broadcast 3 → z_dim by tile+pad.

    Crucially the mapping is *content-sensitive* — different pixel values map
    to different latents, so the parity check is meaningful (a constant-output
    encoder would trivially pass).
    """

    upsampling_factor = 8

    def __init__(self, z_dim: int = 16):
        self.z_dim = z_dim
        self.model = SimpleNamespace(z_dim=z_dim)

    def _encode_single(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, 3, T, H, W) in [-1, 1].
        B, C, T, H, W = pixels.shape
        assert C == 3
        assert H % 8 == 0 and W % 8 == 0

        # Spatial 8x avg-pool, per frame.
        x = pixels.reshape(B * T, C, H, W)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=8, stride=8)
        x = x.reshape(B, C, T, H // 8, W // 8)

        # Temporal causal grouping: latent[0] from frame 0 alone; subsequent
        # latents pool the next 4 frames each.
        if T == 0:
            return torch.zeros(B, self.z_dim, 0, H // 8, W // 8, dtype=pixels.dtype, device=pixels.device)
        head = x[:, :, 0:1]
        tail_frames = x[:, :, 1:]
        T_tail = tail_frames.shape[2]
        T_lat_tail = (T_tail + 3) // 4
        if T_tail > 0:
            # Pad temporal to a multiple of 4 so reshape works, then mean-pool.
            pad_len = T_lat_tail * 4 - T_tail
            if pad_len > 0:
                pad = torch.zeros(B, C, pad_len, H // 8, W // 8, dtype=pixels.dtype, device=pixels.device)
                tail_frames = torch.cat([tail_frames, pad], dim=2)
            tail_grouped = tail_frames.reshape(B, C, T_lat_tail, 4, H // 8, W // 8).mean(dim=3)
            x_lat = torch.cat([head, tail_grouped], dim=2)
        else:
            x_lat = head

        # Channel 3 → z_dim by tiling + bias (content-sensitive).
        reps = self.z_dim // C
        rem = self.z_dim - reps * C
        x_lat = torch.cat([x_lat] * reps + ([x_lat[:, :rem]] if rem > 0 else []), dim=1)
        # Add a small position-dependent bias so different pixel patterns are
        # not collapsed by the tiling.
        bias = torch.linspace(0, 0.1, self.z_dim, dtype=x_lat.dtype, device=x_lat.device)
        x_lat = x_lat + bias.view(1, -1, 1, 1, 1)
        return x_lat

    def batch_encode(self, videos, device):
        videos = videos.to(device)
        return self._encode_single(videos)

    def encode(self, videos, device, **kwargs):
        # Vendored unit + tiled-encode helper both pass a *list* of
        # ``(3, T, H, W)`` tensors here; the parity test passes a single
        # ``(1, 3, T, H, W)`` tensor through the non-list code path.
        # Normalize to a batched (B, 3, T, H, W) tensor before encoding.
        if isinstance(videos, list):
            stacked = torch.stack([v if v.dim() == 4 else v.unsqueeze(0)[0] for v in videos], dim=0)
            return self._encode_single(stacked)
        return self._encode_single(videos)

    def decode(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Builders for the adapter and the vendored unit, sharing the fake VAE.
# ---------------------------------------------------------------------------


def _make_adapter_with_fake_vae(*, vace: bool = True, image_input: bool = False):
    from openwam.model.video_backbone.wan_backbone import Wan21

    vae = _FakeWanVAE(z_dim=16)
    dit = SimpleNamespace(
        seperated_timestep=False,
        fuse_vae_embedding_in_latents=False,
        has_image_input=image_input,
        require_clip_embedding=False,
        require_vae_embedding=False,
    )
    pipe = SimpleNamespace(
        dit=dit,
        vae=vae,
        vace=object() if vace else None,
        text_encoder=None,
        image_encoder=None,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        scheduler=None,
        units=[],
        use_unified_sequence_parallel=False,
    )

    def _preprocess_image(image, *, torch_dtype=None, device=None, min_value=-1, max_value=1, **kw):
        # PIL → (1, 3, H, W) in [min_value, max_value]; mirrors wan base_pipeline.
        import numpy as np

        arr = torch.from_numpy(np.array(image, dtype=np.float32))
        if arr.dim() == 2:
            arr = arr.unsqueeze(-1).expand(-1, -1, 3)
        arr = arr * ((max_value - min_value) / 255.0) + min_value
        arr = arr.to(dtype=torch_dtype or torch.float32, device=device or "cpu")
        return arr.permute(2, 0, 1).unsqueeze(0)

    def _preprocess_video(video, *, min_value=-1, max_value=1, **kw):
        frames = [_preprocess_image(img, min_value=min_value, max_value=max_value) for img in video]
        return torch.stack(frames, dim=2)[0].unsqueeze(0)

    pipe.preprocess_image = _preprocess_image
    pipe.preprocess_video = _preprocess_video

    bb = Wan21(pipe)
    bb._device = torch.device("cpu")
    bb._dtype = torch.float32
    return bb


def _make_pil_first_frame(H: int = 32, W: int = 32, seed: int = 0):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_pixel_inputs_default_first_frame_condition():
    """``_build_vace_pixel_inputs`` with first_frame_image and no
    vace_videos produces the canonical [first_frame, black, ...] +
    [0, 1, ..., 1] form."""
    bb = _make_adapter_with_fake_vae()
    H = W = 32
    T = 13
    first_frame = _make_pil_first_frame(H=H, W=W, seed=42)
    vace_video_pixels, vace_mask_pixels = conditioning.build_vace_pixel_inputs(
        vace_videos=None,
        first_frame_image=[first_frame],
        B=1,
        num_frames=T,
        height=H,
        width=W,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Shape contract.
    assert vace_video_pixels.shape == (1, 3, T, H, W)
    assert vace_mask_pixels.shape == (1, 1, T, H, W)

    # t=0: first frame content, mask=0.
    from openwam.model.video_backbone.wan.preprocess import preprocess_image

    expected_first_pp = preprocess_image(first_frame.resize((W, H)), dtype=bb.dtype, device=bb.device)[0]
    assert torch.allclose(vace_video_pixels[0, :, 0], expected_first_pp)
    assert torch.all(vace_mask_pixels[0, :, 0] == 0)

    # t>0: preprocessed-black padding (-1), mask=1.
    assert torch.all(vace_video_pixels[0, :, 1:] == -1.0)
    assert torch.all(vace_mask_pixels[0, :, 1:] == 1.0)


def test_pixel_inputs_unconditional():
    """No ref + no user vace_video → unconditional padding (all-black, all-ones-mask)."""
    vp, vm = conditioning.build_vace_pixel_inputs(
        vace_videos=None,
        first_frame_image=None,
        B=2,
        num_frames=9,
        height=32,
        width=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.all(vp == -1.0)
    assert torch.all(vm == 1.0)


def test_preprocess_input_vace_drops_first_frame_latents(monkeypatch):
    """Training preprocess for a VACE backbone must NOT emit
    ``first_frame_latents`` (its loss is full-frame; latent[0] is a predicted
    frame, not a clean reference). Companion to the updated
    ``needs_first_frame_skip`` truth table.

    We stub out ``_encode_text`` (no text encoder in the fake pipe) and
    ``_check_resize`` (no pipe.check_resize_height_width) so this stays a
    pure-CPU smoke that exercises the dict contract.
    """
    bb = _make_adapter_with_fake_vae(vace=True)
    import openwam.model.video_backbone.wan.encode as enc_mod
    import openwam.model.video_backbone.wan_backbone as vbb

    monkeypatch.setattr(
        enc_mod,
        "encode_text",
        lambda prompts, **kw: (torch.zeros(1, 4, 32), torch.ones(1, dtype=torch.long)),
    )
    # encode_text is mocked, but ``text_encoder=self.text_encoder`` is still
    # evaluated as a call arg; the fake pipe has no text_encoder, so stub it.
    monkeypatch.setattr(bb, "text_encoder", None, raising=False)
    monkeypatch.setattr(vbb, "check_resize_height_width", lambda h, w, t, **kw: (h, w, t))

    H = W = 32
    T = 13
    frames = [_make_pil_first_frame(H=H, W=W, seed=i) for i in range(T)]
    out = bb.preprocess_input_for_train(
        frames=[frames],
        text=["dummy"],
        vace_videos=[None],
        ref_images=[[frames[0]]],
    )
    assert out["vace_context"] is not None
    assert out["first_frame_latents"] is None, (
        "VACE training path must not emit first_frame_latents — that contract is "
        "reserved for TI2V's clean-replacement loop."
    )
    assert out["fuse_vae_embedding_in_latents"] is False
    assert out["num_clean_prefix_frames"] == 0


def test_build_vace_context_for_deploy_train_parity():
    """Deploy's ``_build_vace_context_for_deploy`` must produce the same
    vace_context as the training path for the same first_frame_image input.

    This is the train/deploy parity guarantee — the symmetric counterpart to
    the parity-vs-vendored test. Catches drift if either branch's pixel-input
    construction starts emitting different (vace_video, vace_mask) pairs.
    """
    bb = _make_adapter_with_fake_vae(vace=True)
    H = W = 32
    T = 13
    first_frame = _make_pil_first_frame(H=H, W=W, seed=11)

    # Training-side pixel construction.
    train_vp, train_vm = conditioning.build_vace_pixel_inputs(
        vace_videos=None,
        first_frame_image=[first_frame],
        B=1,
        num_frames=T,
        height=H,
        width=W,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    train_ctx = conditioning.build_vace_context_from_pixels(
        train_vp, train_vm, vae=bb.vae, encoder=bb.video_encoder, device=bb.device
    )

    # Deploy-side construction.
    inputs_shared = {"num_frames": T, "height": H, "width": W}
    conditioning.build_vace_context_for_deploy(
        inputs_shared,
        first_frame,
        vace_video=None,
        has_vace=bb._has_vace,
        vae=bb.vae,
        encoder=bb.video_encoder,
        dtype=bb.dtype,
        device=bb.device,
    )
    deploy_ctx = inputs_shared["vace_context"]

    assert deploy_ctx.shape == train_ctx.shape
    assert torch.allclose(deploy_ctx, train_ctx, atol=1e-6, rtol=1e-6), (
        f"train/deploy vace_context diverged: max abs = {(deploy_ctx - train_ctx).abs().max().item():.3e}"
    )


def test_build_vace_context_for_deploy_noop_for_non_vace():
    """Non-VACE backbones must leave inputs_shared untouched."""
    bb = _make_adapter_with_fake_vae(vace=False)
    inputs_shared = {"num_frames": 13, "height": 32, "width": 32, "vace_context": None}
    conditioning.build_vace_context_for_deploy(
        inputs_shared,
        first_frame_image=None,
        vace_video=None,
        has_vace=bb._has_vace,
        vae=bb.vae,
        encoder=bb.video_encoder,
        dtype=bb.dtype,
        device=bb.device,
    )
    assert inputs_shared["vace_context"] is None


def test_build_vace_context_for_deploy_forwards_tiled_kwargs(monkeypatch):
    """``_build_vace_context_for_deploy`` must forward ``tiled`` / ``tile_size`` /
    ``tile_stride`` from inputs_shared to the encode helper.

    Otherwise deploy at 480x832 / 720x1280 silently regresses from tiled VAE
    encode (vendored unit's default) to full-frame batch_encode and OOMs on
    a single GPU. This is the issue codex flagged on first review.
    """
    bb = _make_adapter_with_fake_vae(vace=True)
    # Spy on wan.encode.encode_video_for_vace to inspect the tiled kwargs the
    # caller threaded through.
    import openwam.model.video_backbone.wan.encode as enc_mod

    seen: list[dict] = []
    original = enc_mod.encode_video_for_vace

    def _spy(pixels, *, vae, encoder=None, device, tiled, tile_size, tile_stride):
        seen.append({"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride})
        return original(
            pixels, vae=vae, encoder=encoder, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )

    monkeypatch.setattr(enc_mod, "encode_video_for_vace", _spy)

    inputs_shared = {
        "num_frames": 13,
        "height": 32,
        "width": 32,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
    }
    conditioning.build_vace_context_for_deploy(
        inputs_shared,
        first_frame_image=_make_pil_first_frame(H=32, W=32, seed=1),
        vace_video=None,
        has_vace=bb._has_vace,
        vae=bb.vae,
        encoder=bb.video_encoder,
        dtype=bb.dtype,
        device=bb.device,
    )

    # Two encode calls (inactive + reactive); both must see tiled=True.
    assert len(seen) == 2
    for call in seen:
        assert call["tiled"] is True
        assert call["tile_size"] == (30, 52)
        assert call["tile_stride"] == (15, 26)
