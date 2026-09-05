"""First-frame / control conditioning construction for Wan variants (I2V / TI2V
/ VACE), train + deploy.

Free functions — every backbone-owned input (``dit`` / ``vae`` / ``encoder`` /
``image_encoder`` / ``dtype`` / ``device`` / ``latent_spec`` / variant flags) is
passed explicitly so this module stays Wan-internal (no backbone/ABC import).
Encode/decode goes through :mod:`wan.encode`; pixel helpers through
:mod:`wan.preprocess`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor

from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.preprocess import generate_noise, preprocess_image


def resolve_i2v_input_image(first_frame_image, *, dit, is_ti2v, has_vace):
    """Normalize ``first_frame_image`` to a single PIL (or None) for I2V deploy:
    I2V CLIP/VAE units ``.resize`` the value and choke on a list.
    """
    # Defensive ``getattr`` is for test mocks that set these as plain fields.
    is_i2v = bool(getattr(dit, "has_image_input", False)) and not is_ti2v and not has_vace
    if not is_i2v or first_frame_image is None:
        return None
    if isinstance(first_frame_image, (list, tuple)):
        if len(first_frame_image) != 1:
            raise ValueError(f"I2V deploy expects a single first-frame image; got list of {len(first_frame_image)}.")
        return first_frame_image[0]
    return first_frame_image


def build_deploy_noise(*, height, width, num_frames, seed, rand_device, latent_spec, vae, dtype, device) -> Tensor:
    """Initial Gaussian latent noise for deploy (replaces NoiseInitializer)."""
    spec = latent_spec
    if spec is not None:
        z_dim = spec.z_dim
        upsample = spec.spatial_compression
        length = (num_frames - 1) // spec.temporal_compression + (1 if spec.causal_temporal else 0)
    else:
        z_dim = vae.model.z_dim
        upsample = vae.upsampling_factor
        length = (num_frames - 1) // 4 + 1
    shape = (1, z_dim, length, height // upsample, width // upsample)
    return generate_noise(shape, seed=seed, rand_device=rand_device, dtype=dtype, device=device)


def build_deploy_i2v_clip(input_image, *, height, width, dit, image_encoder, dtype, device) -> Optional[Tensor]:
    """I2V CLIP feature (replaces ImageEmbedderCLIP); None if the DiT/encoder gate fails."""
    if image_encoder is None or not dit.require_clip_embedding:
        return None
    image = preprocess_image(input_image.resize((width, height)), dtype=dtype, device=device).to(device)
    clip_context = image_encoder.encode_image([image])
    return clip_context.to(dtype=dtype, device=device)


def build_deploy_i2v_y(
    input_image, *, num_frames, height, width, tiled, tile_size, tile_stride, dit, vae, dtype, device
) -> Optional[Tensor]:
    """I2V VAE conditioning ``y``; None if the DiT gate fails. Uses the tiled
    per-sample ``vae.encode`` (not training's batched ``build_i2v_y``) so deploy
    ``tiled=True`` matches the vendored unit bit-for-bit.
    """
    if not dit.require_vae_embedding:
        return None
    image = preprocess_image(input_image.resize((width, height)), dtype=dtype, device=device).to(device)
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=device)
    msk[:, 1:] = 0
    vae_input = torch.concat(
        [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)], dim=1
    )
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]
    y = vae.encode(
        [vae_input.to(dtype=dtype, device=device)],
        device=device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=device)
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    y = y.to(dtype=dtype, device=device)
    return y


def finalize_ti2v_first_frame_latents(inputs_shared: dict, first_frame_image, *, is_ti2v, encoder, vae, dtype, device):
    """Emit ``first_frame_latents`` for TI2V deploy: its ``seperated_timestep``
    DiT needs both ``fuse_vae_embedding_in_latents=True`` AND ``first_frame_latents``
    so frame-0 tokens get t=0 and base.generate can clean-replace ``latents[:, :, 0:1]``.
    VACE/I2V skip this — their first-frame rides ``vace_context`` / ``y``.
    """
    if not is_ti2v or first_frame_image is None:
        if first_frame_image is None:
            inputs_shared.pop("first_frame_latents", None)
            inputs_shared["fuse_vae_embedding_in_latents"] = False
            inputs_shared["num_clean_prefix_frames"] = 0
        return
    inputs_shared["fuse_vae_embedding_in_latents"] = True
    inputs_shared["num_clean_prefix_frames"] = 0
    ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
    ref_tensor = wan_encode.preprocess_video(ref_frames, encoder=encoder, dtype=dtype, device=device)
    ref_image_latents = wan_encode.encode_video(ref_tensor.to(device), vae=vae, encoder=encoder).to(
        dtype=dtype, device=device
    )
    inputs_shared["first_frame_latents"] = ref_image_latents


# ================================================================
# Native-VACE input convention (training + deploy)
# ================================================================
# These helpers replicate ``WanVideoUnit_VACE.process`` except they (a) accept a
# batch (B >= 1) where the vendored unit assumes B=1, and (b) skip the
# ``vace_reference_image`` prepend — OpenWAM uses the canonical
# ``[first_frame, black...]`` / mask ``[0, 1...]`` form, keeping
# ``vace_context.shape[2] == video_latent.shape[2]``.
def build_vace_pixel_inputs(
    *,
    vace_videos,
    first_frame_image,
    B: int,
    num_frames: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    encoder=None,
    preprocessed_video: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Build the pixel-space ``(vace_video, vace_video_mask)`` pair (shapes
    ``(B, 3, T, H, W)`` in [-1,1] and ``(B, 1, T, H, W)`` in [0,1]).

    Three branches by priority: (1) user-provided ``vace_videos[i]`` verbatim
    (mask all-ones; unreachable in training today); (2) ``first_frame_image`` →
    ``[first_frame, black...]`` / mask ``[0, 1...]`` (default); (3) neither →
    unconditional all-black / all-ones. ``preprocessed_video``, when given under
    branch (2), is sliced ``[:,:,0:1]`` to reuse the already-preprocessed first frame.
    """
    # Padding init = preprocessed black (-1), matching ``preprocess_video``'s
    # RGB(0) → -1. NOT torch.zeros — preprocessed-0 is *gray*.
    vace_video_pixels = torch.full((B, 3, num_frames, height, width), fill_value=-1.0, dtype=dtype, device=device)
    vace_mask_pixels = torch.ones((B, 1, num_frames, height, width), dtype=dtype, device=device)

    has_ref = first_frame_image is not None
    user_provided_any = vace_videos is not None and any(vv is not None for vv in vace_videos)

    if user_provided_any:
        for i in range(B):
            vv = vace_videos[i] if vace_videos is not None else None
            if vv is not None:
                vv_pp = wan_encode.preprocess_video(vv, encoder=encoder, dtype=dtype, device=device).to(
                    dtype=dtype, device=device
                )
                if vv_pp.shape[2] != num_frames:
                    raise ValueError(
                        f"User-provided vace_videos[{i}] has T={vv_pp.shape[2]} but "
                        f"num_frames={num_frames}; this branch does not auto pad/truncate."
                    )
                vace_video_pixels[i] = vv_pp[0]
                # User-supplied vace_video → predict every frame; mask stays all-ones.
            elif has_ref:
                fill_first_frame_condition(
                    vace_video_pixels[i : i + 1],
                    vace_mask_pixels[i : i + 1],
                    ref_image=first_frame_image[i] if isinstance(first_frame_image, list) else first_frame_image,
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=device,
                    preprocessed_first_frame=(
                        preprocessed_video[i : i + 1, :, 0:1] if preprocessed_video is not None else None
                    ),
                )
    elif has_ref:
        # Batch-wide first-frame condition: zero the t=0 mask channel.
        vace_mask_pixels[:, :, 0:1] = 0.0
        if preprocessed_video is not None and preprocessed_video.shape[0] == B:
            # Fast path: reuse the preprocessed input video's first frame.
            vace_video_pixels[:, :, 0:1] = preprocessed_video[:, :, 0:1].to(dtype=dtype, device=device)
        else:
            for i in range(B):
                ref = first_frame_image[i] if isinstance(first_frame_image, list) else first_frame_image
                if isinstance(ref, list):
                    ref = ref[0]
                pp = preprocess_image(ref.resize((width, height)), dtype=dtype, device=device).to(
                    device=device, dtype=dtype
                )
                if pp.dim() == 4 and pp.shape[0] == 1:
                    pp = pp[0]
                vace_video_pixels[i, :, 0] = pp
    # else: unconditional — keep all-black, all-ones-mask.

    return vace_video_pixels, vace_mask_pixels


def fill_first_frame_condition(
    vace_video_slice: Tensor,
    vace_mask_slice: Tensor,
    *,
    ref_image,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    preprocessed_first_frame: Optional[Tensor] = None,
) -> None:
    """In-place fill of one sample's vace_video[t=0] + vace_mask[t=0]. The
    ``(1,3,T,H,W)`` / ``(1,1,T,H,W)`` slices start all-black / all-ones.
    """
    vace_mask_slice[:, :, 0:1] = 0.0
    if preprocessed_first_frame is not None:
        vace_video_slice[:, :, 0:1] = preprocessed_first_frame.to(dtype=dtype, device=device)
        return
    ref = ref_image[0] if isinstance(ref_image, list) else ref_image
    pp = preprocess_image(ref.resize((width, height)), dtype=dtype, device=device).to(device=device, dtype=dtype)
    if pp.dim() == 4 and pp.shape[0] == 1:
        pp = pp[0]
    vace_video_slice[0, :, 0] = pp


def build_vace_context_from_pixels(
    vace_video_pixels: Tensor,
    vace_mask_pixels: Tensor,
    *,
    vae,
    encoder=None,
    device,
    tiled: bool = False,
    tile_size: tuple = (34, 34),
    tile_stride: tuple = (18, 16),
) -> Tensor:
    """Batched pixel→latent conversion of ``WanVideoUnit_VACE.process`` (no
    ref-prepend), output ``(B, 96, T_lat, H_lat, W_lat) = concat([inactive
    (z=16), reactive(z=16), mask(P*Q=64)])``.

    P=Q=8 and ``(T_pix+3)//4`` are baked into the VACE pretrained weights
    (``vace_in_dim=96``) / Wan VAE causal 4x. ``tiled`` must be forwarded from
    deploy so large-frame VACE encode does not regress to full-frame and OOM.
    """
    import torch.nn.functional as F

    # Native's redundant ``+ 0 * ...`` terms dropped (identical at B=1).
    inactive = vace_video_pixels * (1 - vace_mask_pixels)
    reactive = vace_video_pixels * vace_mask_pixels
    inactive_lat = wan_encode.encode_video_for_vace(
        inactive, vae=vae, encoder=encoder, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
    )
    reactive_lat = wan_encode.encode_video_for_vace(
        reactive, vae=vae, encoder=encoder, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
    )
    vace_video_latents = torch.cat([inactive_lat, reactive_lat], dim=1)

    P, Q = 8, 8
    if vace_mask_pixels.shape[3] % P != 0 or vace_mask_pixels.shape[4] % Q != 0:
        raise ValueError(
            f"vace_mask_pixels spatial dims ({vace_mask_pixels.shape[3]}, "
            f"{vace_mask_pixels.shape[4]}) must be divisible by (P=8, Q=8) "
            f"for the native VACE rearrange (pretrained vace_in_dim=96 "
            f"requires this exact tile layout)."
        )
    # Batched form of native's ``rearrange(mask[0,0], "T (H P) (W Q) -> ...")``.
    vace_mask_latents = rearrange(vace_mask_pixels[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=P, Q=Q)
    T_pix = vace_mask_latents.shape[2]
    T_lat = (T_pix + 3) // 4
    vace_mask_latents = F.interpolate(
        vace_mask_latents,
        size=(T_lat, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
        mode="nearest-exact",
    )

    return torch.cat([vace_video_latents, vace_mask_latents], dim=1)


def build_vace_context_for_deploy(
    inputs_shared: dict, first_frame_image, vace_video, *, has_vace, vae, encoder, dtype, device
) -> None:
    """Deploy wrapper around :func:`build_vace_context_from_pixels`, mirroring the
    training path so the two produce bit-equivalent ``vace_context``. Forwards
    ``tiled`` so large-frame deploy keeps the tiled encode and does not OOM.
    """
    if not has_vace:
        return
    num_frames = inputs_shared["num_frames"]
    height = inputs_shared["height"]
    width = inputs_shared["width"]
    dtype = dtype if dtype is not None else torch.bfloat16

    vace_videos = [vace_video] if vace_video is not None else None
    ff_list = first_frame_image if first_frame_image is not None else None
    if ff_list is not None and not isinstance(ff_list, list):
        ff_list = [ff_list]

    vace_video_pixels, vace_mask_pixels = build_vace_pixel_inputs(
        vace_videos=vace_videos,
        first_frame_image=ff_list,
        B=1,
        num_frames=num_frames,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        encoder=encoder,
    )
    vace_context = build_vace_context_from_pixels(
        vace_video_pixels,
        vace_mask_pixels,
        vae=vae,
        encoder=encoder,
        device=device,
        tiled=bool(inputs_shared.get("tiled", False)),
        tile_size=tuple(inputs_shared.get("tile_size") or (34, 34)),
        tile_stride=tuple(inputs_shared.get("tile_stride") or (18, 16)),
    )
    inputs_shared["vace_context"] = vace_context


def build_i2v_y(*, first_frame_image: list, num_frames: int, height: int, width: int, vae, dtype, device) -> Tensor:
    """Build the Wan2.1-I2V ``y`` batch-wise: ``(B, 20, T_lat, H_lat, W_lat) =
    concat([msk(4), vae_y(16)])``. Encodes the whole batch via non-tiled
    ``batch_encode`` (training already runs the VAE non-tiled).
    """
    vae_inputs = []
    msks = []
    for img in first_frame_image:
        image = preprocess_image(img.resize((width, height)), dtype=dtype, device=device).to(device)  # (1, 3, H, W)
        vae_input = torch.cat(
            [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width, device=device)],
            dim=1,
        )  # (3, num_frames, H, W)
        vae_inputs.append(vae_input)

        msk = torch.ones(1, num_frames, height // 8, width // 8, device=device)
        msk[:, 1:] = 0
        msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]  # (4, T_lat, H_lat, W_lat)
        msks.append(msk)

    vae_inputs_b = torch.stack(vae_inputs, dim=0).to(dtype=dtype, device=device)  # (B, 3, T, H, W)
    msks_b = torch.stack(msks, dim=0).to(dtype=dtype, device=device)  # (B, 4, T_lat, H_lat, W_lat)
    y_lat = vae.batch_encode(vae_inputs_b, device=device).to(dtype=dtype, device=device)
    y = torch.cat([msks_b, y_lat], dim=1)  # (B, 20, T_lat, H_lat, W_lat)
    return y
