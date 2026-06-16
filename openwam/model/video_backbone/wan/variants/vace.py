"""VACE variant: conditioning rides entirely on ``vace_context``."""

from __future__ import annotations

from openwam.model.video_backbone.wan import conditioning
from openwam.model.video_backbone.wan.variants.base import WanVariant


class VACEVariant(WanVariant):
    """wan21_vace_1_3b. Build pixel-space (vace_video, vace_mask, ref_image=None)
    inputs — the batched equivalent of ``WanVideoUnit_VACE.process`` — and encode
    them into ``vace_context``. Video latents stay fully noised + fully
    supervised; no ``first_frame_latents`` is emitted (that key is TI2V's
    clean-replacement contract)."""

    def build_train_conditioning(
        self, bb, *, ref_images, vace_videos, stacked_inputs, B, num_frames, height, width, **kw
    ) -> dict:
        # Reuse the already-preprocessed input video so we don't re-decode the
        # first PIL frame from disk; ``stacked_inputs`` is in the same [-1, 1]
        # space the native unit produces after ``pipe.preprocess_video``.
        vace_video_pixels, vace_mask_pixels = conditioning.build_vace_pixel_inputs(
            vace_videos=vace_videos,
            first_frame_image=ref_images,
            B=B,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=stacked_inputs.dtype,
            device=stacked_inputs.device,
            encoder=bb.video_encoder,
            preprocessed_video=stacked_inputs,
        )
        return {
            "vace_context": conditioning.build_vace_context_from_pixels(
                vace_video_pixels, vace_mask_pixels, vae=bb.vae, encoder=bb.video_encoder, device=bb.device
            )
        }
