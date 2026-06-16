"""I2V variant: first-frame condition rides on clip_feature + y."""

from __future__ import annotations

import torch

from openwam.model.video_backbone.wan import conditioning
from openwam.model.video_backbone.wan.preprocess import preprocess_image
from openwam.model.video_backbone.wan.variants.base import WanVariant


class I2VVariant(WanVariant):
    """wan21_i2v_14b_480p. The conditioning image feeds two channels: a CLIP
    embedding (``clip_feature``) and a VAE-encoded first frame (``y``,
    channel-axis concatenated in ``prepare``). Both are gated on what the DiT
    declares it requires (``require_clip_embedding`` / ``require_vae_embedding``)."""

    def build_train_conditioning(
        self, bb, *, frames, ref_images, B, num_frames, height, width, device, dtype, **kw
    ) -> dict:
        dit = bb._dit
        image_encoder = getattr(bb, "image_encoder", None)
        needs_clip = bool(getattr(dit, "require_clip_embedding", False)) and image_encoder is not None
        needs_y = bool(getattr(dit, "require_vae_embedding", False))
        if not (needs_clip or needs_y):
            return {}

        # I2V conditioning image source priority:
        #   1. kw["first_frame_image"]: explicit override (deploy may pass).
        #   2. ref_images: training-time path — base.py always collects
        #      sample["first_frame_image"] into ref_images=...
        #   3. frames[i][0]: fallback to first frame of the GT video clip.
        first_frame_image = kw.get("first_frame_image")
        if first_frame_image is not None and not isinstance(first_frame_image, list):
            first_frame_image = [first_frame_image] * B
        if first_frame_image is None and ref_images is not None:
            first_frame_image = []
            for ref in ref_images:
                if isinstance(ref, list):
                    first_frame_image.append(ref[0])
                else:
                    first_frame_image.append(ref)
        if first_frame_image is None:
            first_frame_image = [clip[0] for clip in frames]
        if len(first_frame_image) != B:
            raise ValueError(f"first_frame_image batch ({len(first_frame_image)}) != frames batch ({B})")

        clip_feature = None
        y = None
        if needs_clip:
            clip_pieces = []
            for img in first_frame_image:
                img_t = preprocess_image(img.resize((width, height)), dtype=bb.dtype, device=bb.device).to(device)
                clip_pieces.append(image_encoder.encode_image([img_t]))
            clip_feature = torch.cat(clip_pieces, dim=0).to(dtype=dtype, device=device)

        if needs_y:
            y = conditioning.build_i2v_y(
                first_frame_image=first_frame_image,
                num_frames=num_frames,
                height=height,
                width=width,
                vae=bb.vae,
                dtype=dtype,
                device=device,
            )
        return {"clip_feature": clip_feature, "y": y}
