"""TI2V variant: first frame is the clean conditioning latent."""

from __future__ import annotations

from openwam.model.video_backbone.wan.variants.base import WanVariant


class TI2VVariant(WanVariant):
    """wan22_ti2v_5b. ``latent[0]`` is the encoded clean first frame; the
    ``seperated_timestep`` path pins t=0 on frame-0 tokens and the loss skips it.
    ``fuse_vae_embedding_in_latents`` is only enabled when a reference frame is
    actually provided."""

    needs_first_frame_skip = True

    def build_train_conditioning(self, bb, *, input_latents, ref_images, **kw) -> dict:
        # Extract latent[0] from the already-encoded video latents (no second
        # VAE call, no prepend). ``base.compute_loss`` clean-replaces
        # ``latents[:, :, 0:1]`` with this on every step so the DiT sees
        # [clean ref, noisy 1..T_lat-1]; ``_compute_video_loss`` trims frame 0.
        has_ref = ref_images is not None and ref_images[0] is not None
        if not has_ref:
            return {}
        return {
            "first_frame_latents": input_latents[:, :, 0:1].clone(),
            "fuse_vae_embedding_in_latents": True,
        }
