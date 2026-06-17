"""Base Wan variant strategy."""

from __future__ import annotations


class WanVariant:
    """Strategy for a Wan family member's first-frame / control conditioning.

    The default (plain) variant adds no conditioning. Subclasses override
    :meth:`build_train_conditioning` to return their variant-specific keys
    (``vace_context`` / ``clip_feature`` + ``y`` / ``first_frame_latents`` + ...);
    ``WanBase.preprocess_input_for_train`` merges them into the shared
    output dict with defaults for the keys a given variant does not set.
    """

    # Whether ``latent[0]`` is unconditionally a conditioning frame for this
    # variant (TI2V's clean-replacement contract). Surfaced by
    # ``WanBase.needs_first_frame_skip``.
    needs_first_frame_skip: bool = False

    def build_train_conditioning(
        self,
        bb,
        *,
        input_latents,
        frames,
        ref_images,
        vace_videos,
        stacked_inputs,
        B,
        num_frames,
        height,
        width,
        device,
        dtype,
        **kw,
    ) -> dict:
        """Return variant-specific conditioning keys (default: none).

        ``bb`` is the :class:`WanBase`; variants call its component
        helpers rather than re-implementing encode/preprocess logic.
        """
        return {}
