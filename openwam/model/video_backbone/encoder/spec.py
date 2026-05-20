"""Structural contract a :class:`VideoEncoder` exposes to its host backbone.

The spec is derived from the loaded encoder weights, NOT from yaml fields.
This is intentional: yaml only lets users name an encoder and its weight
directory, and the encoder's :meth:`from_pretrained` populates the spec from
the actual loaded state. Backbone-side validators (e.g.
:meth:`VideoBackbone.validate_encoder_spec`) then fail-fast when an external
encoder is swapped in but its latent contract doesn't match the surrounding
DiT.

The two newer fields (``is_reversible`` / ``dit_patch_size``) decouple
non-VAE encoders from the historically-VAE assumptions: feature extractors
like DINOv3 declare ``is_reversible=False`` (no pixel decoder) and may set
``dit_patch_size=(1,1,1)`` when they already patchify at the desired scale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoEncoderSpec:
    """Latent contract exposed by a :class:`VideoEncoder`.

    Attributes:
        z_dim: Channel dimension of the latent grid produced by ``batch_encode``.
        spatial_compression: ``H_pixels / H_lat`` (assumes square spatial scaling).
        temporal_compression: ``T_pixels / T_lat``. For Wan VAE this is 4 with a
            causal first-frame token; for V-JEPA2's tubelet it is typically 2.
        causal_temporal: True if the first input frame is encoded into its own
            standalone latent token (Wan-style); False for uniform tubelet
            schedules.
        pixel_range: Range that ``preprocess_video`` is expected to produce.
            Not part of cross-encoder validation — purely informational so
            downstream consumers can sanity-check their inputs.
        is_reversible: Whether the encoder offers a pixel ``decode``. False is
            a hard contract: :meth:`VideoEncoder.decode` / ``to_frames`` are
            allowed to raise ``NotImplementedError``, the backbone-side
            ``decode_video`` and ``BaseWAMArchitecture.generate(decode_video=True)``
            both fail-fast, and the backbone skips the strict ``z_dim``
            equality check (since the DiT's first conv will be rebuilt at the
            encoder's z_dim by :func:`reinit_dit_from_scratch`).
        dit_patch_size: Spatio-temporal patch_size the host DiT applies on top
            of the encoder's already-compressed latent grid. Wan-family DiTs
            historically use ``(1, 2, 2)`` (no further temporal compression, 2x
            spatial); ViT-style encoders that already patch at 16x typically
            pair with ``(1, 1, 1)`` so the DiT's first conv becomes a pure
            channel projection. ``height/width_division_factor`` are derived
            as ``spatial_compression * dit_patch_size[1or2]``.
    """

    z_dim: int
    spatial_compression: int
    temporal_compression: int
    causal_temporal: bool
    pixel_range: tuple[float, float] = (-1.0, 1.0)
    is_reversible: bool = True
    dit_patch_size: tuple[int, int, int] = (1, 2, 2)


__all__ = ["VideoEncoderSpec"]
