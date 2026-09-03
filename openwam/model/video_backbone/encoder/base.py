"""VideoEncoder ABC and the structural spec it exposes to the host backbone.

:class:`VideoEncoderProperties` is the latent contract (z_dim / compression / patch
geometry) derived from the loaded encoder weights — NOT from yaml. The encoder's
``from_pretrained`` populates it from the actual loaded state.

:class:`VideoEncoder` is the ABC each pluggable encoder subclasses. The package
``__init__`` re-exports both alongside the registry + factory.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.encoder.svae import reducer
from openwam.model.video_backbone.encoder.svae.model import SVAE


@dataclass(frozen=True)
class VideoEncoderProperties:
    """Latent contract exposed by a :class:`VideoEncoder`.

    Attributes:
        z_dim: Channel dimension of the latent grid produced by ``batch_encode``.
        spatial_compression: ``H_pixels / H_lat`` (assumes square spatial scaling).
        temporal_compression: ``T_pixels / T_lat``. For Wan VAE this is 4 with a
            causal first-frame token; for V-JEPA 2 / 2.1 it is also 4 (ViT
            tubelet=2 + encoder-side avg-pool over time with stride=2 to match
            Wan VAE causal grouping).
        causal_temporal: True if the first input frame is encoded into its own
            standalone latent token (Wan-style); False for uniform tubelet
            schedules.
        pixel_decode: Whether the encoder offers a pixel ``decode``. False is
            a hard contract: :meth:`VideoEncoder.decode` / ``to_frames`` are
            allowed to raise ``NotImplementedError``, the backbone-side
            ``decode_video`` and ``BaseWAMArchitecture.generate(decode_video=True)``
            both fail-fast, and the backbone skips the strict ``z_dim``
            equality check (since the DiT's first conv will be rebuilt at the
            encoder's z_dim by :func:`reinit_dit_from_scratch`).
        dit_patch_size: Spatio-temporal patch_size the host DiT applies on top
            of the encoder's already-compressed latent grid. Wan-family DiTs
            historically use ``(1, 2, 2)`` (no further temporal compression, 2x
            spatial); the shipped non-VAE encoders (V-JEPA 2, V-JEPA 2.1) also
            use ``(1, 2, 2)`` so their per-frame token grid matches Wan VAE's
            after the DiT's first conv — token-
            count parity is what lets the same Wan DiT consume either latent
            stream interchangeably. A ViT-style encoder that already patches
            at the DiT's target token-grid scale can set ``(1, 1, 1)`` to make
            the DiT's first conv a pure channel projection.
            ``height/width_division_factor`` are derived as
            ``spatial_compression * dit_patch_size[1or2]``.
    """

    z_dim: int
    spatial_compression: int
    temporal_compression: int
    causal_temporal: bool
    pixel_decode: bool = True
    dit_patch_size: tuple[int, int, int] = (1, 2, 2)


class VideoEncoder(ABC, nn.Module):
    """Swappable video latent codec.

    Activated only when ``video_backbone.from_scratch=true`` AND
    ``video_backbone.encoder`` is set in yaml; otherwise the backbone's
    native ``pipe.vae`` is used and ``state_dict`` keys remain bit-exact
    with the upstream pretrained checkpoint.

    Subclasses MUST implement ``properties`` / ``preprocess_video`` / ``batch_encode`` /
    ``from_pretrained``. They MAY implement ``decode`` / ``to_frames`` (only
    when ``properties.pixel_decode=True``) and MAY override
    ``build_dit_input_proj`` / ``build_dit_output_proj`` when the default
    Wan-style projection is not appropriate. An encoder that wants to compress
    its raw features MAY hold an optional frozen S-VAE reducer — see
    :mod:`openwam.model.video_backbone.encoder.svae.reducer`.
    """

    def __init__(self) -> None:
        super().__init__()
        # Optional frozen S-VAE reducer; ``None`` = disabled. A subclass opts in
        # from its own ``__init__`` via ``self._svae = reducer.build(...)`` and
        # routes batch_encode through :meth:`_apply_svae`. The SVAE network +
        # build / sidecar plumbing live in :mod:`...svae.reducer`.
        self._svae: SVAE | None = None

    # ------------------------------------------------------------------
    # Required: latent contract + per-step IO
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def properties(self) -> VideoEncoderProperties:
        """Structural contract derived from loaded weights, not yaml."""

    @abstractmethod
    def preprocess_video(self, frames) -> Tensor:
        """List[PIL.Image] -> ``(B=1, 3, T, H, W)`` tensor."""

    @abstractmethod
    def batch_encode(self, video: Tensor) -> Tensor:
        """Pixel video ``(B, 3, T, H, W)`` -> latent ``(B, z_dim, T_lat, H_lat, W_lat)``.

        Hot path on every training step; tiled encoding is NOT required.
        """

    # ------------------------------------------------------------------
    # Optional: pixel decode. Default raises with a contract-aware message.
    # ------------------------------------------------------------------

    def decode(self, latents: Tensor, *, tiled: bool = True) -> Tensor:
        """Decode latent -> pixel video. Optional; only valid when
        ``properties.pixel_decode=True``."""
        raise NotImplementedError(
            f"{type(self).__name__}.decode unavailable "
            f"(properties.pixel_decode={self.properties.pixel_decode}). "
            "Use latent-level metrics for training, or train a separate "
            "pixel decoder if you need to visualize generated samples."
        )

    def to_frames(self, video_tensor: Tensor) -> list:
        """``(B, 3, T, H, W)`` pixel tensor -> ``list[PIL.Image]``. Optional;
        same constraint as :meth:`decode`."""
        raise NotImplementedError(
            f"{type(self).__name__}.to_frames unavailable (properties.pixel_decode={self.properties.pixel_decode})."
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str, **kw: Any) -> "VideoEncoder":
        """Construct from a weights directory. ``model_path`` is the only
        always-required user-facing argument (read from yaml). Any other
        non-null yaml field is forwarded by ``build_video_encoder`` as a kwarg;
        the subclass signature is the contract — an explicit signature raises
        ``TypeError`` on an unknown field, a ``**kw`` one ignores extras."""

    # ------------------------------------------------------------------
    # Deploy-time skeleton constructor
    # ------------------------------------------------------------------
    # Training saves the underlying module's class + extra_kwargs as a
    # ``components`` entry under ``video_backbone.components`` in
    # config.yaml (see component_specs.py). Deploy needs a way to
    # reconstruct the encoder structure without re-reading the source
    # ``model_path`` (which may be unreachable on the deploy host) —
    # checkpoint weights are loaded immediately after via the
    # architecture's ``load_checkpoint`` strict load. Subclasses whose
    # underlying weights live as a Wan ``components`` entry should
    # override; encoders whose weight files don't fit that mold can
    # leave the default in place and document a different deploy
    # workflow.

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "VideoEncoder":
        """Build a zero-weight encoder skeleton. The host architecture's
        checkpoint strict-load fills in weights immediately after this call.

        Subclasses that override **must** keep all three kwargs in their
        signature — ``base.py:_build_external_encoder_skeleton`` always
        forwards ``encoder_cfg=...`` and ``ckpt_dir=...``, so an override
        that drops either will raise ``TypeError: unexpected keyword
        argument`` at deploy time. Unused kwargs may be accepted and
        ignored (see :class:`WanVideoVAEEncoder.from_skeleton`).

        The three kwargs are a **menu of data sources**, not a single
        priority chain. Each encoder picks the source matching its
        persistence story:

        * ``components_entry`` — the dict shape produced by
          :func:`generate_video_backbone_component_specs`:
          ``{"attr": str, "model_class": str, "extra_kwargs": dict}``.
          Use this when the encoder's structural geometry is fully captured
          by the saved Wan ``components`` entry (e.g. :class:`WanVideoVAEEncoder`).
        * ``ckpt_dir`` — the deploy-side checkpoint directory. Use this for
          per-encoder structural artifacts that the training-side
          :meth:`save_deploy_assets` hook wrote next to the safetensors
          (V-JEPA 2.1's ``manifest.json``, dinov3 / flux2_vae's
          ``config.json``). These encoders are strictly self-contained: they
          read only from ``ckpt_dir`` and a missing artifact raises —
          ``encoder.model_path`` is never consulted at deploy time.
        * ``encoder_cfg`` — the yaml ``model.video_backbone.encoder`` block
          (a dict / DictConfig with ``name`` and ``model_path``). Used for
          non-structural runtime knobs an encoder still needs at deploy time
          (e.g. V-JEPA 2.1's ``vjepa21_forward`` / ``svae_target_dim``), not
          as a weights / geometry source.

        Default implementation raises so non-supporting encoders fail
        loudly at deploy time rather than silently mismatch state_dict
        keys later.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_skeleton not implemented; deploy with this "
            "encoder is not supported. Either implement from_skeleton or train "
            "without an external encoder."
        )

    # ------------------------------------------------------------------
    # Training-side deploy-artifact copy
    # ------------------------------------------------------------------

    def save_deploy_assets(self, output_dir: str, cfg: Any) -> None:
        """Copy per-encoder deploy artifacts into the checkpoint directory.

        Called by the host backbone's ``save_deploy_assets`` after each
        checkpoint save so deploy is self-contained — the deploy host no
        longer needs ``encoder.model_path`` to be reachable. The default is
        a no-op for encoders whose structural state is fully captured by
        the safetensors weights plus the saved ``components`` entry (e.g.
        :class:`WanVideoVAEEncoder`); encoders that depend on side files
        like ``manifest.json`` / ``config.json`` override this to copy them
        next to the ``checkpoint_step_*.safetensors``.

        Overriding encoders are strictly self-contained: a missing source /
        unresolvable cfg / copy IO error must raise so the checkpoint save
        aborts rather than silently producing a deploy-unloadable artifact
        (:meth:`from_skeleton` reads these artifacts only from ``ckpt_dir``,
        with no ``encoder.model_path`` fallback).
        """
        return None

    # ------------------------------------------------------------------
    # DiT-side adapter hooks (modular extension point)
    # ------------------------------------------------------------------
    # Subclasses override these only when the default Wan-style projection
    # is inappropriate (e.g. ViT-style encoders that already patchify
    # spatially and want the DiT's first conv to act as a pure channel
    # projection). The default implementations cover Wan VAE and any
    # encoder whose output is a (B, z_dim, T_lat, H_lat, W_lat) grid.

    def build_dit_input_proj(self, dit_dim: int) -> nn.Module:
        """Return an ``nn.Module`` mapping the encoder's latent grid into
        DiT token embeddings.

        Default implementation produces the Wan-original layout:
        ``nn.Conv3d(properties.z_dim, dit_dim,
                    kernel_size=properties.dit_patch_size, stride=properties.dit_patch_size)``.

        Shape contract:
            input  -- ``(B, properties.z_dim, T_lat, H_lat, W_lat)``
            output -- ``(B, dit_dim, T_out, H_out, W_out)`` where
                      ``T_out = T_lat / dit_patch_size[0]`` etc.
        """
        ps = self.properties.dit_patch_size
        return nn.Conv3d(self.properties.z_dim, dit_dim, kernel_size=ps, stride=ps)

    def build_dit_output_proj(self, dit_dim: int) -> nn.Module:
        """Return an ``nn.Module`` mapping DiT token embeddings back into
        an unpatchify-ready linear vector.

        Default implementation produces the Wan-original layout:
        ``nn.Linear(dit_dim, properties.z_dim * prod(properties.dit_patch_size))``.

        Shape contract:
            input  -- ``(B, L, dit_dim)``
            output -- ``(B, L, properties.z_dim * prod(properties.dit_patch_size))``
                      The host DiT applies ``unpatchify`` on top to recover
                      ``(B, properties.z_dim, T_lat, H_lat, W_lat)``.
        """
        ps = self.properties.dit_patch_size
        return nn.Linear(dit_dim, self.properties.z_dim * math.prod(ps))

    # ------------------------------------------------------------------
    # Optional S-VAE feature reducer (opt-in extension point)
    # ------------------------------------------------------------------
    # A subclass that compresses its raw per-token features sets
    # ``self._svae = reducer.build(...)`` in __init__ and routes batch_encode
    # through :meth:`_apply_svae`. Default (``_svae is None``) is a passthrough,
    # so non-opting encoders stay bit-unchanged. The SVAE network + build /
    # sidecar plumbing live in :mod:`openwam.model.video_backbone.encoder.svae`.

    def _apply_svae(self, z: Tensor) -> Tensor:
        """Reduce features through the optional frozen S-VAE, else passthrough.

        The single opt-in hook a subclass's ``batch_encode`` calls;
        :func:`reducer.reduce` does the deterministic posterior mean. Reducer
        construction / ``svae_target_dim`` sizing / deploy sidecar stay in
        :mod:`...svae.reducer`.
        """
        return reducer.reduce(self._svae, z)
