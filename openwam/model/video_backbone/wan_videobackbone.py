"""Concrete Wan :class:`VideoBackbone` subclasses.

Two members of the Wan family, splitting on construction + VAE-IO:
  - :class:`Wan22Ti2vBackbone` — Wan2.2-TI2V-5B; supports swapping the native
    VAE for an external :class:`VideoEncoder`.
  - :class:`Wan21Backbone` — Wan2.1 I2V / VACE; native VAE only.

All shared behavior (DiT forward, conditioning, deploy) lives in
:class:`WanBackboneBase`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.wan import loader
from openwam.model.video_backbone.wanbackbone_base import WanBackboneBase

logger = logging.getLogger(__name__)


class Wan22Ti2vBackbone(WanBackboneBase):
    """Wan2.2-TI2V-5B backbone with optional external-encoder VAE-IO routing.

    ``external_encoder`` is ``None`` on the default path so ``state_dict()``
    carries only ``vae.*`` keys; setting it activates external-encoder VAE-IO
    routing and aliases the encoder under ``"vae"``.
    """

    def __init__(self, holder, *, external_encoder=None, shift_video=None, text_dim: Optional[int] = None):
        """Internal constructor. Use ``from_pretrained()`` instead."""
        # Base sets self.video_encoder after nn.Module.__init__ (an nn.Module
        # encoder cannot be assigned before that), activating VAE-IO routing.
        super().__init__(holder, external_encoder=external_encoder, shift_video=shift_video, text_dim=text_dim)
        if external_encoder is not None:
            # Override the native (1,2,2)/4×/causal contract with the encoder's;
            # callers consult these attrs and never branch on the encoder.
            self._dit_patch_size = external_encoder.spec.dit_patch_size
            self._temporal_compression = int(external_encoder.spec.temporal_compression)
            self._causal_temporal = bool(external_encoder.spec.causal_temporal)

    @classmethod
    def from_pretrained(
        cls, source, *, external_encoder=None, text_dim: Optional[int] = None, **kw
    ) -> "Wan22Ti2vBackbone":
        """Build a Wan22Ti2vBackbone from a source.

        Sources: ``DictConfig`` (full Hydra cfg → loader), ``str`` dir path /
        ``dict`` with ``model_path`` (lightweight build), else an already-built
        component holder. Construction returns a transient holder that
        ``__init__`` drains into the backbone.

        With ``external_encoder``: derive division factors from the encoder
        spec, release the native VAE, expose latent-shape metadata. See the
        inline comments.
        """
        from omegaconf import DictConfig

        # Skip materializing the native VAE (avoid ~1.5GB waste / a duplicate
        # VAE slot deploy has no weights for) on training-with-irreversible and
        # on deploy-with-ANY external encoder. Reversible-on-training keeps it,
        # needed for the step-(2) spec cross-check against ``v.z_dim`` etc.
        is_deploy = not isinstance(source, DictConfig)
        skip_native_vae = bool(external_encoder is not None and (is_deploy or not external_encoder.spec.is_reversible))

        holder = cls._build_holder(source, skip_native_vae=skip_native_vae, **kw)

        if external_encoder is not None:
            # (3) Division factors from the encoder spec, not a hardcoded ``* 2`` / Wan-VAE grid, else
            # ``check_resize_height_width`` rounds encoder-legal sizes to Wan's grid. Remainder is 1 iff causal.
            patch_size = external_encoder.spec.dit_patch_size
            holder.height_division_factor = external_encoder.spec.spatial_compression * patch_size[1]
            holder.width_division_factor = external_encoder.spec.spatial_compression * patch_size[2]
            holder.time_division_factor = external_encoder.spec.temporal_compression * patch_size[0]
            holder.time_division_remainder = 1 if external_encoder.spec.causal_temporal else 0

            # (4) Release the native VAE so state_dict keys don't double-count with the external encoder. print (not
            # logger.info) because arch init runs before the logger is wired up; rank-0 gated.
            holder.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(os.environ.get("RANK", 0)) == 0:
                print(
                    f"[Wan22Ti2vBackbone] native VAE released; "
                    f"external_encoder={type(external_encoder).__name__} "
                    f"(z_dim={external_encoder.spec.z_dim}, "
                    f"is_reversible={external_encoder.spec.is_reversible}, "
                    f"dit_patch_size={external_encoder.spec.dit_patch_size})",
                    flush=True,
                )

            # (5) Expose latent-shape metadata so deploy noise init reads it without the native VAE (now None).
            holder.latent_spec = external_encoder.spec

        # Resolve optional cfg-side ``shift_video`` here (not in __init__)
        # because the cfg shape depends on the ``source`` type.
        shift_video_cfg = loader.resolve_cfg_shift_video(source)

        return cls(holder, external_encoder=external_encoder, shift_video=shift_video_cfg, text_dim=text_dim)

    # ================================================================
    # External-encoder-aware overrides
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        if name == "vae" and self._uses_external_encoder:
            return self.video_encoder
        return super().get_submodule(name)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        if self._uses_external_encoder and not self.video_encoder.spec.is_reversible:
            raise NotImplementedError(
                f"decode_video on irreversible encoder ({type(self.video_encoder).__name__}; "
                "spec.is_reversible=False). Pass decode_video=False to generate() to "
                "retrieve raw latents, or train a separate pixel decoder."
            )
        return super().decode_video(latents, tiled=tiled)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Wan deploy assets, then forward to the external encoder's own
        deploy-artifact hook so its side files (e.g. V-JEPA ``manifest.json``)
        land alongside.
        """
        super().save_deploy_assets(output_dir, cfg)
        if self.video_encoder is not None:
            self.video_encoder.save_deploy_assets(output_dir, cfg)


class Wan21Backbone(WanBackboneBase):
    """Wan2.1 I2V / VACE backbone — native VAE only (no external encoder)."""

    @classmethod
    def from_pretrained(cls, source, *, text_dim: Optional[int] = None, **kw) -> "Wan21Backbone":
        """Build a Wan21Backbone from a source (see :meth:`WanBackboneBase._build_holder`)."""
        holder = cls._build_holder(source, **kw)
        return cls(holder, shift_video=loader.resolve_cfg_shift_video(source), text_dim=text_dim)


__all__ = ["Wan22Ti2vBackbone", "Wan21Backbone"]
