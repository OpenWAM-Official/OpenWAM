"""Pluggable video encoder subsystem.

Only activated when ``video_backbone.from_scratch=true`` AND
``video_backbone.encoder`` is set in yaml. Under any other combination the
backbone keeps using its built-in ``pipe.vae`` and the encoder package is
inert (registration still runs, but no encoder is instantiated).

Extension contract (the ABC's hook layer)
-----------------------------------------
A new encoder author should only ever touch this package — concretely:

  1. Create ``encoder/<name>.py`` with a ``class XxxEncoder(VideoEncoder)``
     decorated by ``@register_video_encoder("xxx")``.
  2. Implement the four abstract methods: ``spec`` (property),
     ``preprocess_video``, ``batch_encode``, ``from_pretrained``.
  3. Optionally override ``build_dit_input_proj`` / ``build_dit_output_proj``
     when the default Wan-style ``nn.Conv3d`` / ``nn.Linear`` does not fit.
  4. Add ``from .xxx import XxxEncoder  # noqa: F401`` at the bottom of this
     file to trigger registration on import.

The author NEVER needs to touch ``wan_adapter.py`` / ``dit.py`` /
``base.py`` / ``pipeline.py``. See [docs/external_video_encoder.md](docs/external_video_encoder.md).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.encoder.spec import VideoEncoderSpec


class VideoEncoder(ABC, nn.Module):
    """Swappable video latent codec.

    Activated only when ``video_backbone.from_scratch=true`` AND
    ``video_backbone.encoder`` is set in yaml; otherwise the backbone's
    native ``pipe.vae`` is used and ``state_dict`` keys remain bit-exact
    with the upstream pretrained checkpoint.

    Subclasses MUST implement ``spec`` / ``preprocess_video`` / ``batch_encode`` /
    ``from_pretrained``. They MAY implement ``decode`` / ``to_frames`` (only
    when ``spec.is_reversible=True``) and MAY override
    ``build_dit_input_proj`` / ``build_dit_output_proj`` when the default
    Wan-style projection is not appropriate.
    """

    # ------------------------------------------------------------------
    # Required: latent contract + per-step IO
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def spec(self) -> VideoEncoderSpec:
        """Structural contract derived from loaded weights, not yaml."""

    @abstractmethod
    def preprocess_video(self, frames) -> Tensor:
        """List[PIL.Image] -> ``(B=1, 3, T, H, W)`` tensor in ``spec.pixel_range``."""

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
        ``spec.is_reversible=True``."""
        raise NotImplementedError(
            f"{type(self).__name__}.decode unavailable "
            f"(spec.is_reversible={self.spec.is_reversible}). "
            "Use latent-level metrics for training, or train a separate "
            "pixel decoder if you need to visualize generated samples."
        )

    def to_frames(self, video_tensor: Tensor) -> list:
        """``(B, 3, T, H, W)`` pixel tensor -> ``list[PIL.Image]``. Optional;
        same constraint as :meth:`decode`."""
        raise NotImplementedError(
            f"{type(self).__name__}.to_frames unavailable (spec.is_reversible={self.spec.is_reversible})."
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str, **kw: Any) -> "VideoEncoder":
        """Construct from a weights directory. ``model_path`` is the only
        always-required user-facing argument (read from yaml). Additional
        kwargs may be forwarded from yaml fields whose names are returned
        by :meth:`optional_yaml_keys` — those names are part of the yaml
        whitelist applied in ``BaseWAMArchitecture._init_video_backbone``
        and pass through ``build_video_encoder`` to this constructor."""

    # ------------------------------------------------------------------
    # Optional yaml field whitelist
    # ------------------------------------------------------------------

    @classmethod
    def optional_yaml_keys(cls) -> set[str]:
        """Optional yaml fields this encoder accepts beyond ``{name, model_path}``.

        Default: empty set. Override per-encoder to expose runtime knobs
        (not structural weights properties — those belong in ``manifest.json``).
        Returned names are added to the whitelist enforced in
        :meth:`BaseWAMArchitecture._init_video_backbone` and forwarded into
        :meth:`from_pretrained` / :meth:`from_skeleton` as kwargs. Adding a
        key here is the single edit needed to expose it through yaml — the
        forwarding plumbing in ``build_video_encoder`` reads this method
        and packs the kwargs accordingly.
        """
        return set()

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
        'Public implementation.'
        raise NotImplementedError(
            f"{cls.__name__}.from_skeleton not implemented; deploy with this "
            "encoder is not supported. Either implement from_skeleton or train "
            "without an external encoder."
        )

    # ------------------------------------------------------------------
    # Training-side deploy-artifact copy
    # ------------------------------------------------------------------

    def copy_deploy_artifacts(self, output_dir: str, cfg: Any) -> None:
        'Public implementation.'
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
        ``nn.Conv3d(spec.z_dim, dit_dim,
                    kernel_size=spec.dit_patch_size, stride=spec.dit_patch_size)``.

        Shape contract:
            input  -- ``(B, spec.z_dim, T_lat, H_lat, W_lat)``
            output -- ``(B, dit_dim, T_out, H_out, W_out)`` where
                      ``T_out = T_lat / dit_patch_size[0]`` etc.
        """
        ps = self.spec.dit_patch_size
        return nn.Conv3d(self.spec.z_dim, dit_dim, kernel_size=ps, stride=ps)

    def build_dit_output_proj(self, dit_dim: int) -> nn.Module:
        """Return an ``nn.Module`` mapping DiT token embeddings back into
        an unpatchify-ready linear vector.

        Default implementation produces the Wan-original layout:
        ``nn.Linear(dit_dim, spec.z_dim * prod(spec.dit_patch_size))``.

        Shape contract:
            input  -- ``(B, L, dit_dim)``
            output -- ``(B, L, spec.z_dim * prod(spec.dit_patch_size))``
                      The host DiT applies ``unpatchify`` on top to recover
                      ``(B, spec.z_dim, T_lat, H_lat, W_lat)``.
        """
        ps = self.spec.dit_patch_size
        return nn.Linear(dit_dim, self.spec.z_dim * math.prod(ps))









# ----------------------------------------------------------------------
# Registry + factory
# ----------------------------------------------------------------------

_VIDEO_ENCODER_REGISTRY: dict[str, type[VideoEncoder]] = {}


def register_video_encoder(name: str):
    """Decorator that registers a :class:`VideoEncoder` subclass under ``name``.

    Raises:
        ValueError: If ``name`` is already taken.
        TypeError:  If the decorated class is not a :class:`VideoEncoder` subclass.
    """

    def _wrap(cls):
        if not isinstance(cls, type) or not issubclass(cls, VideoEncoder):
            raise TypeError(f"register_video_encoder('{name}') expects a VideoEncoder subclass, got {cls!r}.")
        if name in _VIDEO_ENCODER_REGISTRY:
            raise ValueError(f"Video encoder '{name}' is already registered.")
        _VIDEO_ENCODER_REGISTRY[name] = cls
        return cls

    return _wrap


def build_video_encoder(cfg) -> VideoEncoder:
    """Build a :class:`VideoEncoder` from a config dict / DictConfig.

    Accepts the two required fields ``name`` / ``model_path`` plus any
    optional fields the picked encoder class exposes through
    :meth:`VideoEncoder.optional_yaml_keys` (e.g. ``vjepa2_1_forward``).
    The base-side gate in :meth:`BaseWAMArchitecture._init_video_backbone`
    enforces the same whitelist; this function adds friendly errors for the
    "field present but empty" case (which the whitelist wouldn't catch) and
    forwards the optional fields into :meth:`from_pretrained` as kwargs.
    """

    def _read(key: str):
        # Both dict and OmegaConf DictConfig support both indexing and
        # attribute access; ``isinstance(cfg, dict)`` distinguishes them.
        # OmegaConf with struct mode returns the default from getattr
        # rather than raising, so the second branch is safe.
        if isinstance(cfg, dict):
            value = cfg.get(key)
        else:
            value = getattr(cfg, key, None)
        if value is None:
            raise ValueError(
                f"video_backbone.encoder.{key} is required (got cfg={dict(cfg) if hasattr(cfg, 'keys') else cfg!r})."
            )
        return value

    def _read_optional(key: str):
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)

    name = _read("name")
    model_path = _read("model_path")
    if name not in _VIDEO_ENCODER_REGISTRY:
        available = ", ".join(sorted(_VIDEO_ENCODER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown video encoder '{name}'. Available: {available}")
    encoder_cls = _VIDEO_ENCODER_REGISTRY[name]
    optional_kwargs: dict[str, Any] = {}
    for k in encoder_cls.optional_yaml_keys():
        v = _read_optional(k)
        if v is not None:
            optional_kwargs[k] = v
    return encoder_cls.from_pretrained(str(model_path), **optional_kwargs)


__all__ = ['VideoEncoder', 'VideoEncoderSpec', 'VJEPA2VideoEncoder', 'VJEPA21VideoEncoder', 'WanVideoVAEEncoder', 'build_video_encoder', 'register_video_encoder']

# Built-in registrations (kept at the bottom so subclasses can import names
# from this module without circular issues). Adding a new encoder = adding
# a new line here and a new file alongside.
from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder  # noqa: E402, F401
from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder  # noqa: E402, F401
from openwam.model.video_backbone.encoder.wan_vae import WanVideoVAEEncoder  # noqa: E402, F401
