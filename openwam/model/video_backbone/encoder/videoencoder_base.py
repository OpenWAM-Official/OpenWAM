"""VideoEncoder ABC and the structural spec it exposes to the host backbone.

:class:`VideoEncoderSpec` is the latent contract (z_dim / compression / patch
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


@dataclass(frozen=True)
class VideoEncoderSpec:
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
        pixel_range: Nominal input range. Informational only — not consumed
            by the backbone. Encoders
            that apply additional internal normalization in ``preprocess_video``
            (e.g. ImageNet mean/std for V-JEPA) may legitimately emit
            tensors outside this nominal range; the field documents the
            pre-normalization input expectation, not the post-preprocess output.
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
    pixel_range: tuple[float, float] = (-1.0, 1.0)
    is_reversible: bool = True
    dit_patch_size: tuple[int, int, int] = (1, 2, 2)


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
        """Build a zero-weight encoder skeleton. The host architecture's
        checkpoint strict-load fills in weights immediately after this call.

        Subclasses that override **must** keep all three kwargs in their
        signature — ``base.py:_build_external_encoder_skeleton`` always
        forwards ``encoder_cfg=...`` and ``ckpt_dir=...``, so an override
        that drops either will raise ``TypeError: unexpected keyword
        argument`` at deploy time. Unused kwargs may be accepted and
        ignored (see :class:`WanVideoVAEEncoder.from_skeleton`).

        The three kwargs are a **menu of data sources**, not a single
        priority chain. Each encoder picks one primary source per its
        persistence story; some encoders also pick a secondary source as
        a fallback for older checkpoints (see
        :class:`VJEPA21VideoEncoder`), and some encoders adopt a strict
        self-contained policy with no fallback at all.

        * ``components_entry`` — the dict shape produced by
          :func:`generate_video_backbone_component_specs`:
          ``{"attr": str, "model_class": str, "extra_kwargs": dict}``.
          Use this when the encoder's structural geometry is fully captured
          by the saved Wan ``components`` entry (e.g. :class:`WanVideoVAEEncoder`).
        * ``ckpt_dir`` — the deploy-side checkpoint directory. Use this for
          per-encoder structural artifacts that the training-side
          :meth:`save_deploy_assets` hook wrote next to the safetensors.
          Two flavors are in use today:

            - **Preferred-with-fallback** (V-JEPA 2.1): the encoder reads
              ``<ckpt_dir>/manifest.json`` first and falls back to
              ``<encoder_cfg.model_path>/manifest.json`` for older
              checkpoints saved before the self-containment patch.
            - **Strict self-contained**: the encoder reads
              ``<ckpt_dir>/encoder_meta/encoder_config.json`` and
              ``encoder.model_path`` is *never* consulted at deploy time.
              Missing ckpt_dir or encoder_meta raises immediately.

        * ``encoder_cfg`` — the yaml ``model.video_backbone.encoder`` block
          (a dict / DictConfig with ``name`` and ``model_path``). Use this
          as a fallback source for encoders that adopted the
          preferred-with-fallback policy above, OR as the primary source for
          legacy paths that have not yet migrated to ``ckpt_dir``. Trade-off:
          the deploy host must be able to read ``encoder.model_path`` for
          those paths.

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
        like ``manifest.json`` override this to copy them next to the
        ``checkpoint_step_*.safetensors``.

        Two policies coexist under this ABC; both are documented as
        supported in :meth:`from_skeleton`:

        - **Preferred-with-fallback** (V-JEPA 2.1): a missing source file
          here is logged as a warning and treated as best-effort —
          :meth:`from_skeleton` will fall back to ``encoder.model_path``
          at deploy time. Subclasses that pick this policy MUST NOT raise.
        - **Strict self-contained**: a missing
          source file here is a hard error — re-raise so the checkpoint
          save aborts rather than silently producing a deploy-unloadable
          artifact. Subclasses that pick this policy MUST document their
          re-raise behavior in their own class docstring so external
          callers do not ``try/except`` this method assuming the V-JEPA
          contract.
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
