"""VideoEncoder ABC and the structural spec it exposes to the host backbone.

:class:`VideoEncoderProperties` is the latent contract (z_dim / compression / patch
geometry) derived from the loaded encoder weights — NOT from yaml. The encoder's
``from_pretrained`` populates it from the actual loaded state.

:class:`VideoEncoder` is the ABC each pluggable encoder subclasses. The package
``__init__`` re-exports both alongside the registry + factory.
"""

from __future__ import annotations

import json
import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.encoder.svae import _CHECKPOINT_FORMAT_VERSION, SVAE, build_svae, load_svae

logger = logging.getLogger(__name__)


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
    Wan-style projection is not appropriate. They MAY also attach the optional
    S-VAE feature reducer (default disabled) — see the "Optional S-VAE feature
    reducer" block at the bottom of this class.
    """

    def __init__(self) -> None:
        super().__init__()
        # Optional frozen S-VAE feature reducer; ``None`` = disabled. Encoders
        # that want it opt in from their own ``__init__`` via
        # ``self._svae = self._build_svae(...)``; all others leave it None and
        # every S-VAE helper below is a no-op (state_dict is bit-unchanged).
        self._svae: SVAE | None = None

    # ------------------------------------------------------------------
    # Required: latent contract + per-step IO
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def spec(self) -> VideoEncoderProperties:
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

    # ------------------------------------------------------------------
    # Optional S-VAE feature reducer (general capability; default disabled)
    # ------------------------------------------------------------------
    # A frozen per-token S-VAE (:mod:`openwam.model.video_backbone.encoder.svae`)
    # that compresses an encoder's raw per-token features to a smaller ``z_dim``.
    # Any encoder MAY opt in by calling ``self._svae = self._build_svae(...)`` in
    # its ``__init__`` and routing ``batch_encode`` through
    # :meth:`_apply_svae_if_enabled`; currently only ``VJEPA21VideoEncoder`` does.
    # When ``self._svae is None`` (the default) every helper below is a no-op, so
    # the other encoders' ``batch_encode`` / ``spec`` / ``state_dict`` are
    # bit-unchanged.

    @staticmethod
    def _build_svae(
        svae_path: str | None,
        svae_target_dim: int | None,
        svae_config: dict | None,
    ) -> SVAE | None:
        """Construct the optional frozen S-VAE reducer from one of three sources.

        Mirrors the PCA plumbing's three branches but stores an ``nn.Module``
        (trainable encoder+decoder weights) rather than two static buffers:

        * ``svae_path``   — training: load a standalone-trained checkpoint.
        * ``svae_config`` — deploy skeleton: rebuild a zero-weight shell from the
          sidecar config dict; the architecture's strict ``load_checkpoint``
          fills the weights immediately after construction.
        * neither — disabled (raw passthrough; ``z_dim`` stays ``embed_dim``).

        The reducer is always returned frozen and in eval mode; the world-model
        data path runs it inside ``@torch.no_grad`` preprocessing, and
        ``batch_encode`` calls :meth:`SVAE.encode_mean` (deterministic) so a
        recursive ``host.train()`` cannot flip it into a stochastic path.
        """
        if svae_path is not None and svae_config is not None:
            raise ValueError("Pass only one of svae_path / svae_config, not both.")
        if svae_path is not None:
            svae = load_svae(svae_path)
        elif svae_config is not None:
            svae = build_svae(dict(svae_config))
        else:
            return None
        if svae_target_dim is not None and int(svae_target_dim) != svae.latent_dim:
            raise ValueError(
                f"svae_target_dim ({svae_target_dim}) does not match the S-VAE latent_dim ({svae.latent_dim})."
            )
        svae.eval()
        svae.requires_grad_(False)
        return svae

    def _effective_z_dim(self, raw_dim: int) -> int:
        """The encoder's advertised ``z_dim``: the S-VAE ``latent_dim`` when a
        reducer is attached, else ``raw_dim``. Opt-in encoders call this to size
        ``spec.z_dim`` (and any post-reduce norm) so the DiT first conv /
        unpatchify head / freeze yaml all rebuild against the reduced dim.
        """
        return self._svae.latent_dim if self._svae is not None else int(raw_dim)

    def _apply_svae_if_enabled(self, z: Tensor) -> Tensor:
        """Reduce raw post-pool features with the frozen S-VAE (deterministic
        posterior mean), or pass them through unchanged when none is attached.

        Under DeepSpeed ZeRO-3 the reducer's frozen parameters are partitioned,
        and the forward-pre-hook that would gather them does not fire on this
        preprocessing path (preprocess runs before the architecture forward, so
        no module ``__call__`` on ``self`` has triggered a gather). We therefore
        gather them read-only for the duration of the reduce. No-op off ZeRO-3 —
        the parameters then carry no ``ds_id`` and the gather list is empty.
        """
        if self._svae is None:
            return z
        ds_params = [p for p in self._svae.parameters() if getattr(p, "ds_id", None) is not None]
        if ds_params:
            import deepspeed

            with deepspeed.zero.GatheredParameters(ds_params, modifier_rank=None):
                return self._svae.encode_mean(z)
        return self._svae.encode_mean(z)

    def _write_svae_sidecar(self, output_dir: str) -> None:
        """Write the attached S-VAE's structural config to
        ``<output_dir>/svae_config.json`` so deploy can rebuild a same-shape
        shell. Raises on IO failure — see the opt-in encoder's
        ``save_deploy_assets`` for why this one must abort rather than
        warn-and-skip.

        The payload is versioned with the same ``_CHECKPOINT_FORMAT_VERSION`` as
        the standalone ``svae.pt`` so a stale sidecar (e.g. one written by a
        build whose ``config_dict`` schema differs) is rejected with a clear
        message on read instead of crashing ``SVAE.__init__`` with an unexpected
        keyword.
        """
        dst = os.path.join(output_dir, "svae_config.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"format_version": _CHECKPOINT_FORMAT_VERSION, "model_config": self._svae.config_dict()}, f)
        logger.info("%s: wrote S-VAE sidecar %s", type(self).__name__, dst)

    @staticmethod
    def _read_svae_sidecar(ckpt_dir: str | None) -> dict | None:
        """Read ``<ckpt_dir>/svae_config.json`` (written by
        :meth:`_write_svae_sidecar`). Returns the structural ``model_config``
        dict when the checkpoint carried an S-VAE reducer, else ``None`` (reducer
        disabled). There is intentionally no ``encoder.model_path`` fallback:
        the sidecar is checkpoint-local and self-contained by construction.

        Validates the sidecar ``format_version`` (matching the standalone
        checkpoint), so a legacy unversioned / mismatched sidecar fails fast here
        with a clear message rather than deeper in ``build_svae``.
        """
        if not ckpt_dir:
            return None
        path = os.path.join(ckpt_dir, "svae_config.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        fmt = payload.get("format_version") if isinstance(payload, dict) else None
        if fmt != _CHECKPOINT_FORMAT_VERSION or "model_config" not in payload:
            raise ValueError(
                f"{path!r} has unsupported S-VAE sidecar format_version={fmt!r} "
                f"(this build writes/reads version {_CHECKPOINT_FORMAT_VERSION}). "
                f"Re-export the deploy checkpoint with the current build."
            )
        return payload["model_config"]

    @staticmethod
    def _read_svae_target_dim_from_cfg(encoder_cfg: Any) -> int | None:
        """Pick ``svae_target_dim`` from the saved encoder yaml if present —
        used only as a cross-check against the sidecar's ``latent_dim`` in
        ``__init__``. Absent / null collapses to ``None`` (no cross-check).
        """
        if encoder_cfg is None:
            return None
        if isinstance(encoder_cfg, dict):
            value = encoder_cfg.get("svae_target_dim")
        else:
            value = getattr(encoder_cfg, "svae_target_dim", None)
        return int(value) if value is not None else None
