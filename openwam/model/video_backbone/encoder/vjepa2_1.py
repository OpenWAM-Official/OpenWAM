"""V-JEPA 2.1 video encoder (Mur-Labadia et al., arXiv:2603.14482).

Plugs into the :class:`VideoBackbone` external-encoder path introduced in
PR #60. ``spec.is_reversible=False`` — the host backbone must rebuild its
DiT first conv via the default ``build_dit_input_proj`` hook and skip the
strict native-VAE spec validation. ``spec.causal_temporal=True`` and
``spec.temporal_compression=4`` (ViT tubelet=2 + an encoder-side avg-pool
over time with stride=2) emulate the Wan VAE's causal grouping (1 cond
latent from frame 0 + 1 latent per 4 target pixel frames), so the host
DiT receives the same latent token-count whether the encoder is V-JEPA
or the native Wan VAE.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Literal, get_args

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from openwam.model.video_backbone.encoder import _vjepa_loader
from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.svae import reducer

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# vjepa2_1_forward selects how the condition (frame 0) latent is obtained in
# ``batch_encode`` (see its docstring): ``"video"`` (default) routes a dup'd
# 2-frame clip through the video branch; ``"mixed"`` routes f0 through the
# image branch. The default is a deliberate departure from pre-PR-#92
# (≡ ``"mixed"``); migration notes live in :meth:`from_skeleton`.
_VJEPA21Forward = Literal["video", "mixed"]
_VJEPA21_FORWARD_DEFAULT: _VJEPA21Forward = "video"
# Derived from the Literal so the runtime whitelist can't drift from it.
_VJEPA21_FORWARD_ALLOWED: tuple[_VJEPA21Forward, ...] = get_args(_VJEPA21Forward)


@register_video_encoder("vjepa2_1")
class VJEPA21VideoEncoder(VideoEncoder):
    """V-JEPA 2.1 video encoder.

    Constructor takes an already-built ViT module so unit tests can inject a
    mock without going through ``_vjepa_loader`` (which requires the upstream
    ``app.vjepa_2_1`` package and a local checkpoint file).
    """

    def __init__(
        self,
        vit: nn.Module,
        *,
        embed_dim: int,
        variant: str,
        vjepa2_1_forward: _VJEPA21Forward = _VJEPA21_FORWARD_DEFAULT,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
        svae_config: dict | None = None,
    ):
        super().__init__()
        # ``_vjepa_loader`` installs a RoPE monkey-patch that casts the rotated
        # Q/K back to ``x.dtype``: upstream would promote them to fp32 (fp32
        # sin/cos table), mismatching bf16 V at SDPA and tripping DeepSpeed
        # ZeRO-3's bf16 all_gather on this frozen submodule.
        self._m = vit
        self._variant = variant
        if vjepa2_1_forward not in _VJEPA21_FORWARD_ALLOWED:
            raise ValueError(f"vjepa2_1_forward must be one of {_VJEPA21_FORWARD_ALLOWED}, got {vjepa2_1_forward!r}.")
        self._vjepa2_1_forward: _VJEPA21Forward = vjepa2_1_forward
        self._raw_embed_dim = int(embed_dim)
        # Optional non-linear S-VAE reducer, applied AFTER the cond+target cat
        # (see ``batch_encode``). When enabled it advertises ``latent_dim`` as
        # ``z_dim`` so the DiT conv / unpatchify head / freeze yaml rebuild
        # against the reduced dim.
        self._svae = reducer.build(svae_path, svae_target_dim, svae_config)
        effective_z_dim = reducer.effective_z_dim(self._svae, self._raw_embed_dim)
        self._spec = VideoEncoderProperties(
            z_dim=int(effective_z_dim),
            spatial_compression=16,
            # 4 = ViT tubelet=2 × the ``_TARGET_TEMPORAL_POOL_STRIDE`` avg-pool,
            # matching Wan VAE causal grouping (1 cond + 1 latent / 4 frames) so
            # ``(T_pix - 1) // tc + 1`` lands on the same T_lat.
            temporal_compression=4,
            causal_temporal=True,
            pixel_range=(-1.0, 1.0),
            is_reversible=False,
            # (1, 2, 2) matches Wan VAE's DiT-side patch layout: the default
            # ``build_dit_input_proj`` Conv3d((1,2,2),(1,2,2)) pools 4 V-JEPA
            # spatial neighbors per DiT token for token-count parity.
            dit_patch_size=(1, 2, 2),
        )
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        # Per-token standardization pulling V-JEPA's O(30)-scale features to
        # Wan-latent O(1). Structurally trainable but a fixed standardizer in
        # practice: the encode path runs under ``@torch.no_grad`` and the freeze
        # yaml freezes ``video_backbone.video_encoder`` wholesale. Lives OUTSIDE
        # ``self._m`` so a future PR can make it grad-enabled without touching
        # the ViT freeze granularity. (With the S-VAE on, a near-identity affine
        # over the already-whitened reduced dim.)
        self.feature_norm = nn.LayerNorm(int(effective_z_dim))

    @property
    def spec(self) -> VideoEncoderProperties:
        return self._spec

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def vjepa2_1_forward(self) -> _VJEPA21Forward:
        """How the condition (frame 0) latent is computed in ``batch_encode``.

        ``"video"`` (default) — dup frame 0 to a 2-frame clip and route it
        through the V-JEPA 2.1 video branch (tubelet=2).
        ``"mixed"`` — route frame 0 through the V-JEPA 2.1 image branch
        (tubelet=1).
        """
        return self._vjepa2_1_forward

    @classmethod
    def optional_yaml_keys(cls) -> set[str]:
        # ``vjepa2_1_forward``: condition-frame forward mode (runtime config;
        # ViT weights identical across modes).
        # ``svae_path`` / ``svae_target_dim``: optional S-VAE feature reducer
        # (see ``openwam/model/video_backbone/encoder/svae.py``). All are
        # yaml-level (not in the manifest) — they describe consumer wiring, not
        # weight properties. Defaults applied in ``__init__`` if absent.
        return {"vjepa2_1_forward", "svae_path", "svae_target_dim"}

    def preprocess_video(self, frames: List[Image.Image]) -> torch.Tensor:
        """List[PIL] -> (1, 3, T, H, W) in ImageNet-normalized space."""
        device = next(self._m.parameters()).device
        dtype = next(self._m.parameters()).dtype
        to_tensor = T.ToTensor()
        chans = [to_tensor(img) for img in frames]
        video = torch.stack(chans, dim=1).unsqueeze(0)
        video = video.to(device=device, dtype=dtype)
        video = (video - self._mean.to(dtype)) / self._std.to(dtype)
        return video

    # Extra avg-pool over time (after ViT tubelet=2) so total temporal
    # compression hits 4, matching Wan VAE causal grouping. Locked to 2:
    # tubelet=2 and Wan tc=4 are both production-fixed. Class-level so the
    # "extra 2x" assumption lives in one place.
    _TARGET_TEMPORAL_POOL_STRIDE = 2

    def batch_encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T_pixel, H, W) -> (B, embed_dim, T_lat, H/16, W/16).

        ``T_lat == 1`` when ``T_pixel == 1`` (TI2V ref-frame fast path); else
        ``1 + (T_pixel - 1) // 4`` (1 cond latent + ``(T_pixel-1)/4`` targets),
        emulating Wan VAE causal grouping. Output shape is identical under both
        ``vjepa2_1_forward`` modes; they differ only in the condition pass:

        - Condition (frame 0, no target leakage): ``"video"`` (default) routes
          ``cat([f0, f0])`` through the video branch (tubelet=2); ``"mixed"``
          routes f0 through the image branch (tubelet=1). Both yield 1 latent.
        - Target (both modes): ``cat([f0, f0, t1..tN])`` → video branch → drop
          the first (prepended-pair) latent → ``N/2`` raw targets → avg-pool
          stride 2 → ``N/4``. The prepended pair lets temporal attention see
          the reference frame; dropping its latent keeps cond and target
          independent so deploy/train see the same cond latent.
        """
        z = self._batch_encode_pooled_raw(video)
        z = reducer.reduce(self._svae, z)
        z = self._apply_feature_norm(z)
        return z

    def _batch_encode_pooled_raw(self, video: torch.Tensor) -> torch.Tensor:
        """Encoder forward + temporal mean-pool, BEFORE S-VAE / feature_norm.

        Returns the ``(B, raw_embed_dim, T_lat, H/16, W/16)`` cat of cond +
        mean-pooled target latents — the exact tensor the S-VAE consumes. Split
        from :meth:`batch_encode` so offline S-VAE training
        (:meth:`batch_encode_pooled_for_svae_training`) sees the same post-pool
        distribution.
        """
        B, C, Tp, H, W = video.shape
        if C != 3:
            raise ValueError(f"V-JEPA 2.1 expects 3-channel input; got C={C}.")
        # Align input to ViT param dtype: the RoPE monkey-patch only guarantees
        # Q/K match x.dtype at SDPA, not the input matmuls.
        m_dtype = next(self._m.parameters()).dtype
        if video.dtype != m_dtype:
            video = video.to(m_dtype)
        f0 = video[:, :, 0:1]
        if Tp == 1:
            return self._encode_condition(f0)
        # tubelet=2 + extra time-pool ⇒ (T_pixel - 1) % 4 == 0. RoBoTwin
        # (T_pixel=9): (9-1) % 4 == 0. ✓
        divisor = 2 * self._TARGET_TEMPORAL_POOL_STRIDE
        if (Tp - 1) % divisor != 0:
            raise ValueError(f"V-JEPA 2.1 causal emulation needs (T_pixel - 1) % {divisor} == 0, got T_pixel={Tp}.")
        z_cond = self._encode_condition(f0)
        z_target_raw = self._encode_target_with_prepend(f0, video[:, :, 1:])
        z_target = self._pool_target_temporal(z_target_raw)
        return torch.cat([z_cond, z_target], dim=2)

    def batch_encode_pooled_for_svae_training(self, video: torch.Tensor) -> torch.Tensor:
        """Raw post-pool features for offline S-VAE training / stats collection.

        Returns the ``(B, raw_embed_dim, T_lat, H/16, W/16)`` cat the S-VAE
        consumes — 1 un-pooled cond latent + the mean-pooled target latents —
        BEFORE any reduction or ``feature_norm``. Both sub-populations (cond and
        target) are included so the reducer's prior covers what it sees at
        inference. Fails fast if an S-VAE is already attached: statistics must be
        collected on a raw encoder, never on an already-reduced one.
        """
        if self._svae is not None:
            raise RuntimeError(
                "batch_encode_pooled_for_svae_training requires a raw encoder; "
                "svae_path / svae_config / svae_target_dim must be unset."
            )
        return self._batch_encode_pooled_raw(video)

    def _pool_target_temporal(self, z_target: torch.Tensor) -> torch.Tensor:
        """(B, D, T_target_raw, h, w) -> (B, D, T_target_raw/2, h, w) via
        avg-pool over time with stride ``_TARGET_TEMPORAL_POOL_STRIDE``.

        ``batch_encode`` guarantees ``T_target_raw`` is divisible by the
        stride before calling here. Done BEFORE ``_apply_feature_norm`` so
        the LayerNorm re-standardizes the (slightly-attenuated) post-pool
        feature distribution and the final per-token output stays
        comparable to V-JEPA-native scale.
        """
        s = self._TARGET_TEMPORAL_POOL_STRIDE
        B, D, T, h, w = z_target.shape
        return z_target.reshape(B, D, T // s, s, h, w).mean(dim=3)

    def _encode_condition(self, f0: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) -> (B, D, 1, H/16, W/16). See ``batch_encode`` docstring."""
        if self._vjepa2_1_forward == "mixed":
            return self._encode_image(f0[:, :, 0])
        return self._encode_video_tubelet(torch.cat([f0, f0], dim=2))

    def _encode_target_with_prepend(self, f0: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) + (B, 3, N_even, H, W) -> (B, D, N/2, H/16, W/16).

        Runs ``video_branch(cat([f0, f0, targets]))`` and drops the first
        temporal latent (which encodes the prepended frame-0 pair). The
        kept slices are the target latents whose temporal attention has
        already seen the reference frame — exactly the supervision signal
        we want for the target stream, without leaking target information
        back into the condition lane (which uses its own forward).
        """
        clip = torch.cat([f0, f0, targets], dim=2)
        z_full = self._encode_video_tubelet(clip)
        return z_full[:, :, 1:]

    # V-JEPA 2.1 ViT.forward: input (B,C,T,H,W); T==1 → image branch (tubelet=1),
    # else video branch (tubelet=2). Output (B, L, D) flat, L = T_lat·(H/16)·(W/16)
    # in T-major then (H,W) row-major order.
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, D, 1, H/16, W/16) via V-JEPA 2.1 image branch."""
        x5 = image.unsqueeze(2)  # (B, 3, 1, H, W) -> image branch
        flat = self._m(x5)  # (B, L, D), L = (H/16)*(W/16)
        B, _, H, W = image.shape
        h, w = H // 16, W // 16
        return flat.transpose(1, 2).reshape(B, -1, 1, h, w).contiguous()

    def _encode_video_tubelet(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T_even, H, W) -> (B, D, T_even/2, H/16, W/16) via tubelet=2."""
        flat = self._m(video)  # (B, L, D), L = (T/2)*(H/16)*(W/16)
        B, _, Tp, H, W = video.shape
        h, w = H // 16, W // 16
        return flat.transpose(1, 2).reshape(B, -1, Tp // 2, h, w).contiguous()

    def _apply_feature_norm(self, z: torch.Tensor) -> torch.Tensor:
        B, D, Tp, h, w = z.shape
        z = z.permute(0, 2, 3, 4, 1).reshape(-1, D)
        z = self.feature_norm(z)
        return z.view(B, Tp, h, w, D).permute(0, 4, 1, 2, 3).contiguous()

    def decode(self, latents: torch.Tensor, **kw: Any) -> torch.Tensor:
        raise NotImplementedError(
            "VJEPA21VideoEncoder is irreversible (spec.is_reversible=False); "
            "pixel decode is not defined. Pass decode_video=False to generate()."
        )

    def to_frames(self, video: torch.Tensor) -> list:
        raise NotImplementedError("VJEPA21VideoEncoder is irreversible; to_frames has no meaning.")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        vjepa2_1_forward: _VJEPA21Forward = _VJEPA21_FORWARD_DEFAULT,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
    ) -> "VJEPA21VideoEncoder":
        # Explicit signature (no ``**kw``) so a programmatic kwarg typo raises
        # TypeError instead of silently using the default. The yaml path is
        # already filtered by ``build_video_encoder`` via ``optional_yaml_keys``.
        manifest = _vjepa_loader.read_and_validate_manifest(model_path)
        vit_encoder = _vjepa_loader.prepare_vjepa_imports_and_patch()
        vit = _vjepa_loader.build_vit_from_manifest(vit_encoder, manifest)
        _vjepa_loader.load_vit_weights(vit, model_path, manifest)
        return cls(
            vit,
            embed_dim=int(manifest["embed_dim"]),
            variant=str(manifest["variant"]),
            vjepa2_1_forward=vjepa2_1_forward,
            svae_path=svae_path,
            svae_target_dim=svae_target_dim,
        )

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "VJEPA21VideoEncoder":
        """Deploy-time zero-weight ViT shell, sized by manifest.

        Unlike :class:`WanVideoVAEEncoder` (which reconstructs from the saved
        ``components_entry``), V-JEPA's training-time component-spec
        generator did not persist ViT geometry into ``config.yaml`` for
        PR #67-era runs — ``components[vae]`` actually carries the Wan
        ``WanVideoVAE38`` class (an artifact of ``get_component_specs``
        scanning the Wan ``model_path``). We therefore ignore
        ``components_entry`` and reconstruct from
        ``<ckpt_dir>/manifest.json``, written by :meth:`save_deploy_assets`
        at checkpoint save time. Deploy is strictly self-contained: there is
        no ``encoder.model_path`` fallback, so a checkpoint saved without its
        manifest fails loudly here rather than silently reaching back to a
        training-time path that may be unmounted on the deploy host.

        ViT weights are NOT loaded here — the architecture's strict
        ``load_checkpoint`` populates ``video_encoder._m.*`` from the saved
        safetensors immediately after this call returns.
        """
        manifest_dir = cls._resolve_manifest_dir(ckpt_dir)
        manifest = _vjepa_loader.read_and_validate_manifest(manifest_dir)
        vit_encoder = _vjepa_loader.prepare_vjepa_imports_and_patch()
        with torch.device(device):
            vit = _vjepa_loader.build_vit_from_manifest(vit_encoder, manifest)
        # ``vjepa2_1_forward`` is a runtime knob plumbed through the yaml
        # ``encoder`` block so a checkpoint+yaml pair rebuilds the same encoder
        # the run trained. We warn (only) when a saved yaml omits it: that is
        # the pre-PR-#92 checkpoint signature, and those ckpts are NOT forward-
        # compatible (temporal_compression 2→4 + the 2-pass target rewrite) and
        # must be retrained. ``encoder_cfg is None`` (programmatic callers) is
        # not warned — there is no saved yaml to fix.
        vjepa2_1_forward = cls._read_vjepa2_1_forward_from_cfg(encoder_cfg)
        if encoder_cfg is not None and not cls._cfg_has_vjepa2_1_forward(encoder_cfg):
            logger.warning(
                "VJEPA21VideoEncoder.from_skeleton: saved encoder yaml has no "
                "``vjepa2_1_forward`` field; defaulting to %r. A pre-PR-#92 "
                "checkpoint is NOT forward-compatible (temporal_compression 2→4 "
                "+ 2-pass target rewrite) and must be retrained; setting "
                "``vjepa2_1_forward: mixed`` only restores the cond-frame branch.",
                vjepa2_1_forward,
            )
        # Reducer rebuild: the sidecar config (if the training ckpt carried an
        # S-VAE) sizes a zero-weight shell here; strict ``load_checkpoint``
        # fills ``_svae.*`` right after. ``svae_target_dim`` from the saved yaml
        # is an optional cross-check against the sidecar's ``latent_dim``.
        svae_config = reducer.read_sidecar(ckpt_dir)
        svae_target_dim = reducer.read_target_dim_from_cfg(encoder_cfg)
        logger.info(
            "VJEPA21VideoEncoder.from_skeleton: %s instantiated from %s "
            "(embed_dim=%d, variant=%s, vjepa2_1_forward=%s, svae=%s) — weights pending checkpoint load",
            manifest["arch_name"],
            manifest_dir,
            int(manifest["embed_dim"]),
            str(manifest["variant"]),
            vjepa2_1_forward,
            "on" if svae_config is not None else "off",
        )
        return cls(
            vit,
            embed_dim=int(manifest["embed_dim"]),
            variant=str(manifest["variant"]),
            vjepa2_1_forward=vjepa2_1_forward,
            svae_config=svae_config,
            svae_target_dim=svae_target_dim,
        )

    @staticmethod
    def _cfg_has_vjepa2_1_forward(encoder_cfg: Any) -> bool:
        """Report whether the saved encoder yaml carries the key at all
        (yaml-``null`` counts as present). ``from_skeleton`` needs this — vs.
        :meth:`_read_vjepa2_1_forward_from_cfg`, which collapses absent/null to
        the default — to fire the pre-PR-checkpoint migration warning only when
        the operator truly omitted the key.
        """
        if encoder_cfg is None:
            return False
        if isinstance(encoder_cfg, dict):
            return "vjepa2_1_forward" in encoder_cfg
        _MISSING = object()
        return getattr(encoder_cfg, "vjepa2_1_forward", _MISSING) is not _MISSING

    @staticmethod
    def _read_vjepa2_1_forward_from_cfg(encoder_cfg: Any) -> _VJEPA21Forward:
        """Pick ``vjepa2_1_forward`` from the saved yaml; absent / yaml-null
        collapse to the default."""
        if encoder_cfg is None:
            return _VJEPA21_FORWARD_DEFAULT
        if isinstance(encoder_cfg, dict):
            value = encoder_cfg.get("vjepa2_1_forward")
        else:
            value = getattr(encoder_cfg, "vjepa2_1_forward", None)
        if value is None:
            return _VJEPA21_FORWARD_DEFAULT
        return str(value)  # type: ignore[return-value]  # __init__ validates

    @staticmethod
    def _resolve_manifest_dir(ckpt_dir: str | None) -> str:
        """Return ``ckpt_dir`` when it holds a readable ``manifest.json``.

        Deploy is strictly self-contained: the manifest must live next to the
        checkpoint (written by :meth:`save_deploy_assets`); there is no
        ``encoder.model_path`` fallback. A missing manifest is a hard error.
        ``os.path.isfile`` (not ``exists``) rejects a directory named
        ``manifest.json`` so the failure is named here rather than as a
        confusing ``json.load`` error later.
        """
        ckpt_manifest = os.path.join(ckpt_dir, "manifest.json") if ckpt_dir else None
        if ckpt_manifest and os.path.isfile(ckpt_manifest):
            return str(ckpt_dir)
        raise FileNotFoundError(
            "VJEPA21VideoEncoder.from_skeleton: no readable manifest.json at "
            f"ckpt_dir={ckpt_manifest!r}. Re-save the checkpoint with the "
            "current code, which writes manifest.json into ckpt_dir."
        )

    def save_deploy_assets(self, output_dir: str, cfg: Any) -> None:
        """Copy ``manifest.json`` from ``encoder.model_path`` into
        ``<output_dir>/manifest.json`` so deploy is self-contained.

        Strict self-contained — no deploy-time fallback: an unresolvable cfg,
        a missing source manifest, or a copy IO error all raise, because
        :meth:`from_skeleton` reads the manifest only from ``ckpt_dir``; a
        checkpoint saved without its manifest cannot be deployed. This runs
        once at rank-0 start-up before any weights are saved, so a raise fails
        the run fast instead of producing deploy-unloadable checkpoints.

        The source is re-read from
        ``cfg.model.video_backbone.encoder.model_path`` (the yaml field
        training read at construction) rather than cached on ``self``, so the
        call survives a future cfg-layout refactor.
        """
        import shutil

        # The S-VAE sidecar is likewise strict (``reducer.write_sidecar`` raises):
        # without it ``from_skeleton`` cannot size the reducer.
        if self._svae is not None:
            reducer.write_sidecar(self._svae, output_dir, type(self).__name__)

        try:
            enc_cfg = cfg.model.video_backbone.encoder
            if isinstance(enc_cfg, dict):
                model_path = enc_cfg.get("model_path")
            else:
                model_path = getattr(enc_cfg, "model_path", None)
        except Exception:
            # cfg shape (dict / DictConfig / mock) varies; an unreadable cfg
            # collapses to model_path=None and the hard error below.
            model_path = None

        if not model_path:
            raise FileNotFoundError(
                "VJEPA21VideoEncoder.save_deploy_assets: cannot resolve "
                "model.video_backbone.encoder.model_path from cfg; cannot copy "
                "manifest.json (deploy reads it only from ckpt_dir)."
            )
        src = os.path.join(str(model_path), "manifest.json")
        dst = os.path.join(output_dir, "manifest.json")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"VJEPA21VideoEncoder.save_deploy_assets: manifest.json not found at {src}.")
        if os.path.abspath(src) == os.path.abspath(dst):
            return
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(src, dst)
        logger.info("VJEPA21VideoEncoder.save_deploy_assets: copied %s -> %s", src, dst)

    # Not overriding build_dit_input_proj / build_dit_output_proj: the default
    # Conv3d/Linear pair at dit_patch_size=(1,2,2) already gives the Wan VAE
    # DiT-side layout (token-count parity).


__all__ = ["VJEPA21VideoEncoder"]
