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

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_VJEPA21Forward = Literal["video", "mixed"]
_VJEPA21_FORWARD_DEFAULT: _VJEPA21Forward = "video"
# Derive ALLOWED from the Literal so adding a mode in one place can't drift
# from the runtime whitelist used by ``__init__`` / cfg parsing.
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
        # V-JEPA follows the host dtype set by ``set_dtype_device`` (bf16 in
        # production). The upstream ``rotate_queries_or_keys`` would naturally
        # promote Q/K to fp32 (its sin/cos table is built from a fp32 mask),
        # causing an SDPA dtype mismatch against bf16 V. We fix that with a
        # module-level monkey-patch installed by ``_vjepa_loader`` that casts the
        # RoPE output back to ``x.dtype`` — root-cause fix, and keeps V-JEPA
        # in the host dtype so DeepSpeed ZeRO-3's mixed-precision all_gather
        # stays happy (a fp32-pinned frozen submodule trips
        # ``all_gather_into_tensor`` because its output buffer is bf16).
        self._m = vit
        self._variant = variant
        if vjepa2_1_forward not in _VJEPA21_FORWARD_ALLOWED:
            raise ValueError(f"vjepa2_1_forward must be one of {_VJEPA21_FORWARD_ALLOWED}, got {vjepa2_1_forward!r}.")
        self._vjepa2_1_forward: _VJEPA21Forward = vjepa2_1_forward
        self._raw_embed_dim = int(embed_dim)
        # Optional S-VAE feature reducer. Unlike a linear PCA projection (which
        # commutes with the temporal mean-pool and may sit pre-pool), the S-VAE
        # is non-linear and is applied AFTER the cond+target cat (see
        # ``batch_encode``), so the compact latent that becomes the prediction
        # target is trained on exactly the post-pool distribution. When enabled
        # the encoder advertises the reducer's ``latent_dim`` as ``z_dim`` so the
        # DiT first conv / unpatchify head / freeze yaml / feature_norm all
        # rebuild against the smaller dim.
        self._svae = self._build_svae(svae_path, svae_target_dim, svae_config)
        effective_z_dim = self._effective_z_dim(self._raw_embed_dim)
        self._spec = VideoEncoderProperties(
            z_dim=int(effective_z_dim),
            spatial_compression=16,
            # Effective temporal compression of the encoder is 4 (matching
            # Wan VAE causal grouping): 1 cond latent from frame 0 + 1 latent
            # per 4 target pixel frames. Internally this is two steps —
            # ViT tubelet=2 produces 1 latent per 2 frames, then the
            # ``_TARGET_TEMPORAL_POOL_STRIDE`` avg-pool over time halves the
            # target stream again. With this value the noise-init formula
            # ``(T_pix - 1) // tc + 1`` lands on the same T_lat as Wan VAE.
            temporal_compression=4,
            causal_temporal=True,
            pixel_range=(-1.0, 1.0),
            is_reversible=False,
            # (1, 2, 2) — matches Wan VAE's DiT-side patch layout so the
            # per-frame token grid (H/16/2 × W/16/2) lines up with the
            # native VAE path (H/8/2 × W/8/2 is the same product when the
            # encoder's spatial_compression equals the native VAE's
            # ``upsampling_factor * 2``; for Wan2.2 TI2V-5B both are 16).
            # The default ``build_dit_input_proj`` Conv3d((1,2,2),(1,2,2))
            # head pools 4 V-JEPA spatial neighbors per DiT token; the
            # unpatchify head mirrors with Linear(dit_dim, z_dim * 4).
            dit_patch_size=(1, 2, 2),
        )
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        # Post-norm: a plain LayerNorm at init (weight=1, bias=0) acts as
        # per-token standardization that pulls V-JEPA's O(30)-scale features
        # down to Wan-latent O(1). It is structurally trainable, but in the
        # current OpenWAM data path the entire encoder forward runs inside
        # ``BaseWAMArchitecture.preprocess`` / ``prepare_inputs``, both
        # decorated with ``@torch.no_grad`` (see ``openwam/model/architectures/architecture_base.py``
        # ~L765/L793). So even with ``requires_grad=True`` the LayerNorm
        # params receive no gradient and behave as a fixed standardizer
        # across training. Both the freeze yaml (which freezes
        # ``video_backbone.video_encoder`` wholesale) and the no_grad
        # decorators have to be lifted before this can adapt — out of
        # scope for the V-JEPA 2.1 integration PR. The module still lives
        # OUTSIDE ``self._m`` so a future PR can carve out a grad-enabled
        # path here without restructuring the ViT freeze granularity.
        #
        # When the S-VAE reducer is enabled this LayerNorm operates on the
        # already-near-whitened 48-d posterior mean; it is kept mainly for
        # structural symmetry with the raw 1408-d path (a no-op affine at init).
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

    # ViT tubelet=2 collapses every 2 pixel frames into 1 latent frame; an
    # extra avg-pool over time with stride 2 halves the target stream again
    # so the total effective temporal compression matches Wan VAE's causal
    # grouping (1 cond + 1 latent per 4 target pixel frames). The stride is
    # locked to 2: V-JEPA's tubelet and Wan VAE's temporal_compression are
    # both production-fixed (tubelet=2 / tc=4), so the ratio between them is
    # fixed. The constant is exposed at class level so a reader can locate
    # the only place this "extra 2x" assumption lives.
    _TARGET_TEMPORAL_POOL_STRIDE = 2

    def batch_encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T_pixel, H, W) -> (B, embed_dim, T_lat, H/16, W/16).

        Output ``T_lat == 1`` when ``T_pixel == 1`` (TI2V ref-frame fast
        path); otherwise ``T_lat == 1 + (T_pixel - 1) // 4`` — a frame-0
        condition latent plus ``(T_pixel - 1) / 4`` target latents — which
        emulates the Wan VAE causal grouping (1 + group-of-4) the host
        backbone's first-conv / unpatchify expect. Output shape is the SAME
        under both ``vjepa2_1_forward`` modes; the modes differ only in how
        the condition slice is obtained:

        Condition pass (frame 0 → one latent slice, no target leakage):
          - ``vjepa2_1_forward="video"`` (default): cat([f0, f0], dim=T) →
            video branch (tubelet=2) → 1 latent.
          - ``vjepa2_1_forward="mixed"``: f0 → image branch (tubelet=1) →
            1 latent.

        Target pass (both modes): cat([f0, f0, t1..tN], dim=T) → video branch
        → ``1 + N/2`` latents → discard the first temporal slice (the
        prepended frame-0 pair's latent) → ``N/2`` raw target latents → an
        avg-pool over time with stride ``_TARGET_TEMPORAL_POOL_STRIDE = 2``
        → ``N/4`` target latents. The two prepended frame-0 copies let the
        ViT's temporal attention condition target representations on the
        reference frame; the dropped slice is exactly the one that would
        otherwise leak back into the cond lane, so independence between
        condition and target is preserved and deploy/train see the same
        condition latent. The avg-pool brings the target token-count to
        Wan VAE parity (see ``_TARGET_TEMPORAL_POOL_STRIDE``).
        """
        z = self._batch_encode_pooled_raw(video)
        z = self._apply_svae_if_enabled(z)
        z = self._apply_feature_norm(z)
        return z

    def _batch_encode_pooled_raw(self, video: torch.Tensor) -> torch.Tensor:
        """Encoder forward + temporal mean-pool, BEFORE the S-VAE / feature_norm.

        Returns the channel-first ``(B, raw_embed_dim, T_lat, H/16, W/16)`` cat
        of the (un-pooled) cond latent and the mean-pooled target latents — the
        exact tensor the optional S-VAE reducer consumes. Split out from
        :meth:`batch_encode` so offline S-VAE training / statistics collection
        (:meth:`batch_encode_pooled_for_svae_training`) sees byte-for-byte the
        same post-pool distribution the main path produces.
        """
        B, C, Tp, H, W = video.shape
        if C != 3:
            raise ValueError(f"V-JEPA 2.1 expects 3-channel input; got C={C}.")
        # Defensive: callers normally hand us host dtype already, but the
        # RoPE monkey-patch (installed in ``_vjepa_loader``) only guarantees Q/K
        # match ``x.dtype`` at SDPA — so we still align the input to ViT
        # param dtype to keep the matmuls type-clean.
        m_dtype = next(self._m.parameters()).dtype
        if video.dtype != m_dtype:
            video = video.to(m_dtype)
        f0 = video[:, :, 0:1]
        if Tp == 1:
            return self._encode_condition(f0)
        # ViT tubelet=2 needs (T_pixel - 1) even (target frames make a
        # whole number of tubes); the extra time-pool needs that count
        # of latent target frames itself even — combined,
        # ``(T_pixel - 1) % 4 == 0``. For RoBoTwin (num_frames=33,
        # video_stride=4 → T_pixel=9), (9-1) % 4 == 0. ✓
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

    # V-JEPA 2.1 vision_transformer.forward contract:
    #   input (B, C, T, H, W); T==1 ∧ self._m.img_temporal_dim_size==1 hits
    #   patch_embed_img (PatchEmbed3D, tubelet_size=1) + img_mod_embed; else
    #   patch_embed (PatchEmbed3D, tubelet_size=2) + video_mod_embed. Output
    #   is (B, L, D) flat with L = T_lat * (H/16) * (W/16) in T-major then
    #   (H, W) row-major order.
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
        # Explicit signature (no ``**kw``) so a programmatic typo like
        # ``from_pretrained(path, vjepa_2_1_forward="video")`` raises
        # ``TypeError`` at the call site instead of silently falling back
        # to the default forward mode. The yaml path is already filtered
        # by ``build_video_encoder`` via ``optional_yaml_keys()``.
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
        ``components_entry`` and reconstruct from a ``manifest.json``.

        Manifest source priority (and the rationale for each branch):

        1. ``<ckpt_dir>/manifest.json`` — written by
           :meth:`save_deploy_assets` at checkpoint save time. The
           preferred branch for self-contained deploy: the deploy host
           needs to read the checkpoint dir anyway, and the manifest is
           a small (<1 KB) JSON.
        2. ``<encoder_cfg.model_path>/manifest.json`` — backward-compat
           fallback for PR #85-era checkpoints saved before this self-
           containment patch. The deploy host must reach
           ``encoder.model_path`` for these.

        Existing checkpoints can be migrated by hand-copying the manifest
        into the checkpoint dir (see issue #86 for the recipe). After this
        change is merged, all new checkpoints land on branch (1)
        automatically.

        ViT weights are NOT loaded here — the architecture's strict
        ``load_checkpoint`` populates ``video_encoder._m.*`` from the saved
        safetensors immediately after this call returns.
        """
        manifest_dir = cls._resolve_manifest_dir(ckpt_dir, encoder_cfg)
        manifest = _vjepa_loader.read_and_validate_manifest(manifest_dir)
        vit_encoder = _vjepa_loader.prepare_vjepa_imports_and_patch()
        with torch.device(device):
            vit = _vjepa_loader.build_vit_from_manifest(vit_encoder, manifest)
        # ``vjepa2_1_forward`` is a runtime knob (selects the cond-frame
        # branch); it is plumbed through the yaml ``encoder`` block at
        # deploy time so a checkpoint+yaml pair deployed together always
        # rebuilds the same encoder the training run used.
        #
        # Pre-PR-#92 checkpoint caveat: those ckpts are NOT forward-
        # compatible with the current code and must be retrained. Two
        # unrelated changes broke bit-equivalence and ``vjepa2_1_forward``
        # only covers one of them:
        #
        # 1. ``spec.temporal_compression`` is now 4 (was 2 under the legacy
        #    single-forward path). The cross-check in
        #    ``BaseWAMArchitecture._init_video_backbone`` fails-fast when
        #    the saved yaml still declares
        #    ``video_backbone.temporal_compression: 2``, so a pre-PR ckpt
        #    will not even reach this method without a yaml edit.
        # 2. The target pass now prepends two copies of frame 0, discards
        #    the first latent, and mean-pools the remaining latents over
        #    time with stride 2 — the legacy single-pass path did none of
        #    this. Setting ``vjepa2_1_forward: mixed`` only restores the
        #    cond-frame branch (image branch, tubelet=1); the target
        #    stream content cannot be reproduced.
        #
        # We still warn when the field is absent so operators see a
        # noisy signal rather than a silently-degraded run. New training
        # runs that compose the canonical ``model/encoder=vjepa2_1``
        # group config always pick up the explicit value and do not trip
        # this warning.
        vjepa2_1_forward = cls._read_vjepa2_1_forward_from_cfg(encoder_cfg)
        # ``encoder_cfg is None`` is intentionally NOT warned here: that
        # path is exercised by direct programmatic callers (unit tests,
        # raw ``from_skeleton(components_entry=...)``) where there is no
        # saved yaml to edit and the operator's expectation is "use the
        # current default" — a warning would just be noise. The warning
        # targets the pre-PR-#92-deploy path specifically, where a
        # saved-yaml object exists but the field is absent from it.
        if encoder_cfg is not None and not cls._cfg_has_vjepa2_1_forward(encoder_cfg):
            logger.warning(
                "VJEPA21VideoEncoder.from_skeleton: saved encoder yaml has no "
                "``vjepa2_1_forward`` field; defaulting to %r. If this is a "
                "pre-PR-#92 checkpoint (trained before the 2-pass batch_encode "
                "rewrite), it is NOT forward-compatible with this code and "
                "should be retrained: the target pass has changed (prepend-"
                "and-discard + temporal mean-pool stride=2, ``temporal_"
                "compression`` 2 → 4), so target latents will not match "
                "training regardless of ``vjepa2_1_forward``. Setting "
                "``vjepa2_1_forward: mixed`` in the saved config.yaml's "
                "``model.video_backbone.encoder`` block only restores the "
                "cond-frame bit-equivalence (image branch, tubelet=1).",
                vjepa2_1_forward,
            )
        # Reducer rebuild: the sidecar config (if the training ckpt carried an
        # S-VAE) sizes a zero-weight shell here; strict ``load_checkpoint``
        # fills ``_svae.*`` right after. ``svae_target_dim`` from the saved yaml
        # is an optional cross-check against the sidecar's ``latent_dim``.
        svae_config = cls._read_svae_sidecar(ckpt_dir)
        svae_target_dim = cls._read_svae_target_dim_from_cfg(encoder_cfg)
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
        """Report whether the saved encoder yaml carries the key at all.

        ``_read_vjepa2_1_forward_from_cfg`` collapses absent / yaml-null /
        ``None`` into "use default", which is what callers want for the
        actual lookup. ``from_skeleton`` separately needs to know whether
        the operator omitted the key (the pre-PR-checkpoint signature)
        to drive a one-time migration warning, so this helper returns a
        boolean for that question — yaml-``null`` counts as present.
        """
        if encoder_cfg is None:
            return False
        if isinstance(encoder_cfg, dict):
            return "vjepa2_1_forward" in encoder_cfg
        _MISSING = object()
        return getattr(encoder_cfg, "vjepa2_1_forward", _MISSING) is not _MISSING

    @staticmethod
    def _read_vjepa2_1_forward_from_cfg(encoder_cfg: Any) -> _VJEPA21Forward:
        """Pick ``vjepa2_1_forward`` from the saved yaml at deploy time.

        ``yaml: null`` is treated as "use default" (collapses with absent
        via ``.get(...)`` / ``getattr(..., None)``). This intentionally
        differs from the V-JEPA 2 sibling, which rejects a *present*
        ``vjepa2_1_forward`` key regardless of value — because there the
        field is the wrong-encoder signature (operator copy-paste from a
        vjepa2_1 yaml), while here null is just an explicit "no value,
        please default". A note in the user's yaml (``vjepa2_1_forward:``)
        therefore behaves like omission, which is the least-surprise
        reading for the V-JEPA 2.1 path."""
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
    def _resolve_manifest_dir(ckpt_dir: str | None, encoder_cfg: Any) -> str:
        """Pick which directory holds a readable ``manifest.json`` at deploy time.

        See :meth:`from_skeleton` for the priority rationale. The returned
        directory is guaranteed to contain a regular ``manifest.json`` file
        (``os.path.isfile`` — directories with that name are NOT accepted;
        keeps the check consistent with the training-side copy below and
        avoids a confusing ``json.load`` error if the path somehow exists as
        a directory). When neither source yields one, the raised
        ``FileNotFoundError`` names BOTH attempted paths so the operator can
        see exactly where we looked without having to read the source.
        """
        ckpt_manifest = os.path.join(ckpt_dir, "manifest.json") if ckpt_dir else None
        if ckpt_manifest and os.path.isfile(ckpt_manifest):
            return str(ckpt_dir)

        fallback_dir: str | None = None
        if encoder_cfg is not None:
            if isinstance(encoder_cfg, dict):
                fallback_dir = encoder_cfg.get("model_path")
            else:
                fallback_dir = getattr(encoder_cfg, "model_path", None)
        fallback_manifest = os.path.join(str(fallback_dir), "manifest.json") if fallback_dir else None
        if fallback_manifest and os.path.isfile(fallback_manifest):
            return str(fallback_dir)

        raise FileNotFoundError(
            "VJEPA21VideoEncoder.from_skeleton: no readable manifest.json. "
            f"Tried ckpt_dir={ckpt_manifest!r} and "
            f"encoder.model_path={fallback_manifest!r}. Neither source is "
            "reachable / has a manifest. Either re-save the checkpoint with "
            "the current code (which writes manifest.json into ckpt_dir), or "
            "hand-copy manifest.json into the checkpoint dir."
        )

    def save_deploy_assets(self, output_dir: str, cfg: Any) -> None:
        """Copy ``manifest.json`` from ``encoder.model_path`` into
        ``<output_dir>/manifest.json`` so deploy is self-contained.

        Resolves the manifest source via
        ``cfg.model.video_backbone.encoder.model_path`` — the same yaml
        field training read at construction time. We re-read the cfg (vs.
        caching the path on ``self``) so the call doesn't break if a
        future refactor moves things; missing/empty config and IO errors
        log a warning and skip the copy rather than raising (a copy
        failure must never crash an otherwise-good training run; deploy
        then falls back to the legacy ``encoder.model_path`` branch in
        :meth:`from_skeleton`).

        IO failure handling matters because :meth:`save_deploy_assets`
        runs inside the trainer's checkpoint save flow — losing the
        safetensors save over a manifest-copy ``PermissionError`` /
        ``ENOSPC`` / disappearing-mount ``OSError`` would be a strict
        regression vs. the pre-self-containment behavior.
        """
        import shutil

        # The S-VAE sidecar has NO deploy-time fallback (unlike the manifest,
        # which can fall back to ``encoder.model_path``): without it
        # ``from_skeleton`` cannot size the reducer and the strict checkpoint
        # load fails. ``save_deploy_assets`` runs once at run start-up, before
        # any weights are saved, so letting a write failure raise here fails the
        # run fast rather than silently producing checkpoints that cannot be
        # deployed. (It is a rank-0-only start-up step, so the raise surfaces on
        # rank 0 — operators should treat it as a setup failure.)
        if self._svae is not None:
            self._write_svae_sidecar(output_dir)

        model_path = None
        try:
            enc_cfg = cfg.model.video_backbone.encoder
            if isinstance(enc_cfg, dict):
                model_path = enc_cfg.get("model_path")
            else:
                model_path = getattr(enc_cfg, "model_path", None)
        except Exception:
            # Intentionally broad: cfg shape (dict / DictConfig / mock)
            # varies across call sites, and the contract above forbids
            # raising. Anything that prevents us from reading model_path
            # collapses to "skip the copy, deploy falls back".
            pass

        if not model_path:
            logger.warning(
                "VJEPA21VideoEncoder.save_deploy_assets: cannot resolve "
                "model.video_backbone.encoder.model_path from cfg; skipping "
                "manifest copy. Deploy will fall back to encoder.model_path."
            )
            return
        src = os.path.join(str(model_path), "manifest.json")
        dst = os.path.join(output_dir, "manifest.json")
        if not os.path.isfile(src):
            logger.warning(
                "VJEPA21VideoEncoder.save_deploy_assets: manifest.json "
                "not found at %s; skipping copy. Deploy will fall back to "
                "encoder.model_path.",
                src,
            )
            return
        if os.path.abspath(src) == os.path.abspath(dst):
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copyfile(src, dst)
        except OSError as e:
            logger.warning(
                "VJEPA21VideoEncoder.save_deploy_assets: copying %s -> %s "
                "failed (%s); skipping. Deploy will fall back to encoder.model_path.",
                src,
                dst,
                e,
            )
            return
        logger.info(
            "VJEPA21VideoEncoder.save_deploy_assets: copied %s -> %s",
            src,
            dst,
        )

    # Intentionally NOT overriding build_dit_input_proj / build_dit_output_proj:
    # spec.dit_patch_size=(1,2,2) makes the default Conv3d/Linear pair produce
    # Conv3d(z_dim, dit_dim, (1,2,2), (1,2,2)) and Linear(dit_dim, z_dim * 4)
    # — a 2x2 spatial pool per DiT token that mirrors Wan VAE's DiT-side
    # patch layout (token-count parity; see PR description).


__all__ = ["VJEPA21VideoEncoder"]
