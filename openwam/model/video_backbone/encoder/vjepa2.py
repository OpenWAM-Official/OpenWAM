"""V-JEPA 2 video encoder (Assran et al., arXiv:2506.09985).

Plugs into the :class:`VideoBackbone` external-encoder path with the same
spec contract as V-JEPA 2.1 (``spec.is_reversible=False``,
``spec.causal_temporal=True``, ``spec.temporal_compression=4`` — ViT
tubelet=2 + encoder-side avg-pool over time with stride=2 to match Wan
VAE causal grouping) so the host backbone treats it identically end-to-end.

The single behavioral difference vs V-JEPA 2.1 is that the upstream
``src.models.vision_transformer.VisionTransformer`` ships **without an
image branch** — there is no ``img_temporal_dim_size`` / ``patch_embed_img``
fast path for ``T_pixel == 1`` input. Every forward routes through the
tubelet=2 video branch, so the condition (frame 0) latent is always
obtained by dup-ing frame 0 once and running the resulting 2-frame clip
through the encoder. No equivalent of V-JEPA 2.1's ``vjepa2_1_forward``
knob exists; setting that yaml field with the vjepa2 encoder is a
fail-fast at construction time.

Target latents follow the same prepend-and-discard policy as V-JEPA 2.1:
``cat([f0, f0, targets])`` through the video branch, drop the first
temporal latent. Condition and target each go through their own forward
so target information never leaks into the condition lane.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from openwam.model.video_backbone.encoder import VideoEncoder, register_video_encoder
from openwam.model.video_backbone.encoder.spec import VideoEncoderSpec

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@register_video_encoder("vjepa2")
class VJEPA2VideoEncoder(VideoEncoder):
    """V-JEPA 2 video encoder.

    Constructor takes an already-built ViT module so unit tests can inject
    a mock without going through ``from_pretrained`` (which requires the
    upstream ``src.models`` package — see ``third_party/vjepa2/`` — and a
    local checkpoint file).
    """

    def __init__(self, vit: nn.Module, *, embed_dim: int, variant: str):
        super().__init__()
        # Follows the host dtype set by ``set_dtype_device`` (bf16 in
        # production). The upstream ``rotate_queries_or_keys`` would
        # promote Q/K to fp32 (sin/cos table built from a fp32 mask),
        # causing an SDPA dtype mismatch against bf16 V. We fix that with
        # a module-level monkey-patch installed in
        # ``_prepare_vjepa_imports_and_patch`` that casts the RoPE output
        # back to ``x.dtype`` — root-cause fix, keeps the ViT in host
        # dtype so DeepSpeed ZeRO-3's mixed-precision all_gather stays
        # happy (a fp32-pinned frozen submodule trips
        # ``all_gather_into_tensor`` because its output buffer is bf16).
        self._m = vit
        self._variant = variant
        self._spec = VideoEncoderSpec(
            z_dim=int(embed_dim),
            spatial_compression=16,
            # Effective temporal compression of the encoder is 4 (matching
            # Wan VAE causal grouping): 1 cond latent from frame 0 + 1 latent
            # per 4 target pixel frames. Internally this is two steps —
            # ViT tubelet=2 produces 1 latent per 2 frames, then the
            # ``_TARGET_TEMPORAL_POOL_STRIDE`` avg-pool over time halves the
            # target stream again. See the V-JEPA 2.1 sibling for the
            # rationale; the math is identical here.
            temporal_compression=4,
            causal_temporal=True,
            pixel_range=(-1.0, 1.0),
            is_reversible=False,
            # (1, 2, 2) — matches Wan VAE's DiT-side patch layout so the
            # per-frame token grid lines up with the native VAE path. See
            # the V-JEPA 2.1 sibling for the spatial-token math; the same
            # geometry applies (ViT-g/16 patch_embed + DiT patch=(1,2,2)
            # gives the same H/32 × W/32 grid as Wan VAE upsample=16 +
            # DiT patch=(1,2,2) on Wan2.2 TI2V-5B).
            dit_patch_size=(1, 2, 2),
        )
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        # Post-norm matches the sibling encoders: a plain LayerNorm at
        # init (weight=1, bias=0) acts as per-token standardization that
        # pulls raw ViT-scale features down to Wan-latent O(1). Frozen
        # for now by the freeze_modules path on ``video_backbone._encoder``
        # plus the @torch.no_grad-decorated preprocess/prepare_inputs
        # hooks. A future PR can carve out a grad-enabled path.
        self.feature_norm = nn.LayerNorm(int(embed_dim))

    @property
    def spec(self) -> VideoEncoderSpec:
        return self._spec

    @property
    def variant(self) -> str:
        return self._variant

    @classmethod
    def optional_yaml_keys(cls) -> set[str]:
        # No ``vjepa2_1_forward`` analog — V-JEPA 2 has no image branch,
        # so the "video" path is the only meaningful forward strategy.
        # Setting the field on this encoder is rejected in
        # ``from_pretrained`` / ``from_skeleton`` rather than silently
        # ignored, so a yaml that mistakenly carries the V-JEPA 2.1 knob
        # over to a vjepa2 run fails fast.
        return set()

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

    # See V-JEPA 2.1 sibling for the rationale: ViT tubelet=2 + this
    # avg-pool over time = effective temporal compression 4 (Wan VAE
    # parity). Locked at 2 because tubelet/tc are production-fixed.
    _TARGET_TEMPORAL_POOL_STRIDE = 2

    def batch_encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T_pixel, H, W) -> (B, embed_dim, T_lat, H/16, W/16).

        Output ``T_lat == 1`` when ``T_pixel == 1`` (TI2V ref-frame fast
        path); otherwise ``T_lat == 1 + (T_pixel - 1) // 4`` — same
        causal-grouped layout as V-JEPA 2.1's ``batch_encode``. Both
        slices come from independent video-branch forwards:

        - Condition pass: cat([f0, f0], dim=T) → 2-frame clip → tubelet=2
          → 1 latent.
        - Target pass: cat([f0, f0, t1..tN], dim=T) → ``1 + N/2`` latents
          → drop the first temporal slice (the prepended pair's latent) →
          ``N/2`` raw target latents → avg-pool over time with stride
          ``_TARGET_TEMPORAL_POOL_STRIDE = 2`` → ``N/4`` target latents.

        The two passes never see each other's outputs, so the condition
        latent is independent of targets (deploy / train match) while
        the target latents have already attended to the reference frame
        and produce a richer supervision signal. The avg-pool brings the
        target token-count to Wan VAE parity.
        """
        B, C, Tp, H, W = video.shape
        if C != 3:
            raise ValueError(f"V-JEPA 2 expects 3-channel input; got C={C}.")
        # Align input to ViT param dtype to keep matmuls type-clean even
        # if a caller hands us a stray dtype.
        m_dtype = next(self._m.parameters()).dtype
        if video.dtype != m_dtype:
            video = video.to(m_dtype)
        f0 = video[:, :, 0:1]
        if Tp == 1:
            z = self._encode_condition(f0)
        else:
            # ViT tubelet=2 + extra time-pool stride=2 → (T_pixel-1)%4==0
            # in production. ``divisor`` stays parametric so the error
            # message tracks the constant if it's ever bumped.
            divisor = 2 * self._TARGET_TEMPORAL_POOL_STRIDE
            if (Tp - 1) % divisor != 0:
                raise ValueError(f"V-JEPA 2 causal emulation needs (T_pixel - 1) % {divisor} == 0, got T_pixel={Tp}.")
            z_cond = self._encode_condition(f0)
            z_target_raw = self._encode_target_with_prepend(f0, video[:, :, 1:])
            z_target = self._pool_target_temporal(z_target_raw)
            z = torch.cat([z_cond, z_target], dim=2)
        z = self._apply_feature_norm(z)
        return z

    def _pool_target_temporal(self, z_target: torch.Tensor) -> torch.Tensor:
        """(B, D, T_target_raw, h, w) -> (B, D, T_target_raw/2, h, w) via
        avg-pool over time with stride ``_TARGET_TEMPORAL_POOL_STRIDE``.
        ``batch_encode`` enforces divisibility before calling here.
        """
        s = self._TARGET_TEMPORAL_POOL_STRIDE
        B, D, T, h, w = z_target.shape
        return z_target.reshape(B, D, T // s, s, h, w).mean(dim=3)

    def _encode_condition(self, f0: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) -> (B, D, 1, H/16, W/16). f0 dup → video branch."""
        return self._encode_video_tubelet(torch.cat([f0, f0], dim=2))

    def _encode_target_with_prepend(self, f0: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) + (B, 3, N_even, H, W) -> (B, D, N/2, H/16, W/16).

        Runs ``video_branch(cat([f0, f0, targets]))`` and drops the first
        temporal latent (the one encoding the prepended frame-0 pair).
        """
        clip = torch.cat([f0, f0, targets], dim=2)
        z_full = self._encode_video_tubelet(clip)
        return z_full[:, :, 1:]

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
            "VJEPA2VideoEncoder is irreversible (spec.is_reversible=False); "
            "pixel decode is not defined. Pass decode_video=False to generate()."
        )

    def to_frames(self, video: torch.Tensor) -> list:
        raise NotImplementedError("VJEPA2VideoEncoder is irreversible; to_frames has no meaning.")

    # Geometry constants the encoder's reshape paths and spec are hard-wired
    # against. Manifests carrying different ``patch`` / ``tubelet`` values
    # are rejected at load time (vs producing a confusing reshape error
    # later inside ``batch_encode``).
    _REQUIRED_MANIFEST_PATCH = 16
    _REQUIRED_MANIFEST_TUBELET = 2

    @classmethod
    def from_pretrained(cls, model_path: str, **kw: Any) -> "VJEPA2VideoEncoder":
        # Order matters: ``_reject_vjepa2_1_only_kwargs`` raises a more
        # specific ``ValueError`` for the high-traffic "V-JEPA 2.1 yaml
        # copy-pasted onto a V-JEPA 2 encoder" mistake (`vjepa2_1_forward`).
        # The generic ``TypeError`` below catches *every* other unexpected
        # kwarg but with a less actionable message, so it must run second.
        cls._reject_vjepa2_1_only_kwargs(kw)
        # Match V-JEPA 2.1's strict signature: anything else here is a
        # programmer typo (the yaml path is already filtered by
        # ``build_video_encoder`` via ``optional_yaml_keys() == set()``).
        if kw:
            raise TypeError(
                f"VJEPA2VideoEncoder.from_pretrained got unexpected kwargs: {sorted(kw)}. "
                "This encoder accepts no optional fields beyond {name, model_path}."
            )
        manifest = cls._read_and_validate_manifest(model_path)
        vit_encoder = cls._prepare_vjepa_imports_and_patch()
        vit = cls._build_vit_from_manifest(vit_encoder, manifest)
        cls._load_vit_weights(vit, model_path, manifest)
        return cls(
            vit,
            embed_dim=int(manifest["embed_dim"]),
            variant=str(manifest["variant"]),
        )

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "VJEPA2VideoEncoder":
        """Deploy-time zero-weight ViT shell, sized by manifest.

        Mirrors :class:`VJEPA21VideoEncoder.from_skeleton`'s
        preferred-with-fallback manifest resolution:

        1. ``<ckpt_dir>/manifest.json`` — written by
           :meth:`copy_deploy_artifacts` at checkpoint save time. This
           is the primary path that keeps the deploy host self-contained.
        2. ``<encoder_cfg.model_path>/manifest.json`` — second source
           used when ``<ckpt_dir>/manifest.json`` is unreachable (e.g.
           the checkpoint was copied without its sidecar artifacts).
           Requires the user-side V-JEPA 2 weight directory to be
           reachable on the deploy host.

        ViT weights are NOT loaded here — the architecture's strict
        ``load_checkpoint`` populates ``_encoder._m.*`` from the saved
        safetensors immediately after this call returns.
        """
        cls._reject_vjepa2_1_only_cfg_keys(encoder_cfg)
        manifest_dir = cls._resolve_manifest_dir(ckpt_dir, encoder_cfg)
        manifest = cls._read_and_validate_manifest(manifest_dir)
        vit_encoder = cls._prepare_vjepa_imports_and_patch()
        with torch.device(device):
            vit = cls._build_vit_from_manifest(vit_encoder, manifest)
        logger.info(
            "VJEPA2VideoEncoder.from_skeleton: %s instantiated from %s "
            "(embed_dim=%d, variant=%s) — weights pending checkpoint load",
            manifest["arch_name"],
            manifest_dir,
            int(manifest["embed_dim"]),
            str(manifest["variant"]),
        )
        return cls(
            vit,
            embed_dim=int(manifest["embed_dim"]),
            variant=str(manifest["variant"]),
        )

    @staticmethod
    def _reject_vjepa2_1_only_kwargs(kw: dict) -> None:
        """``vjepa2_1_forward`` is a V-JEPA 2.1-only runtime knob; raise
        with a clear hint if it leaks through to the vjepa2 path so the
        operator does not have to chase a silently-ignored field."""
        if "vjepa2_1_forward" in kw:
            raise ValueError(
                "video_backbone.encoder.vjepa2_1_forward is V-JEPA 2.1-only — "
                "V-JEPA 2 has no image branch, so the cond pass is locked to "
                "the dup+video forward and the field has no meaning here. "
                "Drop it from your yaml (or switch encoder.name to vjepa2_1)."
            )

    @classmethod
    def _reject_vjepa2_1_only_cfg_keys(cls, encoder_cfg: Any) -> None:
        """Deploy-side mirror of ``_reject_vjepa2_1_only_kwargs``.

        Reject the key whenever it is **present** in the saved yaml block,
        regardless of value — even a ``vjepa2_1_forward: null`` entry on a
        vjepa2 encoder is an operator mistake (the V-JEPA 2.1 knob has no
        meaning here), and silently accepting null would diverge from the
        training-side semantic where ``"vjepa2_1_forward" in kw`` always
        raises.
        """
        if encoder_cfg is None:
            return
        if isinstance(encoder_cfg, dict):
            present = "vjepa2_1_forward" in encoder_cfg
            value = encoder_cfg.get("vjepa2_1_forward") if present else None
        else:
            # OmegaConf / SimpleNamespace: distinguish "absent" from
            # "present as None" via the presence sentinel.
            _MISSING = object()
            value = getattr(encoder_cfg, "vjepa2_1_forward", _MISSING)
            present = value is not _MISSING
        if present:
            cls._reject_vjepa2_1_only_kwargs({"vjepa2_1_forward": value})

    @staticmethod
    def _resolve_manifest_dir(ckpt_dir: str | None, encoder_cfg: Any) -> str:
        """Pick which directory holds a readable ``manifest.json`` at deploy
        time (preferred: ``<ckpt_dir>/manifest.json``; fallback:
        ``<encoder.model_path>/manifest.json``). Mirrors V-JEPA 2.1's helper.
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
            "VJEPA2VideoEncoder.from_skeleton: no readable manifest.json. "
            f"Tried ckpt_dir={ckpt_manifest!r} and "
            f"encoder.model_path={fallback_manifest!r}. Neither source is "
            "reachable / has a manifest. Either re-save the checkpoint with "
            "the current code (which writes manifest.json into ckpt_dir), or "
            "hand-copy manifest.json into the checkpoint dir."
        )

    def copy_deploy_artifacts(self, output_dir: str, cfg: Any) -> None:
        """Copy ``manifest.json`` from ``encoder.model_path`` into
        ``<output_dir>/manifest.json`` so deploy is self-contained.

        Best-effort: any failure (missing source, IO error, unresolvable
        cfg) is logged and skipped — deploy then falls back to the
        ``encoder.model_path`` branch in :meth:`from_skeleton`. Never
        raises so a copy hiccup cannot crash an otherwise-good training
        checkpoint save.
        """
        import shutil

        model_path = None
        try:
            enc_cfg = cfg.model.video_backbone.encoder
            if isinstance(enc_cfg, dict):
                model_path = enc_cfg.get("model_path")
            else:
                model_path = getattr(enc_cfg, "model_path", None)
        except Exception:
            # Intentionally broad: cfg shape (dict / DictConfig / mock)
            # varies across call sites, and the contract forbids raising.
            pass

        if not model_path:
            logger.warning(
                "VJEPA2VideoEncoder.copy_deploy_artifacts: cannot resolve "
                "model.video_backbone.encoder.model_path from cfg; skipping "
                "manifest copy. Deploy will fall back to encoder.model_path."
            )
            return
        src = os.path.join(str(model_path), "manifest.json")
        dst = os.path.join(output_dir, "manifest.json")
        if not os.path.isfile(src):
            logger.warning(
                "VJEPA2VideoEncoder.copy_deploy_artifacts: manifest.json "
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
                "VJEPA2VideoEncoder.copy_deploy_artifacts: copying %s -> %s "
                "failed (%s); skipping. Deploy will fall back to encoder.model_path.",
                src,
                dst,
                e,
            )
            return
        logger.info(
            "VJEPA2VideoEncoder.copy_deploy_artifacts: copied %s -> %s",
            src,
            dst,
        )

    # V-JEPA 2.1-only manifest fields. Their presence in a manifest fed to
    # the vjepa2 encoder is almost always "operator copy-pasted a V-JEPA 2.1
    # manifest into the vjepa2 directory". The upstream V-JEPA 2 ViT
    # constructor silently drops unknown kwargs via ``**kwargs``, so without
    # this check the operator would only learn about the mistake later as
    # an opaque state_dict mismatch (V-JEPA 2.1 ckpt has ``patch_embed_img.*``
    # keys that V-JEPA 2 ViT doesn't, and vice versa).
    _VJEPA21_ONLY_MANIFEST_FIELDS = ("img_temporal_dim_size", "interpolate_rope")

    @classmethod
    def _read_and_validate_manifest(cls, model_path: str) -> dict:
        manifest_path = os.path.join(model_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"VJEPA2 encoder requires manifest.json in {model_path}.")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        patch = int(manifest["patch"])
        tubelet = int(manifest["tubelet"])
        if patch != cls._REQUIRED_MANIFEST_PATCH or tubelet != cls._REQUIRED_MANIFEST_TUBELET:
            raise ValueError(
                f"VJEPA2 manifest patch/tubelet must be "
                f"({cls._REQUIRED_MANIFEST_PATCH}, {cls._REQUIRED_MANIFEST_TUBELET}); "
                f"got ({patch}, {tubelet}). The encoder's spec (spatial_compression=16, "
                f"temporal_compression=4 = ViT tubelet=2 × encoder pool stride=2) and "
                f"reshape logic (H//16, W//16, Tp//2) are hard-wired against these values."
            )
        present_2_1_fields = [k for k in cls._VJEPA21_ONLY_MANIFEST_FIELDS if k in manifest]
        if present_2_1_fields:
            raise ValueError(
                f"VJEPA2 manifest at {manifest_path} carries V-JEPA 2.1-only fields "
                f"{sorted(present_2_1_fields)}. These are added by the V-JEPA 2.1 ViT "
                "wrapper (image branch + RoPE interpolation) and have no analog in the "
                "upstream V-JEPA 2 ViT. The most likely cause is a copy-pasted V-JEPA 2.1 "
                "manifest in a vjepa2 weight dir — point ``encoder.model_path`` at the "
                "correct V-JEPA 2 weight dir, or strip the fields and verify the manifest "
                "actually matches a V-JEPA 2 checkpoint."
            )
        cls._check_arch_use_rope_consistency(manifest)
        return manifest

    @staticmethod
    def _prepare_vjepa_imports_and_patch():
        """Bootstrap ``third_party/vjepa2`` import path + install the RoPE
        dtype monkey-patch. Idempotent. Returns the imported
        ``vision_transformer`` module from ``src.models``.

        Unlike V-JEPA 2.1 (which imports the vendored fork at
        ``app.vjepa_2_1.models.vision_transformer``), V-JEPA 2 uses the
        upstream ``src.models`` package — the same submodule, different
        entry point. The RoPE monkey-patch covers
        ``src.models.utils.modules.rotate_queries_or_keys`` (the V-JEPA 2
        path) and is independent of the V-JEPA 2.1 patch installed on
        ``app.vjepa_2_1.models.utils.modules``; both modules co-exist in
        the submodule and one encoder can be active without affecting
        the other.
        """
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        vjepa2_root = repo_root / "third_party" / "vjepa2"
        if vjepa2_root.is_dir() and str(vjepa2_root) not in sys.path:
            sys.path.insert(0, str(vjepa2_root))

        from src.models import vision_transformer as vit_encoder
        from src.models.utils import modules as vjepa_modules

        if not getattr(vjepa_modules.rotate_queries_or_keys, "_openwam_dtype_safe", False):
            _orig_rotate = vjepa_modules.rotate_queries_or_keys

            def _safe_rotate(*args, **kwargs):
                out = _orig_rotate(*args, **kwargs)
                # Find the input tensor (positional or keyword) to cast back to its dtype.
                ref = args[0] if args else kwargs.get("x", None)
                if isinstance(ref, torch.Tensor) and isinstance(out, torch.Tensor):
                    return out.to(ref.dtype)
                return out

            _safe_rotate._openwam_dtype_safe = True
            vjepa_modules.rotate_queries_or_keys = _safe_rotate

        return vit_encoder

    @staticmethod
    def _check_arch_use_rope_consistency(manifest: dict) -> None:
        """Manifest-internal contradiction check, isolated from the
        upstream ``src.models`` import so the error stays correct in
        environments where the submodule is not initialized.
        """
        arch_name = manifest["arch_name"]
        manifest_use_rope = manifest.get("use_rope", True)
        if arch_name.endswith("_rope") and not manifest_use_rope:
            raise ValueError(
                f"Manifest arch_name={arch_name!r} hardcodes use_rope=True "
                "but the manifest sets use_rope=False. Pick a non-_rope "
                "arch (e.g. 'vit_giant_xformers') or set use_rope=True."
            )

    @staticmethod
    def _build_vit_from_manifest(vit_encoder, manifest: dict) -> nn.Module:
        """Construct a zero-weight ViT per the manifest. No weight load.

        Unlike V-JEPA 2.1, the V-JEPA 2 ViT constructor accepts neither
        ``img_temporal_dim_size`` nor ``interpolate_rope`` — those
        kwargs were added in V-JEPA 2.1's vendored fork to power its
        image branch + RoPE interpolation. The base ViT silently drops
        unknown kwargs through ``**kwargs``, but we pass only the
        signature-supported fields here to keep the contract clean.
        """
        arch_name = manifest["arch_name"]  # e.g. "vit_giant_xformers_rope"
        manifest_use_rope = manifest.get("use_rope", True)
        vit_kwargs: dict[str, Any] = dict(
            patch_size=manifest["patch"],
            img_size=(manifest["img_size"], manifest["img_size"]),
            num_frames=manifest["training_num_frames"],
            tubelet_size=manifest["tubelet"],
            use_sdpa=True,
        )
        # ``*_rope`` wrappers hardcode ``use_rope=True``; only forward the
        # manifest value for non-_rope arches to avoid the
        # "got multiple values for keyword argument 'use_rope'" TypeError.
        if not arch_name.endswith("_rope"):
            vit_kwargs["use_rope"] = manifest_use_rope
        return vit_encoder.__dict__[arch_name](**vit_kwargs)

    @staticmethod
    def _load_vit_weights(vit: nn.Module, model_path: str, manifest: dict) -> None:
        """Populate a constructed ViT with pretrained weights from disk."""
        ckpt = torch.load(
            os.path.join(model_path, manifest["checkpoint_file"]),
            map_location="cpu",
        )
        state_dict = ckpt[manifest.get("checkpoint_key", "target_encoder")]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        # ``strict=False`` tolerates exactly the ``pos_embed`` buffer that
        # the absolute-pos-embedding variants of the published checkpoint
        # ship: the RoPE variants we always load do not consume it. Any
        # other missing / unexpected key is a manifest-vs-checkpoint
        # mismatch and we fail fast rather than silently leave the
        # frozen ViT partially randomly initialized.
        load_result = vit.load_state_dict(state_dict, strict=False)
        unexpected = set(load_result.unexpected_keys) - {"pos_embed"}
        if unexpected or load_result.missing_keys:
            raise RuntimeError(
                "VJEPA2 checkpoint load left the ViT inconsistent with the "
                "constructed module. This usually means the manifest "
                "``arch_name`` does not match the checkpoint, or the "
                "``checkpoint_key`` extracts the wrong sub-dict. Details: "
                f"missing_keys={sorted(load_result.missing_keys)[:8]} "
                f"unexpected_keys={sorted(unexpected)[:8]}."
            )


__all__ = ["VJEPA2VideoEncoder"]
