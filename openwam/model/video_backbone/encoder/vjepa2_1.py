"""V-JEPA 2.1 video encoder (Mur-Labadia et al., arXiv:2603.14482).

Plugs into the :class:`VideoBackbone` external-encoder path introduced in
PR #60. ``spec.is_reversible=False`` — the host backbone must rebuild its
DiT first conv via the default ``build_dit_input_proj`` hook and skip the
strict native-VAE spec validation. ``spec.causal_temporal=True`` and
``spec.temporal_compression=2`` emulate the Wan VAE's first-frame
separability with V-JEPA 2.1's image branch (tubelet=1) on frame 0 and the
video branch (tubelet=2) on the remainder.
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


@register_video_encoder("vjepa2_1")
class VJEPA21VideoEncoder(VideoEncoder):
    """V-JEPA 2.1 video encoder.

    Constructor takes an already-built ViT module so unit tests can inject a
    mock without going through ``_load_vit`` (which requires the upstream
    ``app.vjepa_2_1`` package and a local checkpoint file).
    """

    def __init__(self, vit: nn.Module, *, embed_dim: int, variant: str):
        super().__init__()
        # V-JEPA follows the host dtype set by ``set_dtype_device`` (bf16 in
        # production). The upstream ``rotate_queries_or_keys`` would naturally
        # promote Q/K to fp32 (its sin/cos table is built from a fp32 mask),
        # causing an SDPA dtype mismatch against bf16 V. We fix that with a
        # module-level monkey-patch installed in ``_load_vit`` that casts the
        # RoPE output back to ``x.dtype`` — root-cause fix, and keeps V-JEPA
        # in the host dtype so DeepSpeed ZeRO-3's mixed-precision all_gather
        # stays happy (a fp32-pinned frozen submodule trips
        # ``all_gather_into_tensor`` because its output buffer is bf16).
        self._m = vit
        self._variant = variant
        self._spec = VideoEncoderSpec(z_dim=int(embed_dim), spatial_compression=16, temporal_compression=2, causal_temporal=True, pixel_range=(-1.0, 1.0), is_reversible=False, dit_patch_size=(1, 2, 2))
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        # Post-norm: a plain LayerNorm at init (weight=1, bias=0) acts as
        # per-token standardization that pulls V-JEPA's O(30)-scale features
        # down to Wan-latent O(1). It is structurally trainable, but in the
        # current OpenWAM data path the entire encoder forward runs inside
        # ``BaseWAMArchitecture.preprocess`` / ``prepare_inputs``, both
        # decorated with ``@torch.no_grad`` (see ``openwam/model/base.py``
        # ~L765/L793). So even with ``requires_grad=True`` the LayerNorm
        # params receive no gradient and behave as a fixed standardizer
        # across training. Both the freeze yaml (which freezes
        # ``video_backbone._encoder`` wholesale) and the no_grad
        # decorators have to be lifted before this can adapt — out of
        # scope for the V-JEPA 2.1 integration PR. The module still lives
        # OUTSIDE ``self._m`` so a future PR can carve out a grad-enabled
        # path here without restructuring the ViT freeze granularity.
        self.feature_norm = nn.LayerNorm(int(embed_dim))

    @property
    def spec(self) -> VideoEncoderSpec:
        return self._spec

    @property
    def variant(self) -> str:
        return self._variant

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

    def batch_encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T_pixel, H, W) -> (B, embed_dim, T_lat, H/16, W/16).

        ``T_lat == 1`` when ``T_pixel == 1`` (TI2V ref-frame fast path);
        otherwise ``T_lat == 1 + (T_pixel - 1) // 2`` — frame 0 goes through
        the V-JEPA 2.1 image branch (tubelet=1) and frames 1.. through the
        tubelet=2 video branch, emulating the Wan VAE first-frame
        separability the host backbone assumes.
        """
        B, C, Tp, H, W = video.shape
        if C != 3:
            raise ValueError(f"V-JEPA 2.1 expects 3-channel input; got C={C}.")
        # Defensive: callers normally hand us host dtype already, but the
        # RoPE monkey-patch (installed in ``_load_vit``) only guarantees Q/K
        # match ``x.dtype`` at SDPA — so we still align the input to ViT
        # param dtype to keep the matmuls type-clean.
        m_dtype = next(self._m.parameters()).dtype
        if video.dtype != m_dtype:
            video = video.to(m_dtype)
        if Tp == 1:
            z = self._encode_image(video[:, :, 0])
        else:
            if (Tp - 1) % 2 != 0:
                raise ValueError(f"V-JEPA 2.1 causal emulation needs (T_pixel - 1) % 2 == 0, got T_pixel={Tp}.")
            z0 = self._encode_image(video[:, :, 0])
            zR = self._encode_video_tubelet(video[:, :, 1:])
            z = torch.cat([z0, zR], dim=2)
        z = self._apply_feature_norm(z)
        return z

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

    # Geometry constants the encoder's reshape paths and spec are hard-wired
    # against. The manifest can carry different ``patch`` / ``tubelet`` values
    # only if a future PR also generalizes the (h = H // 16) / (Tp // 2)
    # reshape and the (spatial=16, temporal=2) ``spec`` block. Today the
    # encoder is locked to ViT-g/16 tubelet=2 — manifests that disagree get
    # a fail-fast at load time instead of a confusing reshape error later.
    _REQUIRED_MANIFEST_PATCH = 16
    _REQUIRED_MANIFEST_TUBELET = 2

    @classmethod
    def from_pretrained(cls, model_path: str, **kw: Any) -> "VJEPA21VideoEncoder":
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
           :meth:`copy_deploy_artifacts` at checkpoint save time. The
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
        ``load_checkpoint`` populates ``_encoder._m.*`` from the saved
        safetensors immediately after this call returns.
        """
        manifest_dir = cls._resolve_manifest_dir(ckpt_dir, encoder_cfg)
        manifest = cls._read_and_validate_manifest(manifest_dir)
        vit_encoder = cls._prepare_vjepa_imports_and_patch()
        with torch.device(device):
            vit = cls._build_vit_from_manifest(vit_encoder, manifest)
        logger.info(
            "VJEPA21VideoEncoder.from_skeleton: %s instantiated from %s "
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
        fallback_manifest = (
            os.path.join(str(fallback_dir), "manifest.json") if fallback_dir else None
        )
        if fallback_manifest and os.path.isfile(fallback_manifest):
            return str(fallback_dir)

        raise FileNotFoundError(
            "VJEPA21VideoEncoder.from_skeleton: no readable manifest.json. "
            f"Tried ckpt_dir={ckpt_manifest!r} and "
            f"encoder.model_path={fallback_manifest!r}. The checkpoint was "
            "saved before the self-containment patch and the fallback is "
            "unreachable or has no manifest. Either re-save with the current "
            "code, or hand-copy manifest.json into the checkpoint dir "
            "(see issue #86)."
        )

    def copy_deploy_artifacts(self, output_dir: str, cfg: Any) -> None:
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

        IO failure handling matters because :meth:`copy_deploy_artifacts`
        runs inside the trainer's checkpoint save flow — losing the
        safetensors save over a manifest-copy ``PermissionError`` /
        ``ENOSPC`` / disappearing-mount ``OSError`` would be a strict
        regression vs. the pre-self-containment behavior.
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
            # varies across call sites, and the contract above forbids
            # raising. Anything that prevents us from reading model_path
            # collapses to "skip the copy, deploy falls back".
            pass

        if not model_path:
            logger.warning(
                "VJEPA21VideoEncoder.copy_deploy_artifacts: cannot resolve "
                "model.video_backbone.encoder.model_path from cfg; skipping "
                "manifest copy. Deploy will fall back to encoder.model_path."
            )
            return
        src = os.path.join(str(model_path), "manifest.json")
        dst = os.path.join(output_dir, "manifest.json")
        if not os.path.isfile(src):
            logger.warning(
                "VJEPA21VideoEncoder.copy_deploy_artifacts: manifest.json "
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
                "VJEPA21VideoEncoder.copy_deploy_artifacts: copying %s -> %s "
                "failed (%s); skipping. Deploy will fall back to encoder.model_path.",
                src,
                dst,
                e,
            )
            return
        logger.info(
            "VJEPA21VideoEncoder.copy_deploy_artifacts: copied %s -> %s",
            src,
            dst,
        )

    @classmethod
    def _read_and_validate_manifest(cls, model_path: str) -> dict:
        manifest_path = os.path.join(model_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"VJEPA21 encoder requires manifest.json in {model_path}.")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        patch = int(manifest["patch"])
        tubelet = int(manifest["tubelet"])
        if patch != cls._REQUIRED_MANIFEST_PATCH or tubelet != cls._REQUIRED_MANIFEST_TUBELET:
            raise ValueError(
                f"VJEPA21 manifest patch/tubelet must be "
                f"({cls._REQUIRED_MANIFEST_PATCH}, {cls._REQUIRED_MANIFEST_TUBELET}); "
                f"got ({patch}, {tubelet}). The encoder's spec (spatial_compression=16, "
                f"temporal_compression=2) and reshape logic (H//16, W//16, Tp//2) are "
                f"hard-wired against these values. Use a different manifest or extend "
                "the encoder to honor the manifest geometry."
            )
        cls._check_arch_use_rope_consistency(manifest)
        return manifest

    @staticmethod
    def _prepare_vjepa_imports_and_patch():
        """Bootstrap ``third_party/vjepa2`` import path + install the RoPE
        dtype monkey-patch. Idempotent. Returns the imported
        ``vision_transformer`` module.

        Avoids ``torch.hub.load(...)``: upstream ``VJEPA_BASE_URL`` currently
        points to a localhost test endpoint and is not pullable. The
        ``facebookresearch/vjepa2`` repo is vendored as a git submodule under
        ``third_party/vjepa2`` (branch ``vjepa2_1``) and exposes its model
        code as ``app.vjepa_2_1.*`` — the repo root itself is the package.

        ``tests/conftest.py`` already inserts ``third_party/vjepa2`` into
        ``sys.path``; for non-pytest entry points (``scripts/train.py`` /
        REPL / deploy) we bootstrap the same path lazily on first call so
        the encoder works without forcing every launcher to know about the
        layout. No-op if the submodule isn't checked out — the import below
        then raises with a clear ``ModuleNotFoundError`` telling the user
        to run ``git submodule update --init third_party/vjepa2``.

        The RoPE dtype monkey-patch root-cause-fixes a V-JEPA / SDPA
        dtype mismatch under mixed-precision: upstream
        ``rotate_queries_or_keys`` builds its sin/cos table from a fp32
        mask (``1.0 * frame_ids``) and einsums it against an fp32
        ``omega``, so the rotated Q/K leave the function in fp32 even
        when ``x`` is bf16. The host backbone keeps V in bf16, and
        PyTorch SDPA refuses ``query.dtype != value.dtype``. The patch
        casts the output back to ``x.dtype`` on exit — covers all six
        call sites in ``AttentionRoPE.forward`` (qd/kd, qh/kh, qw/kw)
        without editing the vendored submodule. Idempotent via the
        ``_openwam_dtype_safe`` sentinel so repeated calls (training
        reload, deploy skeleton + later weight load, EMA replicas) do
        not re-wrap.
        """
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        vjepa2_root = repo_root / "third_party" / "vjepa2"
        if vjepa2_root.is_dir() and str(vjepa2_root) not in sys.path:
            sys.path.insert(0, str(vjepa2_root))

        from app.vjepa_2_1.models import vision_transformer as vit_encoder
        from app.vjepa_2_1.models.utils import modules as vjepa_modules

        if not getattr(vjepa_modules.rotate_queries_or_keys, "_openwam_dtype_safe", False):
            _orig_rotate = vjepa_modules.rotate_queries_or_keys

            def _safe_rotate(x, pos, n_registers, has_cls_first):
                out = _orig_rotate(x, pos, n_registers=n_registers, has_cls_first=has_cls_first)
                return out.to(x.dtype)

            _safe_rotate._openwam_dtype_safe = True
            vjepa_modules.rotate_queries_or_keys = _safe_rotate

        return vit_encoder

    @staticmethod
    def _check_arch_use_rope_consistency(manifest: dict) -> None:
        """Manifest-internal contradiction check, isolated from vjepa2 imports.

        Runs without touching ``third_party/vjepa2`` so the error stays
        correct in CI/dev environments where the submodule isn't
        initialized. Called by ``_read_and_validate_manifest`` (the
        ``from_pretrained`` / ``from_skeleton`` path) and by the
        ``_load_vit`` back-compat shim (PR #83 V9/V10 regression
        tests), so all paths get the same fail-fast.
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

        Upstream wrappers ending in ``_rope`` (e.g.
        ``vit_giant_xformers_rope``) hardcode ``use_rope=True`` in their
        ``VisionTransformer(...)`` call and forward ``**kwargs`` to the
        same constructor — passing ``use_rope`` again from here raises
        ``TypeError: got multiple values for keyword argument 'use_rope'``.
        For non-``_rope`` arches the wrapper does not set it, so we
        forward the manifest value; we default to ``True`` (opt-out)
        because every V-JEPA 2.1 manifest we ship uses RoPE —
        ``VisionTransformer``'s own ``use_rope=False`` default is the
        wrong choice for this encoder.
        """
        arch_name = manifest["arch_name"]  # e.g. "vit_giant_xformers"
        manifest_use_rope = manifest.get("use_rope", True)
        vit_kwargs: dict[str, Any] = dict(
            patch_size=manifest["patch"],
            img_size=(manifest["img_size"], manifest["img_size"]),
            num_frames=manifest["training_num_frames"],
            tubelet_size=manifest["tubelet"],
            use_sdpa=True,
            img_temporal_dim_size=manifest.get("img_temporal_dim_size", 1),
            interpolate_rope=manifest.get("interpolate_rope", True),
        )
        if not arch_name.endswith("_rope"):
            vit_kwargs["use_rope"] = manifest_use_rope
        return vit_encoder.__dict__[arch_name](**vit_kwargs)

    @classmethod
    def _load_vit(cls, model_path: str, manifest: dict) -> nn.Module:
        """Back-compat shim — chains the new helpers so PR #83 V9/V10
        regression tests (which call ``_load_vit`` directly) keep working.

        ``_check_arch_use_rope_consistency`` runs BEFORE
        ``_prepare_vjepa_imports_and_patch`` so the manifest-internal
        ValueError stays correct in environments where the
        ``third_party/vjepa2`` submodule isn't initialized — matches the
        PR #83 review invariant.
        """
        cls._check_arch_use_rope_consistency(manifest)
        vit_encoder = cls._prepare_vjepa_imports_and_patch()
        vit = cls._build_vit_from_manifest(vit_encoder, manifest)
        cls._load_vit_weights(vit, model_path, manifest)
        return vit

    @staticmethod
    def _load_vit_weights(vit: nn.Module, model_path: str, manifest: dict) -> None:
        """Populate a constructed ViT with pretrained weights from disk."""
        ckpt = torch.load(
            os.path.join(model_path, manifest["checkpoint_file"]),
            map_location="cpu",
        )
        state_dict = ckpt[manifest.get("checkpoint_key", "target_encoder")]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        # ``strict=False`` is intentional but narrow: the checkpoint ships a
        # learned ``pos_embed`` for the absolute-pos-embedding variants, and
        # we always load the RoPE variants whose forward does not consume it
        # (and so the buffer/parameter does not exist on the constructed
        # ``vit`` either). Anything else missing or unexpected is a
        # manifest / checkpoint mismatch that would silently leave the frozen
        # ViT partially randomly initialized — fail fast instead. The
        # tolerated unexpected set is exactly ``{"pos_embed"}``; missing keys
        # must always be empty.
        load_result = vit.load_state_dict(state_dict, strict=False)
        unexpected = set(load_result.unexpected_keys) - {"pos_embed"}
        if unexpected or load_result.missing_keys:
            raise RuntimeError(
                "VJEPA21 checkpoint load left the ViT inconsistent with the "
                "constructed module. This usually means the manifest "
                "``arch_name`` does not match the checkpoint, or the "
                "``checkpoint_key`` extracts the wrong sub-dict. Details: "
                f"missing_keys={sorted(load_result.missing_keys)[:8]} "
                f"unexpected_keys={sorted(unexpected)[:8]}."
            )

    # Intentionally NOT overriding build_dit_input_proj / build_dit_output_proj:
    # the PR #60 defaults at dit_patch_size=(1,2,2) produce Conv3d((1,2,2),(1,2,2))
    # and Linear(dit_dim, z_dim * 4) — exactly what we need.


__all__ = ["VJEPA21VideoEncoder"]
