"""DINOv3 video feature encoder.

DINOv3 ViT (sizes / patch read from HF config) applied frame-by-frame,
followed by a parameter-free causal mean pool along the temporal axis to
mirror Wan VAE's ``T_lat = 1 + (T - 1) / 4`` semantics, then a
parameter-free per-token LayerNorm so the diffusion target lives in a
roughly N(0, I) space (the Wan flow-matching schedule's design assumption).

Shape contract (Robotwin default, ViT-B/16):

    pixel  (B, 3, T, H, W),   T ≡ 1 (mod 4)
      ↓ ImageNet normalize, per-frame ViT
    grid   (B, D, T, H/P, W/P)
      ↓ causal mean pool over T
    pooled (B, D, 1 + (T-1)/4, H/P, W/P)
      ↓ non-affine LayerNorm over the D axis (per token)
    latent (B, D, 1 + (T-1)/4, H/P, W/P)

where ``D = config.hidden_size`` and ``P = config.patch_size``.

The trailing LayerNorm uses ``elementwise_affine=False`` (zero learnable
parameters) and is applied per spatial-temporal token across the channel
axis. This keeps the diffusion target's per-token mean ≈ 0 and
std ≈ 1 without needing a one-shot dataset-level statistics calibration
or extra files: same input frames produce the same latent regardless of
batch size, with no batch-dependent drift.

The DiT's ``patch_embedding`` / ``head.head`` are rebuilt from the default
hooks on :class:`VideoEncoder` (``Conv3d(D, dit_dim, (1,2,2), (1,2,2))``
and ``Linear(dit_dim, D * 4)``), giving a token count identical to the
Wan VAE path on the same input frames.

The framework contract is defined by :class:`VideoEncoder`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.svae import reducer

# Checkpoint-local namespace for the DINOv3 HF config, so a self-contained deploy
# reads ``<ckpt>/dinov3/config.json`` instead of needing the original
# ``encoder.model_path`` reachable (mirrors flux2_vae / V-JEPA's sidecar).
_DINOV3_CKPT_SUBDIR = "dinov3"

logger = logging.getLogger(__name__)


# ImageNet statistics — the standard preprocessing for DINOv3 / DINOv2 / ViT
# checkpoints. Kept as plain tuples; ``_preprocess_image`` builds matching
# ``mean`` / ``std`` tensors on each call (the path is only hit once per
# clip during dataloading, so the per-call allocation is negligible).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_image(image: Image.Image, *, dtype, device) -> Tensor:
    """PIL image -> ``(1, 3, H, W)`` tensor in ImageNet-normalized space.

    Mirrors :func:`openwam.model.video_backbone.encoder.wan22_vae._preprocess_image`
    in structure (``np.array`` + ``repeat`` to broadcast a leading batch axis)
    but swaps the ``[-1, 1]`` linear rescale for ImageNet mean/std normalize.
    """
    arr = torch.tensor(np.array(image, dtype=np.float32), dtype=dtype, device=device)
    arr = arr / 255.0
    mean = torch.tensor(_IMAGENET_MEAN, dtype=dtype, device=device)
    std = torch.tensor(_IMAGENET_STD, dtype=dtype, device=device)
    arr = (arr - mean) / std
    return repeat(arr, "H W C -> B C H W", B=1)


def _causal_temporal_pool(x: Tensor) -> Tensor:
    """Wan-style causal time pool: keep frame 0 as-is, mean-pool the rest in 4s.

    Args:
        x: ``(B, D, T, H, W)`` with ``T ≡ 1 (mod 4)``.

    Returns:
        ``(B, D, 1 + (T - 1) // 4, H, W)``.
    """
    if (x.shape[2] - 1) % 4 != 0:
        raise ValueError(
            f"DinoV3VideoEncoder expects T ≡ 1 (mod 4); got T={x.shape[2]}. "
            "This matches Wan VAE's causal first-frame + 4x temporal grouping."
        )
    first = x[:, :, :1]
    rest = x[:, :, 1:]
    rest = rearrange(rest, "B D (k g) H W -> B D k g H W", g=4)
    rest = rest.mean(dim=3)
    return torch.cat([first, rest], dim=2)


@register_video_encoder("dinov3")
class DinoV3VideoEncoder(VideoEncoder):
    """:class:`VideoEncoder` wrapping a DINOv3 ViT backbone.

    Sizes (``embed_dim`` / ``patch_size`` / register-token count) are read
    from the on-disk HF config in :meth:`from_pretrained` /
    :meth:`from_skeleton`, so this class is not pinned to a specific ViT
    width or patch — any DINOv3 release that exposes ``hidden_size``,
    ``patch_size`` and ``num_register_tokens`` on its config works.

    Per-frame ViT encoding + parameter-free causal mean pool along T to align
    with Wan VAE's temporal semantics. The DiT-side ``patch_embedding`` and
    ``head.head`` are rebuilt from :class:`VideoEncoder`'s default hooks so
    dinov3 reuses the exact same projection structure as the wan22_vae path.

    No ``decode`` / ``to_frames`` (``properties.pixel_decode=False``) — the ABC
    defaults raise ``NotImplementedError`` with a contract-aware message.
    """

    def __init__(
        self,
        vit: nn.Module,
        *,
        embed_dim: int,
        patch_size: int,
        num_register_tokens: int,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
        svae_config: dict | None = None,
    ):
        super().__init__()
        self._m = vit
        self._num_register_tokens = int(num_register_tokens)
        # Raw post-pool channel dim before any S-VAE reduction; kept separate
        # from the effective z_dim so offline S-VAE collection
        # (``batch_encode_pooled_for_svae_training``) reports the un-reduced dim.
        self._raw_embed_dim = int(embed_dim)
        # Optional frozen S-VAE reducer (mirrors VJEPA21VideoEncoder). When
        # attached, ``z_dim`` / ``_out_norm`` rebuild against its ``latent_dim``.
        self._svae = reducer.build(svae_path, svae_target_dim, svae_config)
        effective_z_dim = reducer.effective_z_dim(self._svae, self._raw_embed_dim)
        # Per-token output normalization. ``elementwise_affine=False`` means no
        # learnable γ/β — this is a pure geometric rescale, registered as a
        # ``nn.Module`` only so ``set_dtype_device`` / ``state_dict`` treat it
        # uniformly with the rest of the encoder. ViT's ``last_hidden_state``
        # already comes from a final LayerNorm in most HF implementations, so
        # this layer is near-identity in the common case but still guarantees
        # the target distribution invariant for downstream flow-matching. With
        # the S-VAE on it operates over the already-whitened ``latent_dim``.
        self._out_norm = nn.LayerNorm(int(effective_z_dim), elementwise_affine=False, eps=1e-6)
        # Spec mirrors Wan VAE's geometry on Wan2.2-TI2V-5B (spatial=16,
        # temporal=4 causal, dit_patch=(1,2,2)) so token counts match while
        # ``z_dim`` differs (768 vs 48). pixel_decode=False causes the
        # backbone to skip the strict spec equality check.
        self._spec = VideoEncoderProperties(
            z_dim=int(effective_z_dim),
            spatial_compression=int(patch_size),
            temporal_compression=4,
            causal_temporal=True,
            pixel_decode=False,
            dit_patch_size=(1, 2, 2),
        )

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    def preprocess_video(self, frames) -> Tensor:
        dtype = next(self._m.parameters()).dtype
        device = next(self._m.parameters()).device
        images = [_preprocess_image(img, dtype=dtype, device=device) for img in frames]
        return torch.stack(images, dim=2)

    def batch_encode(self, video: Tensor) -> Tensor:
        """``(B, 3, T, H, W) → (B, z_dim, T_lat, H/16, W/16)``.

        Per-frame ViT forward (folded into the batch axis), drop CLS +
        register tokens, reshape to a 5D grid, causal-mean-pool along T,
        optionally reduce via the frozen S-VAE (base hook), then per-token
        (channel-axis) LayerNorm.
        """
        z = self._batch_encode_pooled_raw(video)
        z = self._apply_svae(z)
        return self._apply_feature_norm(z)

    def _batch_encode_pooled_raw(self, video: Tensor) -> Tensor:
        """``(B, 3, T, H, W) → (B, embed_dim, T_lat, H/16, W/16)`` BEFORE the
        S-VAE reduce / trailing LayerNorm.

        Split from :meth:`batch_encode` so offline S-VAE training
        (:meth:`batch_encode_pooled_for_svae_training`) sees byte-for-byte the
        same post-pool, pre-LayerNorm distribution the main path produces.
        """
        if video.dim() != 5 or video.shape[1] != 3:
            raise ValueError(f"batch_encode expects (B, 3, T, H, W); got {tuple(video.shape)}")
        b, _, t, h, w = video.shape
        ps = self._spec.spatial_compression
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"DinoV3VideoEncoder requires H,W divisible by patch_size={ps}; got H={h}, W={w}.")

        flat = rearrange(video, "B C T H W -> (B T) C H W")
        # Gradient enablement is the caller's responsibility — the host
        # backbone's ``preprocess`` / ``prepare_inputs`` already wrap the
        # encode path in ``@torch.no_grad`` for the frozen-feature use case
        # (matches ``wan22_vae`` / ``vjepa21``). Not pinning ``no_grad`` here
        # leaves room for adapter / LoRA / partial-unfreeze experiments.
        outputs = self._m(flat)
        # ``trust_remote_code=True`` lets each DINOv3 snapshot ship its own
        # modeling code; most return ``BaseModelOutput`` (or a subclass) but
        # some custom forks return a bare Tensor or a different dataclass.
        # Probe defensively so we fail with a clear message instead of an
        # opaque ``AttributeError`` on a mismatched HF mirror.
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None:
            if torch.is_tensor(outputs):
                tokens = outputs
            else:
                raise TypeError(
                    "DINOv3 forward returned an object without "
                    "``last_hidden_state`` and it is not a Tensor either "
                    f"(got {type(outputs).__name__}). The bundled "
                    "modeling code at encoder.model_path is likely "
                    "non-standard; expose ``last_hidden_state`` or return "
                    "the patch-token tensor directly."
                )
        # tokens: (B*T, 1 + R + N, D)

        # DINOv3 layout: [CLS, register_0..R-1, patch_0..N-1].
        n_drop = 1 + self._num_register_tokens
        patch_tokens = tokens[:, n_drop:, :]
        expected = (h // ps) * (w // ps)
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"DINOv3 returned {patch_tokens.shape[1]} patch tokens but expected "
                f"(H/{ps}) * (W/{ps}) = {expected}. Check input resolution and "
                f"register-token count ({self._num_register_tokens})."
            )
        grid = rearrange(
            patch_tokens,
            "(B T) (Hl Wl) D -> B D T Hl Wl",
            B=b,
            T=t,
            Hl=h // ps,
            Wl=w // ps,
        )
        return _causal_temporal_pool(grid)

    def _apply_feature_norm(self, pooled: Tensor) -> Tensor:
        """Per-token LayerNorm along the current channel dim: move D last, apply
        the non-affine ``_out_norm``, restore ``(B, D, T_lat, H_lat, W_lat)``."""
        pooled = rearrange(pooled, "B D T H W -> B T H W D")
        pooled = self._out_norm(pooled)
        return rearrange(pooled, "B T H W D -> B D T H W")

    def batch_encode_pooled_for_svae_training(self, video: Tensor) -> Tensor:
        """Raw post-pool features for offline S-VAE training / stats collection.

        Returns the ``(B, embed_dim, T_lat, H/16, W/16)`` tensor the S-VAE
        consumes — BEFORE the trailing LayerNorm — so the reducer trains on the
        un-whitened distribution (matches the V-JEPA 2.1 contract used by
        :mod:`scripts.svae_train.collect_svae_features`). Fails fast if an S-VAE is already
        attached: statistics must be collected on a raw encoder.
        """
        if self._svae is not None:
            raise RuntimeError(
                "batch_encode_pooled_for_svae_training requires a raw encoder; "
                "svae_path / svae_config / svae_target_dim must be unset."
            )
        return self._batch_encode_pooled_raw(video)

    # decode / to_frames intentionally omitted — defaults from VideoEncoder
    # raise NotImplementedError because properties.pixel_decode=False.

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
        **kw: Any,
    ) -> "DinoV3VideoEncoder":
        from transformers import AutoConfig, AutoModel

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"encoder model_path is not a directory: {model_path}")

        # AutoConfig is loaded explicitly so the structural fields (embed_dim,
        # patch_size, register tokens) are read from the on-disk config, never
        # from yaml. ``trust_remote_code=True`` is a no-op for native-format
        # DINOv3 (``model_type: dinov3_vit``, no ``auto_map`` → built-in
        # ``DINOv3ViTModel`` from transformers); it still supports older
        # snapshots that ship custom modeling code via ``auto_map``.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        vit = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        vit = vit.eval()

        embed_dim, patch_size, num_register_tokens = cls._extract_structural_fields(config, model_path)

        logger.info(
            "DinoV3VideoEncoder loaded %s (embed_dim=%d, patch_size=%d, register_tokens=%d, svae=%s)",
            model_path,
            embed_dim,
            patch_size,
            num_register_tokens,
            "on" if svae_path is not None else "off",
        )
        return cls(
            vit,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
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
    ) -> "DinoV3VideoEncoder":
        """Deploy-time zero-weight ViT shell, sized by the DINOv3 HF ``config.json``.

        No weights are loaded here (``AutoModel.from_config``) — the
        architecture's strict ``load_checkpoint`` fills them in. ``components_entry``
        is ignored (its ``vae`` entry is the Wan VAE placeholder). The config is
        read strictly from ``<ckpt_dir>/dinov3/config.json`` (written by
        :meth:`save_deploy_assets`); deploy is self-contained, with no
        ``encoder.model_path`` fallback — a checkpoint saved without its config
        sidecar fails loudly here.

        For native-format DINOv3 (``model_type: dinov3_vit``, no ``auto_map``,
        ``DINOv3ViTModel`` built into transformers) the config alone is enough and
        ``trust_remote_code=True`` is a no-op. A snapshot whose config carries an
        ``auto_map`` pointing at bundled ``modeling_*.py`` would additionally need
        those files reachable — config-only self-containment does not cover that
        rarer case. Structural fields (``hidden_size`` / ``patch_size`` /
        ``num_register_tokens``) come from the HF config.
        """
        from transformers import AutoConfig, AutoModel

        config_dir = cls._resolve_config_dir(ckpt_dir)
        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
        with torch.device(device):
            vit = AutoModel.from_config(config, trust_remote_code=True)
        vit = vit.to(dtype=torch.bfloat16).eval()

        embed_dim, patch_size, num_register_tokens = cls._extract_structural_fields(config, config_dir)

        # Reducer rebuild: the sidecar (if the training ckpt carried an S-VAE)
        # sizes a zero-weight shell here; strict ``load_checkpoint`` fills
        # ``_svae.*`` right after. ``svae_target_dim`` from the saved yaml is an
        # optional cross-check against the sidecar's ``latent_dim``.
        svae_config = reducer.read_sidecar(ckpt_dir)
        svae_target_dim = reducer.read_target_dim_from_cfg(encoder_cfg)

        logger.info(
            "DinoV3VideoEncoder.from_skeleton: instantiated from %s "
            "(embed_dim=%d, patch_size=%d, register_tokens=%d, svae=%s) — weights pending checkpoint load",
            config_dir,
            embed_dim,
            patch_size,
            num_register_tokens,
            "on" if svae_config is not None else "off",
        )
        return cls(
            vit,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            svae_config=svae_config,
            svae_target_dim=svae_target_dim,
        )

    def save_deploy_assets(self, output_dir: str, cfg: Any) -> None:
        """Copy the DINOv3 HF ``config.json`` into ``<output_dir>/dinov3/config.json``
        so deploy is self-contained.

        Strict self-contained: an unresolvable cfg / missing source / copy IO
        error all raise, because :meth:`from_skeleton` reads the config only
        from ``ckpt_dir`` — a checkpoint saved without its config sidecar
        cannot be deployed. Runs once at rank-0 start-up before any weights are
        saved, so a raise fails the run fast.
        """
        import shutil

        # The S-VAE sidecar is strict (``reducer.write_sidecar`` raises): without
        # it ``from_skeleton`` cannot size the reducer.
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
                "DinoV3VideoEncoder.save_deploy_assets: cannot resolve "
                "model.video_backbone.encoder.model_path from cfg; cannot copy "
                "config.json (deploy reads it only from ckpt_dir)."
            )
        src = os.path.join(str(model_path), "config.json")
        dst = os.path.join(output_dir, _DINOV3_CKPT_SUBDIR, "config.json")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"DinoV3VideoEncoder.save_deploy_assets: config.json not found at {src}.")
        if os.path.abspath(src) == os.path.abspath(dst):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        logger.info("DinoV3VideoEncoder.save_deploy_assets: copied %s -> %s", src, dst)

    # ------------------------------------------------------------------
    # Private load / deploy helpers (the VideoEncoder contract is above)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_structural_fields(config: Any, model_path: str) -> tuple[int, int, int]:
        """Read ``(embed_dim, patch_size, num_register_tokens)`` from an HF config.

        Centralised so :meth:`from_pretrained` and :meth:`from_skeleton`
        agree on which attributes to probe and how to fail. ``hidden_size``
        is the canonical DINOv3 / HF ViT field; ``embed_dim`` is kept as a
        fallback for forks that rename it. Both are checked with explicit
        ``None`` discrimination (rather than truthy fallback) so a
        misconfigured ``hidden_size=0`` would surface here instead of being
        silently overwritten by ``embed_dim``.
        """
        hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(config, "embed_dim", None)
        if hidden is None:
            raise ValueError(
                f"Could not infer embed_dim from DINOv3 config at {model_path} (checked hidden_size, embed_dim)."
            )
        embed_dim = int(hidden)
        if embed_dim <= 0:
            raise ValueError(f"DINOv3 config at {model_path} reports non-positive hidden_size/embed_dim={embed_dim!r}.")
        patch_size = int(getattr(config, "patch_size", 16))
        num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        return embed_dim, patch_size, num_register_tokens

    @staticmethod
    def _resolve_config_dir(ckpt_dir: str | None) -> str:
        """Return the dir holding a readable DINOv3 ``config.json`` at deploy time.

        Strictly self-contained: only ``<ckpt_dir>/dinov3/config.json`` (written
        by :meth:`save_deploy_assets`) is consulted; there is no
        ``encoder.model_path`` fallback. A missing config is a hard error.
        """
        ckpt_cfg = os.path.join(ckpt_dir, _DINOV3_CKPT_SUBDIR, "config.json") if ckpt_dir else None
        if ckpt_cfg and os.path.isfile(ckpt_cfg):
            return os.path.join(str(ckpt_dir), _DINOV3_CKPT_SUBDIR)
        raise FileNotFoundError(
            "DinoV3VideoEncoder.from_skeleton: no readable config.json at "
            f"ckpt_dir={ckpt_cfg!r}. Re-save the checkpoint with the current "
            "code, which writes dinov3/config.json into ckpt_dir."
        )


__all__ = ["DinoV3VideoEncoder"]
