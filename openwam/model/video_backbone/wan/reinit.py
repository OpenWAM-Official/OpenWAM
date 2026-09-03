"""DiT re-initialization / external-encoder adaptation.

Free functions operating on a backbone's ``dit`` / ``dit2`` — lifted out of
``wan_backbone.py`` so the backbone class carries only its ABC
implementation. ``reinit_dit_from_scratch`` (training ``from_scratch``) and
``adapt_dit_to_external_encoder`` (deploy reshape) are the public entry points;
``base.py`` calls them.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _probe_dit_stats(dit) -> dict:
    """Snapshot representative tensors for before/after verification, covering
    both reinit paths: ``q.weight`` (stdlib reset) and ``blocks[0].modulation``
    / ``head.modulation`` (hand-reset nn.Parameters).
    """
    return {
        "q.weight_mean": float(dit.blocks[0].self_attn.q.weight.float().mean().item()),
        "q.weight_std": float(dit.blocks[0].self_attn.q.weight.float().std().item()),
        "blocks[0].modulation_mean": float(dit.blocks[0].modulation.float().mean().item()),
        "blocks[0].modulation_std": float(dit.blocks[0].modulation.float().std().item()),
        "head.modulation_mean": float(dit.head.modulation.float().mean().item()),
        "head.modulation_std": float(dit.head.modulation.float().std().item()),
    }


def adapt_dit_to_external_encoder(
    backbone,
    external_encoder,
    dit_patch_size: Optional[Tuple[int, int, int]],
) -> None:
    """Rebuild ``backbone.dit`` ``patch_embedding`` / ``head.head`` / ``patch_size``
    / ``in_dim`` to match an external encoder's latent shape (``wan22_vae`` is a
    no-op shape-wise; non-VAE encoders adapt the first conv / final Linear).

    ``patch_size`` must be synced because ``unpatchify`` rearranges by it; a
    different ``dit_patch_size`` would otherwise shape-mismatch on first forward.
    Called from reinit (training) and deploy ``_init_video_backbone``.

    ``dit_patch_size`` MUST come from the backbone (single source of truth);
    reading ``external_encoder.properties`` directly here bypasses the abstraction.
    """
    dits = [m for m in (getattr(backbone, "dit", None), getattr(backbone, "dit2", None)) if m is not None]
    if not dits:
        logger.warning("adapt_dit_to_external_encoder: backbone has no dit/dit2")
        return
    if dit_patch_size is None:
        raise ValueError(
            "adapt_dit_to_external_encoder: dit_patch_size is required. "
            "Source it from the backbone (e.g. "
            "self.video_backbone.dit_patch_size) rather than reading "
            "external_encoder.properties.dit_patch_size directly."
        )
    ps = tuple(dit_patch_size)
    for dit in dits:
        dit.patch_embedding = external_encoder.build_dit_input_proj(dit.dim)
        dit.patch_size = ps
        head_mod = getattr(dit, "head", None)
        # MotWanModel-style DiTs may lack a head; guard the assignment.
        if head_mod is not None and hasattr(head_mod, "head"):
            head_mod.head = external_encoder.build_dit_output_proj(dit.dim)
            head_mod.patch_size = ps
        dit.in_dim = external_encoder.properties.z_dim


def reinit_dit_from_scratch(
    backbone,
    *,
    external_encoder=None,
    dit_patch_size: Optional[Tuple[int, int, int]] = None,
    verbose: bool = True,
) -> None:
    """Re-initialize all learnable params in ``backbone.dit`` (and ``dit2``) with
    PyTorch standard init; VAE / text_encoder / image_encoder / vace untouched.
    Used by the ``from_scratch`` switch to ablate pretrained-vs-scratch DiT.

    Two steps: (1) ``reset_parameters()`` for stdlib layers; (2) hand-reset the
    directly-mounted ``nn.Parameter`` that ``modules()`` does NOT yield (~30
    ``DiTBlock.modulation`` + ~180 ``RMSNorm.weight`` per 30-layer DiT) — without
    this they would silently retain pretrained values. Buffers (``freqs`` cache)
    are deterministic and left alone.

    ``external_encoder``: when provided, rebuild patch_embedding / head.head via
    its hooks BEFORE the reset loop (no-op shape-wise for ``wan22_vae``); ``None``
    keeps the "reset weights only, don't touch shapes" behavior. ``verbose``
    prints a rank-0 BEFORE/AFTER summary via ``print`` (independent of logging).
    """
    import os

    import torch.nn as nn

    from openwam.model.video_backbone.wan.models.dit import MLP, DiTBlock, Head, RMSNorm

    stdlib_resettable = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.LayerNorm)

    dits = [m for m in (getattr(backbone, "dit", None), getattr(backbone, "dit2", None)) if m is not None]
    if not dits:
        logger.warning("reinit_dit_from_scratch: backbone has no dit/dit2 to re-init")
        return

    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    # Rebuild patch_embedding + head.head BEFORE reset_parameters (no-op
    # shape-wise for wan22_vae). The subsequent reset re-inits them again —
    # harmless duplicate random init in the same distribution.
    if external_encoder is not None:
        adapt_dit_to_external_encoder(backbone, external_encoder, dit_patch_size)

    before_stats = [] if (verbose and is_main) else None
    after_stats = [] if (verbose and is_main) else None

    for root in dits:
        if before_stats is not None:
            before_stats.append(_probe_dit_stats(root))
        for sub in root.modules():
            if isinstance(sub, stdlib_resettable):
                sub.reset_parameters()
        with torch.no_grad():
            for sub in root.modules():
                if isinstance(sub, RMSNorm):
                    sub.weight.fill_(1.0)
                elif isinstance(sub, DiTBlock):
                    dim = sub.modulation.shape[-1]
                    sub.modulation.normal_(mean=0.0, std=dim**-0.5)
                elif isinstance(sub, Head):
                    dim = sub.modulation.shape[-1]
                    sub.modulation.normal_(mean=0.0, std=dim**-0.5)
                elif isinstance(sub, MLP) and getattr(sub, "has_pos_emb", False):
                    sub.emb_pos.zero_()
        if after_stats is not None:
            after_stats.append(_probe_dit_stats(root))

    logger.info(
        "reinit_dit_from_scratch: re-initialized %d DiT module(s); VAE/T5 untouched",
        len(dits),
    )

    if verbose and is_main:
        # print(flush=True) so the trace surfaces under non-INFO logging.
        bar = "=" * 78
        print(bar, flush=True)
        print(
            "[reinit_dit_from_scratch] video DiT weights re-initialized from scratch. "
            f"VAE / text_encoder kept pretrained. ({len(dits)} DiT module(s), rank=0 summary)",
            flush=True,
        )
        for i, (before, after, root) in enumerate(zip(before_stats, after_stats, dits)):
            label = "dit" if i == 0 else f"dit{i + 1}"
            expected_mod_std = root.dim**-0.5
            print(
                f"  [{label}] q.weight                 BEFORE mean={before['q.weight_mean']:+.4e} std={before['q.weight_std']:.4e}  "
                f"-> AFTER mean={after['q.weight_mean']:+.4e} std={after['q.weight_std']:.4e}",
                flush=True,
            )
            print(
                f"  [{label}] blocks[0].modulation     BEFORE mean={before['blocks[0].modulation_mean']:+.4e} std={before['blocks[0].modulation_std']:.4e}  "
                f"-> AFTER mean={after['blocks[0].modulation_mean']:+.4e} std={after['blocks[0].modulation_std']:.4e}  "
                f"(expected std≈{expected_mod_std:.4e})",
                flush=True,
            )
            print(
                f"  [{label}] head.modulation          BEFORE mean={before['head.modulation_mean']:+.4e} std={before['head.modulation_std']:.4e}  "
                f"-> AFTER mean={after['head.modulation_mean']:+.4e} std={after['head.modulation_std']:.4e}  "
                f"(expected std≈{expected_mod_std:.4e})",
                flush=True,
            )
        print(
            "  Reproducibility: with the same cfg.project.seed, these AFTER numbers "
            "are bit-exact across runs (rank-0 broadcast covers other ranks).",
            flush=True,
        )
        print(bar, flush=True)
