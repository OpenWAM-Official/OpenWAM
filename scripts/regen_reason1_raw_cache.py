"""Regenerate Reason1 text embeddings as RAW last-layer hidden (L=512, 3584).

The mainline ``openwam.dataloader.reason1_embedding_computation`` caches
``(L=512, 1024) bf16`` post-projection embeddings, applying the pretrained
``net.crossattn_proj: Linear(100352, 1024) + GELU`` to compress the
full 28-layer concat down to 1024.

This script bypasses that projection and stores ONLY the LAST layer's
mean-normalized hidden state — i.e., the raw 3584-d Reason1 output. The
intended use is the "A.1" cross-attention widening: widen Cosmos25 DiT
``crossattn_emb_channels`` to 3584 so that the model consumes the full
per-token Reason1 bandwidth instead of the bottlenecked 1024.

Output cache layout matches the mainline tool (sha256-keyed, bucketed by
2-char prefix), so the existing dataloader cache lookup works unchanged
— only the tensor shape and ``cache_dir`` change.

Usage:
    .venv/bin/python scripts/regen_reason1_raw_cache.py \\
        --reason1-ckpt /path/to/assets/Cosmos-Reason1-7B \\
        --dataset-config configs/dataloader/robotwin.yaml \\
        --output-dir /path/to/cache/reason1_robotwin_raw_3584
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# Re-use the well-tested heavy helpers without touching them.
from openwam.dataloader.reason1_embedding_computation import (
    _NUM_EMBEDDING_PADDING_TOKENS,
    _REASON1_HIDDEN_SIZE,
    _build_reason1,
    _cache_path_for_sha,
    _encode_reason1,
    _enumerate_prompts,
    _find_unmanaged_flat_cache_files,
    _guard_against_stale_output_dir,
    _load_dataset_config,
    _relative_cache_path_for_sha,
    _save_embedding,
    _sha256_text,
)
from openwam.dataloader.transforms.text_embedding_cache import CACHE_LAYOUT

logger = logging.getLogger("regen_reason1_raw_cache")


def _to_raw_last_layer(full_concat):
    """Take ``(1, L, 100352)`` full-concat → ``(L, 3584)`` bf16 last layer.

    The mainline `_encode_reason1` returns a concatenation over the 28
    transformer layer hiddens (each already mean-normalized along channel).
    Layer 28 sits at the end of the channel axis, so slicing the last
    ``_REASON1_HIDDEN_SIZE`` channels recovers the last-layer mean-normalized
    hidden state — no extra GPU work.
    """
    import torch

    last = full_concat[..., -_REASON1_HIDDEN_SIZE:]
    return last.squeeze(0).to(dtype=torch.bfloat16).contiguous().cpu()


def precompute_raw(
    *,
    reason1_ckpt: Path,
    dataset_config: Path,
    output_dir: Path,
    device: str = "cuda:0",
    overwrite: bool = False,
    allow_stale_output_dir: bool = False,
    dataset_dir_override: str | None = None,
) -> dict:
    import torch

    dataset_cfg = _load_dataset_config(str(dataset_config))
    if dataset_dir_override:
        dataset_cfg["dataset_dir"] = dataset_dir_override
        logger.info("Overriding dataset_dir → %s", dataset_dir_override)
    prompts = _enumerate_prompts(dataset_cfg)
    logger.info("Enumerated %d unique formatted prompts from dataset", len(prompts))
    prompt_shas = {_sha256_text(prompt) for prompt in prompts}
    _guard_against_stale_output_dir(
        output_dir,
        len(prompts),
        allow_stale_output_dir=allow_stale_output_dir,
    )
    if not allow_stale_output_dir:
        unmanaged_flat = _find_unmanaged_flat_cache_files(output_dir, prompt_shas)
        if unmanaged_flat:
            examples = ", ".join(p.name for p in unmanaged_flat[:3])
            raise RuntimeError(
                f"Refusing to write into cache dir with {len(unmanaged_flat)} unmanaged flat "
                f".safetensors files (examples: {examples}). Use a fresh --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    model, tokenizer = _build_reason1(reason1_ckpt, device=device, dtype=dtype)

    try:
        from tqdm import tqdm
        prompt_iter = tqdm(prompts, desc="Encoding (raw last-layer)", unit="prompt")
    except ImportError:
        prompt_iter = prompts

    saved = 0
    skipped = 0
    for prompt in prompt_iter:
        sha = _sha256_text(prompt)
        path = _cache_path_for_sha(output_dir, sha)
        if path.exists() and not overwrite:
            skipped += 1
            continue
        emb = _encode_reason1(model, tokenizer, prompt, device=device)
        raw = _to_raw_last_layer(emb)
        if raw.shape != (_NUM_EMBEDDING_PADDING_TOKENS, _REASON1_HIDDEN_SIZE):
            raise RuntimeError(
                f"raw last-layer shape {tuple(raw.shape)} != "
                f"({_NUM_EMBEDDING_PADDING_TOKENS}, {_REASON1_HIDDEN_SIZE})"
            )
        _save_embedding(output_dir, _relative_cache_path_for_sha(sha), raw, prompt)
        saved += 1

    # Empty prompt for CFG dropout.
    empty_path = output_dir / "empty.safetensors"
    if overwrite or not empty_path.exists():
        emb = _encode_reason1(model, tokenizer, "", device=device)
        raw = _to_raw_last_layer(emb)
        if raw.shape != (_NUM_EMBEDDING_PADDING_TOKENS, _REASON1_HIDDEN_SIZE):
            raise RuntimeError(
                f"empty raw last-layer shape {tuple(raw.shape)} != "
                f"({_NUM_EMBEDDING_PADDING_TOKENS}, {_REASON1_HIDDEN_SIZE})"
            )
        _save_embedding(output_dir, "empty.safetensors", raw, "")

    # Write a manifest so (a) ``_guard_against_stale_output_dir`` is armed on
    # reruns — it early-returns without one — and (b) the cache carries
    # provenance, most importantly ``dim: 3584`` which distinguishes this RAW
    # cache from the mainline post-projection (1024-d) cache that shares the
    # same sha256-keyed filenames.
    manifest = {
        "format": "reason1_raw_last_layer",
        "reason1_ckpt": str(reason1_ckpt),
        "dtype": "bf16",
        "layout": CACHE_LAYOUT,
        "seq_len": _NUM_EMBEDDING_PADDING_TOKENS,
        "dim": _REASON1_HIDDEN_SIZE,
        "prompt_count": len(prompts),
        "newly_saved": saved,
        "skipped_existing": skipped,
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info("Done: %s", manifest)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reason1-ckpt", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-stale-output-dir", action="store_true")
    parser.add_argument(
        "--dataset-dir-override",
        type=str,
        default=None,
        help="Override dataset_dir from the YAML (use for env-specific paths).",
    )
    return parser


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)
    return precompute_raw(
        reason1_ckpt=args.reason1_ckpt,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
        allow_stale_output_dir=args.allow_stale_output_dir,
        dataset_dir_override=args.dataset_dir_override,
    )


if __name__ == "__main__":
    # Returns a manifest dict on success; failures bubble up as exceptions
    # (non-zero exit) rather than a boolean status.
    main()
