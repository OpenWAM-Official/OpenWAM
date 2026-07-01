"""Pre-compute Cosmos-Reason1-7B text embeddings for the CosmosPredict25 backbone.

For every unique RoboTwin caption (enumerated across all
``instructions/<task>/episode<N>.json`` files in ``seen`` and ``unseen``,
plus the task-name fallback used by ``_resolve_prompt``), this tool:

1. Wraps the caption with ``format_prompt_for_inference`` so the cache key
   matches what training will look up at runtime.
2. Encodes via Reason1-7B exactly as upstream does
   (``cosmos_predict2/_src/predict2/text_encoders/text_encoder.py:131-220``,
   tracked via the ``third_party/cosmos-predict2.5`` submodule): apply chat
   template with the
   "image-generator helper" system prompt, pad/truncate to L=512 with
   ``pad_token_id``, forward with ``output_hidden_states=True``,
   mean-normalize each of the 28 transformer-block hidden states, then
   ``full_concat`` along the channel dim into ``(L=512, 100352)``.
3. Loads ONLY ``net.crossattn_proj.0.{weight,bias}`` (no DiT body)
   from the Cosmos ``<uuid>_ema_bf16.pt`` checkpoint, builds the upstream
   ``Sequential(Linear(100352,1024), GELU())``, and applies it to the
   100352 tensor to land at ``(L=512, 1024)`` bf16.
4. Saves each result as ``<output_dir>/<sha[:2]>/<sha256(prompt)>.safetensors``
   (tensor key ``"pre_encoded_text"``; safetensors metadata carries the
   raw caption for debug). Also caches ``empty.safetensors`` for the
   classifier-free guidance dropout target consumed by
   ``TextEmbeddingCacheTransform``.

Usage::

    python -m openwam.dataloader.utils.stats_computation.reason1_embedding_computation \\
        --reason1-ckpt /path/to/assets/Cosmos-Reason1-7B \\
        --cosmos-ckpt  /path/to/assets/Cosmos-Predict2.5-2B/base/post-trained/<uuid>_ema_bf16.pt \\
        --dataset-config configs/dataloader/robotwin.yaml \\
        --output-dir   /path/to/cache/reason1_robotwin_postproj_v1

The script writes a ``manifest.json`` with the Cosmos and Reason1 paths
+ sha256s + dtype + count so consumers can detect a stale cache after a
Cosmos DiT checkpoint swap (the ``crossattn_proj`` weights would have
changed and the post-projection embeddings need re-baking).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openwam.dataloader.transforms.text_embedding_cache import (
    BUCKET_PREFIX_LEN,
    CACHE_LAYOUT,
    bucketed_cache_path_for_sha,
)

logger = logging.getLogger("reason1_precompute")

_STALE_FILE_MULTIPLIER = 100

# Copied verbatim from upstream
# cosmos_predict2/_src/predict2/text_encoders/text_encoder.py:145-150
# (via the third_party/cosmos-predict2.5 submodule). Kept as a module constant
# so we don't import third_party at runtime.
_COSMOS_REASON1_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."

# Upstream pad/truncation target — matches text_encoder.py:34.
_NUM_EMBEDDING_PADDING_TOKENS = 512

# Reason1 / Qwen2.5-VL-7B geometry: 28 transformer layers × hidden_size=3584
# → full_concat produces 100352 channels (matches Cosmos `crossattn_proj_in_channels`).
_REASON1_NUM_TRANSFORMER_LAYERS = 28
_REASON1_HIDDEN_SIZE = 3584
_REASON1_FULL_CONCAT_DIM = _REASON1_NUM_TRANSFORMER_LAYERS * _REASON1_HIDDEN_SIZE  # 100352

# Post-projection target dim — Cosmos 2B Stage-c `crossattn_emb_channels=1024`.
_COSMOS_POSTPROJ_DIM = 1024


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path_for_sha(output_dir: Path, sha: str) -> Path:
    return Path(bucketed_cache_path_for_sha(str(output_dir), sha))


def _relative_cache_path_for_sha(sha: str) -> Path:
    return Path(sha[:BUCKET_PREFIX_LEN]) / f"{sha}.safetensors"


def _load_dataset_config(path: str) -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _enumerate_prompts(dataset_cfg: dict) -> list[str]:
    """Walk the dataset's instruction JSONs and return every formatted prompt
    that ``RoboTwinDataset._get_prompt`` could ever return.

    ``_resolve_prompt`` (robotwin.py:221-274) picks via
    ``random.choice`` over the ``seen`` / ``unseen`` pools per call, so the
    cache must enumerate every entry — not just a sampled instance. We also
    include the task-name fallback string and the empty caption (the latter
    is saved separately as ``empty.safetensors``).
    """
    from openwam.dataloader.robotwin import (
        ROBOTWIN_ALL_TASKS,
        ROBOTWIN_TRAIN_TASKS,
        discover_robotwin_roots,
    )
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    dataset_dir = dataset_cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset_dir is required in the dataset config")
    robot = dataset_cfg.get("robot", "aloha-agilex")
    variant = dataset_cfg.get("variant", "both")
    task_name = dataset_cfg.get("task_name")

    if task_name:
        tasks = [task_name]
    else:
        train_tasks = dataset_cfg.get("train_tasks")
        holdout_tasks = dataset_cfg.get("holdout_tasks")
        if train_tasks:
            tasks = list(train_tasks)
        elif holdout_tasks:
            tasks = sorted(t for t in ROBOTWIN_ALL_TASKS if t not in holdout_tasks)
        else:
            tasks = ROBOTWIN_TRAIN_TASKS

    variant_list = ["clean_50", "randomized_500"] if variant == "both" else [variant]

    prompts: set[str] = set()
    for v in variant_list:
        roots = discover_robotwin_roots(dataset_dir, robot, v, tasks)
        for task_label, data_root in roots:
            # Always include the task-name fallback — _resolve_prompt falls
            # back to it when an episode has no instruction entry, so the
            # cache must contain it regardless of whether instructions/ exists.
            local_task = task_label.split("/")[0]
            prompts.add(format_prompt_for_inference(f"The bimanual robot is performing a {local_task} task."))

            instr_dir = os.path.join(os.path.dirname(data_root), "instructions")
            if not os.path.isdir(instr_dir):
                logger.warning("No instructions/ next to %s — task-name fallback only", data_root)
                continue
            for fname in sorted(os.listdir(instr_dir)):
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(instr_dir, fname), "r") as f:
                        instr = json.load(f)
                except Exception as exc:
                    logger.warning("Skipping malformed %s: %s", fname, exc)
                    continue
                bases: list[str] = []
                if isinstance(instr, dict):
                    for pool_name in ("seen", "unseen"):
                        pool = instr.get(pool_name)
                        if pool:
                            bases.extend(p for p in pool if isinstance(p, str))
                    extra = instr.get("instruction")
                    if isinstance(extra, str):
                        bases.append(extra)
                elif isinstance(instr, str):
                    bases.append(instr)
                elif isinstance(instr, list):
                    bases.extend(b for b in instr if isinstance(b, str))
                for b in bases:
                    prompts.add(format_prompt_for_inference(b))

    return sorted(prompts)


def _build_crossattn_proj(cosmos_ckpt: Path, device: str, dtype):
    """Lift only ``net.crossattn_proj.0.{weight,bias}`` from the Cosmos
    EMA checkpoint and build the upstream ``Sequential(Linear, GELU)``.

    Avoids loading the entire DiT (~5 GB on GPU). Mirrors
    ``minimal_v4_dit.py:1566-1568`` exactly: ``Linear(in, out, bias=True) +
    nn.GELU()``.
    """
    import torch
    import torch.nn as nn

    if not cosmos_ckpt.is_file():
        raise FileNotFoundError(f"Cosmos checkpoint not found: {cosmos_ckpt}")
    sd = torch.load(cosmos_ckpt, map_location="cpu", weights_only=False)
    w_key = "net.crossattn_proj.0.weight"
    b_key = "net.crossattn_proj.0.bias"
    if w_key not in sd or b_key not in sd:
        peek = sorted(k for k in sd if "crossattn_proj" in k)
        raise KeyError(f"Expected {w_key!r} and {b_key!r} in {cosmos_ckpt}; found crossattn_proj keys: {peek}")
    w = sd[w_key]
    b = sd[b_key]
    out_dim, in_dim = w.shape
    if in_dim != _REASON1_FULL_CONCAT_DIM or out_dim != _COSMOS_POSTPROJ_DIM:
        raise ValueError(
            f"crossattn_proj has unexpected shape: weight={tuple(w.shape)}, "
            f"expected ({_COSMOS_POSTPROJ_DIM}, {_REASON1_FULL_CONCAT_DIM})."
        )
    linear = nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        linear.weight.copy_(w.float())
        linear.bias.copy_(b.float())
    proj = nn.Sequential(linear, nn.GELU()).to(device=device, dtype=dtype).eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


def _build_reason1(reason1_ckpt: Path, device: str, dtype):
    """Load Reason1-7B via HF transformers in text-only mode.

    Returns ``(model, tokenizer)``. The model is ``Qwen2_5_VLForConditionalGeneration``;
    we'll call it with ``input_ids`` only (no ``pixel_values``), so it runs
    the language-model branch end-to-end. ``output_hidden_states=True`` is
    set per-call.
    """
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    if not reason1_ckpt.is_dir():
        raise FileNotFoundError(f"Reason1 weights not found at {reason1_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(str(reason1_ckpt), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(reason1_ckpt),
        torch_dtype=dtype,
        device_map={"": device},
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # transformers ≥5 ``Qwen2_5_VLConfig`` exposes the language-model geometry
    # under ``config.text_config``. ``or model.config`` is a defensive fallback
    # for the (unsupported) case where ``text_config`` is missing or ``None``.
    text_cfg = getattr(model.config, "text_config", None) or model.config
    hidden_size = getattr(text_cfg, "hidden_size", None)
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    if hidden_size != _REASON1_HIDDEN_SIZE:
        raise ValueError(
            f"Reason1 hidden_size={hidden_size} != expected {_REASON1_HIDDEN_SIZE} "
            "(this script targets the Cosmos-Reason1-7B / Qwen2.5-VL-7B geometry only)."
        )
    if num_layers != _REASON1_NUM_TRANSFORMER_LAYERS:
        raise ValueError(f"Reason1 num_hidden_layers={num_layers} != expected {_REASON1_NUM_TRANSFORMER_LAYERS}")
    return model, tokenizer


def _tokenize_with_chat_template(tokenizer, prompt: str):
    """Match upstream's chat-template wrap: system prompt + user content."""
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": _COSMOS_REASON1_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]
    # HF chat-template returns the formatted *string*; tokenize after.
    chat_string = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(
        chat_string,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    input_ids = enc["input_ids"][0].tolist()

    pad_id = int(tokenizer.pad_token_id)
    if len(input_ids) < _NUM_EMBEDDING_PADDING_TOKENS:
        pad_len = _NUM_EMBEDDING_PADDING_TOKENS - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
    else:
        input_ids = input_ids[:_NUM_EMBEDDING_PADDING_TOKENS]
    return input_ids


def _mean_normalize_along_last(hs):
    """Per-token, per-layer normalize: (x - mean) / (std + 1e-8) along channel dim."""
    return (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)


def _encode_reason1(model, tokenizer, prompt: str, device: str):
    """Run one prompt through Reason1 and return mean-normalized full_concat
    ``(1, L=512, 100352)``."""
    import torch

    input_ids = _tokenize_with_chat_template(tokenizer, prompt)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids_t)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_t,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states  # tuple of (B, L, H) tensors

    # Sanity: 1 embedding layer + 28 transformer blocks = 29 entries.
    expected = _REASON1_NUM_TRANSFORMER_LAYERS + 1
    if len(hidden_states) != expected:
        raise RuntimeError(
            f"Reason1 hidden_states has {len(hidden_states)} entries; "
            f"expected {expected} (1 embed + {_REASON1_NUM_TRANSFORMER_LAYERS} transformer)."
        )
    # full_concat skips the embedding layer (index 0).
    normalized = [_mean_normalize_along_last(h) for h in hidden_states[1:]]
    full_concat = torch.cat(normalized, dim=-1)  # (1, L, 100352)
    if full_concat.shape[-1] != _REASON1_FULL_CONCAT_DIM:
        raise RuntimeError(
            f"full_concat dim mismatch: got {full_concat.shape[-1]}, expected {_REASON1_FULL_CONCAT_DIM}."
        )
    return full_concat


def _project_to_postproj(reason1_emb, crossattn_proj):
    """Apply the upstream ``Sequential(Linear, GELU)`` → ``(L, 1024)`` bf16."""
    import torch

    in_dtype = next(crossattn_proj.parameters()).dtype
    out = crossattn_proj(reason1_emb.to(dtype=in_dtype))
    return out.squeeze(0).to(dtype=torch.bfloat16).contiguous().cpu()


def _save_embedding(out_dir: Path, rel_path: Path | str, tensor, prompt: str):
    from safetensors.torch import save_file

    metadata = {"prompt": prompt[:512]}  # safetensors metadata caps at 1MB total
    path = out_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"pre_encoded_text": tensor}, str(path), metadata=metadata)


def _count_safetensors_files(root: Path, *, limit: int) -> int:
    """Count safetensors up to *limit* for stale-output-dir detection."""
    count = 0
    for _path in root.rglob("*.safetensors"):
        count += 1
        if count >= limit:
            return count
    return count


def _guard_against_stale_output_dir(output_dir: Path, prompt_count: int, *, allow_stale_output_dir: bool) -> None:
    """Refuse obviously stale cache dirs unless the operator opts in.

    A correct Reason1 cache is one ~1 MiB file per unique prompt plus
    ``empty.safetensors``. The observed RoboTwin blow-up was a directory with
    a manifest for 1593 prompts but >1.2M files. This bounded guard catches
    that class without fully scanning huge directories.
    """
    manifest_path = output_dir / "manifest.json"
    if allow_stale_output_dir or not output_dir.exists() or not manifest_path.exists():
        return
    threshold = max(prompt_count + 1, prompt_count * _STALE_FILE_MULTIPLIER, 5)
    seen = _count_safetensors_files(output_dir, limit=threshold + 1)
    if seen > threshold:
        raise RuntimeError(
            f"Refusing to write into likely stale Reason1 cache dir: {output_dir}\n"
            f"Current prompt_count={prompt_count}, but found more than {threshold} existing "
            ".safetensors files while scanning. A healthy cache should be close to "
            "prompt_count + 1 (empty.safetensors). Use a fresh --output-dir, clean the "
            "old directory manually, or pass --allow-stale-output-dir if you intentionally "
            "want to append to it."
        )


def _find_unmanaged_flat_cache_files(output_dir: Path, prompt_shas: set[str]) -> list[Path]:
    """Return flat prompt cache files that are not part of this precompute set."""
    unmanaged: list[Path] = []
    if not output_dir.exists():
        return unmanaged
    for path in output_dir.glob("*.safetensors"):
        if path.name == "empty.safetensors":
            continue
        stem = path.stem
        if len(stem) == 64 and stem not in prompt_shas:
            unmanaged.append(path)
    return unmanaged


def precompute(
    *,
    reason1_ckpt: Path,
    cosmos_ckpt: Path,
    dataset_config: Path,
    output_dir: Path,
    device: str = "cuda:0",
    overwrite: bool = False,
    allow_stale_output_dir: bool = False,
):
    """High-level entry — exposed for tests that mock the heavy imports."""
    import torch

    dataset_cfg = _load_dataset_config(str(dataset_config))
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
                f".safetensors files (examples: {examples}). New caches are bucketed under "
                f"<sha[:{BUCKET_PREFIX_LEN}]>/<sha>.safetensors. Use a fresh --output-dir, "
                "clean the stale flat files manually, or pass --allow-stale-output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    crossattn_proj = _build_crossattn_proj(cosmos_ckpt, device=device, dtype=dtype)
    model, tokenizer = _build_reason1(reason1_ckpt, device=device, dtype=dtype)

    try:
        from tqdm import tqdm

        prompt_iter = tqdm(prompts, desc="Encoding prompts", unit="prompt")
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
        proj = _project_to_postproj(emb, crossattn_proj)
        if proj.shape != (_NUM_EMBEDDING_PADDING_TOKENS, _COSMOS_POSTPROJ_DIM):
            raise RuntimeError(
                f"projected embedding shape {tuple(proj.shape)} != "
                f"({_NUM_EMBEDDING_PADDING_TOKENS}, {_COSMOS_POSTPROJ_DIM})"
            )
        _save_embedding(output_dir, _relative_cache_path_for_sha(sha), proj, prompt)
        saved += 1

    # Always (re)compute empty.safetensors for CFG dropout target.
    empty_path = output_dir / "empty.safetensors"
    if overwrite or not empty_path.exists():
        emb = _encode_reason1(model, tokenizer, "", device=device)
        proj = _project_to_postproj(emb, crossattn_proj)
        _save_embedding(output_dir, "empty.safetensors", proj, "")

    manifest = {
        "reason1_ckpt": str(reason1_ckpt),
        "cosmos_ckpt": str(cosmos_ckpt),
        "cosmos_ckpt_sha256": _sha256_file(cosmos_ckpt),
        "dtype": "bf16",
        "layout": CACHE_LAYOUT,
        "bucket_prefix_len": BUCKET_PREFIX_LEN,
        "seq_len": _NUM_EMBEDDING_PADDING_TOKENS,
        "dim": _COSMOS_POSTPROJ_DIM,
        "prompt_count": len(prompts),
        "newly_saved": saved,
        "skipped_existing": skipped,
        "system_prompt": _COSMOS_REASON1_SYSTEM_PROMPT,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info("Precompute complete — saved=%d, skipped=%d, output=%s", saved, skipped, output_dir)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reason1-ckpt", required=True, type=Path)
    parser.add_argument("--cosmos-ckpt", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite cache files that already exist (default: skip and keep them).",
    )
    parser.add_argument(
        "--allow-stale-output-dir",
        action="store_true",
        help=(
            "Allow writing into a manifest-bearing output dir with far more existing "
            ".safetensors files than the current prompt set. Prefer a fresh output dir."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    return precompute(
        reason1_ckpt=args.reason1_ckpt,
        cosmos_ckpt=args.cosmos_ckpt,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
        allow_stale_output_dir=args.allow_stale_output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - exercised via __main__
    main()


# Re-export low-level helpers for tests that want to bypass the CLI.
__all__ = [
    "precompute",
    "main",
    "_COSMOS_REASON1_SYSTEM_PROMPT",
    "_enumerate_prompts",
    "_build_crossattn_proj",
    "_mean_normalize_along_last",
    "_tokenize_with_chat_template",
]


def _aux_for_tests() -> dict[str, Any]:
    """Stable handle for tests to inspect module-level constants."""
    return {
        "NUM_EMBEDDING_PADDING_TOKENS": _NUM_EMBEDDING_PADDING_TOKENS,
        "REASON1_FULL_CONCAT_DIM": _REASON1_FULL_CONCAT_DIM,
        "COSMOS_POSTPROJ_DIM": _COSMOS_POSTPROJ_DIM,
        "SYSTEM_PROMPT": _COSMOS_REASON1_SYSTEM_PROMPT,
    }
