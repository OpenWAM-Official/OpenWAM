"""Wan VAE / text-encoder IO with native-VAE ↔ external-encoder routing.

Free functions — ``vae`` / ``encoder`` / ``tokenizer`` / ``text_encoder`` passed
explicitly so the backbone class holds no IO logic. ``encoder is not None``
selects the external-encoder path; otherwise the native Wan VAE is used.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video

logger = logging.getLogger(__name__)


def encode_text(prompts: list, *, tokenizer, text_encoder, device, cache=None) -> Tuple[Tensor, Tensor]:
    """umT5 ``(context, seq_lens)``; with a :class:`TrainTextEmbedCache`, see ``encode_text_cached``."""
    if cache is not None:
        return encode_text_cached(prompts, tokenizer=tokenizer, text_encoder=text_encoder, device=device, cache=cache)
    return _encode_text_uncached(prompts, tokenizer=tokenizer, text_encoder=text_encoder, device=device)


def _encode_text_uncached(prompts: list, *, tokenizer, text_encoder, device, cache=None) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = _run_encoder(text_encoder, ids, mask, device=device, cache=cache)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


class TrainTextEmbedCache:
    """Bounded, per-process cache of the frozen umT5 encoder's outputs for training
    (``OPENWAM_TEXT_CACHE=1``).

    Keyed by the exact encoder input of one sample (its token ids and attention
    mask, as bytes). Filled only from the batch being encoded inside a training
    iteration (never at start-up, never persisted). On a miss the encoder runs on
    the full batch, exactly as without the cache (unless ``OPENWAM_TEXT_MISS_FAST``),
    so a cached row is the row the encoder produced for that input at this batch
    shape; per-sample rows of the encoder do not depend on the other samples
    (per-row GEMMs, per-sample masked attention, per-token norms). ``max_entries``
    caps the memory (one row is 512 x 4096 bf16 = 4 MiB); beyond it the oldest
    entries are evicted.
    """

    def __init__(self, max_entries: int = 256):
        import collections

        self.max_entries = int(max_entries)
        self.rows = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
        self.last_max_len = None  # longest prompt (tokens) of the last encoded batch, host-side

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, row: Tensor) -> None:
        if key in self.rows or self.max_entries <= 0:
            return
        self.rows[key] = row
        while len(self.rows) > self.max_entries:
            self.rows.popitem(last=False)

    def clear(self) -> None:
        self.rows.clear()
        self.last_max_len = None


class TextEncoderOffload:
    """Frozen umT5 weights in pinned host memory between text-cache misses (``training.text_encoder_offload=true``,
    with ``OPENWAM_TEXT_CACHE=1``).

    With the text cache, umT5 only runs on cache misses (LIBERO: about 40 prompts, so only in the first iterations of
    a run), but its 10.6 GiB of bf16 weights would stay on every GPU for the whole run. Here the first miss runs with
    the weights where they are (on the GPU); afterwards every parameter and buffer is copied once into one flat
    pinned host buffer per dtype and the module's tensors point at views of it. A later miss copies the flat buffer
    to the GPU in one asynchronous host-to-device copy, points the tensors at views of the GPU copy, runs the encoder
    and points them back at the host views (the caching allocator reuses the GPU copy in stream order). The encoder
    computes with the same weights and the same kernels on the GPU: bitwise the same rows. Costs one host-to-device
    copy of the weights per iteration with a miss.
    """

    def __init__(self, module):
        self.module = module
        self.host = None  # {dtype: flat pinned tensor}
        self.slots = []  # (tensor, dtype, offset, numel, shape)
        self.on_gpu = True  # the first miss runs with the weights where they were loaded
        self.loads = 0
        self.enabled = True
        self.cache = None
        # Inference and uncached entry points do not receive the training cache.
        # Keep the weight lifecycle discoverable from the encoder itself.
        module._openwam_text_offload = self

    def _tensors(self):
        seen = set()
        for t in list(self.module.parameters()) + list(self.module.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                yield t

    def acquire(self, device):
        if self.on_gpu:
            return
        dev = {dt: flat.to(device=device, non_blocking=True) for dt, flat in self.host.items()}
        for t, dt, off, n, shape in self.slots:
            t.data = dev[dt][off : off + n].view(shape)
        self.on_gpu = True
        self.loads += 1

    def release(self):
        if self.host is None:  # first release: build the pinned host copy (once per process)
            sizes = {}
            for t in self._tensors():
                sizes[t.dtype] = sizes.get(t.dtype, 0) + t.numel()
            self.host = {dt: torch.empty(n, dtype=dt, pin_memory=True) for dt, n in sizes.items()}
            offs = dict.fromkeys(sizes, 0)
            for t in self._tensors():
                off, n = offs[t.dtype], t.numel()
                self.host[t.dtype][off : off + n].copy_(t.data.reshape(-1))
                self.slots.append((t, t.dtype, off, n, tuple(t.shape)))
                offs[t.dtype] = off + n
            logger.info(
                "umT5 offload: %.2f GiB of weights moved to pinned host memory",
                sum(f.numel() * f.element_size() for f in self.host.values()) / 2**30,
            )
        for t, dt, off, n, shape in self.slots:
            t.data = self.host[dt][off : off + n].view(shape)
        self.on_gpu = False

    def disable(self, device):
        """Leave live weights on the compute device once the encoder can change.

        Restoring the old frozen host snapshot after an optimizer step would
        discard updates. Also, moving parameters during a differentiable
        forward would invalidate backward's saved tensors.
        """
        self.acquire(device)
        self.enabled = False
        self.host = None
        self.slots.clear()
        if self.cache is not None:
            self.cache.clear()


def _run_encoder(text_encoder, ids, mask, *, device, cache=None):
    offload = getattr(text_encoder, "_openwam_text_offload", None)
    if offload is None:
        offload = getattr(cache, "offload", None)
    if offload is None:
        return text_encoder(ids, mask)
    if isinstance(cache, TrainTextEmbedCache):
        offload.cache = cache
    if not _encoder_is_frozen_eval(text_encoder):
        offload.disable(device)
        return text_encoder(ids, mask)
    if not offload.enabled:
        return text_encoder(ids, mask)
    offload.acquire(device)
    try:
        return text_encoder(ids, mask)
    finally:
        offload.release()


def _to_device(t: Tensor, device) -> Tensor:
    """Host-to-device copy without a host sync (through pinned memory) where CUDA is available."""
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _encoder_is_frozen_eval(text_encoder) -> bool:
    if not isinstance(text_encoder, torch.nn.Module):
        return False
    if any(m.training for m in text_encoder.modules()):
        return False
    return not any(p.requires_grad for p in text_encoder.parameters())


def encode_text_cached(prompts: list, *, tokenizer, text_encoder, device, cache) -> Tuple[Tensor, Tensor]:
    """``encode_text`` with a :class:`TrainTextEmbedCache` (same outputs, bit for bit).

    When every sample of the batch is cached, the rows are stacked; otherwise the
    encoder runs on the whole batch as in ``encode_text`` and its rows are cached.
    The padding rows are zeroed with ``masked_fill_`` (the same +0.0 as the
    per-sample slice assignment of ``encode_text``, without its per-sample host syncs).
    """
    if cache is None or not _encoder_is_frozen_eval(text_encoder):
        if cache is not None:
            cache.clear()
        return _encode_text_uncached(
            prompts, tokenizer=tokenizer, text_encoder=text_encoder, device=device, cache=cache
        )
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    keys = [(ids[i].numpy().tobytes(), mask[i].numpy().tobytes()) for i in range(ids.shape[0])]
    seq_lens_cpu = mask.gt(0).sum(dim=1).long()
    cache.last_max_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() else 0
    seq_lens = _to_device(seq_lens_cpu, device)
    rows = [cache.get(k) for k in keys]
    n = cache.hits + cache.misses + 1
    if n & (n - 1) == 0 and os.environ.get("RANK", "0") == "0":
        logger.info(
            "text-embed cache: call %d, hits %d, misses %d, entries %d", n, cache.hits, cache.misses, len(cache.rows)
        )
    if all(r is not None for r in rows):
        cache.hits += 1
        return torch.stack(rows, dim=0), seq_lens
    cache.misses += 1
    if os.environ.get("OPENWAM_TEXT_MISS_FAST", "0") == "1":
        return _encode_misses(
            ids, mask, keys, rows, seq_lens_cpu, seq_lens, text_encoder=text_encoder, device=device, cache=cache
        )
    context = _run_encoder(text_encoder, _to_device(ids, device), _to_device(mask, device), device=device, cache=cache)
    pad = torch.arange(context.shape[1], device=context.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
    context.masked_fill_(pad.unsqueeze(-1), 0)
    for i, k in enumerate(keys):
        if cache.get(k) is None:
            cache.put(k, context[i].clone())
    return context, seq_lens


def _encode_misses(ids, mask, keys, rows, seq_lens_cpu, seq_lens, *, text_encoder, device, cache):
    """Cache-miss path with ``OPENWAM_TEXT_MISS_FAST=1``: run the frozen umT5 only on the distinct
    prompts of the batch that are not cached yet, on their token prefix padded to a multiple of 64
    instead of 512.

    The same function per row: the encoder's rows are independent (per-row GEMMs, per-row attention),
    padded keys get ``finfo.min`` before the fp32 softmax (weight exactly 0) and the relative position
    bias depends only on i - j, so the outputs at the valid positions do not depend on how much padding
    follows; positions >= seq_len are zeroed and the row is zero-padded back to 512. The GEMMs run with
    other shapes, so rounding can differ from a full-batch 512-token encode. Rows are cached as on the
    full-batch path.
    """
    miss, seen = [], set()
    for i, k in enumerate(keys):
        if rows[i] is None and k not in seen:
            seen.add(k)
            miss.append(i)
    full_len = ids.shape[1]
    sl = seq_lens_cpu[miss]
    L = min(full_len, max(64, ((int(sl.max()) + 63) // 64) * 64))
    sub_ids = _to_device(ids[miss, :L].contiguous(), device)
    sub_mask = _to_device(mask[miss, :L].contiguous(), device)
    out = _run_encoder(text_encoder, sub_ids, sub_mask, device=device, cache=cache)
    sl_d = _to_device(sl, device)
    pad = torch.arange(L, device=out.device).unsqueeze(0) >= sl_d.unsqueeze(1)
    out.masked_fill_(pad.unsqueeze(-1), 0)
    if L < full_len:
        out = torch.nn.functional.pad(out, (0, 0, 0, full_len - L))
    new_rows = {}
    for j, i in enumerate(miss):
        row = out[j].clone()
        new_rows[keys[i]] = row
        cache.put(keys[i], row)
    context = torch.stack([r if r is not None else new_rows[k] for r, k in zip(rows, keys)], dim=0)
    return context, seq_lens


def encode_text_for_inference(
    prompt, *, vace_cache, prompt_embed_cache, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    """Deploy ``(context, seq_lens)``, reusing a prompt-keyed cached embed."""
    if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt:
        return vace_cache["context"], vace_cache["seq_lens"]
    if prompt_embed_cache is not None and prompt in prompt_embed_cache:
        return prompt_embed_cache[prompt]
    context, seq_lens = encode_text([prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device)
    if prompt_embed_cache is not None:
        prompt_embed_cache[prompt] = (context, seq_lens)
    return context, seq_lens


def preprocess_video(frames, *, encoder=None, dtype, device) -> Tensor:
    """One clip (list of PIL frames) -> ``(1, C, T, H, W)``; or, for a batch of uint8
    ``(T, H, W, 3)`` clip tensors (``openwam.dataloader.collate``), the stacked
    ``(B, C, T, H, W)`` batch (see ``preprocess_video_uint8_batch``)."""
    if encoder is None and isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], Tensor):
        return preprocess_video_uint8_batch(frames, dtype=dtype, device=device)
    if encoder is not None:
        return encoder.preprocess_video(frames)
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def preprocess_video_uint8_batch(clips, *, dtype, device, min_value=-1, max_value=1) -> Tensor:
    """Batched GPU form of ``torch.cat([preprocess_video(clip) for clip in clips])``.

    ``preprocess_video`` per frame: float32 pixel array -> ``dtype`` (exact for 0..255) on the
    device, ``* ((max - min) / 255)``, ``+ min`` (each op rounding to ``dtype``),
    laid out ``(1, C, T, H, W)``; clips concatenated on dim 0. Here the uint8
    pixels go to the device (asynchronously when pinned), are converted to
    ``dtype`` there (exact) and go through the same two ops in the same order, then
    are laid out ``(B, C, T, H, W)`` contiguously: bitwise the same tensor.
    """
    on_dev = [c.to(device=device, non_blocking=True) for c in clips]
    x = torch.stack(on_dev, dim=0).to(dtype=dtype)
    x = x * ((max_value - min_value) / 255) + min_value
    return x.permute(0, 4, 1, 2, 3).contiguous()


def encode_video(video_tensor: Tensor, *, vae, encoder=None) -> Tensor:
    if encoder is not None:
        return encoder.batch_encode(video_tensor)
    return vae.batch_encode(video_tensor, device=video_tensor.device)


def encode_video_for_vace(
    pixels: Tensor,
    *,
    vae,
    encoder=None,
    device,
    tiled: bool,
    tile_size: tuple,
    tile_stride: tuple,
) -> Tensor:
    """Tiled-aware VAE encode for the VACE pixel→latent helper, returning
    ``(B, z_dim, T_lat, H_lat, W_lat)``. ``tiled=False`` (training) batches in
    one call; ``tiled=True`` (deploy) loops per-sample with bounded peak memory,
    mirroring the vendored unit so large-frame deploy does not OOM.
    """
    if not tiled:
        return encode_video(pixels, vae=vae, encoder=encoder).to(dtype=pixels.dtype, device=pixels.device)
    if encoder is not None:
        # No generic tiled-encode contract; fall back to batch_encode.
        # Unreachable today (VACE + external_encoder is fail-fast).
        return encoder.batch_encode(pixels).to(dtype=pixels.dtype, device=pixels.device)
    # Native Wan VAE: per-sample tiled encode (deploy is B=1).
    outs = []
    for i in range(pixels.shape[0]):
        lat = vae.encode(
            [pixels[i]],
            device=device,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        outs.append(lat)
    return torch.cat(outs, dim=0).to(dtype=pixels.dtype, device=pixels.device)


def decode_latents(latents: Tensor, *, vae, encoder=None, device, tiled: bool = True) -> Tensor:
    if encoder is not None:
        return encoder.decode(latents.to(device), tiled=tiled)
    return vae.decode(latents.to(device), device=device, tiled=tiled)


def latents_to_frames(video_tensor: Tensor, *, encoder=None) -> list:
    if encoder is not None:
        return encoder.to_frames(video_tensor)
    return vae_output_to_video(video_tensor)
