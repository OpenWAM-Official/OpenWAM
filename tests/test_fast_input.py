"""OPENWAM_FAST_INPUT / OPENWAM_TEXT_CACHE paths give the same tensors as the default input pipeline."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.dataloader.collate import openwam_collate
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.preprocess import preprocess_video as preprocess_video_native
from openwam.model.video_backbone.wan_backbone import _truncate_padded_context
from openwam.train.openwam_trainer import OpenWAMTrainer


def _clip(seed, frames=5, height=16, width=24):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB") for _ in range(frames)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_uint8_collate_matches_per_frame_preprocessing(dtype):
    clips = [_clip(0), _clip(1)]
    reference = torch.cat([preprocess_video_native(c, dtype=dtype, device="cpu") for c in clips], dim=0)

    batch = openwam_collate([{"video": c, "prompt": "p"} for c in clips])
    assert all(s["video"].dtype == torch.uint8 and s["video"].shape == (5, 16, 24, 3) for s in batch)
    fast = wan_encode.preprocess_video([s["video"] for s in batch], encoder=None, dtype=dtype, device="cpu")

    assert fast.dtype == reference.dtype and fast.shape == reference.shape
    assert fast.stride() == reference.stride()
    assert torch.equal(fast, reference)


def test_collate_leaves_other_samples_as_they_are():
    gray = [Image.new("L", (8, 8)) for _ in range(3)]
    mixed = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 4))]
    samples = [{"video": gray}, {"video": mixed}, {"prompt": "no video"}]
    out = openwam_collate(samples)
    assert [s is o for s, o in zip(samples, out)] == [True, True, True]


class _Tokenizer:
    """Character codes, padded to ``max_length``; mask 1 on the prompt tokens."""

    def __call__(self, prompts, *, return_mask, add_special_tokens, max_length, padding, truncation):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            codes = [ord(c) % 250 + 1 for c in p][:max_length]
            ids[i, : len(codes)] = torch.tensor(codes)
            mask[i, : len(codes)] = 1
        return ids, mask


class _TextEncoder(torch.nn.Module):
    """Per-token and per-row independent (as umT5's rows are), nonzero at padding positions."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(256, 8)

    def forward(self, ids, mask):
        return torch.tanh(self.embed(ids)) * 2 + mask.unsqueeze(-1)


def _encode(prompts, cache=None):
    encoder = _TextEncoder().eval().requires_grad_(False)
    return wan_encode.encode_text(prompts, tokenizer=_Tokenizer(), text_encoder=encoder, device="cpu", cache=cache)


@pytest.mark.parametrize("miss_fast", ["0", "1"])
def test_text_cache_rows_match_direct_encode(monkeypatch, miss_fast):
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", miss_fast)
    cache = wan_encode.TrainTextEmbedCache(max_entries=8)
    batches = [
        ["pick up the bowl", "open the drawer", "pick up the bowl"],  # all misses, one duplicate
        ["open the drawer", "pick up the bowl", "open the drawer"],  # all hits
        ["put the mug on the plate", "open the drawer", "turn on the stove"],  # partial hit
    ]
    for prompts in batches:
        ref_context, ref_lens = _encode(prompts)
        context, seq_lens = _encode(prompts, cache=cache)
        assert torch.equal(context, ref_context)
        assert torch.equal(seq_lens, ref_lens)
        assert cache.last_max_len == max(len(p) for p in prompts)
    assert (cache.hits, cache.misses, len(cache.rows)) == (1, 2, 4)


def test_text_cache_evicts_oldest_rows():
    cache = wan_encode.TrainTextEmbedCache(max_entries=2)
    for prompts in (["a"], ["b"], ["c"]):
        _encode(prompts, cache=cache)
    assert len(cache.rows) == 2 and cache.misses == 3


def test_context_truncation_keeps_every_valid_position(monkeypatch):
    context = torch.randn(2, 512, 4)
    assert _truncate_padded_context(context, 70) is context  # switch off
    monkeypatch.setenv("OPENWAM_CONTEXT_TRUNC", "1")
    for max_len, keep in [(1, 63), (63, 63), (64, 127), (300, 319), (511, 511), (512, 512)]:
        out = _truncate_padded_context(context, max_len)
        assert out.shape[1] == keep >= max_len
        assert torch.equal(out, context[:, :keep])
    assert _truncate_padded_context(context, None) is context


def test_fast_input_collate_needs_a_backbone_that_takes_uint8_clips(monkeypatch):
    def collate_fn(accepts):
        trainer = SimpleNamespace(
            architecture=SimpleNamespace(video_backbone=SimpleNamespace(accepts_uint8_clips=accepts))
        )
        return OpenWAMTrainer._collate_fn(trainer)

    assert collate_fn(True) is list  # switch off
    monkeypatch.setenv("OPENWAM_FAST_INPUT", "1")
    assert collate_fn(True) is openwam_collate
    assert collate_fn(False) is list
