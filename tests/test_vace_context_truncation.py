"""Context truncation must preserve VACE's unmasked padding contribution.

CPU-only: exercise the actual training preprocess/cache/conditioning path with
small deterministic encoders, then compare real VACE and DiT forward/backward.
No pretrained weights, CUDA kernels or network access are required.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from openwam.model.video_backbone.wan.models.dit import DiTBlock
from openwam.model.video_backbone.wan.models.vace import VaceWanModel
from openwam.model.video_backbone.wan_backbone import Wan21, Wan22Ti2v, _truncate_padded_context


class _Tokenizer:
    def __call__(self, prompts, *, return_mask, add_special_tokens, max_length, padding, truncation):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            codes = [ord(c) % 250 + 1 for c in prompt][:max_length]
            ids[i, : len(codes)] = torch.tensor(codes)
            mask[i, : len(codes)] = 1
        return ids, mask


class _FrozenTextEncoder(torch.nn.Module):
    """Token-independent outputs, including nonzero padding before encode_text."""

    def __init__(self):
        super().__init__()
        weights = torch.arange(256 * 8, dtype=torch.float64).reshape(256, 8).sin()
        self.embed = torch.nn.Embedding.from_pretrained(weights, freeze=True)
        self.calls = 0
        self.eval()

    def forward(self, ids, mask):
        self.calls += 1
        return self.embed(ids) + mask.unsqueeze(-1)


class _VideoVAE:
    """Content-sensitive 16-channel, causal 4x temporal / 8x spatial stand-in."""

    def batch_encode(self, videos, device):
        pixels = videos.to(device)[:, :, ::4]
        pooled = torch.nn.functional.avg_pool3d(pixels, kernel_size=(1, 8, 8))
        return pooled.mean(dim=1, keepdim=True).expand(-1, 16, -1, -1, -1).contiguous()


@pytest.fixture(autouse=True)
def _isolate_optimization_switches(monkeypatch):
    for name, value in {
        "OPENWAM_TEXT_CACHE": "1",
        "OPENWAM_TEXT_CACHE_MAX_ENTRIES": "8",
        "OPENWAM_TEXT_MISS_FAST": "0",
        "OPENWAM_CONTEXT_TRUNC": "0",
        "OPENWAM_VAE_FIRST": "0",
    }.items():
        monkeypatch.setenv(name, value)


def _make_backbone(variant):
    holder = SimpleNamespace(
        dit=SimpleNamespace(
            has_image_input=False,
            fuse_vae_embedding_in_latents=variant == "ti2v",
            seperated_timestep=variant == "ti2v",
        ),
        vace=object() if variant == "vace" else None,
        vae=_VideoVAE(),
        text_encoder=_FrozenTextEncoder(),
        tokenizer=_Tokenizer(),
        height_division_factor=16,
        width_division_factor=16,
        time_division_factor=4,
        time_division_remainder=1,
    )
    cls = Wan22Ti2v if variant == "ti2v" else Wan21
    bb = cls(holder, text_dim=8)
    bb._device = torch.device("cpu")
    bb._dtype = torch.float64
    return bb


def _preprocess(bb, prompts):
    frames = [[Image.new("RGB", (16, 16), (20 + i, 40 + t, 60)) for t in range(5)] for i in range(len(prompts))]
    return bb.preprocess_input_for_train(
        frames=frames,
        text=prompts,
        ref_images=[[clip[0]] for clip in frames],
    )


@pytest.mark.parametrize("miss_fast", ["0", "1"])
@pytest.mark.parametrize("vae_first", ["0", "1"])
def test_vace_preserves_full_context_on_cache_miss_and_hit(monkeypatch, miss_fast, vae_first):
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", miss_fast)
    monkeypatch.setenv("OPENWAM_VAE_FIRST", vae_first)
    bb = _make_backbone("vace")
    prompts = ["pick up", "close drawer"]

    monkeypatch.setenv("OPENWAM_TEXT_CACHE", "0")
    reference = _preprocess(bb, prompts)
    assert "_train_text_cache" not in bb.__dict__
    assert bb.text_encoder.calls == 1

    monkeypatch.setenv("OPENWAM_TEXT_CACHE", "1")
    for truncation in ("1", "0", "1"):  # first miss, then hits with either switch value
        monkeypatch.setenv("OPENWAM_CONTEXT_TRUNC", truncation)
        actual = _preprocess(bb, prompts)
        assert actual["context"].shape == (2, 512, 8)
        for key in ("context", "seq_lens", "input_latents", "vace_context"):
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        assert bb.text_encoder.calls == 2  # retain the cache's avoided encoder work

    cache = bb.__dict__["_train_text_cache"]
    assert (cache.misses, cache.hits, len(cache.rows)) == (1, 2, 2)
    assert cache.last_max_len == len(prompts[1])
    assert all(row.shape == (512, 8) for row in cache.rows.values())
    assert reference["seq_lens"].tolist() == [len(p) for p in prompts]


@pytest.mark.parametrize("variant", ["plain", "ti2v"])
@pytest.mark.parametrize(("prompt_length", "bucket_length"), [(7, 63), (70, 127)])
def test_masked_wan_variants_still_truncate_context(monkeypatch, variant, prompt_length, bucket_length):
    bb = _make_backbone(variant)
    prompts = ["a" * prompt_length, "pick"]
    reference = _preprocess(bb, prompts)
    assert reference["context"].shape == (2, 512, 8)

    monkeypatch.setenv("OPENWAM_CONTEXT_TRUNC", "1")
    actual = _preprocess(bb, prompts)
    assert actual["context"].shape == (2, bucket_length, 8)
    torch.testing.assert_close(actual["context"], reference["context"][:, :bucket_length], rtol=0, atol=0)
    torch.testing.assert_close(actual["seq_lens"], reference["seq_lens"], rtol=0, atol=0)
    assert actual["seq_lens"].tolist() == [prompt_length, 4]
    assert bb.text_encoder.calls == 1


def test_vace_hints_and_gradients_match_with_truncation_enabled(monkeypatch):
    bb = _make_backbone("vace")
    reference = _preprocess(bb, ["pick up"])
    monkeypatch.setenv("OPENWAM_CONTEXT_TRUNC", "1")
    actual = _preprocess(bb, ["pick up"])

    # Real, randomly initialized VACE/text projection/DiT modules: unlike zero
    # control weights, these expose changed padding contributions in the hints.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        vace = VaceWanModel(
            vace_layers=(0,), vace_in_dim=96, patch_size=(1, 1, 1), dim=8, num_heads=2, ffn_dim=16
        ).double()
        text_projection = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(8, 8)
        ).double()
        dit = DiTBlock(False, 8, 2, 16).double()
        x0 = torch.randn(1, 8, 8, dtype=torch.float64)
        time0 = torch.randn(1, 6, 8, dtype=torch.float64)
    freqs = torch.ones(8, 1, 2, dtype=torch.complex128)
    parameters = tuple(vace.parameters()) + tuple(text_projection.parameters()) + tuple(dit.parameters())

    def forward_backward(context):
        x = x0.clone().requires_grad_()
        time = time0.clone().requires_grad_()
        control = reference["vace_context"].clone().requires_grad_()
        projected = text_projection(context)
        hint = vace(x, control, projected, time, freqs)[0]
        context_mask = (torch.arange(context.shape[1]).unsqueeze(0) < reference["seq_lens"].unsqueeze(1))[
            :, None, None, :
        ]
        output = dit(x + hint, projected, time, freqs, context_mask=context_mask)
        loss = output.square().mean() + hint.square().mean()
        gradients = torch.autograd.grad(loss, (*parameters, x, time, control))
        assert all(torch.isfinite(g).all() for g in gradients)
        return hint, output, gradients

    ref_hint, ref_output, ref_gradients = forward_backward(reference["context"])
    hint, output, gradients = forward_backward(actual["context"])
    torch.testing.assert_close(hint, ref_hint, rtol=0, atol=0)
    torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
    for actual_grad, ref_grad in zip(gradients, ref_gradients):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=0, atol=0)

    # Sensitivity control: the old unconditional truncation changes both the
    # unmasked VACE hint and backward, despite the main DiT masking its padding.
    incorrectly_truncated = _truncate_padded_context(reference["context"], 7)
    assert incorrectly_truncated.shape[1] == 63
    bad_hint, _, bad_gradients = forward_backward(incorrectly_truncated)
    assert not torch.allclose(bad_hint, ref_hint, rtol=1e-7, atol=1e-9)
    assert any(not torch.allclose(a, b, rtol=1e-7, atol=1e-9) for a, b in zip(bad_gradients, ref_gradients))
