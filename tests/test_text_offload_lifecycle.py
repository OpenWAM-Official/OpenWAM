"""The frozen text encoder can be reused after cached training offloads it."""

import pytest
import torch

from openwam.model.video_backbone.wan import encode as wan_encode


class _Tokenizer:
    def __call__(self, prompts, *, return_mask, add_special_tokens, max_length, padding, truncation):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            tokens = [ord(c) % 63 + 1 for c in prompt][:max_length]
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        return ids, mask


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(64, 8)
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer("bias", torch.randn(8))
        self.fail = False
        self.calls = 0

    def forward(self, ids, mask):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected encoder failure")
        return self.proj(self.embed(ids)) + self.bias


def _encode(encoder, prompt, *, device="cpu", cache=None):
    return wan_encode.encode_text([prompt], tokenizer=_Tokenizer(), text_encoder=encoder, device=device, cache=cache)[0]


class _RecordingOffload:
    """CPU lifecycle probe; CUDA tests below exercise the real weight transfers."""

    def __init__(self, encoder, cache):
        self.events = []
        self.enabled = True
        self.cache = cache
        encoder._openwam_text_offload = self
        cache.offload = self

    def acquire(self, device):
        self.events.append("acquire")

    def release(self):
        self.events.append("release")

    def disable(self, device):
        self.events.append("disable")
        self.cache.clear()
        self.enabled = False


def test_uncached_entry_uses_encoder_lifecycle_and_releases_after_failure():
    encoder = _Encoder().eval().requires_grad_(False)
    cache = wan_encode.TrainTextEmbedCache(8)
    offload = _RecordingOffload(encoder, cache)
    _encode(encoder, "train", cache=cache)
    offload.events.clear()
    _encode(encoder, "infer")
    assert offload.events == ["acquire", "release"]
    offload.events.clear()
    encoder.fail = True
    with pytest.raises(RuntimeError, match="injected encoder failure"):
        _encode(encoder, "failure")
    assert offload.events == ["acquire", "release"]


@pytest.mark.parametrize("use_cache", [False, True])
def test_training_bypass_invalidates_cached_rows_and_disables_offload(use_cache):
    encoder = _Encoder().eval().requires_grad_(False)
    cache = wan_encode.TrainTextEmbedCache(8)
    offload = _RecordingOffload(encoder, cache)
    before = _encode(encoder, "same", cache=cache).clone()
    assert cache.rows
    encoder.train()
    _encode(encoder, "same", cache=cache if use_cache else None)
    assert not offload.enabled
    assert not cache.rows and cache.last_max_len is None
    with torch.no_grad():
        encoder.proj.bias.add_(1)
    encoder.eval()
    after = _encode(encoder, "same", cache=cache)
    assert not torch.equal(after, before)
    assert torch.equal(after, _encode(encoder, "same"))


def test_nonoffloaded_cache_is_invalidated_on_training_bypass():
    encoder = _Encoder().eval().requires_grad_(False)
    cache = wan_encode.TrainTextEmbedCache(8)
    _encode(encoder, "same", cache=cache)
    encoder.train()
    _encode(encoder, "same", cache=cache)
    assert not cache.rows and cache.last_max_len is None


def _assert_device(encoder, expected):
    assert all(t.device.type == expected for t in list(encoder.parameters()) + list(encoder.buffers()))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("miss_fast", ["0", "1"])
def test_real_offload_cached_inference_uncached_and_cached_again(monkeypatch, miss_fast):
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", miss_fast)
    torch.manual_seed(29)
    encoder = _Encoder().cuda().eval().requires_grad_(False)
    reference = {prompt: _encode(encoder, prompt, device="cuda") for prompt in ("train", "new prompt")}
    cache = wan_encode.TrainTextEmbedCache(8)
    cache.offload = wan_encode.TextEncoderOffload(encoder)
    actual = _encode(encoder, "train", device="cuda", cache=cache)
    torch.testing.assert_close(actual, reference["train"])
    _assert_device(encoder, "cpu")
    calls = encoder.calls
    actual = _encode(encoder, "train", device="cuda", cache=cache)
    assert encoder.calls == calls and cache.offload.loads == 0
    torch.testing.assert_close(actual, reference["train"])
    prompt_cache = {}
    for _ in range(2):
        actual, _ = wan_encode.encode_text_for_inference(
            "new prompt",
            vace_cache=None,
            prompt_embed_cache=prompt_cache,
            tokenizer=_Tokenizer(),
            text_encoder=encoder,
            device="cuda",
        )
        torch.testing.assert_close(actual, reference["new prompt"])
        _assert_device(encoder, "cpu")
        assert cache.offload.loads == 1
    actual = _encode(encoder, "new prompt", device="cuda")
    torch.testing.assert_close(actual, reference["new prompt"])
    _assert_device(encoder, "cpu")
    assert cache.offload.loads == 2
    calls = encoder.calls
    _encode(encoder, "train", device="cuda", cache=cache)
    assert encoder.calls == calls and cache.offload.loads == 2
    actual = _encode(encoder, "new prompt", device="cuda", cache=cache)
    torch.testing.assert_close(actual, reference["new prompt"])
    _assert_device(encoder, "cpu")
    assert cache.offload.loads == 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("use_cache", [False, True])
def test_real_offload_bypass_preserves_backward_updates_after_refreezing(monkeypatch, use_cache):
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", "0")
    torch.manual_seed(31)
    encoder = _Encoder().cuda().eval().requires_grad_(False)
    cache = wan_encode.TrainTextEmbedCache(8)
    cache.offload = wan_encode.TextEncoderOffload(encoder)
    before = _encode(encoder, "same", device="cuda", cache=cache).clone()
    _assert_device(encoder, "cpu")
    # Keep eval mode to test the independent requires_grad guard.
    encoder.requires_grad_(True)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    actual = _encode(encoder, "same", device="cuda", cache=cache if use_cache else None)
    _assert_device(encoder, "cuda")
    assert not cache.offload.enabled and cache.offload.host is None and not cache.offload.slots
    assert not cache.rows and cache.last_max_len is None
    actual.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = _encode(encoder, "same", device="cuda").detach()
    assert not torch.equal(before, expected)
    encoder.requires_grad_(False)
    actual = _encode(encoder, "same", device="cuda", cache=cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    calls, loads = encoder.calls, cache.offload.loads
    _encode(encoder, "same", device="cuda", cache=cache)
    assert encoder.calls == calls and cache.offload.loads == loads
    _assert_device(encoder, "cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fail_first", [False, True])
def test_real_offload_releases_after_encoder_exception_and_can_retry(monkeypatch, fail_first):
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", "0")
    encoder = _Encoder().cuda().eval().requires_grad_(False)
    cache = wan_encode.TrainTextEmbedCache(8)
    cache.offload = wan_encode.TextEncoderOffload(encoder)
    if not fail_first:
        _encode(encoder, "train", device="cuda", cache=cache)
    encoder.fail = True
    with pytest.raises(RuntimeError, match="injected encoder failure"):
        _encode(encoder, "failure", device="cuda", cache=cache)
    _assert_device(encoder, "cpu")
    assert not cache.offload.on_gpu
    encoder.fail = False
    actual = _encode(encoder, "failure", device="cuda", cache=cache)
    assert torch.isfinite(actual).all()
    _assert_device(encoder, "cpu")
