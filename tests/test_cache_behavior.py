"""Unit tests for vace_cache / prompt_embed_cache in
WanBase.preprocess_input_for_inference.

The deploy method builds every conditioning signal with explicit helpers (no
WanVideoPipeline unit-runner). The only cached quantity is the text embedding
``(context, seq_lens)``, produced by ``_encode_text`` and reused via either
cache. These tests spy on ``_encode_text`` to verify:

1. Cold start: both caches miss → text is encoded.
2. Same prompt (vace_cache hit): text encode skipped.
3. New prompt (vace_cache key mismatch): text re-encoded.
4. Seen prompt via prompt_embed_cache (vace cleared): text encode skipped.
5. Unseen prompt: both miss → text encoded.
6. prompt_A → prompt_B → prompt_A: prompt_A embed still cached.
7. vace_cache records prompt_key + populated.
8. No caches: runs cleanly every call.

Plus the I2V deploy first-frame unwrap path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openwam.model.video_backbone.wan_backbone import WanBase

# ---------------------------------------------------------------------------
# Minimal mock infrastructure
# ---------------------------------------------------------------------------


class _MockScheduler:
    def set_timesteps(self, num_inference_steps, shift):
        pass


class _MockPipe:
    def __init__(self):
        self.scheduler = _MockScheduler()


class _MockTokenizer:
    def __call__(self, prompts, **kw):
        n = len(prompts)
        return torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4, dtype=torch.long)


class _MockTextEncoder:
    """Records each call so a test can count how often text was encoded."""

    def __init__(self, calls):
        self._calls = calls

    def __call__(self, ids, mask):
        self._calls.append(list(range(ids.shape[0])))
        return torch.zeros(ids.shape[0], 4, 8)


class _MockWanVB:
    """WanVideoBackbone-like object exercising the deploy cache seam.

    The encode seams (text / noise / I2V clip+y) are stubbed so the test needs
    no real weights or GPU; ``_encode_text`` records its calls so a test can
    assert whether the text path was hit or served from a cache.
    """

    def __init__(self, pipe=None):
        self._pipe = pipe if pipe is not None else _MockPipe()
        self._is_ti2v = False
        self._has_vace = False
        self._device = "cpu"
        self._dtype = torch.float32
        self._encode_text_calls: list[list] = []
        # Encode + conditioning seams now live in wan.encode / wan.conditioning
        # free functions; provide the state they read explicitly.
        self._tokenizer = _MockTokenizer()
        self.text_encoder = _MockTextEncoder(self._encode_text_calls)
        self._height_division_factor = 16
        self._width_division_factor = 16
        self._time_division_factor = 4
        self._time_division_remainder = 1
        self._latent_spec = SimpleNamespace(
            z_dim=4, spatial_compression=8, temporal_compression=4, causal_temporal=True
        )
        self._dit = SimpleNamespace(has_image_input=False)
        self.video_encoder = None
        # preprocess passes vae= unconditionally, but build_deploy_noise only
        # dereferences it when _latent_spec is None (never here). The I2V tests
        # override this with a real encode stub via _make_i2v_mock.
        self.vae = None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def scheduler(self):
        return self._pipe.scheduler

    @property
    def dit(self):
        # backbone build_deploy_i2v_clip/y read self.dit; mirror self._dit.
        return self._dit

    # --- real method under test (conditioning seams now in wan.conditioning) ---
    def preprocess_input_for_inference(self, **kw):
        return WanBase.preprocess_input_for_inference(self, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(vb, prompt, vace_cache=None, prompt_embed_cache=None, seed=0):
    return vb.preprocess_input_for_inference(
        prompt=prompt,
        vace_video=None,
        first_frame_image=None,
        num_frames=17,
        height=32,
        width=32,
        seed=seed,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
        prompt_embed_cache=prompt_embed_cache,
    )


def _text_calls(vb) -> int:
    return len(vb._encode_text_calls)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cold_start_encodes_text():
    """No cache provided — text is encoded."""
    vb = _MockWanVB()
    _prep(vb, "prompt_A")
    assert _text_calls(vb) == 1


def test_vace_cache_hit_skips_text_encode():
    """After first call, same prompt → vace_cache hit → text encode skipped."""
    vb = _MockWanVB()
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache)
    assert vace_cache.get("populated")
    assert vace_cache.get("prompt_key") == "prompt_A"
    assert _text_calls(vb) == 1

    _prep(vb, "prompt_A", vace_cache=vace_cache)
    assert _text_calls(vb) == 1, "text should be served from vace_cache on a hit"


def test_vace_cache_prompt_change_causes_miss():
    """Prompt change → vace_cache key mismatch → text re-encoded."""
    vb = _MockWanVB()
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache)
    assert vace_cache["prompt_key"] == "prompt_A"

    _prep(vb, "prompt_B", vace_cache=vace_cache)
    assert _text_calls(vb) == 2, "expected text re-encode on prompt change"
    assert vace_cache["prompt_key"] == "prompt_B"


def test_prompt_embed_cache_hit_skips_text_encode():
    """After first call with prompt_A, a second episode (vace_cache cleared)
    still skips text encode via prompt_embed_cache."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache
    assert _text_calls(vb) == 1

    vace_cache.clear()
    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == 1, "text should be served from prompt_embed_cache"


def test_prompt_embed_cache_miss_on_new_prompt():
    """First time seeing prompt_B: both caches miss → text encoded."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache

    _prep(vb, "prompt_B", prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == 2, "expected text encode for an unseen prompt"
    assert "prompt_B" in prompt_embed_cache


def test_prompt_embed_cache_survives_vace_overwrite():
    """After prompt_A → prompt_B → prompt_A: prompt_A embed is still cached."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    _prep(vb, "prompt_B", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert vace_cache["prompt_key"] == "prompt_B"
    calls_before = _text_calls(vb)

    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == calls_before, "prompt_A embed should still be cached"
    assert vace_cache["prompt_key"] == "prompt_A"


def test_vace_cache_stores_prompt_key():
    """vace_cache must record prompt_key + the cached embedding."""
    vb = _MockWanVB()
    vace_cache: dict = {}

    _prep(vb, "unique_prompt_xyz", vace_cache=vace_cache)
    assert vace_cache.get("prompt_key") == "unique_prompt_xyz"
    assert vace_cache.get("populated") is True
    assert vace_cache.get("context") is not None
    assert vace_cache.get("seq_lens") is not None


def test_both_caches_none_does_not_crash():
    """With no caches, the function encodes text every call."""
    vb = _MockWanVB()
    for i in range(3):
        _prep(vb, f"prompt_{i}")
    assert _text_calls(vb) == 3


def _make_i2v_mock():
    """Mock satisfying the I2V predicate of ``resolve_i2v_input_image`` plus the
    clip+y conditioning builders (image_encoder + vae stubs)."""
    mock_vb = _MockWanVB()
    mock_vb._dit = SimpleNamespace(has_image_input=True, require_clip_embedding=True, require_vae_embedding=True)
    mock_vb._is_ti2v = False
    mock_vb._has_vace = False
    mock_vb.image_encoder = SimpleNamespace(encode_image=lambda imgs: torch.zeros(1, 1, 8))
    mock_vb.vae = SimpleNamespace(encode=lambda vids, device, **kw: [torch.zeros(16, 5, 4, 4)])
    return mock_vb


def test_i2v_deploy_list_unwrap():
    """Deploy passes first_frame_image=[PIL]; the I2V path must write a single
    PIL (not a list) into inputs_shared["input_image"], keep
    vace_reference_image None (so no phantom prefix latent), and populate the
    clip_feature + y conditioning slots."""
    from PIL import Image

    mock_vb = _make_i2v_mock()
    img = Image.new("RGB", (320, 384))
    inputs = mock_vb.preprocess_input_for_inference(
        prompt="prompt_I2V",
        first_frame_image=[img],
        num_frames=17,
        height=32,
        width=32,
        seed=0,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
    )
    assert isinstance(inputs["input_image"], Image.Image), (
        f"I2V deploy should unwrap [PIL] to a single PIL, got {type(inputs['input_image'])}"
    )
    assert inputs["vace_reference_image"] is None
    assert inputs.get("clip_feature") is not None
    assert inputs.get("y") is not None


def test_i2v_deploy_unwrap_stable_across_cache():
    """Across a vace_cache-populated call, the I2V unwrap + conditioning are
    rebuilt identically (only the text embed is cached)."""
    from PIL import Image

    mock_vb = _make_i2v_mock()
    img = Image.new("RGB", (320, 384))
    vace_cache: dict = {}

    mock_vb.preprocess_input_for_inference(
        prompt="prompt_I2V",
        first_frame_image=[img],
        num_frames=17,
        height=32,
        width=32,
        seed=0,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
    )
    assert vace_cache.get("populated")

    inputs = mock_vb.preprocess_input_for_inference(
        prompt="prompt_I2V",
        first_frame_image=[img],
        num_frames=17,
        height=32,
        width=32,
        seed=1,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
    )
    assert isinstance(inputs["input_image"], Image.Image)
    assert inputs["vace_reference_image"] is None
    assert inputs.get("clip_feature") is not None
    assert inputs.get("y") is not None
