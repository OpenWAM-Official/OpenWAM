"""Unit tests for vace_cache / prompt_embed_cache in
WanVideoBackbone.preprocess_input_for_inference.

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

import torch

from openwam.model.inference_inputs import InferenceInputs
from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

# ---------------------------------------------------------------------------
# Minimal mock infrastructure
# ---------------------------------------------------------------------------


class _MockScheduler:
    def set_timesteps(self, num_inference_steps, shift):
        pass


class _MockPipe:
    def __init__(self):
        self.scheduler = _MockScheduler()


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
        self._dtype = None
        self._encode_text_calls: list[list] = []

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    # --- stubbed encode seams (no real weights / GPU) ---
    def _encode_text(self, prompts):
        self._encode_text_calls.append(list(prompts))
        return torch.zeros(1, 4, 8), torch.ones(1, dtype=torch.long)

    def _check_resize(self, h, w, num_frames):
        return h, w, num_frames

    def _build_deploy_noise(self, *, height, width, num_frames, seed, rand_device):
        return torch.zeros(1, 4, 1, height // 8, width // 8)

    def _build_deploy_i2v_clip(self, input_image, *, height, width):
        return torch.zeros(1, 1, 8)

    def _build_deploy_i2v_y(self, input_image, *, num_frames, height, width, tiled, tile_size, tile_stride):
        return torch.zeros(1, 20, 1, height // 8, width // 8)

    # --- real methods under test ---
    def preprocess_input_for_inference(self, inputs):
        return WanVideoBackbone.preprocess_input_for_inference(self, inputs)

    def _encode_text_for_inference(self, prompt, *, vace_cache, prompt_embed_cache):
        return WanVideoBackbone._encode_text_for_inference(
            self, prompt, vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache
        )

    def _resolve_i2v_input_image(self, first_frame_image):
        return WanVideoBackbone._resolve_i2v_input_image(self, first_frame_image)

    def _finalize_ti2v_first_frame_latents(self, inputs_shared, first_frame_image):
        return WanVideoBackbone._finalize_ti2v_first_frame_latents(self, inputs_shared, first_frame_image)

    def _build_vace_context_for_deploy(self, inputs_shared, first_frame_image, vace_video):
        # _has_vace=False → real method is a no-op fast path.
        return WanVideoBackbone._build_vace_context_for_deploy(self, inputs_shared, first_frame_image, vace_video)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(vb, prompt, vace_cache=None, prompt_embed_cache=None, seed=0):
    return vb.preprocess_input_for_inference(
        InferenceInputs(
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
    """Mock that satisfies the I2V predicate of ``_resolve_i2v_input_image``."""
    from types import SimpleNamespace

    mock_vb = _MockWanVB()
    mock_vb._dit = SimpleNamespace(has_image_input=True)
    mock_vb._is_ti2v = False
    mock_vb._has_vace = False
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
        InferenceInputs(
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
        InferenceInputs(
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
    )
    assert vace_cache.get("populated")

    inputs = mock_vb.preprocess_input_for_inference(
        InferenceInputs(
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
    )
    assert isinstance(inputs["input_image"], Image.Image)
    assert inputs["vace_reference_image"] is None
    assert inputs.get("clip_feature") is not None
    assert inputs.get("y") is not None
