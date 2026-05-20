"""Unit tests for vace_cache and prompt_embed_cache in WanVideoBackbone.prepare_inputs_for_inference.

Tests verify:
1. Cold start: both caches miss, all units run.
2. Same prompt, same call (vace_cache hit): only non-text units run.
3. New prompt (never seen): both miss, all units run.
4. New prompt (seen before via prompt_embed_cache): vace miss + embed hit, only non-text units run.
5. Back to original prompt after vace_cache was overwritten: embed hit.
"""

from __future__ import annotations

from openwam.model.inference_inputs import InferenceInputs

# ---------------------------------------------------------------------------
# Minimal mock infrastructure
# ---------------------------------------------------------------------------


class _MockTextUnit:
    is_text_unit = True


class _MockObsUnit:
    pass


class _MockScheduler:
    def set_timesteps(self, num_inference_steps, shift):
        pass


class _MockPipe:
    """Minimal pipe stub that tracks which units were called per invocation."""

    def __init__(self):
        self.scheduler = _MockScheduler()
        self.text_unit = _MockTextUnit()
        self.obs_unit = _MockObsUnit()
        self.units = [self.text_unit, self.obs_unit]
        self.in_iteration_models = []
        self._calls: list[str] = []

    def unit_runner(self, unit, pipe, inputs_shared, inputs_posi, inputs_nega):
        if isinstance(unit, _MockTextUnit):
            self._calls.append("text")
            inputs_posi["_text_context"] = f"ctx_for_{inputs_posi.get('prompt', '?')}"
        else:
            self._calls.append("nontext")
        return inputs_shared, inputs_posi, inputs_nega

    def load_models_to_device(self, models):
        pass

    def reset_calls(self):
        self._calls.clear()


class _MockWanVB:
    """Minimal WanVideoBackbone-like object with prepare_inputs_for_inference."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._is_ti2v = False
        self._has_vace = False
        self._device = "cpu"
        self._dtype = None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def prepare_inputs_for_inference(self, inputs):
        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

        return WanVideoBackbone.prepare_inputs_for_inference(self, inputs)

    def _finalize_ti2v_inputs(self, inputs_shared, first_frame_image):
        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

        return WanVideoBackbone._finalize_ti2v_inputs(self, inputs_shared, first_frame_image)

    def _resolve_i2v_input_image(self, first_frame_image):
        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

        return WanVideoBackbone._resolve_i2v_input_image(self, first_frame_image)

    @staticmethod
    def _is_text_unit(unit):
        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

        return WanVideoBackbone._is_text_unit(unit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(pipe, prompt, vace_cache=None, prompt_embed_cache=None, seed=0):
    vb = _MockWanVB(pipe)
    return vb.prepare_inputs_for_inference(
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cold_start_runs_all_units():
    """No cache provided — all units execute."""
    pipe = _MockPipe()
    _prep(pipe, "prompt_A")
    assert pipe._calls == ["text", "nontext"], f"expected all units, got {pipe._calls}"


def test_vace_cache_hit_skips_text_unit():
    """After first call, same prompt → vace_cache hit → text unit skipped."""
    pipe = _MockPipe()
    vace_cache: dict = {}

    _prep(pipe, "prompt_A", vace_cache=vace_cache)
    assert vace_cache.get("populated")
    assert vace_cache.get("prompt_key") == "prompt_A"

    pipe.reset_calls()
    _prep(pipe, "prompt_A", vace_cache=vace_cache)
    assert "text" not in pipe._calls, f"text unit should be skipped on vace hit, got {pipe._calls}"
    assert "nontext" in pipe._calls


def test_vace_cache_prompt_change_causes_miss():
    """Prompt change → vace_cache key mismatch → miss."""
    pipe = _MockPipe()
    vace_cache: dict = {}

    _prep(pipe, "prompt_A", vace_cache=vace_cache)
    assert vace_cache["prompt_key"] == "prompt_A"

    pipe.reset_calls()
    _prep(pipe, "prompt_B", vace_cache=vace_cache)
    assert "text" in pipe._calls, f"expected text unit on prompt change, got {pipe._calls}"
    assert vace_cache["prompt_key"] == "prompt_B"


def test_prompt_embed_cache_hit_skips_text_unit():
    """After first call with prompt_A, second episode (vace_cache cleared) still
    skips text unit via prompt_embed_cache."""
    pipe = _MockPipe()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache

    vace_cache.clear()
    pipe.reset_calls()
    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)

    assert "text" not in pipe._calls, f"text unit should be skipped via prompt_embed_cache hit, got {pipe._calls}"
    assert "nontext" in pipe._calls


def test_prompt_embed_cache_miss_on_new_prompt():
    """First time seeing prompt_B: both caches miss, all units run."""
    pipe = _MockPipe()
    prompt_embed_cache: dict = {}

    _prep(pipe, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache

    pipe.reset_calls()
    _prep(pipe, "prompt_B", prompt_embed_cache=prompt_embed_cache)
    assert "text" in pipe._calls, f"expected full run for unseen prompt, got {pipe._calls}"
    assert "prompt_B" in prompt_embed_cache


def test_prompt_embed_cache_survives_vace_overwrite():
    """After prompt_A → prompt_B → prompt_A cycle: prompt_A embed is still cached."""
    pipe = _MockPipe()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    _prep(pipe, "prompt_B", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert vace_cache["prompt_key"] == "prompt_B"

    pipe.reset_calls()
    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert "text" not in pipe._calls, f"text unit should be skipped (prompt_A embed cached), got {pipe._calls}"
    assert vace_cache["prompt_key"] == "prompt_A"


def test_vace_cache_stores_prompt_key():
    """vace_cache must record prompt_key so it detects changes on next call."""
    pipe = _MockPipe()
    vace_cache: dict = {}

    _prep(pipe, "unique_prompt_xyz", vace_cache=vace_cache)
    assert vace_cache.get("prompt_key") == "unique_prompt_xyz"
    assert vace_cache.get("populated") is True


def test_both_caches_none_does_not_crash():
    """With no caches passed, function runs cleanly every time."""
    pipe = _MockPipe()
    for i in range(3):
        pipe.reset_calls()
        _prep(pipe, f"prompt_{i}")
        assert "text" in pipe._calls and "nontext" in pipe._calls


def _make_i2v_mock():
    """Mock that satisfies the helper's is_i2v predicate."""
    from types import SimpleNamespace

    mock_vb = _MockWanVB(_MockPipe())
    mock_vb._dit = SimpleNamespace(has_image_input=True)
    mock_vb._is_ti2v = False
    mock_vb._has_vace = False
    return mock_vb


def test_i2v_deploy_list_unwrap_cold_start():
    """Deploy passes first_frame_image=[PIL]; the cold-start branch must
    (a) write a single PIL (not a list) into inputs_shared["input_image"] and
    (b) null out vace_reference_image so the upstream NoiseInitializer /
    InputVideoEmbedder do NOT prepend a phantom latent frame (which would
    break the channel-cat with y inside prepare())."""
    from PIL import Image

    mock_vb = _make_i2v_mock()
    img = Image.new("RGB", (832, 480))
    inputs = mock_vb.prepare_inputs_for_inference(
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
        f"cold-start branch should unwrap list to single PIL, got {type(inputs['input_image'])}"
    )
    assert inputs["vace_reference_image"] is None, (
        f"I2V deploy must null vace_reference_image to avoid extra prefix frame, "
        f"got {type(inputs['vace_reference_image'])}"
    )


def test_i2v_deploy_list_unwrap_cache_hit():
    """Same as above but on the vace_cache-hit branch: the cached
    inputs_shared dict must be rewritten so a stale list/None does not
    survive across calls."""
    from PIL import Image

    mock_vb = _make_i2v_mock()
    img = Image.new("RGB", (832, 480))
    vace_cache: dict = {}

    # 1st call populates the cache.
    mock_vb.prepare_inputs_for_inference(
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
    assert vace_cache.get("populated"), "first call should populate vace_cache"

    # 2nd call must hit the cache and still unwrap the list + null vace ref.
    inputs = mock_vb.prepare_inputs_for_inference(
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
    assert isinstance(inputs["input_image"], Image.Image), (
        f"cache-hit branch should also unwrap to PIL, got {type(inputs['input_image'])}"
    )
    assert inputs["vace_reference_image"] is None, (
        f"I2V deploy cache-hit must null vace_reference_image, "
        f"got {type(inputs['vace_reference_image'])}"
    )
