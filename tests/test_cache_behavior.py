"""Unit tests for vace_cache and prompt_embed_cache in prepare_pipeline_inputs.

Tests verify:
1. Cold start: both caches miss, all units run.
2. Same prompt, same call (vace_cache hit): only non-text units run.
3. New prompt (never seen): both miss, all units run.
4. New prompt (seen before via prompt_embed_cache): vace miss + embed hit, only non-text units run.
5. Back to original prompt after vace_cache was overwritten: embed hit.
"""

from __future__ import annotations

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
        self._calls: list[str] = []  # "text" or "nontext" per unit_runner call

    def unit_runner(self, unit, pipe, inputs_shared, inputs_posi, inputs_nega):
        if isinstance(unit, _MockTextUnit):
            self._calls.append("text")
            # Simulate text encoder writing context tensors into inputs_posi
            inputs_posi["_text_context"] = f"ctx_for_{inputs_posi.get('prompt', '?')}"
        else:
            self._calls.append("nontext")
        return inputs_shared, inputs_posi, inputs_nega

    def reset_calls(self):
        self._calls.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(pipe, prompt, vace_cache=None, prompt_embed_cache=None, seed=0):
    from openwam.deployment.joint_generation import prepare_pipeline_inputs

    return prepare_pipeline_inputs(
        pipe=pipe,
        prompt=prompt,
        negative_prompt="",
        vace_video=None,
        first_frame_image=None,
        num_frames=17,
        height=32,
        width=32,
        seed=seed,
        cfg_scale=1.0,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
        prompt_embed_cache=prompt_embed_cache,
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
    assert vace_cache.get("prompt_key") == ("prompt_A", "")

    pipe.reset_calls()
    _prep(pipe, "prompt_A", vace_cache=vace_cache)
    assert "text" not in pipe._calls, f"text unit should be skipped on vace hit, got {pipe._calls}"
    assert "nontext" in pipe._calls


def test_vace_cache_prompt_change_causes_miss():
    """Prompt change → vace_cache key mismatch → miss."""
    pipe = _MockPipe()
    vace_cache: dict = {}

    _prep(pipe, "prompt_A", vace_cache=vace_cache)
    assert vace_cache["prompt_key"] == ("prompt_A", "")

    pipe.reset_calls()
    _prep(pipe, "prompt_B", vace_cache=vace_cache)
    # Full run expected: text unit must have been called
    assert "text" in pipe._calls, f"expected text unit on prompt change, got {pipe._calls}"
    # vace_cache should now hold prompt_B
    assert vace_cache["prompt_key"] == ("prompt_B", "")


def test_prompt_embed_cache_hit_skips_text_unit():
    """After first call with prompt_A, second episode (vace_cache cleared) still
    skips text unit via prompt_embed_cache."""
    pipe = _MockPipe()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    # Episode 1: cold start — populates both caches
    inputs_shared, inputs_posi, _ = _prep(
        pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache
    )
    assert ("prompt_A", "") in prompt_embed_cache
    assert "_text_context" in prompt_embed_cache[("prompt_A", "")][0]

    # Episode 2: clear vace_cache only (simulates reset path)
    vace_cache.clear()
    pipe.reset_calls()
    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)

    assert "text" not in pipe._calls, f"text unit should be skipped via prompt_embed_cache hit, got {pipe._calls}"
    assert "nontext" in pipe._calls


def test_prompt_embed_cache_miss_on_new_prompt():
    """First time seeing prompt_B: both caches miss, all units run."""
    pipe = _MockPipe()
    prompt_embed_cache: dict = {}

    # Warm up with prompt_A
    _prep(pipe, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert ("prompt_A", "") in prompt_embed_cache

    # First call with brand-new prompt_B
    pipe.reset_calls()
    _prep(pipe, "prompt_B", prompt_embed_cache=prompt_embed_cache)
    assert "text" in pipe._calls, f"expected full run for unseen prompt, got {pipe._calls}"
    assert ("prompt_B", "") in prompt_embed_cache


def test_prompt_embed_cache_survives_vace_overwrite():
    """After prompt_A → prompt_B → prompt_A cycle: prompt_A embed is still cached."""
    pipe = _MockPipe()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    # prompt_A first
    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    # prompt_B (overwrites vace_cache)
    _prep(pipe, "prompt_B", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert vace_cache["prompt_key"] == ("prompt_B", "")

    # Back to prompt_A — vace miss but embed hit
    pipe.reset_calls()
    _prep(pipe, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert "text" not in pipe._calls, f"text unit should be skipped (prompt_A embed cached), got {pipe._calls}"
    assert vace_cache["prompt_key"] == ("prompt_A", "")


def test_vace_cache_stores_prompt_key():
    """vace_cache must record prompt_key so it detects changes on next call."""
    pipe = _MockPipe()
    vace_cache: dict = {}

    _prep(pipe, "unique_prompt_xyz", vace_cache=vace_cache)
    assert vace_cache.get("prompt_key") == ("unique_prompt_xyz", "")
    assert vace_cache.get("populated") is True


def test_both_caches_none_does_not_crash():
    """With no caches passed, function runs cleanly every time."""
    pipe = _MockPipe()
    for i in range(3):
        pipe.reset_calls()
        _prep(pipe, f"prompt_{i}")
        assert "text" in pipe._calls and "nontext" in pipe._calls
