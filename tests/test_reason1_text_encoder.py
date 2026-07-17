"""Unit tests for ``Reason1LiveTextEncoder`` and its module-level helpers.

Hits the pure-Python helpers with fakes — does NOT need the real Reason1-7B
weights. The GPU-gated smoke at the bottom exercises the real encoder on
hardware that has the bundle.
"""

from __future__ import annotations

import os
import types

import pytest
import torch

from openwam.model.video_backbone.cosmos_predict25 import text_encoder as te
from openwam.model.video_backbone.cosmos_predict25.text_encoder import Reason1LiveTextEncoder


def test_constants_match_upstream_geometry():
    assert te._NUM_EMBEDDING_PADDING_TOKENS == 512
    assert te._REASON1_HIDDEN_SIZE == 3584
    assert te._REASON1_NUM_TRANSFORMER_LAYERS == 28
    assert te._REASON1_FULL_CONCAT_DIM == 28 * 3584
    # Upstream system prompt — copied verbatim from cosmos text_encoder.py.
    assert "image generator" in te._COSMOS_REASON1_SYSTEM_PROMPT


def test_mean_normalize_zero_mean_unit_std():
    x = torch.randn(2, 4, 8)
    y = te._mean_normalize_along_last(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros_like(y.mean(dim=-1)), atol=1e-5)
    # std along last dim should be ~1 (up to eps).
    assert torch.allclose(y.std(dim=-1), torch.ones_like(y.std(dim=-1)), atol=1e-3)


def test_mean_normalize_handles_constant_input():
    # All-zero input: mean=0 exactly, std=0 exactly, so numerator is 0 and
    # the eps in the denominator keeps the result finite (=0). Guards
    # against accidentally producing NaN/Inf on degenerate hidden states.
    x = torch.zeros((1, 1, 8))
    y = te._mean_normalize_along_last(x)
    assert torch.isfinite(y).all()
    assert torch.equal(y, torch.zeros_like(y))


def test_tokenize_with_chat_template_pads_to_512():
    """Without real HF tokenizer, mock the apply_chat_template + tokenize
    behavior and verify the pad-or-truncate-to-512 semantics."""

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self, token_count: int):
            self._token_count = token_count

        def apply_chat_template(self, conversations, **kw):
            assert kw.get("tokenize") is False
            assert kw.get("add_generation_prompt") is False
            # The exact wrapped string doesn't matter — we don't pass it back through.
            user = conversations[1]["content"][0]["text"]
            return f"<sys>{conversations[0]['content'][0]['text']}<user>{user}<eos>"

        def __call__(self, text, **kw):
            return {"input_ids": torch.full((1, self._token_count), 7, dtype=torch.long)}

    # Short prompt → padding.
    short = FakeTokenizer(token_count=10)
    ids = te._tokenize_with_chat_template(short, "x")
    assert len(ids) == 512
    assert ids[:10] == [7] * 10
    assert ids[10:] == [short.pad_token_id] * (512 - 10)

    # Long prompt → truncation.
    long = FakeTokenizer(token_count=1000)
    ids = te._tokenize_with_chat_template(long, "long input")
    assert len(ids) == 512
    assert all(t == 7 for t in ids)


def _fake_model(hidden_size: int, num_layers: int):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_layers)
        )
    )


def test_validate_geometry_accepts_nested_text_config():
    """``hidden_size`` / ``num_hidden_layers`` must be read via
    ``config.text_config`` for transformers ≥5 ``Qwen2_5_VLConfig``, which no
    longer exposes them on the top-level config."""
    Reason1LiveTextEncoder._validate_geometry(_fake_model(3584, 28))


def test_validate_geometry_rejects_wrong_hidden_size():
    with pytest.raises(ValueError, match="hidden_size"):
        Reason1LiveTextEncoder._validate_geometry(_fake_model(1024, 28))


def test_validate_geometry_rejects_wrong_layer_count():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        Reason1LiveTextEncoder._validate_geometry(_fake_model(3584, 32))


# ----------------------------------------------------------------------
# Reason1 real-load smoke — only runs when Reason1 weights AND CUDA exist.
# ----------------------------------------------------------------------

REASON1_ASSET_PATH = os.environ.get("REASON1_ASSET_PATH", "/path/to/assets/Cosmos-Reason1-7B")


@pytest.mark.gpu
@pytest.mark.skipif(
    not (os.path.isdir(REASON1_ASSET_PATH) and torch.cuda.is_available()),
    reason="needs Cosmos-Reason1-7B weights + CUDA",
)
def test_real_reason1_encode_smoke():
    """Run one prompt through the real Reason1-7B live encoder and assert the geometry."""
    enc = Reason1LiveTextEncoder(REASON1_ASSET_PATH, dtype=torch.bfloat16, device="cuda:0")
    out = enc("pick up the block")
    assert out.shape == (1, 512, 100352)
    assert torch.isfinite(out.float()).all()
