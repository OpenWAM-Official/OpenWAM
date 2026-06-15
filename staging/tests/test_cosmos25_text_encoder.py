"""CPU + GPU tests for :class:`Reason1LiveTextEncoder`.

CPU tests use a stand-in ``Qwen2_5_VLForConditionalGeneration`` + ``AutoTokenizer``
injected via ``monkeypatch`` so we exercise the wiring (geometry validation,
batched tokenize, hidden-state slicing, ``mean_normalize``, concat → 100352)
without loading the real 14 GB Reason1-7B weights.

The single GPU smoke at the bottom (``test_real_reason1_live_smoke``) gates
on the Cosmos-Reason1-7B asset being on disk and CUDA being available — same
pattern as ``tests/test_reason1_embedding_precompute.py::test_real_reason1_encode_smoke``.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# transformers is a base dep (used by the offline `reason1_embedding_computation`
# path) but the Qwen-VL classes may not be importable in every environment.
# Skip the whole CPU block if transformers is missing rather than failing hard.
transformers = pytest.importorskip("transformers", reason="transformers required for Reason1 tests")


_HIDDEN = 3584
_NUM_LAYERS = 28
_L_PAD = 512


def _fake_hidden_states(B: int, L: int, dtype: torch.dtype, device: torch.device) -> tuple:
    """29 tensors of shape (B, L, 3584): 1 embed + 28 transformer layers."""
    return tuple(torch.zeros(B, L, _HIDDEN, dtype=dtype, device=device) for _ in range(_NUM_LAYERS + 1))


class _FakeTokenizer:
    """Drop-in for ``transformers.AutoTokenizer``.

    Returns exactly ``token_count`` tokens (independent of input text) so the
    encoder's pad/truncate logic can be deterministically exercised.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, token_count: int = 32):
        self._token_count = token_count

    def apply_chat_template(self, conversations, **kw):
        user = conversations[1]["content"][0]["text"]
        return f"<sys>...<user>{user}<eos>"

    def __call__(self, text, **kw):
        return {"input_ids": torch.full((1, self._token_count), 7, dtype=torch.long)}


class _FakeReason1Model:
    """Drop-in for ``Qwen2_5_VLForConditionalGeneration``.

    Mimics enough of the surface (``config``, ``parameters``, ``eval``, ``to``,
    ``__call__``) for :class:`Reason1LiveTextEncoder` to drive it through one
    batched forward pass and return 29 hidden states.

    ``nested_config=True`` puts ``hidden_size`` / ``num_hidden_layers`` under
    ``config.text_config`` (no top-level attrs), matching real
    transformers ≥5 ``Qwen2_5_VLConfig`` layout. ``nested_config=False`` keeps
    them at the top level for the older transformers layout.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        hidden_size: int = _HIDDEN,
        num_hidden_layers: int = _NUM_LAYERS,
        nested_config: bool = False,
    ):
        if nested_config:
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
            )
        else:
            self.config = types.SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        # A single trainable-then-frozen param so `.parameters()` is non-empty
        # and device/dtype probing works.
        self._param = nn.Parameter(torch.empty(1, dtype=dtype, device=device), requires_grad=True)

    def parameters(self):
        yield self._param

    def eval(self):
        return self

    def to(self, **kw):
        if "device" in kw or "dtype" in kw:
            self._param.data = self._param.data.to(**kw)
        return self

    def __call__(self, *, input_ids, attention_mask, output_hidden_states, use_cache):
        # Echo whatever device/dtype the param landed on; this matches the
        # behaviour of a real frozen Qwen-VL when bf16 weights live on CUDA.
        assert output_hidden_states is True
        assert use_cache is False
        B, L = input_ids.shape
        hs = _fake_hidden_states(B, L, dtype=self._param.dtype, device=self._param.device)
        return types.SimpleNamespace(hidden_states=hs)


@pytest.fixture
def patched_transformers(monkeypatch, tmp_path):
    """Patch transformers' AutoTokenizer + Qwen2_5_VLForConditionalGeneration so
    Reason1LiveTextEncoder's `from transformers import ...` inside __init__
    picks up our fakes."""
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *_args, **_kw: _FakeTokenizer()),
    )

    def _from_pretrained(*_args, **kw):
        dtype = kw.get("torch_dtype", torch.float32)
        device_map = kw.get("device_map") or "cpu"
        if isinstance(device_map, dict):
            device = next(iter(device_map.values()))
        else:
            device = device_map
        device = torch.device(device) if not isinstance(device, torch.device) else device
        return _FakeReason1Model(dtype=dtype, device=device)

    monkeypatch.setattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
        types.SimpleNamespace(from_pretrained=_from_pretrained),
    )

    # Reason1LiveTextEncoder requires the ckpt dir to exist on disk.
    ckpt = tmp_path / "Cosmos-Reason1-7B"
    ckpt.mkdir()
    return ckpt


def test_reason1_live_encoder_returns_correct_shape(patched_transformers):
    """Batched forward returns (B, 512, 100352) — the pre-projection geometry
    the wrapper's auto-gate expects when D == crossattn_proj_in_channels."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    te = Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")
    out = te(["pick up the block", "fold the towel"])
    assert out.shape == (2, _L_PAD, _NUM_LAYERS * _HIDDEN)  # (2, 512, 100352)
    assert torch.isfinite(out).all()


def test_reason1_live_encoder_accepts_single_string(patched_transformers):
    """A bare str input is coerced to a 1-element batch."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    te = Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")
    out = te("pick up the block")
    assert out.shape == (1, _L_PAD, _NUM_LAYERS * _HIDDEN)


def test_reason1_live_encoder_is_not_nn_module(patched_transformers):
    """Pin the state_dict invariant — the encoder is intentionally NOT an
    nn.Module so its 14 GB of weights do not bleed into the wrapper's
    state_dict (mirrors `Wan2pt1VAEInterface`; see docs §14)."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    te = Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")
    assert not isinstance(te, nn.Module), (
        "Reason1LiveTextEncoder must remain a plain Python class. Wrapping it "
        "in nn.Module would register 14 GB of Reason1 weights into "
        "Cosmos25PipelineWrapper.state_dict() — see text_encoder.py docstring."
    )


def test_reason1_live_encoder_frozen(patched_transformers):
    """The inner model must be `requires_grad=False` after construction —
    matches the offline-path freeze contract."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    te = Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")
    for p in te.model.parameters():
        assert not p.requires_grad, "Reason1 model parameters must be frozen at construction."


def test_reason1_live_encoder_to_updates_state(patched_transformers):
    """`to()` updates both the cached `dtype` / `device` attrs and the inner
    model — exercised by `adapter._move_cosmos_reason1` during
    `set_dtype_device` so the encoder follows the rest of the pipeline."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    te = Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")
    te.to(dtype=torch.bfloat16, device="cpu")
    assert te.dtype == torch.bfloat16
    assert te.device == torch.device("cpu")
    assert next(te.model.parameters()).dtype == torch.bfloat16


def test_reason1_live_encoder_from_empty_registers_meta_shell(monkeypatch, tmp_path):
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    ckpt = tmp_path / "reason1"
    ckpt.mkdir()

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *_args, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kw: types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=_HIDDEN, num_hidden_layers=_NUM_LAYERS)
            )
        ),
    )
    monkeypatch.setattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
        types.SimpleNamespace(
            _from_config=lambda *_args, **_kw: _FakeReason1Model(
                dtype=torch.float32, device=torch.device("meta"), nested_config=True
            )
        ),
    )

    te = Reason1LiveTextEncoder.from_empty(ckpt, dtype=torch.float32)

    assert te.device == torch.device("meta")
    assert next(te.model.parameters()).device.type == "meta"
    assert te.tokenizer is not None


def test_reason1_live_encoder_geometry_validation(monkeypatch, patched_transformers):
    """A Reason1 variant with wrong `hidden_size` must be rejected at load
    time, mirroring the offline path's `_build_reason1` geometry check."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    def _bad_from_pretrained(*_args, **kw):
        dtype = kw.get("torch_dtype", torch.float32)
        return _FakeReason1Model(dtype=dtype, device=torch.device("cpu"), hidden_size=4096)

    monkeypatch.setattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
        types.SimpleNamespace(from_pretrained=_bad_from_pretrained),
    )
    with pytest.raises(ValueError, match="hidden_size"):
        Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")


def test_reason1_live_encoder_accepts_nested_text_config(monkeypatch, patched_transformers):
    """Real transformers ≥5 ``Qwen2_5_VLConfig`` exposes ``hidden_size`` only
    on ``config.text_config`` (no top-level attr) — the geometry check must
    follow that path. Regression for the previously-broken access pattern
    that read directly from ``model.config.hidden_size``."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    def _nested_from_pretrained(*_args, **kw):
        dtype = kw.get("torch_dtype", torch.float32)
        return _FakeReason1Model(dtype=dtype, device=torch.device("cpu"), nested_config=True)

    monkeypatch.setattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
        types.SimpleNamespace(from_pretrained=_nested_from_pretrained),
    )
    # Must construct without raising — regression for the AttributeError on
    # real Reason1 loads under transformers ≥5.
    te = Reason1LiveTextEncoder(patched_transformers, dtype=torch.float32, device="cpu")
    out = te("pick up the block")
    assert out.shape == (1, _L_PAD, _NUM_LAYERS * _HIDDEN)


def test_reason1_live_encoder_missing_ckpt_raises(tmp_path):
    """Loading from a non-existent path raises a clear FileNotFoundError
    before any transformers import is attempted."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    with pytest.raises(FileNotFoundError, match="Cosmos-Reason1"):
        Reason1LiveTextEncoder(tmp_path / "does-not-exist", dtype=torch.float32, device="cpu")


# ----------------------------------------------------------------------
# GPU smoke — runs only when the real Reason1-7B weights are on disk.
# ----------------------------------------------------------------------


_REASON1_PATH = "/path/to/assets/Cosmos-Reason1-7B"


@pytest.mark.gpu
@pytest.mark.skipif(
    not (os.path.isdir(_REASON1_PATH) and torch.cuda.is_available()),
    reason="needs Cosmos-Reason1-7B weights + CUDA",
)
def test_real_reason1_live_smoke():
    """Real-weights smoke: load Reason1-7B, encode one prompt, assert the
    geometry that the wrapper auto-gate expects (pre-projection 100352)."""
    from openwam.model.video_backbone.cosmos25.text_encoder import Reason1LiveTextEncoder

    te = Reason1LiveTextEncoder(Path(_REASON1_PATH), dtype=torch.bfloat16, device="cuda:0")
    out = te("pick up the block")
    assert out.shape == (1, _L_PAD, _NUM_LAYERS * _HIDDEN)  # (1, 512, 100352)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
