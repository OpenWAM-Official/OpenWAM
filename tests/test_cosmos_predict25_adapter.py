"""Cosmos-Predict2.5 adapter: capability flags + unsupported-path errors.

Uses a tiny fake pipeline so the adapter can be exercised without the
upstream ``cosmos_predict2`` package being installed.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.cosmos_predict25 import CosmosFlowSchedulerAdapter
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone


class _FakeCosmosPipeline:
    """Bare-minimum stand-in for a real cosmos_predict2 pipeline.

    The adapter only touches ``dim``/``num_layers``/``num_heads``/``head_dim``
    /``context_dim`` for its property surface; the block-loop methods are not
    invoked by these tests.
    """

    dim = 1280
    num_layers = 24
    num_heads = 10
    head_dim = 128
    context_dim = 2048


def _build_backbone(**overrides) -> CosmosPredict25VideoBackbone:
    pipe = _FakeCosmosPipeline()
    kwargs = dict(
        net=pipe,
        vae=None,
        text_encoder=None,
        dim=pipe.dim,
        num_layers=pipe.num_layers,
        num_heads=pipe.num_heads,
        head_dim=pipe.head_dim,
        context_dim=pipe.context_dim,
        scheduler=CosmosFlowSchedulerAdapter(shift_video=3.0),
        freeze=True,
    )
    kwargs.update(overrides)
    return CosmosPredict25VideoBackbone(**kwargs)


def test_properties_match_pipeline_geometry():
    bb = _build_backbone()
    assert bb.dim == 1280
    assert bb.num_layers == 24
    assert bb.num_heads == 10
    assert bb.head_dim == 128
    assert bb.text_dim == 2048
    assert isinstance(bb.scheduler, CosmosFlowSchedulerAdapter)


def test_self_attn_paths_raise_with_clear_message():
    bb = _build_backbone()
    dummy_state = BlockLoopState(
        hidden_states=torch.zeros(1, 4, bb.dim),
        time_mod=torch.zeros(1, 6, bb.dim),
        rope_freqs=torch.zeros(4, bb.head_dim, dtype=torch.complex64),
        context=torch.zeros(1, 1, bb.text_dim),
    )
    with pytest.raises(NotImplementedError, match="pre_attn_at_layer"):
        bb.pre_attn_at_layer(0, dummy_state)
    with pytest.raises(NotImplementedError, match="post_attn_at_layer"):
        bb.post_attn_at_layer(0, dummy_state, attn_out=torch.zeros(1, 4, bb.dim), post_state={})


def test_single_system_now_supported():
    """Cosmos now supports single-system (via the 3D block forward).

    The inject/extract methods are overridden (no longer the raising ABC
    default) and ``assert_ready_for_shared_tokens`` is a no-op (Cosmos carries
    per-token modulation in ``extras``, not ``time_mod``). The full inject →
    run → extract round-trip is covered in
    ``test_cosmos_predict25_single_system.py`` (needs a runnable DiT)."""
    from openwam.model.video_backbone.base import VideoBackbone

    bb = _build_backbone()
    assert type(bb).inject_shared_tokens is not VideoBackbone.inject_shared_tokens
    assert type(bb).extract_shared_tokens is not VideoBackbone.extract_shared_tokens
    # No-op readiness check: must not raise the Wan 4D-time_mod requirement.
    bb.assert_ready_for_shared_tokens(
        BlockLoopState(
            hidden_states=torch.zeros(1, 1, 1, 1, bb.dim),
            time_mod=torch.zeros(()),
            rope_freqs=torch.zeros(()),
            context=torch.zeros(1, 1, bb.text_dim),
        )
    )


def test_vace_rejected_for_mvp():
    bb = _build_backbone()
    with pytest.raises(NotImplementedError, match="VACE"):
        bb.preprocess_input_for_train(frames=[], text=[], vace_videos=[object()])


def test_ref_images_forwarded_to_wrapper():
    """``ref_images`` is now passed through to the wrapper for TI2V.

    The adapter previously stripped this field; with the TI2V path live,
    it must forward ``ref_images`` to the wrapper's ``preprocess_input``.
    The fake pipeline has no ``preprocess_input`` method, so the adapter
    raises a *generic* NotImplementedError about the missing method — what
    we assert here is that no rejection about reference-image/first_frame
    fires from the adapter itself (the old gate is gone).
    """
    bb = _build_backbone()
    import pytest

    # No VAE configured → generic VAE error; the point is no reference-image/
    # first_frame-specific rejection fires.
    with pytest.raises((NotImplementedError, RuntimeError, ValueError)) as exc_info:
        bb.preprocess_input_for_train(frames=[], text=[], ref_images=[object()])
    msg = str(exc_info.value)
    assert "reference-image" not in msg and "first_frame_image" not in msg
