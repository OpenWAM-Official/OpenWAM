"""Cosmos-Predict2.5 adapter: capability flags + unsupported-path errors.

Uses a tiny fake pipeline so the adapter can be exercised without the
upstream ``cosmos_predict2`` package being installed.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.video_backbone.adapter import BlockLoopState
from openwam.model.video_backbone.cosmos25 import (
    Cosmos25VideoBackbone,
    CosmosFlowSchedulerAdapter,
)


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


def _build_backbone(**overrides) -> Cosmos25VideoBackbone:
    pipe = _FakeCosmosPipeline()
    kwargs = dict(
        pipeline=pipe,
        dim=pipe.dim,
        num_layers=pipe.num_layers,
        num_heads=pipe.num_heads,
        head_dim=pipe.head_dim,
        context_dim=pipe.context_dim,
        scheduler=CosmosFlowSchedulerAdapter(flow_shift=3.0),
        freeze=True,
    )
    kwargs.update(overrides)
    return Cosmos25VideoBackbone(**kwargs)


def test_properties_match_pipeline_geometry():
    bb = _build_backbone()
    assert bb.dim == 1280
    assert bb.num_layers == 24
    assert bb.num_heads == 10
    assert bb.head_dim == 128
    assert bb.context_dim == 2048
    assert isinstance(bb.scheduler, CosmosFlowSchedulerAdapter)


def test_self_attn_paths_raise_with_clear_message():
    bb = _build_backbone()
    dummy_state = BlockLoopState(
        x=torch.zeros(1, 4, bb.dim),
        t_mod=torch.zeros(1, 6, bb.dim),
        freqs=torch.zeros(4, bb.head_dim, dtype=torch.complex64),
        context=torch.zeros(1, 1, bb.context_dim),
    )
    with pytest.raises(NotImplementedError, match="pre_attn_at_layer"):
        bb.pre_attn_at_layer(0, dummy_state)
    with pytest.raises(NotImplementedError, match="post_attn_at_layer"):
        bb.post_attn_at_layer(0, dummy_state, attn_out=torch.zeros(1, 4, bb.dim), post_state={})


def test_shared_backbone_paths_raise():
    bb = _build_backbone()
    dummy_state = BlockLoopState(
        x=torch.zeros(1, 4, bb.dim),
        t_mod=torch.zeros(1, 6, bb.dim),
        freqs=torch.zeros(4, bb.head_dim, dtype=torch.complex64),
        context=torch.zeros(1, 1, bb.context_dim),
    )
    with pytest.raises(NotImplementedError, match="inject_action_tokens"):
        bb.inject_action_tokens(dummy_state, torch.zeros(1, 2, bb.dim), 2)
    with pytest.raises(NotImplementedError, match="extract_action_tokens"):
        bb.extract_action_tokens(dummy_state, 2)
    with pytest.raises(NotImplementedError, match="inject_shared_tokens"):
        bb.inject_shared_tokens(dummy_state, torch.zeros(1, 2, bb.dim), 2)
    with pytest.raises(NotImplementedError, match="extract_shared_tokens"):
        bb.extract_shared_tokens(dummy_state, 2)


def test_vace_rejected_for_mvp():
    bb = _build_backbone()
    with pytest.raises(NotImplementedError, match="VACE"):
        bb.preprocess_input(frames=[], text=[], vace_videos=[object()])


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
    try:
        bb.preprocess_input(frames=[], text=[], ref_images=[object()])
    except NotImplementedError as exc:
        assert "reference-image" not in str(exc) and "first_frame_image" not in str(exc)
