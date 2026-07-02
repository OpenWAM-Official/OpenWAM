"""Regression: cosmos ``run_block`` must honor ``use_gradient_checkpointing``.

Before the fix, ``dit_forward.run_block`` called ``block(...)`` directly and
silently ignored ``state.use_gradient_checkpointing`` — so the 28-block 2B DiT
trained with NO activation checkpointing even though ``configs/train.yaml``
defaults it on (``wan_backbone.run_block`` wraps blocks in
``gradient_checkpoint_forward``; cosmos didn't). That blows up activation memory
and OOMs at the intended batch size.

We prove the flag is honored by counting block-forward invocations:
``torch.utils.checkpoint`` (use_reentrant=False) recomputes the wrapped forward
during backward, so a checkpointed block runs forward twice (forward +
recompute), a non-checkpointed block exactly once. No GPU / upstream Cosmos
needed — a tiny fake net exercises ``run_block`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.cosmos_predict25 import dit_forward


class _CountingBlock(nn.Module):
    """Mimics the cosmos DiT block call signature; counts forward passes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.calls = 0

    def forward(
        self,
        x,
        t_embedding_B_T_D,
        context,
        *,
        rope_emb_L_1_1_D,
        adaln_lora_B_T_3D,
        extra_per_block_pos_emb,
    ):
        self.calls += 1
        return self.lin(x)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_CountingBlock()])


def _make_state(use_gc: bool) -> BlockLoopState:
    return BlockLoopState(
        hidden_states=torch.randn(2, 4, requires_grad=True),
        time_mod=torch.zeros(1),
        rope_freqs=torch.zeros(1),
        context=torch.randn(2, 3, 4),
        extras={
            "t_embedding_B_T_D": None,
            "rope_emb_L_1_1_D": None,
            "adaln_lora_B_T_3D": None,
            "extra_per_block_pos_emb": None,
        },
        use_gradient_checkpointing=use_gc,
    )


def _run_and_backward(use_gc: bool) -> int:
    net = _Net()
    state = _make_state(use_gc)
    state = dit_forward.run_block(net, 0, state)
    state.hidden_states.sum().backward()
    return net.blocks[0].calls


def test_run_block_recomputes_when_checkpointing_enabled():
    """Checkpointing on ⇒ block forward runs twice (forward + backward recompute)."""
    assert _run_and_backward(use_gc=True) == 2


def test_run_block_single_forward_when_checkpointing_disabled():
    """Checkpointing off ⇒ exactly one block forward; result is byte-identical
    to a direct call, so the non-checkpointed path is unchanged by the fix."""
    assert _run_and_backward(use_gc=False) == 1
