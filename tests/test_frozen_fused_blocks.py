"""Fused halves must preserve freeze_modules' forward/no_grad contract.

These CPU tests run the real Wan/Action split-block entry points. The compiler
wrapper is replaced with its eager function to isolate dispatch and autograd
semantics; this is deliberately not an Inductor numerical equivalence test.
"""

from types import SimpleNamespace

import pytest
import torch

from openwam.model.action_backbone.separate_action_dit import ActionDiT, ActionDiTState, SelfAttnActionDiTBlock
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.kernels import fused_blocks
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan.models.dit import DiTBlock
from openwam.model.video_backbone.wan_backbone import WanBase


@pytest.fixture(autouse=True)
def _isolate_fusion(monkeypatch):
    monkeypatch.setattr(fused_blocks, "_ENABLED", True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(deterministic)


def _block(kind):
    torch.manual_seed(831)
    if kind == "wan":
        return DiTBlock(has_image_input=False, dim=8, num_heads=2, ffn_dim=16)
    return SelfAttnActionDiTBlock(hidden_dim=8, num_heads=2, attn_head_dim=4, ffn_dim=16, kv_hidden_dim=8)


def _inputs(kind):
    generator = torch.Generator().manual_seed(812)
    shape = (1, 3, 6, 8) if kind == "wan" else (1, 6, 8)
    return tuple(torch.randn(size, generator=generator, requires_grad=True) for size in ((1, 3, 8), shape, (1, 5, 8)))


def _supported(kind, block):
    check = fused_blocks.wan_block_supported if kind == "wan" else fused_blocks.act_block_supported
    return check(block, _inputs(kind)[1])


def _freeze(block, path):
    holder = torch.nn.Module()
    holder.block = block
    dotted = "block" + (f".{path}" if path else "")
    assert BaseWAMArchitecture.freeze_modules(holder, [dotted]) == [dotted]


def _pre(kind, block, inputs):
    x, time_mod, context = inputs
    mask = torch.ones(1, 5, dtype=torch.bool)
    freqs = torch.ones(3, 2, dtype=torch.complex128)
    if kind == "wan":
        state = BlockLoopState(
            hidden_states=x,
            time_mod=time_mod,
            rope_freqs=freqs.unsqueeze(1),
            context=context,
            context_mask=mask,
            extras={"dit": SimpleNamespace(blocks=[block])},
        )
        backbone = SimpleNamespace()
        result = WanBase.pre_attn_at_layer_for_compile(backbone, 0, state)
    else:
        payload = ActionDiTState(
            x_action=x,
            t_mod=time_mod,
            t_embed=torch.zeros(1, 8),
            action_freqs=freqs,
            context=context,
            context_mask=mask,
        )
        state = SimpleNamespace(payload=payload)
        backbone = SimpleNamespace(blocks=[block], _num_heads=2)
        result = ActionDiT.pre_attn_at_layer_for_compile(backbone, 0, state)
    return backbone, state, result


def _run(kind, block, monkeypatch, enabled):
    monkeypatch.setattr(fused_blocks, "_ENABLED", enabled)
    block.zero_grad(set_to_none=True)
    inputs = _inputs(kind)
    backbone, state, (q, k, v, post) = _pre(kind, block, inputs)
    attn_out = fused_blocks.masked_mha(q, k, v, 2, None)
    if kind == "wan":
        out = WanBase.post_attn_at_layer_for_compile(backbone, 0, state, attn_out, post).hidden_states
    else:
        out = ActionDiT.post_attn_at_layer_for_compile(backbone, 0, state, attn_out, post).payload.x_action
    if out.requires_grad:
        out.square().sum().backward()
    gradients = [x.grad for x in inputs] + [p.grad for p in block.parameters()]
    return out.detach(), [None if grad is None else grad.clone() for grad in gradients]


@pytest.mark.parametrize("kind", ["wan", "action"])
@pytest.mark.parametrize("path", ["", "self_attn", "self_attn.q", "cross_attn", "cross_attn.o", "ffn", "ffn.0"])
def test_frozen_block_or_leaf_uses_eager_halves_with_identical_gradients(kind, path, monkeypatch):
    block = _block(kind)
    # The guard must see a freeze applied after its first successful check.
    assert _supported(kind, block)
    _freeze(block, path)
    assert not _supported(kind, block)

    def unexpected_compile(*args):
        pytest.fail("a no_grad-wrapped module must not enter fused functional code")

    monkeypatch.setattr(fused_blocks, "_compiled", unexpected_compile)
    eager_out, eager_gradients = _run(kind, block, monkeypatch, enabled=False)
    fast_out, fast_gradients = _run(kind, block, monkeypatch, enabled=True)
    torch.testing.assert_close(fast_out, eager_out, rtol=0, atol=0)
    for actual, expected in zip(fast_gradients, eager_gradients):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["wan", "action"])
@pytest.mark.parametrize("path", ["", "self_attn"])
def test_frozen_pre_attention_projections_stop_upstream_gradients(kind, path, monkeypatch):
    block = _block(kind)
    _freeze(block, path)
    monkeypatch.setattr(fused_blocks, "_compiled", lambda name, fn: fn)
    _, _, (q, k, v, _) = _pre(kind, block, _inputs(kind))
    assert not q.requires_grad
    assert not k.requires_grad
    assert not v.requires_grad


@pytest.mark.parametrize("kind", ["wan", "action"])
@pytest.mark.parametrize("mode", ["train", "eval", "parameters_only"])
def test_regular_and_parameter_frozen_blocks_keep_fusion_and_input_gradients(kind, mode, monkeypatch):
    block = _block(kind)
    if mode == "eval":
        block.eval()
    elif mode == "parameters_only":
        block.requires_grad_(False)
    assert _supported(kind, block)
    calls = []

    def eager_compiled(name, fn):
        calls.append(name)
        return fn

    monkeypatch.setattr(fused_blocks, "_compiled", eager_compiled)
    eager_out, eager_gradients = _run(kind, block, monkeypatch, enabled=False)
    assert not calls
    fast_out, fast_gradients = _run(kind, block, monkeypatch, enabled=True)
    assert len(calls) == 3  # pre, post attention, post FFN
    torch.testing.assert_close(fast_out, eager_out, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(fast_gradients, eager_gradients):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert all(grad is not None and grad.abs().sum() > 0 for grad in fast_gradients[:3])


@pytest.mark.parametrize("kind", ["wan", "action"])
@pytest.mark.parametrize("path", ["", "self_attn", "self_attn.q", "self_attn.norm_q", "cross_attn", "ffn", "ffn.1"])
def test_custom_instance_forward_disables_fusion(kind, path):
    block = _block(kind)
    module = block.get_submodule(path) if path else block
    original = module.forward
    module.forward = lambda *args, **kwargs: original(*args, **kwargs)
    assert not _supported(kind, block)


@pytest.mark.parametrize("kind", ["wan", "action"])
def test_custom_projection_subclass_and_hooks_disable_fusion(kind):
    class CustomLinear(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x) + 1

    block = _block(kind)
    old_projection = block.self_attn.q
    block.self_attn.q = CustomLinear(8, 8)
    assert not _supported(kind, block)
    block.self_attn.q = old_projection
    assert _supported(kind, block)
    hook = old_projection.register_forward_hook(lambda module, args, output: output + 1)
    assert not _supported(kind, block)
    hook.remove()
    assert _supported(kind, block)


@pytest.mark.parametrize("kind", ["wan", "action"])
@pytest.mark.parametrize("hook_kind", ["forward_pre", "forward", "backward_pre", "backward"])
def test_global_hooks_preserve_outputs_and_gradients_by_disabling_fusion(kind, hook_kind, monkeypatch):
    block = _block(kind)
    assert _supported(kind, block)
    calls = []
    target = block.self_attn.q

    def hook(module, values, *extra):
        if module is not target:
            return None
        calls.append(module)
        if hook_kind == "forward":
            return extra[0] * 0.5
        # Forward pre-hooks change the inputs; backward hooks change the
        # gradients. Assert actual behavior, not only the guard's return value.
        return tuple(value * 0.5 if value is not None else None for value in values)

    registrations = {
        "forward_pre": torch.nn.modules.module.register_module_forward_pre_hook,
        "forward": torch.nn.modules.module.register_module_forward_hook,
        "backward_pre": torch.nn.modules.module.register_module_full_backward_pre_hook,
        "backward": torch.nn.modules.module.register_module_full_backward_hook,
    }
    handle = registrations[hook_kind](hook)
    try:
        assert not _supported(kind, block)

        def unexpected_compile(*args):
            pytest.fail("functional fusion must not bypass global module hooks")

        monkeypatch.setattr(fused_blocks, "_compiled", unexpected_compile)
        eager_out, eager_gradients = _run(kind, block, monkeypatch, enabled=False)
        assert calls
        calls.clear()
        fast_out, fast_gradients = _run(kind, block, monkeypatch, enabled=True)
        assert calls
        torch.testing.assert_close(fast_out, eager_out, rtol=0, atol=0)
        for actual, expected in zip(fast_gradients, eager_gradients):
            assert (actual is None) == (expected is None)
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        handle.remove()
    assert _supported(kind, block)
