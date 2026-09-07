"""CPU checks for Cosmos3 inference compilation."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.compile_options import cosmos3_blocks_compile_cfg  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit_forward  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import (  # noqa: E402
    Cosmos3OmniTransformer,
)


def _backbone():
    torch.manual_seed(17)
    net = Cosmos3OmniTransformer(
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        latent_channel=2,
        latent_patch_size=1,
        patch_latent_dim=2,
        vocab_size=32,
        rope_axes_dim=[1, 1, 1],
        attention_bias=False,
        qk_norm_for_text=False,
        use_und_k_norm_for_gen=True,
        hidden_act="relu2",
    )
    return Cosmos3EdgeVideoBackbone(net=net, dim=12, num_layers=2, num_heads=2, head_dim=6, context_dim=12).eval()


def _state(*, batch=1, seq=6, padded=False, masked=False, mask_4d=False):
    torch.manual_seed(42 + seq)
    und_mask = torch.ones(batch, 4, dtype=torch.bool) if padded else None
    if padded:
        und_mask[-1, -1] = False
    gen_mask = torch.ones(seq, seq, dtype=torch.bool) if masked else None
    if masked:
        gen_mask[: seq // 2, seq // 2 :] = False
        if mask_4d:
            gen_mask = gen_mask[None, None].expand(batch, 1, -1, -1)
    angle = torch.randn(1, seq, 3).repeat(1, 1, 2)
    return BlockLoopState(
        hidden_states=torch.randn(batch, seq, 12),
        time_mod=torch.empty(0),
        rope_freqs=torch.empty(0),
        context=torch.randn(batch, 4, 12),
        grid_frames=1,
        grid_height=1,
        grid_width=seq,
        extras={
            "und_kv": tuple((torch.randn(batch, 4, 1, 6), torch.randn(batch, 4, 1, 6)) for _ in range(2)),
            "und_mask": und_mask,
            "cos_gen": angle.cos(),
            "sin_gen": angle.sin(),
            "shared_attention_mask": gen_mask,
            "orig_hw": (1, seq),
        },
    )


def test_default_options():
    cfg = cosmos3_blocks_compile_cfg({"enabled": True})
    assert cfg.enabled and not cfg.dynamic and cfg.torch_mode == "default"


def test_custom_compile_options_are_forwarded(monkeypatch):
    vb = _backbone()
    compiler = Mock(return_value=dit_forward._gen_block_forward)
    monkeypatch.setattr(torch, "compile", compiler)
    vb.apply_compile_optimizations(
        {"enabled": True, "cosmos3_blocks": {"torch_mode": "max-autotune-no-cudagraphs", "dynamic": True}}
    )
    with torch.no_grad():
        vb.run_block(0, _state())
    compiler.assert_called_once_with(dit_forward._gen_block_forward, dynamic=True, mode="max-autotune-no-cudagraphs")


def test_invalid_public_switch_is_rejected():
    vb = _backbone()
    with pytest.raises(ValueError, match="Unknown compile enabled"):
        vb.apply_compile_optimizations({"enabled": "typo"})
    assert not vb._gen_block_compile_enabled


def test_base_compile_hook_is_optional_and_noop():
    vb = _backbone()
    assert VideoBackbone.apply_compile_optimizations(vb, {"enabled": True}) is None
    assert not vb._gen_block_compile_enabled and vb._compiled_gen_block is None


@pytest.mark.parametrize("cfg", [None, {"enabled": False}, {"enabled": True, "cosmos3_blocks": {"enabled": False}}])
def test_disabled_is_eager(monkeypatch, cfg):
    vb = _backbone()
    compiler = Mock(side_effect=AssertionError("must remain eager"))
    monkeypatch.setattr(torch, "compile", compiler)
    vb.apply_compile_optimizations(cfg)
    with torch.no_grad():
        vb.run_block(0, _state())
    compiler.assert_not_called()


def test_lazy_reuse_reset_and_checkpoint_keys(monkeypatch):
    vb = _backbone()
    before = {key: value.clone() for key, value in vb.state_dict().items()}
    compiled = Mock(wraps=dit_forward._gen_block_forward)
    compiler = Mock(return_value=compiled)
    monkeypatch.setattr(torch, "compile", compiler)
    vb.apply_compile_optimizations({"enabled": True})
    compiler.assert_not_called()
    with torch.no_grad():
        for _ in range(2):
            state = _state()
            for i in range(2):
                vb.run_block(i, state)
    compiler.assert_called_once_with(dit_forward._gen_block_forward, dynamic=False, mode="default")
    assert compiled.call_count == 4
    after = vb.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    vb.apply_compile_optimizations({"enabled": False})
    assert vb._compiled_gen_block is None and not vb._gen_block_compile_enabled
    vb.apply_compile_optimizations({"enabled": True})
    with torch.no_grad():
        vb.run_block(0, _state())
    assert compiler.call_count == 2


@pytest.mark.parametrize("gate", ["training", "grad", "checkpoint", "offload", "outer_compile"])
def test_inference_only(monkeypatch, gate):
    vb = _backbone()
    state = _state()
    vb.apply_compile_optimizations({"enabled": True})
    compiler = Mock(side_effect=AssertionError("must remain eager"))
    monkeypatch.setattr(torch, "compile", compiler)
    if gate == "training":
        vb.train()
    elif gate == "checkpoint":
        state.use_gradient_checkpointing = True
    elif gate == "offload":
        state.use_gradient_checkpointing_offload = True
    elif gate == "outer_compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with torch.set_grad_enabled(gate == "grad"):
        output = vb.run_block(0, state).hidden_states
        if gate == "grad":
            output.sum().backward()
            assert vb.dit.layers[0].self_attn.add_q_proj.weight.grad is not None
    compiler.assert_not_called()


@pytest.mark.parametrize("at_setup", [False, True])
def test_failure_retries_eager_once_and_can_reenable(monkeypatch, caplog, at_setup):
    vb = _backbone()
    with torch.no_grad():
        expected = vb.run_block(0, _state()).hidden_states
    broken = Mock(side_effect=RuntimeError("compiler failed"))
    compiler = broken if at_setup else Mock(return_value=broken)
    monkeypatch.setattr(torch, "compile", compiler)
    vb.apply_compile_optimizations({"enabled": True})
    with torch.no_grad():
        for _ in range(2):
            torch.testing.assert_close(vb.run_block(0, _state()).hidden_states, expected)
    assert compiler.call_count == 1
    assert vb._compiled_gen_block is None and not vb._gen_block_compile_enabled
    assert "falling back to eager" in caplog.text
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    vb.apply_compile_optimizations({"enabled": True})
    with torch.no_grad():
        torch.testing.assert_close(vb.run_block(0, _state()).hidden_states, expected)
    assert vb._compiled_gen_block is not None


@pytest.mark.parametrize(
    "padded,masked,mask_4d", [(False, False, False), (True, False, False), (False, True, False), (True, True, True)]
)
def test_fullgraph_block_loop_parity(monkeypatch, padded, masked, mask_4d):
    """Check Dynamo capture with an eager backend, not GPU code generation."""
    torch._dynamo.reset()
    vb = _backbone()
    real_compile = torch.compile
    graphs = []

    def backend(gm, inputs, **kwargs):
        graphs.append(gm)
        return gm.forward

    def compile_fullgraph(fn, **kwargs):
        return real_compile(fn, backend=backend, fullgraph=True, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_fullgraph)
    vb.apply_compile_optimizations({"enabled": True})
    with torch.no_grad():
        for seq in (6, 8, 6):
            kwargs = dict(batch=2, seq=seq, padded=padded, masked=masked, mask_4d=mask_4d)
            eager, fast = _state(**kwargs), _state(**kwargs)
            for i in range(2):
                dit_forward.run_block(vb.dit, i, eager)
                vb.run_block(i, fast)
                assert vb._gen_block_compile_enabled  # Do not let fallback hide a capture failure.
                torch.testing.assert_close(fast.hidden_states, eager.hidden_states, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(vb.finalize(fast), vb.finalize(eager), rtol=1e-5, atol=1e-5)
    assert graphs
    torch._dynamo.reset()


def test_late_failure_preserves_previous_layer_progress(monkeypatch, caplog):
    vb = _backbone()

    def fail_at_second_layer(layer, *args):
        if layer is vb.dit.layers[1]:
            raise RuntimeError("second layer failed")
        return dit_forward._gen_block_forward(layer, *args)

    compiled = Mock(side_effect=fail_at_second_layer)
    monkeypatch.setattr(torch, "compile", Mock(return_value=compiled))
    vb.apply_compile_optimizations({"enabled": True})
    eager, fast = _state(), _state()
    with torch.no_grad():
        for i in range(2):
            dit_forward.run_block(vb.dit, i, eager)
            vb.run_block(i, fast)
            torch.testing.assert_close(fast.hidden_states, eager.hidden_states)
    assert compiled.call_count == 2
    assert vb._compiled_gen_block is None and not vb._gen_block_compile_enabled
    assert caplog.text.count("falling back to eager") == 1


@pytest.mark.parametrize("warm_inner", [False, True])
def test_actual_outer_compile_does_not_nest_inner_compile(monkeypatch, warm_inner):
    torch._dynamo.reset()
    vb = _backbone()
    real_compile = torch.compile
    compiler = Mock(side_effect=lambda fn, **kwargs: real_compile(fn, backend="eager", **kwargs))
    monkeypatch.setattr(torch, "compile", compiler)
    vb.apply_compile_optimizations({"enabled": True})
    with torch.no_grad():
        if warm_inner:
            vb.run_block(0, _state())
        calls_before = compiler.call_count
        expected = dit_forward.run_block(vb.dit, 0, _state()).hidden_states
        outer = real_compile(lambda state: vb.run_block(0, state).hidden_states, backend="eager", fullgraph=True)
        torch.testing.assert_close(outer(_state()), expected)
        assert compiler.call_count == calls_before
    torch._dynamo.reset()
