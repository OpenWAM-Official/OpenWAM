"""CPU parity: the batched cosmos3 engine vs the vendored native forward.

Runs the tiny random Edge-flavoured config through both paths with identical
inputs and asserts the noisy-frame velocity fields match at fp32 tolerance —
the canary for any drift in the und-tower / gen-block / patchify / rope /
timestep-scatter reimplementation. (The native forward zero-fills clean frames
and decodes only noisy tokens; our finalize returns the full grid, so clean
frames are compared for the upstream-zeros invariant only.)
"""

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.video_backbone.cosmos3 import dit_forward, text_pack  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import (  # noqa: E402
    Cosmos3OmniTransformer,
)

MINI = dict(
    attention_bias=False,
    head_dim=6,
    hidden_size=12,
    intermediate_size=24,
    latent_channel=2,
    latent_patch_size=1,
    num_attention_heads=2,
    num_hidden_layers=2,
    num_key_value_heads=1,
    patch_latent_dim=2,
    qk_norm_for_text=False,
    use_und_k_norm_for_gen=True,
    hidden_act="relu2",
    rms_norm_eps=1e-5,
    rope_axes_dim=[1, 1, 1],
    rope_theta=1e8,
    vocab_size=32,
)


def _build(cfg):
    torch.manual_seed(0)
    return Cosmos3OmniTransformer(**cfg).eval()


def _native_velocity(net, ids, lat, text_pos, vis_pos, grid, ncp, t_val):
    grid_t, grid_h, grid_w = grid
    length = ids.numel()
    stride = grid_h * grid_w
    s = grid_t * stride
    noisy = torch.arange(ncp, grid_t)
    mse_idx = torch.cat([torch.arange(length + f * stride, length + (f + 1) * stride) for f in noisy.tolist()])
    with torch.no_grad():
        out = net(
            input_ids=ids,
            text_indexes=torch.arange(length),
            position_ids=torch.cat([text_pos, vis_pos], dim=1),
            und_len=length,
            sequence_length=length + s,
            vision_tokens=[lat],
            vision_token_shapes=[grid],
            vision_sequence_indexes=torch.arange(length, length + s),
            vision_mse_loss_indexes=mse_idx,
            vision_timesteps=torch.full((noisy.numel() * stride,), t_val),
            vision_noisy_frame_indexes=[noisy],
        )
    return out.sample[0]


def _mine_velocity(net, ids, lat, text_pos, vis_pos, ncp, t_val):
    with torch.no_grad():
        cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
        und_mask = torch.ones(1, ids.numel(), dtype=torch.bool)
        context, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0), und_mask, cos_und, sin_und)
        state = dit_forward.prepare_block_loop(
            net,
            latents=lat,
            timestep=torch.tensor([t_val]),
            context=context,
            context_mask=und_mask,
            und_kv=und_kv,
            vision_positions=vis_pos.unsqueeze(1),
            num_clean_prefix_frames=ncp,
        )
        for i in range(len(net.layers)):
            state = dit_forward.run_block(net, i, state)
        return dit_forward.finalize_block_loop(net, state)


def _positions(net, und_len, latent_grid):
    p = int(net.config.latent_patch_size)
    grid = text_pack.patch_grid(*latent_grid, p)
    text_pos = text_pack.text_mrope_positions(und_len, float_positions=True)
    _, vis_pos = text_pack.build_joint_positions(
        und_len,
        grid,
        modality_margin=int(net.config.unified_3d_mrope_temporal_modality_margin),
        fps=24.0,
        base_fps=float(net.config.base_fps),
        temporal_compression_factor=4,
    )
    return text_pos, vis_pos, grid


def test_engine_matches_native_forward_p1():
    net = _build(MINI)
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    text_pos, vis_pos, grid = _positions(net, ids.numel(), (3, 2, 2))

    native = _native_velocity(net, ids, lat, text_pos, vis_pos, grid, ncp=1, t_val=500.0)
    mine = _mine_velocity(net, ids, lat, text_pos, vis_pos, ncp=1, t_val=500.0)

    assert native[:, :, :1].abs().sum() == 0
    diff = (mine[:, :, 1:] - native[:, :, 1:]).abs().max().item()
    assert torch.allclose(mine[:, :, 1:], native[:, :, 1:], atol=1e-5), f"max diff {diff}"


def test_engine_matches_native_forward_p2_with_padding():
    cfg = dict(MINI, latent_patch_size=2, patch_latent_dim=2 * 2 * MINI["latent_channel"])
    net = _build(cfg)
    ids = torch.tensor([5, 6, 7])
    lat = torch.randn(1, cfg["latent_channel"], 2, 3, 5)  # odd H/W → zero-pad path
    text_pos, vis_pos, grid = _positions(net, ids.numel(), (2, 3, 5))

    native = _native_velocity(net, ids, lat, text_pos, vis_pos, grid, ncp=1, t_val=995.0)
    mine = _mine_velocity(net, ids, lat, text_pos, vis_pos, ncp=1, t_val=995.0)

    diff = (mine[:, :, 1:] - native[:, :, 1:]).abs().max().item()
    assert torch.allclose(mine[:, :, 1:], native[:, :, 1:], atol=1e-5), f"max diff {diff}"


def test_batched_engine_consistent_with_single():
    net = _build(MINI)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    lat1 = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    lat = torch.cat([lat1, lat1], dim=0)
    text_pos, vis_pos, _ = _positions(net, 4, (3, 2, 2))

    with torch.no_grad():
        cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
        und_mask = torch.ones(2, 4, dtype=torch.bool)
        context, und_kv = dit_forward.run_und_tower(net, ids, und_mask, cos_und, sin_und)
        state = dit_forward.prepare_block_loop(
            net,
            latents=lat,
            timestep=torch.tensor([500.0, 500.0]),
            context=context,
            context_mask=und_mask,
            und_kv=und_kv,
            vision_positions=vis_pos.unsqueeze(1),
            num_clean_prefix_frames=1,
        )
        for i in range(len(net.layers)):
            state = dit_forward.run_block(net, i, state)
        batched = dit_forward.finalize_block_loop(net, state)

    single = _mine_velocity(net, ids[0], lat1, text_pos, vis_pos, ncp=1, t_val=500.0)
    assert torch.allclose(batched[0], batched[1], atol=1e-6)
    assert torch.allclose(batched[:1], single, atol=1e-6)


def test_none_und_mask_matches_all_true_tensor():
    """``und_mask=None`` (the fused-kernel fast path) must be numerically identical
    to passing an explicit all-True mask — it is an optimization, not a variant."""
    net = _build(MINI)
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    text_pos, vis_pos, _ = _positions(net, ids.numel(), (3, 2, 2))
    all_true = torch.ones(1, ids.numel(), dtype=torch.bool)

    outs = []
    for und_mask in (all_true, None):
        with torch.no_grad():
            cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
            ctx, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0), und_mask, cos_u, sin_u)
            state = dit_forward.prepare_block_loop(
                net,
                latents=lat,
                timestep=torch.tensor([500.0]),
                context=ctx,
                und_mask=und_mask,
                und_kv=und_kv,
                vision_positions=vis_pos.unsqueeze(1),
                num_clean_prefix_frames=1,
            )
            for i in range(len(net.layers)):
                state = dit_forward.run_block(net, i, state)
            outs.append((ctx, dit_forward.finalize_block_loop(net, state), state.prefix_kv_mask))

    (ctx_a, out_a, pm_a), (ctx_b, out_b, pm_b) = outs
    assert torch.allclose(ctx_a, ctx_b, atol=1e-6), "und tower diverges between masked and causal paths"
    assert torch.allclose(out_a, out_b, atol=1e-6), "gen stream diverges when the all-True mask is dropped"
    assert pm_a is not None and pm_b is None  # None propagates to the widen helper


def _mini_backbone(net):
    from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone

    return Cosmos3EdgeVideoBackbone(
        net=net,
        vae=None,
        tokenizer=None,
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
    )


def test_rotary_inv_freq_survives_dtype_cast():
    """set_dtype_device must keep the rotary frequency table in fp32.

    bf16 costs ~0.37% on inv_freq, which the vision stream's 15000-token
    modality offset multiplies into a phase error that scrambles cos/sin.
    """
    net = _build(MINI)
    vb = _mini_backbone(net)
    before = net.rotary_emb.inv_freq.clone()
    vb.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    assert net.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(net.rotary_emb.inv_freq, before)
    # The timestep embedder keeps its own fp32 pin too.
    assert next(net.time_embedder.parameters()).dtype == torch.float32


def test_rotary_inv_freq_survives_the_trainer_call_order():
    """The real training order, which a single set_dtype_device call cannot model.

    ``openwam_trainer`` calls set_dtype_device at :126, then
    ``accelerator.prepare()``, then again at :530. DeepSpeed's
    ``_configure_distributed_model`` runs ``self.module.bfloat16()`` inside
    prepare, and ``nn.Module.bfloat16()`` casts floating-point *buffers* — so a
    table snapshotted by the second call is already rounded. Anything that
    restores from a snapshot taken inside set_dtype_device passes the
    single-call test above and still trains on bf16 phases.
    """
    net = _build(MINI)
    pristine = net.rotary_emb.inv_freq.detach().clone()
    vb = _mini_backbone(net)

    vb.set_dtype_device(torch.bfloat16, torch.device("cpu"))  # trainer :126
    net.bfloat16()  # what DeepSpeed does inside accelerator.prepare()
    assert net.rotary_emb.inv_freq.dtype == torch.bfloat16, "precondition: the cast must reach the buffer"
    vb.set_dtype_device(torch.bfloat16, torch.device("cpu"))  # trainer :530

    assert net.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(net.rotary_emb.inv_freq, pristine), "fp32 table not recovered after the DeepSpeed cast"


def test_pristine_inv_freq_is_not_a_buffer():
    """The copy has to be unreachable by ``nn.Module._apply``.

    Registering it as a buffer (or a parameter) would put it right back in the
    path of every ``.to(dtype=)`` / ``.bfloat16()`` and defeat the point; it
    would also leak into the checkpoint.
    """
    net = _build(MINI)
    vb = _mini_backbone(net)
    assert vb._pristine_inv_freq is not None and vb._pristine_inv_freq.dtype == torch.float32
    assert "_pristine_inv_freq" not in dict(vb.named_buffers())
    assert not any("_pristine_inv_freq" in k for k in vb.state_dict())
    vb.bfloat16()
    assert vb._pristine_inv_freq.dtype == torch.float32


def test_rotary_inv_freq_recomputed_when_build_is_already_cast():
    """A net handed over already bf16 must not seed the copy from rounded bits."""
    net = _build(MINI).bfloat16()
    expected = dit_forward.rotary_inv_freq(int(net.config.head_dim), float(net.config.rope_theta))
    vb = _mini_backbone(net)
    assert vb._pristine_inv_freq.dtype == torch.float32
    assert torch.equal(vb._pristine_inv_freq, expected)
    vb.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    assert torch.equal(net.rotary_emb.inv_freq, expected)


def test_freeze_unused_native_heads():
    from openwam.model.video_backbone.cosmos3.pipeline_builder import _freeze_unused_native_heads

    net = _build(dict(MINI, action_gen=True, action_dim=4, num_embodiment_domains=3))
    frozen = _freeze_unused_native_heads(net)
    expected = sum(p.numel() for p in net.action_proj_in.parameters())
    expected += sum(p.numel() for p in net.action_proj_out.parameters())
    expected += net.action_modality_embed.numel()
    assert frozen == expected
    assert not any(p.requires_grad for p in net.action_proj_in.parameters())
    assert not any(p.requires_grad for p in net.action_proj_out.parameters())
    assert not net.action_modality_embed.requires_grad
    # The gen pathway stays trainable.
    assert net.layers[0].self_attn.add_q_proj.weight.requires_grad
    assert net.proj_out.weight.requires_grad


def test_gradients_reach_gen_pathway():
    net = _build(MINI)
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    text_pos, vis_pos, _ = _positions(net, ids.numel(), (3, 2, 2))

    cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(1, 4, dtype=torch.bool)
    context, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0), und_mask, cos_und, sin_und)
    state = dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.tensor([500.0]),
        context=context,
        context_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=1,
    )
    for i in range(len(net.layers)):
        state = dit_forward.run_block(net, i, state)
    out = dit_forward.finalize_block_loop(net, state)
    out[:, :, 1:].square().mean().backward()

    gen = net.layers[0].self_attn.add_q_proj.weight.grad
    und = net.layers[0].self_attn.to_q.weight.grad
    assert gen is not None and gen.abs().sum() > 0
    assert und is None  # und tower ran under no_grad


class _VocabSafeTokenizer:
    """Chat-template stub whose ids fit MINI's 32-token vocab."""

    eos_token_id = 30
    pad_token_id = 0
    unk_token_id = 31

    def apply_chat_template(self, conversations, *, tokenize, add_generation_prompt, return_dict):
        body = [1] + [ord(c) % 12 + 2 for c in conversations[-1]["content"]]
        return body + ([28] if add_generation_prompt else [])

    def convert_tokens_to_ids(self, token):
        return 29


def _tokenizer_backbone():
    from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone

    vb = Cosmos3EdgeVideoBackbone(
        net=_build(MINI),
        vae=None,
        tokenizer=_VocabSafeTokenizer(),
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
        prompt_templates=False,
    ).eval()
    # ``device``/``dtype`` default to cuda/bf16 on the ABC; pin them so
    # ``_encode_prompts`` runs on CPU.
    vb.set_dtype_device(torch.float32, torch.device("cpu"))
    return vb


def test_encode_prompts_signals_padding_only_when_lengths_differ():
    """``_encode_prompts`` is the sole producer of the ``und_mask is None`` contract.

    Equal-length prompts must yield ``None`` (the fused-kernel path); unequal
    ones must yield a real per-sample gate. Replacing the predicate with a bare
    ``None`` leaves every other cosmos3 test green while the padded sample
    silently attends its own padding — ``text_encoder_dropout`` rewrites prompts
    to ``""``, so the predicate flips on real training data.
    """
    vb = _tokenizer_backbone()
    kw = dict(num_frames=9, height=8, width=8, fps=24.0)

    with torch.no_grad():
        same = vb._encode_prompts(["abcd", "efgh"], **kw)
        diff = vb._encode_prompts(["abcd", "ef"], **kw)

    assert same["context_mask"] is None, "equal-length prompts must take the no-padding path"
    gate = diff["context_mask"]
    assert gate is not None
    lens = diff["seq_lens"].tolist()
    assert lens[0] > lens[1], "the shorter prompt must be the padded one"
    assert gate.shape == (2, lens[0])
    assert gate[0].all()
    assert gate[1, : lens[1]].all() and not gate[1, lens[1] :].any()


def test_padding_gate_changes_the_padded_sample_only():
    """The consequence of dropping that gate, measured rather than asserted.

    Without it the padded sample attends pad-token keys; the unpadded sample in
    the same batch is unaffected, which is why a batch-level smoke test cannot
    see the difference.
    """
    vb = _tokenizer_backbone()
    net = vb.dit
    tok = vb._require_tokenizer()
    ids_list = [text_pack.tokenize_prompt(tok, t, max_length=512) for t in ("abcd", "ef")]
    input_ids, real_gate, _ = text_pack.pad_und_batch(ids_list, pad_token_id=int(tok.pad_token_id))
    und_len = input_ids.shape[1]
    torch.manual_seed(3)
    lat = torch.randn(2, MINI["latent_channel"], 2, 2, 2)
    text_pos, vis_pos, _ = _positions(net, und_len, tuple(lat.shape[2:]))

    def velocity(gate):
        with torch.no_grad():
            cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
            _, und_kv = dit_forward.run_und_tower(net, input_ids, gate, cos_u, sin_u)
            state = dit_forward.prepare_block_loop(
                net,
                latents=lat,
                timestep=torch.full((2,), 500.0),
                context=torch.zeros(2, und_len, MINI["hidden_size"]),
                und_mask=gate,
                und_kv=und_kv,
                vision_positions=vis_pos.unsqueeze(1),
                num_clean_prefix_frames=0,
            )
            for i in range(len(net.layers)):
                state = dit_forward.run_block(net, i, state)
            return dit_forward.finalize_block_loop(net, state)

    gated, ungated = velocity(real_gate), velocity(None)
    padded = (gated[1] - ungated[1]).abs().max().item()
    unpadded = (gated[0] - ungated[0]).abs().max().item()
    assert padded > 1e-4, f"the gate made no difference to the padded sample ({padded:.3e})"
    assert unpadded < 1e-6, f"the unpadded sample must be untouched ({unpadded:.3e})"
