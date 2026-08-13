"""IDM teacher-forcing support for cosmos3_edge — CPU tests.

Covers the backbone's branch merge/split contract and the prefix-KV mask
widening the IDM driver needs, since Cosmos3's per-layer keys carry the cached
und (text) stream that has no matching query rows.
"""

import types

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.architectures.utils.mask_modes import widen_mask_for_prefix_kv  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit_forward, idm_merge, text_pack  # noqa: E402
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


def _net():
    torch.manual_seed(0)
    return Cosmos3OmniTransformer(**MINI).eval()


def _state(net, ids, lat, ncp=0):
    text_pos = text_pack.text_mrope_positions(ids.numel(), float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.numel(), grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(lat.shape[0], ids.numel(), dtype=torch.bool)
    ctx, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0).expand(lat.shape[0], -1), und_mask, cos_u, sin_u)
    return dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((lat.shape[0],), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=ncp,
    )


def test_merge_concats_tokens_and_rotary():
    net = _net()
    ids = torch.tensor([1, 2, 3, 4])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))

    s_n_before, s_c_before = noisy.hidden_states.shape[1], cond.hidden_states.shape[1]
    merged, s_noisy, s_cond = noisy_merged = idm_merge.merge_branches(noisy, cond)
    assert (s_noisy, s_cond) == (s_n_before, s_c_before)
    assert merged.hidden_states.shape[1] == s_noisy + s_cond
    assert merged.extras["cos_gen"].shape[1] == s_noisy + s_cond
    assert merged.extras["sin_gen"].shape[1] == s_noisy + s_cond
    assert merged.grid_frames == noisy.grid_frames + cond.grid_frames
    # und cache is shared (same prompt) and the prefix declaration survives.
    assert merged.extras["und_kv"] is noisy.extras["und_kv"]
    assert merged.prefix_kv_len == noisy.prefix_kv_len
    del noisy_merged


def test_split_is_inverse_of_merge():
    net = _net()
    ids = torch.tensor([5, 6, 7])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    hn, hc = noisy.hidden_states.clone(), cond.hidden_states.clone()

    merged, _, _ = idm_merge.merge_branches(noisy, cond)
    noisy2, cond2 = idm_merge.split_branches(merged, noisy, cond)
    assert torch.equal(noisy2.hidden_states, hn)
    assert torch.equal(cond2.hidden_states, hc)


def test_merge_rejects_mismatched_spatial_layout():
    net = _net()
    ids = torch.tensor([1, 2])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 4, 2))
    with pytest.raises(ValueError, match="spatial token layout"):
        idm_merge.merge_branches(noisy, cond)


def test_merged_state_runs_through_blocks():
    net = _net()
    ids = torch.tensor([1, 2, 3, 4])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    with torch.no_grad():
        merged, s_noisy, s_cond = idm_merge.merge_branches(noisy, cond)
        for i in range(len(net.layers)):
            merged = dit_forward.run_block(net, i, merged)
        noisy2, cond2 = idm_merge.split_branches(merged, noisy, cond)
    assert noisy2.hidden_states.shape[1] == s_noisy
    assert cond2.hidden_states.shape[1] == s_cond
    assert torch.isfinite(noisy2.hidden_states).all() and torch.isfinite(cond2.hidden_states).all()


def test_prefix_widening_shapes_and_semantics():
    # Square mask + a declared prefix -> rectangular mask with visible prefix cols.
    st = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=None)
    m = torch.zeros((5, 5), dtype=torch.bool)
    w = widen_mask_for_prefix_kv(m, st)
    assert w.shape == (5, 8)
    assert w[:, :3].all() and not w[:, 3:].any()

    # Per-sample padding gate broadcasts to (B, 1, S, prefix + S).
    pm = torch.tensor([[True, True, False]])
    st2 = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=pm)
    w2 = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st2)
    assert w2.shape == (1, 1, 5, 8)
    assert w2[0, 0, :, 2].sum() == 0  # padded und column masked for every query row

    # No prefix declared -> untouched (byte-identical for Wan / predict2.5).
    st3 = types.SimpleNamespace(prefix_kv_len=0, prefix_kv_mask=None)
    m3 = torch.ones((4, 4), dtype=torch.bool)
    assert widen_mask_for_prefix_kv(m3, st3) is m3


def test_backbone_overrides_idm_stubs():
    """``callable(...)`` is not a test: ``VideoBackbone`` defines both hooks as
    raising stubs, so it is true for every subclass. Assert the override."""
    from openwam.model.video_backbone.base import VideoBackbone

    for name in ("merge_idm_video_branches", "split_idm_video_branches"):
        base_fn = getattr(VideoBackbone, name)
        assert getattr(Cosmos3EdgeVideoBackbone, name) is not base_fn, f"{name} is still the base stub"
    # And the base really does raise — otherwise the check above proves nothing.
    with pytest.raises(NotImplementedError):
        VideoBackbone.merge_idm_video_branches(Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone), None, None)


def test_prefix_widening_preserves_rank_and_stays_shared():
    """Rank in == rank out, and ``None`` (no padding) stays batch-shared.

    The no-padding signal is ``prefix_kv_mask is None``, decided host-side at
    pack time — deliberately NOT an ``.all()`` test on the tensor, which would
    cost a device sync and a dynamo graph break inside the compiled MoT region.
    """
    # 4D mask with no per-sample gate used to crash (2D pad cat'd onto 4D).
    st = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=None)
    m4 = torch.ones((2, 1, 5, 5), dtype=torch.bool)
    w4 = widen_mask_for_prefix_kv(m4, st)
    assert w4.shape == (2, 1, 5, 8)

    # None keeps the cheap batch-shared 2D mask.
    w2 = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st)
    assert w2.shape == (5, 8) and w2.all()

    # A tensor gate is honored per sample even when it happens to be all-True:
    # the helper never inspects its values, so callers must pass None to opt
    # into the shared path.
    st_tensor = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=torch.ones(4, 3, dtype=torch.bool))
    w_t = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st_tensor)
    assert w_t.shape == (4, 1, 5, 8) and w_t.all()


def _b2_state_with_padding(net, pad_len=2):
    """A B=2 cosmos3 video state whose second prompt is right-padded.

    Sample 0 uses all 5 und tokens; sample 1 uses 3 and pads the tail, which is
    the only configuration in which the Stage-2 prefix gate does any work.
    """
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    und_mask = torch.tensor([[True] * 5, [True] * (5 - pad_len) + [False] * pad_len])
    lat = torch.randn(2, MINI["latent_channel"], 2, 2, 2)
    text_pos = text_pack.text_mrope_positions(ids.shape[1], float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.shape[1], grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    ctx, und_kv = dit_forward.run_und_tower(net, ids, und_mask, cos_u, sin_u)
    state = dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((2,), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=0,
    )
    return state, und_mask


def _idm_driver(net):
    from openwam.model.architectures.dual_system.idm import IDMMoTDriver
    from tests.test_cosmos3_joint_self_attn import _TinyActionBackbone

    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        vae=None,
        tokenizer=None,
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    return IDMMoTDriver(vb, ab, attention_mask_mode="action_sees_video")


def test_idm_stage2_wiring_keeps_padded_und_keys_out():
    """End-to-end: prefill_video_cache -> run_action_with_video_cache on a padded batch.

    Drives the actual plumbing rather than ``build_cached_action_mask`` in
    isolation, by poisoning the cached K/V at the padded und columns: if the
    gate is dropped anywhere along the way (prefill not returning it, Stage 2
    ignoring it), the poison reaches the action stream and the output moves.
    """
    torch.manual_seed(0)
    net = _net()
    driver = _idm_driver(net)
    pad_len = 2

    def stage2(poison: bool, gate_override="keep"):
        torch.manual_seed(7)  # identical latents across calls; only the poison varies
        state, und_mask = _b2_state_with_padding(net, pad_len=pad_len)
        with torch.no_grad():
            kv_cache, key_len, gate = driver.prefill_video_cache(state)
            assert gate is not None, "prefill must hand Stage 2 the prefix gate for a padded batch"
            assert torch.equal(gate, und_mask)
            if poison:
                # Sample 1's padded und slots only; every other key is untouched.
                for layer in kv_cache:
                    for t in (layer["k"], layer["v"]):
                        t[1, und_mask.shape[1] - pad_len : und_mask.shape[1]] = 1e4
            astate = types.SimpleNamespace(
                payload=types.SimpleNamespace(x_action=torch.zeros(2, 3, MINI["hidden_size"]))
            )
            astate.payload.x_action.copy_(torch.arange(2 * 3 * MINI["hidden_size"]).float().view(2, 3, -1) * 1e-2)
            out = driver.run_action_with_video_cache(
                astate,
                video_kv_cache=kv_cache,
                video_seq_len=key_len,
                prefix_kv_mask=gate if gate_override == "keep" else gate_override,
            )
        return out.payload.x_action

    clean = stage2(poison=False)
    poisoned = stage2(poison=True)
    assert torch.allclose(clean, poisoned, atol=1e-6), (
        f"padded und keys leaked into the action stream: max diff {(clean - poisoned).abs().max().item():.3e}"
    )

    # Positive control: without the gate the same poison must move the output,
    # otherwise the assertion above would pass vacuously.
    leaked = stage2(poison=True, gate_override=None)
    assert (clean - leaked).abs().max().item() > 1e-3, "poison is inert; the test cannot detect a leak"


def _stage2_action_mask(driver, s_action, key_len, prefix_mask):
    return driver.build_cached_action_mask(
        s_action=s_action, video_key_len=key_len, device=torch.device("cpu"), prefix_kv_mask=prefix_mask
    )


def test_stage2_action_mask_gates_padded_und_columns():
    """Regression: the cached Stage-2 mask must close padded und columns.

    An all-ones mask over the cached key length would open every und slot,
    including right-padding that the joint loop masks out — a silent wrong
    answer the moment IDM inference is batched with unequal prompt lengths.
    """
    from openwam.model.architectures.dual_system.idm import IDMMoTDriver

    driver = IDMMoTDriver.__new__(IDMMoTDriver)
    prefix, s_video, s_action = 4, 10, 3
    prefix_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    mask = _stage2_action_mask(driver, s_action=s_action, key_len=prefix + s_video, prefix_mask=prefix_mask)

    # Key axis is [prefix | video | action]; queries are the action rows only.
    assert mask.shape == (2, 1, s_action, prefix + s_video + s_action)
    # Sample 0: all four und columns open. Sample 1: last two are padding.
    assert mask[0, 0, :, :prefix].all()
    assert mask[1, 0, :, :2].all() and not mask[1, 0, :, 2:prefix].any()
    # Video + action columns stay fully visible for both samples.
    assert mask[:, :, :, prefix:].all()


def test_stage2_action_mask_matches_joint_loop_action_rows():
    """The cached mask must equal the action-query row slice of the joint mask."""
    from openwam.model.architectures.dual_system.idm import IDMMoTDriver

    driver = IDMMoTDriver.__new__(IDMMoTDriver)
    prefix, s_video, s_action = 4, 10, 3
    prefix_mask = torch.tensor([[True, True, False, False]])

    cached = _stage2_action_mask(driver, s_action, prefix + s_video, prefix_mask)
    # Joint loop: square [video, action] mask (action rows all-True under
    # action_sees_video), then widened by the same prefix gate.
    joint = torch.ones((s_video + s_action, s_video + s_action), dtype=torch.bool)
    joint = widen_mask_for_prefix_kv(joint, types.SimpleNamespace(prefix_kv_len=prefix, prefix_kv_mask=prefix_mask))
    assert torch.equal(cached, joint[:, :, s_video:, :])


def test_stage2_action_mask_prefix_free_is_all_ones():
    """Wan / predict2.5 (no prefix) keep the byte-identical 2D all-ones mask."""
    from openwam.model.architectures.dual_system.idm import IDMMoTDriver

    driver = IDMMoTDriver.__new__(IDMMoTDriver)
    mask = _stage2_action_mask(driver, s_action=3, key_len=10, prefix_mask=None)
    assert mask.shape == (3, 13) and mask.all()
