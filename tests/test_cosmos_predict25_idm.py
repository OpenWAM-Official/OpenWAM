"""IDM (dual_system/idm) support for the CosmosPredict25 video backbone.

IDM teacher-forcing merges a noisy + cond video branch into one MoT pass. On
Wan the branches are flat ``(B, L, D)`` sequences; on Cosmos they are 5D grids
``(B, T, H, W, D)`` with per-frame modulation in ``state.extras``. These CPU
tests use the ``_RichCosmosBlock`` fakes (shared with the joint_self_attn
suite) to validate:

* ``force_per_token_t_mod`` yields a per-frame ``t_embedding_B_T_D``;
* the backbone-owned branch merge/split round-trips and reports token counts
  (``T·H·W``), not ``hidden_states.shape[1]`` (``T``);
* the 3-branch ``run_idm_training_loop`` runs end-to-end on the 5D state;
* the teacher-forcing mask is token-granular (action↔cond, not action↔noisy);
* train (3-branch) and deploy (Stage-2 frozen-video KV cache) action
  predictions match — the FastWAM-IDM core invariant, on Cosmos.
"""

from __future__ import annotations

import torch

from tests.test_cosmos_predict25_joint_self_attn import _build_rich_wrapper


def _make_cosmos_idm(num_blocks=2, *, action_dim=3, text_dim=12):
    """Build a DualSystemIDMArchitecture on a CosmosPredict25 rich-fake backbone."""
    from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture

    backbone = _build_rich_wrapper(num_blocks=num_blocks)
    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": action_dim,
        "dim": 16,
        "ffn_dim": 32,
        "num_heads": 4,
        "attn_head_dim": 4,
        "video_dim": 16,
        "text_dim": text_dim,
        "bridge_layers": tuple(range(num_blocks)),
        "idm_video_cond_noise_prob": 0.0,
        "mot_checkpoint_mixed_attn": False,
    }
    arch = DualSystemIDMArchitecture(cfg)
    arch.video_backbone = backbone
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.build_mot_driver()
    return arch, backbone


def _prep(backbone, *, timestep, force_per_token_t_mod, seed=0):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 16, 2, 4, 4, generator=g)
    context = torch.randn(1, 4, 12, generator=g)
    return backbone.prepare(
        input_latents=latents,
        context=context,
        timestep=timestep,
        force_per_token_t_mod=force_per_token_t_mod,
    )


# ----------------------------------------------------------------------
# force_per_token_t_mod → per-frame t_embedding
# ----------------------------------------------------------------------


def test_force_per_token_t_mod_yields_per_frame_t_embedding():
    _, backbone = _make_cosmos_idm()
    # latents (1,16,2,4,4) → grid frames f=2 (temporal patch 1).
    state_pf = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True)
    assert state_pf.grid_frames == 2
    assert state_pf.extras["t_embedding_B_T_D"].shape[1] == 2  # per-frame
    assert state_pf.extras["adaln_lora_B_T_3D"].shape[1] == 2

    state_default = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=False)
    assert state_default.extras["t_embedding_B_T_D"].shape[1] == 1  # broadcast


# ----------------------------------------------------------------------
# merge / split round-trip + token-count contract
# ----------------------------------------------------------------------


def test_merge_branches_reports_token_counts_and_concatenates_extras():
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)

    H, W = noisy.grid_height, noisy.grid_width
    merged, s_noisy, s_cond = backbone.merge_idm_video_branches(noisy, cond)

    # token counts are T·H·W, NOT hidden_states.shape[1] (== T for the 5D grid)
    assert s_noisy == noisy.grid_frames * H * W
    assert s_cond == cond.grid_frames * H * W
    assert s_noisy != noisy.hidden_states.shape[1]  # guard: shape[1] is frames

    # hidden state + frame extras concat along the frame axis
    assert merged.hidden_states.shape[1] == noisy.grid_frames + cond.grid_frames
    assert merged.grid_frames == noisy.grid_frames + cond.grid_frames
    assert merged.extras["t_embedding_B_T_D"].shape[1] == noisy.grid_frames + cond.grid_frames
    # per-token RoPE concat along the token axis (dim=0)
    assert merged.extras["rope_emb_L_1_1_D"].shape[0] == s_noisy + s_cond


def test_merge_split_round_trips():
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)
    noisy_ref = noisy.hidden_states.clone()
    cond_ref = cond.hidden_states.clone()
    # Per-frame modulation extras must survive the merge/split untouched (they feed
    # finalize per branch); split_branches deliberately does NOT write them back.
    t_emb_noisy_ref = noisy.extras["t_embedding_B_T_D"].clone()
    t_emb_cond_ref = cond.extras["t_embedding_B_T_D"].clone()

    merged, _, _ = backbone.merge_idm_video_branches(noisy, cond)
    # No block loop runs the merged state changes here; splitting must recover both.
    noisy_out, cond_out = backbone.split_idm_video_branches(merged, noisy, cond)
    assert torch.allclose(noisy_out.hidden_states, noisy_ref)
    assert torch.allclose(cond_out.hidden_states, cond_ref)
    assert torch.allclose(noisy_out.extras["t_embedding_B_T_D"], t_emb_noisy_ref)
    assert torch.allclose(cond_out.extras["t_embedding_B_T_D"], t_emb_cond_ref)


def test_merge_rejects_mismatched_spatial_layout():
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True)
    cond = backbone.prepare(
        input_latents=torch.randn(1, 16, 2, 4, 6),  # different W
        context=torch.randn(1, 4, 12),
        timestep=torch.zeros(1),
        force_per_token_t_mod=True,
    )
    import pytest

    with pytest.raises(ValueError, match="spatial token layout"):
        backbone.merge_idm_video_branches(noisy, cond)


def test_merge_rejects_broadcast_t_mod():
    # The module docstring requires both branches prepared with
    # force_per_token_t_mod=True; a broadcast (B, 1, D) t_embedding must be
    # rejected up front rather than silently producing a (B, 2, D) emb for the
    # 2T-frame grid (mirrors the Wan-side ndim==4 guard).
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=False, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)
    assert noisy.grid_frames == 2  # but t_embedding_B_T_D is broadcast (B, 1, D)
    import pytest

    with pytest.raises(ValueError, match="per-frame modulation"):
        backbone.merge_idm_video_branches(noisy, cond)


# ----------------------------------------------------------------------
# teacher-forcing mask is token-granular
# ----------------------------------------------------------------------


def test_teacher_forcing_mask_token_granularity():
    arch, backbone = _make_cosmos_idm()
    driver = arch._mot_driver
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)

    _, s_noisy, s_cond = backbone.merge_idm_video_branches(noisy, cond)
    s_action = 3
    tpf = driver._video_tokens_per_frame(noisy)
    mask = driver._build_teacher_forcing_mask(
        s_noisy_video=s_noisy,
        s_cond_video=s_cond,
        s_action=s_action,
        video_tokens_per_frame=tpf,
        device=torch.device("cpu"),
    )
    total = s_noisy + s_cond + s_action
    assert mask.shape == (total, total)
    # action (tail rows) attends to cond block but NOT the noisy block
    a0 = s_noisy + s_cond
    assert mask[a0:, s_noisy:a0].all()  # action → cond
    assert not mask[a0:, :s_noisy].any()  # action → noisy blocked
    # noisy and cond never attend across to each other
    assert not mask[:s_noisy, s_noisy:a0].any()


# ----------------------------------------------------------------------
# 3-branch training loop runs on the 5D state
# ----------------------------------------------------------------------


def test_idm_training_loop_runs_and_preserves_5d_state():
    arch, backbone = _make_cosmos_idm()
    arch.eval()
    driver = arch._mot_driver

    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)
    actions = torch.randn(1, 3, arch.action_backbone.action_dim)
    astate = arch.action_backbone.prepare_state(
        actions, torch.tensor([0.5]), context=torch.randn(1, 4, 12), context_mask=torch.ones(1, 4, dtype=torch.bool)
    )

    vstate_noisy, vstate_cond, astate = driver.run_idm_training_loop(noisy, cond, astate)
    pred = arch.action_backbone.extract_prediction(astate)

    assert vstate_noisy.hidden_states.shape == (1, 2, 2, 2, 16)  # 5D preserved
    assert torch.isfinite(vstate_noisy.hidden_states).all()
    assert torch.isfinite(pred).all()


# ----------------------------------------------------------------------
# train (3-branch) ↔ deploy (Stage-2 frozen-video KV cache) consistency
# ----------------------------------------------------------------------


def test_idm_train_action_matches_deploy_stage2_cosmos():
    """3-branch train action_pred must equal Stage-2 deploy action_pred on Cosmos.

    Train cond branch uses per-frame t=0 (force_per_token_t_mod); deploy prefill
    uses the scalar t=0 broadcast path. Both yield identical per-frame modulation
    (t=0 is uniform across frames), so the cached video K/V — and the action
    prediction — must match.
    """
    arch, backbone = _make_cosmos_idm()
    arch.eval()
    driver = arch._mot_driver

    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=5)
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=6)
    cond_deploy = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=False, seed=5)

    actions = torch.randn(1, 3, arch.action_backbone.action_dim)
    a_timestep = torch.tensor([0.5])
    context = torch.randn(1, 4, 12)
    context_mask = torch.ones(1, 4, dtype=torch.bool)

    astate_train = arch.action_backbone.prepare_state(actions, a_timestep, context=context, context_mask=context_mask)
    astate_deploy = arch.action_backbone.prepare_state(actions, a_timestep, context=context, context_mask=context_mask)

    _, _, astate_train = driver.run_idm_training_loop(noisy, cond, astate_train)
    pred_train = arch.action_backbone.extract_prediction(astate_train)

    video_seq_len = int(cond_deploy.grid_frames) * driver._video_tokens_per_frame(cond_deploy)
    kv_cache, _, _ = driver.prefill_video_cache(cond_deploy)
    astate_deploy = driver.run_action_with_video_cache(
        astate_deploy, video_kv_cache=kv_cache, video_seq_len=video_seq_len
    )
    pred_deploy = arch.action_backbone.extract_prediction(astate_deploy)

    assert torch.allclose(pred_train, pred_deploy, atol=1e-5, rtol=1e-5), (
        "Cosmos IDM train↔deploy action_pred diverged.\n"
        f"  train: {pred_train.flatten()[:8]}\n"
        f"  deploy: {pred_deploy.flatten()[:8]}"
    )
