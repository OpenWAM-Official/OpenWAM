"""Registry + fail-fast config validation for the cosmos3_edge backbone.

Every bad-config case must raise from the pure-Python validation phase (before
any heavy import), so these tests run identically on CPU CI without diffusers.
"""

import pytest

from openwam.model.video_backbone import (
    _VIDEO_BACKBONE_REGISTRY,
    Cosmos3EdgeVideoBackbone,
    build_video_backbone,
    register_video_backbone,
)


def test_cosmos3_edge_registered():
    assert _VIDEO_BACKBONE_REGISTRY["cosmos3_edge"] is Cosmos3EdgeVideoBackbone


def test_double_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_video_backbone("cosmos3_edge")(Cosmos3EdgeVideoBackbone)


def _cfg(**vb):
    return {"model": {"video_backbone": vb}}


def test_unknown_name_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="cosmos3_edge"):
        build_video_backbone("cosmos3_edge", _cfg(name="cosmos3_nano", model_path="/tmp/x"))


def test_missing_model_path_raises():
    with pytest.raises(ValueError, match="model_path"):
        build_video_backbone("cosmos3_edge", _cfg(name="cosmos3_edge"))


def test_bad_text_dropout_raises():
    with pytest.raises(ValueError, match="text_encoder_dropout"):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path="/tmp/x", text_encoder_dropout=1.5),
        )


def test_bad_max_text_tokens_raises():
    with pytest.raises(ValueError, match="max_text_tokens"):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path="/tmp/x", max_text_tokens=2),
        )


def test_bad_fps_raises():
    with pytest.raises(ValueError, match="fps"):
        build_video_backbone("cosmos3_edge", _cfg(name="cosmos3_edge", model_path="/tmp/x", fps=0))


def test_freeze_und_false_rejected():
    with pytest.raises(NotImplementedError, match="freeze_und"):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path="/tmp/x", freeze_und=False),
        )


def test_freeze_is_not_gated_on_ckpt_dir(tmp_path, monkeypatch):
    """``ckpt_dir`` is set by finetune/resume, not just deploy.

    Regression: the freeze used to sit behind ``if freeze_und and not deploy``
    with ``deploy = ckpt_dir is not None``, so any
    ``training.finetune_ckpt_path`` / ``resume_ckpt_path`` run handed the whole
    und tower to the optimizer — 2.19x the trainable surface, for parameters
    that run under ``no_grad`` and can never receive a gradient.

    Driven on the meta shell (no weights, no GPU) with the tokenizer stubbed.
    """
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    from transformers import PreTrainedTokenizerFast

    from openwam.model.video_backbone.cosmos3.pipeline_builder import build_cosmos3_pipeline

    (tmp_path / "text_tokenizer").mkdir()
    monkeypatch.setattr(
        PreTrainedTokenizerFast, "from_pretrained", classmethod(lambda cls, *a, **k: object()), raising=True
    )

    holder = build_cosmos3_pipeline(_cfg(name="cosmos3_edge", model_path=str(tmp_path)), ckpt_dir=str(tmp_path))
    net = holder.net

    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert trainable, "everything frozen — the gen pathway must stay trainable"
    # und tower: shared embedding, final norm, per-layer und attn/MLP/norms.
    leaked = sorted(
        n
        for n in trainable
        if n.startswith(("embed_tokens", "norm.", "action_proj_", "audio_proj_"))
        or any(f".self_attn.{c}." in n for c in ("to_q", "to_k", "to_v", "to_out", "k_norm_und_for_gen"))
        or any(f".{c}." in n for c in ("input_layernorm", "post_attention_layernorm", "mlp"))
    )
    assert not leaked, f"und/native-head params left trainable on the ckpt_dir path: {leaked[:6]}"
    # The gen half is untouched by the freeze.
    assert any(".self_attn.add_q_proj." in n for n in trainable)


def test_nonexistent_model_path_fails_fast(tmp_path):
    # With diffusers installed this is a FileNotFoundError on the missing
    # transformer/ subfolder; without diffusers it is the install-hint
    # ImportError. Both are acceptable fail-fast outcomes on CPU.
    with pytest.raises((FileNotFoundError, ImportError)):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path=str(tmp_path / "nonexistent")),
        )


def test_resume_path_materializes_the_shell(tmp_path, monkeypatch):
    """``resume_ckpt_path`` must not be left holding a meta shell.

    Resume builds from ``ckpt_dir`` like deploy and finetune, but deliberately
    skips the safetensors load (accelerate's ``load_state`` restores the weights
    only after ``prepare``). Nothing therefore materializes the shell before
    ``set_dtype_device``, which raises "Cannot copy out of meta tensor" on a
    meta parameter — i.e. a cosmos3 run could be trained but never resumed.
    """
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    import torch
    from transformers import PreTrainedTokenizerFast

    from openwam.model.video_backbone.cosmos3 import pipeline_builder as pb

    (tmp_path / "text_tokenizer").mkdir()
    monkeypatch.setattr(
        PreTrainedTokenizerFast, "from_pretrained", classmethod(lambda cls, *a, **k: object()), raising=True
    )
    # The 3.1B shell is free on meta but ~6 GiB once materialized, so swap in a
    # tiny transformer for the allocation while keeping the real code path.
    tiny = dict(
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
    monkeypatch.setattr(pb, "_COSMOS3_EDGE_NET_KWARGS", tiny)
    monkeypatch.setattr(
        pb, "_COSMOS3_EDGE_GEOMETRY", dict(dim=12, num_layers=2, num_heads=2, head_dim=6, context_dim=12)
    )
    monkeypatch.setattr(pb, "_WEIGHTLESS_CONFIG_KEYS", ())

    class _TinyVAE(torch.nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 1, 1)
            self.config = type("C", (), {"latents_mean": [0.0], "latents_std": [1.0]})()

    monkeypatch.setattr(pb, "_COSMOS3_VAE_KWARGS", {})
    # The builder imports AutoencoderKLWan inside the function, so patch it at
    # the source module rather than on pb.
    monkeypatch.setattr("diffusers.AutoencoderKLWan", _TinyVAE, raising=False)

    cfg = _cfg(name="cosmos3_edge", model_path=str(tmp_path))
    deploy_holder = pb.build_cosmos3_pipeline(cfg, ckpt_dir=str(tmp_path))
    assert any(p.is_meta for p in deploy_holder.net.parameters()), "deploy should still get the cheap meta shell"

    resume_holder = pb.build_cosmos3_pipeline(cfg, ckpt_dir=str(tmp_path), materialize_weights=True)
    assert not any(p.is_meta for p in resume_holder.net.parameters())
    assert not any(b.is_meta for b in resume_holder.net.buffers())
    # Zeroed, not uninitialized: load_state overwrites everything, but NaN/Inf
    # from raw `to_empty` storage can be read by DeepSpeed while flattening.
    assert all(torch.all(p == 0) for p in resume_holder.net.parameters())
    # The rotary table is re-registered in fp32 after materialization.
    assert resume_holder.net.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.any(resume_holder.net.rotary_emb.inv_freq != 0)
    # set_dtype_device is what used to blow up on the meta shell.
    from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone

    vb = Cosmos3EdgeVideoBackbone(
        net=resume_holder.net,
        vae=None,
        tokenizer=None,
        dim=12,
        num_layers=2,
        num_heads=2,
        head_dim=6,
        context_dim=12,
    )
    vb.set_dtype_device(torch.bfloat16, torch.device("cpu"))


def test_ckpt_loader_marks_resume_but_not_finetune(tmp_path, monkeypatch):
    """The flag must come from the finetune/resume distinction, not from ckpt_dir.

    Both paths set ``_ckpt_dir``; only resume skips the weight load, so only
    resume needs real storage.
    """
    from omegaconf import OmegaConf

    from openwam.train.utils import ckpt_model_loader

    OmegaConf.save(
        OmegaConf.create({"model": {"video_backbone": {"name": "cosmos3_edge", "model_path": "/nonexistent"}}}),
        tmp_path / "config.yaml",
    )
    captured = {}

    class _Arch:
        registry_name = "dual_system_joint_self_attn"
        params = {"video_backbone": {}}

    class _Built:
        def load_checkpoint(self, weights):
            captured["loaded"] = weights

    def _capture(name, params):
        captured["vb"] = params["video_backbone"]
        return _Built()

    monkeypatch.setattr("openwam.model.resolve_architecture_config", lambda _m: _Arch(), raising=True)
    monkeypatch.setattr("openwam.model.build_architecture", _capture, raising=True)
    monkeypatch.setattr(ckpt_model_loader, "find_latest_weights", lambda d: "w.safetensors", raising=True)

    ckpt_model_loader.build_architecture_from_ckpt_dir(str(tmp_path), weights_required=False)
    assert captured["vb"]["_materialize_weights"] is True, "resume must request real storage"
    assert captured["vb"]["_ckpt_dir"] == str(tmp_path)
    assert "loaded" not in captured, "resume must not load safetensors — that is why it needs real storage"

    captured.clear()
    ckpt_model_loader.build_architecture_from_ckpt_dir(str(tmp_path), weights_required=True)
    assert captured["vb"]["_materialize_weights"] is False, "finetune loads weights; the meta shell is fine"
    assert captured["loaded"] == "w.safetensors"
