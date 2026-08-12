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
