"""``finetune_ckpt_path``: the ckpt's config.yaml is the reconstruction base, the
live cfg.model is the authority.

Regression: the finetune path built the architecture from
``ckpt_cfg.model`` alone, so every ``model.*`` value the operator set in
``configs/`` (or on the CLI) was silently discarded — a run configured with
``attention_mask_mode=isolated`` trained the checkpoint's ``mutual`` instead.
Worse, ``save_config`` wrote the LIVE cfg into the new run dir, so the new
checkpoint's config.yaml advertised values the model had never been built with
and deploying it rebuilt a different model than the one that trained.

``build_architecture`` is stubbed out (the real one loads ~25 GB of weights), so
these run on CPU with no checkpoint; everything up to it — config merge,
``resolve_architecture_config``, the ``_source`` dict the backbone build reads —
is the real code path.
"""

import pytest
from omegaconf import OmegaConf, open_dict

from openwam.train.utils import ckpt_model_loader as cml

# Minimal stand-in for a saved run's config.yaml: the two ``components`` /
# ``tokenizer`` keys only a saved config carries, plus one overridable value per
# section (video_backbone / action_backbone / architecture).
CKPT_CFG = {
    "model": {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/ckpt-host/Wan2.2-TI2V-5B",
            "from_scratch": False,
            "shift_video": 5.0,
            "components": [
                {
                    "attr": "dit",
                    "model_class": "openwam.model.video_backbone.wan.models.dit.WanModel",
                    "extra_kwargs": {"dim": 3072, "num_layers": 30, "num_heads": 24},
                }
            ],
            "tokenizer": {"class": "x.Tok", "attr": "tokenizer", "subdir": "tokenizer/google/umt5-xxl"},
        },
        "action_backbone": {"dim": 1024, "ffn_dim": 4096, "shift_action": 5.0},
        "freeze": ["video_backbone.vae"],
        "architecture": {
            "framework": "dual_system",
            "variant": "joint_self_attn",
            "action_dim": 80,
            "use_proprioception": True,
            "state_dim": 80,
            "bridge_layers": [0, 1, 2],
            "bridge_interval": 1,
            "mot_checkpoint_mixed_attn": True,
            "attention_mask_mode": "mutual",
            "video_attention_mask_mode": "first_frame_causal",
        },
    },
    "training": {"batch_size": 24},
}


def _write_ckpt_cfg(tmp_path, **model_patch):
    """Write a ckpt config.yaml, optionally patching its model section."""
    cfg = OmegaConf.create(CKPT_CFG)
    for dotted, val in model_patch.items():
        OmegaConf.update(cfg, f"model.{dotted.replace('__', '.')}", val)
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    return str(tmp_path)


def _live_cfg(**model_patch):
    """A live Hydra-shaped run config: same yaml family as the ckpt, minus the
    saved-only reconstruction specs, plus struct mode as Hydra hands it over."""
    model = OmegaConf.to_container(OmegaConf.create(CKPT_CFG["model"]), resolve=True)
    del model["video_backbone"]["components"]
    del model["video_backbone"]["tokenizer"]
    model["video_backbone"]["model_path"] = "/live-host/Wan2.2-TI2V-5B"
    cfg = OmegaConf.create({"model": model, "training": {"batch_size": 4}, "project": {"seed": 42}})
    for dotted, val in model_patch.items():
        OmegaConf.update(cfg, f"model.{dotted.replace('__', '.')}", val)
    OmegaConf.set_struct(cfg, True)
    return cfg


@pytest.fixture
def capture(monkeypatch):
    """Stub ``build_architecture`` / weight discovery; capture the build params."""
    captured: dict = {}

    class _Built:
        def load_checkpoint(self, weights):
            captured["loaded"] = weights

    def _build(name, params, **kw):
        captured["registry_name"] = name
        captured["params"] = params
        return _Built()

    monkeypatch.setattr("openwam.model.build_architecture", _build, raising=True)
    monkeypatch.setattr(cml, "find_latest_weights", lambda d: "w.safetensors", raising=True)
    return captured


def test_live_model_cfg_overrides_the_ckpt_config(tmp_path, capture):
    """Every live ``model.*`` value must reach the built architecture."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(
        architecture__attention_mask_mode="isolated",
        architecture__mot_checkpoint_mixed_attn=False,
        architecture__video_attention_mask_mode="bidirectional",
        action_backbone__shift_action=9.0,
        video_backbone__shift_video=1.5,
    )

    cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=True, override_cfg=cfg)

    params = capture["params"]
    assert params["attention_mask_mode"] == "isolated"
    assert params["mot_checkpoint_mixed_attn"] is False
    assert params["video_attention_mask_mode"] == "bidirectional"
    assert params["shift_action"] == 9.0
    assert params["video_backbone"]["shift_video"] == 1.5
    # The backbone build reads video_backbone values back off ``_source``, not
    # off ``params`` — an override that reaches only one of the two would give
    # the architecture and its backbone different schedules.
    assert params["video_backbone"]["_source"]["shift_video"] == 1.5
    assert capture["loaded"] == "w.safetensors"


def test_ckpt_reconstruction_specs_survive_the_merge(tmp_path, capture):
    """``components`` / ``tokenizer`` exist only in the ckpt config: the merge
    must not drop them, or the skeletons can't be rebuilt without model_path."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__attention_mask_mode="isolated")

    cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=True, override_cfg=cfg)

    source = capture["params"]["video_backbone"]["_source"]
    assert source["components"][0]["extra_kwargs"]["num_layers"] == 30
    assert source["tokenizer"]["subdir"] == "tokenizer/google/umt5-xxl"
    assert capture["params"]["video_backbone"]["_ckpt_dir"] == ckpt_dir
    # ...and they must also land in cfg, which is what save_config() writes —
    # otherwise the NEW run's checkpoints stop being self-contained.
    assert OmegaConf.select(cfg, "model.video_backbone.components") is not None
    assert OmegaConf.select(cfg, "model.video_backbone.tokenizer") is not None


def test_cfg_model_is_rewritten_to_exactly_what_was_built(tmp_path, capture):
    """save_config() writes cfg.model. It must describe the built architecture.

    Otherwise the new checkpoint's config.yaml advertises values the model was
    never built with, and the deploy loader rebuilds a different model.
    """
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__attention_mask_mode="isolated", video_backbone__shift_video=1.5)

    cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=True, override_cfg=cfg)

    params = capture["params"]
    assert cfg.model.architecture.attention_mask_mode == params["attention_mask_mode"] == "isolated"
    assert cfg.model.video_backbone.shift_video == params["video_backbone"]["shift_video"] == 1.5
    # The ckpt-sourced DiT geometry is recorded too, so a later deploy sizes the
    # skeleton the same way this run did.
    assert cfg.model.video_backbone.components[0].extra_kwargs.num_layers == 30
    # Struct mode survives the node swap: a typo'd model key must still raise.
    with pytest.raises(Exception):
        cfg.model.not_a_real_key = 1


def test_explicit_null_in_live_cfg_clears_a_ckpt_value(tmp_path, capture):
    """``bridge_layers: null`` in train.yaml means "derive from bridge_interval".

    A merge that treated the live ``None`` as "unset" would keep the ckpt's
    explicit list and silently size ActionDiT off the wrong layer selection.
    """
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__bridge_layers=None, architecture__bridge_interval=2)

    cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=True, override_cfg=cfg)

    assert capture["params"]["bridge_layers"] is None
    assert capture["params"]["bridge_interval"] == 2


def test_video_backbone_name_mismatch_raises(tmp_path, capture):
    """The ckpt's component specs are bound to its backbone family."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(video_backbone__name="cosmos3_edge")

    with pytest.raises(ValueError, match="model.video_backbone.name"):
        cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=True, override_cfg=cfg)


def test_architecture_framework_mismatch_raises(tmp_path, capture):
    """A different framework is a different architecture class and state_dict."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    cfg = _live_cfg(architecture__framework="single_system")

    with pytest.raises(ValueError, match="model.architecture.framework"):
        cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=True, override_cfg=cfg)


def test_resume_keeps_the_ckpt_config(tmp_path, capture):
    """Resume continues ONE run whose config.yaml is reused untouched, so the
    ckpt config stays authoritative — a live override there would make that
    file lie about the model being trained."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)

    cml.build_architecture_from_ckpt_dir(ckpt_dir, weights_required=False, override_cfg=None)

    params = capture["params"]
    assert params["attention_mask_mode"] == "mutual"
    assert params["video_backbone"]["shift_video"] == 5.0
    assert params["video_backbone"]["_materialize_weights"] is True
    assert "loaded" not in capture, "resume must not load safetensors"


def test_resume_reports_the_live_keys_it_discards(tmp_path, caplog):
    """Ckpt-wins on resume must not be silent."""
    ckpt_dir = _write_ckpt_cfg(tmp_path)
    ckpt_cfg = OmegaConf.load(tmp_path / "config.yaml")
    cfg = _live_cfg(architecture__attention_mask_mode="isolated")

    with caplog.at_level("WARNING"):
        cml.warn_live_model_cfg_ignored(ckpt_cfg, cfg, ckpt_dir)

    assert "model.architecture.attention_mask_mode" in caplog.text
    assert "'mutual'" in caplog.text and "'isolated'" in caplog.text


def test_ckpt_config_without_a_model_section_raises(tmp_path, capture):
    """Not a run dir written by this trainer — say so instead of an AttributeError."""
    OmegaConf.save(OmegaConf.create({"training": {"batch_size": 4}}), tmp_path / "config.yaml")

    with pytest.raises(ValueError, match="no `model:` section"):
        cml.build_architecture_from_ckpt_dir(str(tmp_path), weights_required=True, override_cfg=None)


def test_trainer_reads_freeze_from_the_merged_cfg(tmp_path, monkeypatch):
    """``OpenWAMTrainer.__init__`` binds ``m = cfg.model`` before the merge
    replaces that node, so it must rebind afterwards. Otherwise the run freezes
    off the pre-merge config while save_config() writes the merged one — the
    saved config.yaml would claim a freeze list the run never applied."""
    import torch.nn as nn

    ckpt_dir = _write_ckpt_cfg(tmp_path)
    # The ckpt carries `freeze`; the live config omits it, so only the merged
    # config has one. Pre-merge `m` would yield [].
    cfg = _live_cfg()
    with open_dict(cfg):
        del cfg.model["freeze"]
        cfg.training = {
            "finetune_ckpt_path": ckpt_dir,
            "resume_ckpt_path": None,
            "initialize_model_on_cpu": False,
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
        }
        cfg.project = {"seed": None}

    frozen: dict = {}

    class _Arch(nn.Module):
        backbones: dict = {}
        dtype = None
        device = None

        def set_dtype_device(self, *a):
            pass

        def freeze_modules(self, names):
            frozen["names"] = list(names)
            return []

        def init_training_schedulers(self, n):
            pass

        def set_training_runtime(self, **kw):
            pass

        def load_checkpoint(self, weights):
            pass

    monkeypatch.setattr("openwam.model.build_architecture", lambda name, params, **kw: _Arch(), raising=True)
    monkeypatch.setattr(cml, "find_latest_weights", lambda d: "w.safetensors", raising=True)

    from openwam.train.openwam_trainer import OpenWAMTrainer

    OpenWAMTrainer(cfg, accelerator=None, dataset=None)

    assert frozen["names"] == ["video_backbone.vae"], "freeze must come from the merged (ckpt+live) cfg.model"
    assert list(cfg.model.freeze) == ["video_backbone.vae"], "save_config() must record the same list"


def test_diff_model_cfgs_buckets_keys():
    """Lists stay leaves so a components diff doesn't bury the real changes."""
    ckpt = OmegaConf.create({"a": {"x": 1, "y": 2}, "specs": [{"k": 1}], "only_ckpt": 3})
    live = OmegaConf.create({"a": {"x": 1, "y": 9}, "specs": [{"k": 1}], "only_live": 4})

    overridden, live_only, ckpt_only = cml.diff_model_cfgs(ckpt, live)

    assert overridden == [("a.y", 2, 9)]
    assert live_only == [("only_live", 4)]
    assert ckpt_only == [("only_ckpt", 3)]
