"""Deploy self-containment artifacts for cosmos3_edge (CPU).

save_deploy_assets must embed the vae component spec + copy the tokenizer's
structural files when model_path is a readable bundle, write-once semantics on
the components key, and no-op without mutation otherwise (the no-op case lives
in test_cosmos3_api_hygiene).
"""

import json

from omegaconf import OmegaConf

from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone
from openwam.model.video_backbone.cosmos3.component_specs import (
    copy_cosmos3_artifacts,
    generate_cosmos3_component_specs,
)


def _fake_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    tok = bundle / "text_tokenizer"
    tok.mkdir(parents=True)
    (tok / "tokenizer.json").write_text(json.dumps({"version": "fake"}))
    (tok / "tokenizer_config.json").write_text("{}")
    (tok / "special_tokens_map.json").write_text("{}")
    (tok / "chat_template.jinja").write_text("{{ messages }}")
    return bundle


def test_component_specs_emission(tmp_path):
    bundle = _fake_bundle(tmp_path)
    specs = generate_cosmos3_component_specs(str(bundle), has_vae=True)
    assert specs == {"components": [{"attr": "vae", "source": "state_dict", "sub_module": "vae"}]}
    assert generate_cosmos3_component_specs(str(bundle), has_vae=False) == {"components": []}
    assert generate_cosmos3_component_specs(str(tmp_path / "missing"), has_vae=True) is None


def test_artifact_copy_repairs_partial_state(tmp_path):
    bundle = _fake_bundle(tmp_path)
    out = tmp_path / "ckpt"
    out.mkdir()
    copy_cosmos3_artifacts(str(out), str(bundle))
    copied = sorted(p.name for p in (out / "text_tokenizer").iterdir())
    assert copied == [
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    # Idempotent: second call leaves the populated dir untouched.
    copy_cosmos3_artifacts(str(out), str(bundle))
    assert sorted(p.name for p in (out / "text_tokenizer").iterdir()) == copied
    # Repair: a partial copy (interrupted save) is completed, not frozen.
    (out / "text_tokenizer" / "tokenizer.json").unlink()
    copy_cosmos3_artifacts(str(out), str(bundle))
    assert (out / "text_tokenizer" / "tokenizer.json").exists()


def test_save_deploy_assets_embeds_spec_and_copies(tmp_path):
    bundle = _fake_bundle(tmp_path)
    out = tmp_path / "ckpt2"
    out.mkdir()
    vb = Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)
    object.__setattr__(vb, "vae", object())  # marker: vae child present

    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": str(bundle)}}})
    vb.save_deploy_assets(str(out), cfg)
    comps = OmegaConf.to_container(OmegaConf.select(cfg, "model.video_backbone.components"))
    assert comps == [{"attr": "vae", "source": "state_dict", "sub_module": "vae"}]
    assert (out / "text_tokenizer" / "tokenizer.json").exists()

    # Write-once: a second save with a different bundle must not overwrite.
    cfg2 = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "model_path": str(bundle),
                    "components": [{"attr": "vae", "source": "state_dict", "sub_module": "custom"}],
                }
            }
        }
    )
    vb.save_deploy_assets(str(out), cfg2)
    comps2 = OmegaConf.to_container(OmegaConf.select(cfg2, "model.video_backbone.components"))
    assert comps2[0]["sub_module"] == "custom"
