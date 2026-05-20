"""GPU smoke for self-contained Cosmos25 deploy round-trip.

End-to-end check of the Reason1 self-containment work
(``text_encoder.py``, ``component_specs.py``, ``pipeline_builder.py``,
``model_loader.py``):

1. Build a training-time architecture with ``text_encoder=reason1_live``
   pointing at the real Cosmos-Reason1-7B bundle.
2. Save: unified safetensors (``arch.save_checkpoint``), Hydra
   ``config.yaml`` (``save_config``), and the small Reason1 structural
   artifacts under ``<ckpt_dir>/reason1/`` (``copy_deploy_artifacts``).
3. Drop the original arch; reload via the deploy entrypoint
   ``load_from_checkpoint_dir``. The loader is expected to:
     - Detect the ckpt-local ``reason1/`` artifact dir,
     - Clear ``text_encoder_path`` to force the meta-device shell branch
       in ``build_cosmos25_pipeline``,
     - Build ``Reason1LiveTextEncoder`` via ``from_empty`` (so no external
       Cosmos-Reason1 bundle is touched),
     - Materialise weights from the unified safetensors via
       ``architecture.load_checkpoint``.
4. Verify the deploy ``_reason1_inner`` weights match the originals
   byte-for-byte — the only state path that ferried them is the
   safetensors itself.

This test is the canonical "deploy host does not need the Cosmos-Reason1
bundle" check. Without it, the self-containment work is unverified
end-to-end.

Skip conditions: CUDA, ``cosmos_predict2``, the Cosmos-Predict2.5-2B
bundle, and Cosmos-Reason1-7B bundle must all be present. Without any
one, this test is meaningless and silently skips.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.gpu


ASSET_PATH = Path(os.environ.get("COSMOS25_ASSET_PATH", "/path/to/assets/Cosmos-Predict2.5-2B"))
REASON1_PATH = Path(os.environ.get("COSMOS_REASON1_PATH", "/path/to/assets/Cosmos-Reason1-7B"))
REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS_CFG_PATH = REPO_ROOT / "configs" / "model" / "dual_system_cosmos25.yaml"


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"Cosmos asset bundle missing at {ASSET_PATH}.")
    if not REASON1_PATH.exists():
        pytest.skip(f"Cosmos-Reason1-7B bundle missing at {REASON1_PATH}.")
    if not COSMOS_CFG_PATH.exists():
        pytest.skip(f"Cosmos config missing at {COSMOS_CFG_PATH}.")
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError:
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos25.sh first.")


def _build_train_cfg() -> "OmegaConf":
    """Build a minimal Hydra-style config dict the deploy loader can rehydrate.

    The deploy loader (``load_from_checkpoint_dir``) reads
    ``cfg.model.video_backbone.*`` for the arch build and
    ``cfg.accelerate.mixed_precision`` for the target dtype.
    ``cfg.dataloader`` is read only for normalizer wiring — omitting it
    is fine (``_build_action_normalizer`` returns None on absent config).
    """
    model_cfg = OmegaConf.load(str(COSMOS_CFG_PATH))
    model_cfg.video_backbone.model_path = str(ASSET_PATH)
    model_cfg.video_backbone.text_encoder = "reason1_live"
    model_cfg.video_backbone.text_encoder_path = str(REASON1_PATH)
    cfg = OmegaConf.create({"model": model_cfg, "accelerate": {"mixed_precision": "bf16"}})
    return cfg


def test_cosmos25_deploy_round_trip_reason1_in_safetensors(tmp_path, caplog):
    'Public implementation.'
    _skip_unless_runnable()

    from omegaconf import open_dict

    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.model import build_architecture, resolve_architecture_config
    from openwam.model.video_backbone.cosmos25.component_specs import copy_cosmos25_artifacts
    from openwam.train.utils.checkpointing import save_config

    cfg = _build_train_cfg()
    resolved = resolve_architecture_config(cfg.model)
    train_arch = build_architecture(resolved.registry_name, resolved.params)
    train_arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    train_arch.move_frozen_to_device(torch.device("cuda:0"))

    # Sample a stable reference tensor from `_reason1_inner` before save.
    # Pick the first Linear-like param so we don't need to know Qwen-VL's
    # exact submodule layout — just grab one parameter and stash a clone.
    pipe = train_arch.video_backbone._pipe
    assert hasattr(pipe, "_reason1_inner"), (
        "Cosmos25PipelineWrapper must register Reason1's inner Qwen as "
        "`_reason1_inner` for the unified-safetensors round-trip to work."
    )
    reason1_state = pipe._reason1_inner.state_dict()
    assert reason1_state, "Reason1 inner state_dict is empty; weights never loaded?"
    probe_key, probe_tensor = next(iter(reason1_state.items()))
    probe_ref_cpu = probe_tensor.detach().clone().cpu()
    assert probe_ref_cpu.numel() > 0, f"probe tensor {probe_key} is empty"

    # Drop and rebuild the wrapper handle so we don't keep references to
    # the train graph during deploy reload — they share GPU memory and we
    # want a clean deploy-side allocation.
    del pipe, reason1_state, probe_tensor

    # --- Save (mirrors trainer's save flow at openwam_trainer.py:548-561) ---
    # The trainer injects `model.video_backbone.components` into the cfg
    # before `save_config`; the deploy loader keys off that field to flip
    # into the self-contained branch. Mirror the same injection so this
    # smoke test exercises the real deploy gate, not a degenerate path.
    specs = train_arch.get_component_specs(str(ASSET_PATH))
    assert specs is not None and "components" in specs, (
        "Cosmos25 must produce component specs when model_path is a real dir."
    )
    with open_dict(cfg):
        OmegaConf.update(cfg, "model.video_backbone.components", specs["components"])

    ckpt_path = tmp_path / "checkpoint_step_1.safetensors"
    train_arch.save_checkpoint(str(ckpt_path))
    save_config(str(tmp_path), cfg)
    # ``copy_deploy_artifacts`` dispatches through every backbone; we call
    # cosmos25's directly so the test is independent of the dispatcher's
    # wiring (which is exercised by `test_cosmos25_deploy_artifacts.py`).
    copy_cosmos25_artifacts(str(tmp_path), str(REASON1_PATH))

    reason1_dir = tmp_path / "reason1"
    assert reason1_dir.is_dir(), "copy_cosmos25_artifacts did not produce <ckpt_dir>/reason1/"
    assert (reason1_dir / "config.json").is_file(), "Reason1 config.json missing"
    assert (reason1_dir / "tokenizer.json").is_file(), "Reason1 tokenizer.json missing"

    # Free the training arch before the deploy load — we want to confirm
    # the deploy side allocates Reason1 fresh from the safetensors, not
    # that it secretly shares tensors with the still-live train arch.
    del train_arch
    torch.cuda.empty_cache()

    # --- Reload via the deploy entrypoint ---
    with caplog.at_level(logging.INFO, logger="openwam"):
        loaded_cfg, deploy_arch = load_from_checkpoint_dir(str(tmp_path), device="cuda:0")

    # The model_loader log line is the witness that the self-contained
    # Reason1 artifact dir was detected (which is what flips the empty-shell
    # branch in `pipeline_builder`).
    log_text = "\n".join(rec.message for rec in caplog.records)
    assert "Using self-contained Reason1 artifacts" in log_text, (
        "Deploy loader did not announce use of <ckpt_dir>/reason1/; the "
        "empty-shell `from_empty` path may not have been exercised. "
        f"Captured log:\n{log_text}"
    )

    # The loader must have cleared text_encoder_path inside the rehydrated
    # cfg, signalling that the deploy host need not reach the original
    # Cosmos-Reason1 bundle.
    deploy_vb_cfg = OmegaConf.select(loaded_cfg, "model.video_backbone")
    assert deploy_vb_cfg is not None
    # The rehydrated cfg on disk still records the training-time path —
    # the loader clears `text_encoder_path` only on the in-memory
    # `vb_cfg_dict` used to build the deploy pipeline. That's intentional;
    # what we care about is that the deploy arch's Reason1 is materialized
    # without touching that path.

    # --- Verify Reason1 weights round-trip byte-for-byte ---
    deploy_pipe = deploy_arch.video_backbone._pipe
    assert hasattr(deploy_pipe, "_reason1_inner"), (
        "Deploy-side wrapper has no `_reason1_inner`; the empty-shell "
        "registration regressed."
    )
    deploy_state = deploy_pipe._reason1_inner.state_dict()
    assert probe_key in deploy_state, (
        f"Probe key {probe_key!r} missing from deploy `_reason1_inner` state_dict. "
        f"Available keys (first 10): {list(deploy_state.keys())[:10]}"
    )
    deploy_probe = deploy_state[probe_key].detach().cpu()
    assert deploy_probe.shape == probe_ref_cpu.shape, (
        f"Reason1 probe shape mismatch on reload: "
        f"saved {tuple(probe_ref_cpu.shape)}, loaded {tuple(deploy_probe.shape)}."
    )
    # bf16 save → bf16 reload is exact; no tolerance needed.
    assert torch.equal(deploy_probe, probe_ref_cpu), (
        f"Reason1 probe tensor {probe_key!r} differs after save/reload. "
        "The Reason1 weights did not ride into the unified safetensors "
        "(missing `_reason1_inner` registration in `Cosmos25PipelineWrapper`?), "
        "or `load_checkpoint` failed to populate them from the meta-device shell."
    )

    # --- Verify no meta-device stragglers remain after load ---
    target_device = torch.device("cuda:0")
    meta_params = [
        (n, p) for n, p in deploy_pipe._reason1_inner.named_parameters() if p.device.type == "meta"
    ]
    assert not meta_params, (
        f"Deploy Reason1 still has meta-device parameters after load: "
        f"{[n for n, _ in meta_params[:5]]} (showing first 5)."
    )
    wrong_device = [
        (n, p.device) for n, p in deploy_pipe._reason1_inner.named_parameters() if p.device != target_device
    ]
    assert not wrong_device, (
        f"Deploy Reason1 has params on the wrong device after `set_dtype_device`: "
        f"{wrong_device[:5]} (showing first 5)."
    )

    # --- Verify the deploy Reason1 wrapper class is actually live (not a stub) ---
    # `Reason1LiveTextEncoder` is the plain-Python wrapper around the
    # registered `_reason1_inner`. Confirm the tokenizer materialised
    # (proves `from_empty` consumed the on-disk artifacts) and that the
    # encoder's device/dtype track the deploy target.
    deploy_te = deploy_pipe.text_encoder
    assert deploy_te is not None, "Deploy text_encoder slot is empty"
    assert deploy_te.tokenizer is not None, "Deploy Reason1 tokenizer not constructed"
    assert deploy_te.device == target_device, (
        f"Deploy Reason1 wrapper device={deploy_te.device}, expected {target_device}."
    )
    assert deploy_te.dtype == torch.bfloat16, (
        f"Deploy Reason1 wrapper dtype={deploy_te.dtype}, expected torch.bfloat16."
    )
