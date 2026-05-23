"""Phase 4 §4.4 last checkbox — deploy round-trip sanity check.

Validates the contract that ``scripts/deploy.py --ckpt-dir <sana_ckpt>`` relies
on:

1. ``architecture.save_checkpoint(<path>)`` → safetensors that round-trips
   strict-load through :func:`openwam.deploy.model_loader.load_from_checkpoint_dir`
   on a SANA-Video 2B backbone.
2. The deploy loader can rebuild a SANA-backed ``DualSystemSelfAttn``
   architecture from a minimal training-time ``config.yaml`` snapshot
   (``model:`` block + ``accelerate.mixed_precision``) plus the saved
   safetensors.
3. ``JointInferenceEngine(cfg, architecture)`` instantiates without
   touching the policy server — the engine wrap is the next deploy step
   after the loader.

The published SANA asset bundle at ``/path/to/assets/SANA-Video_2B_480p``
is *not* an OpenWAM training checkpoint — it's pretrained weights only. The
test fabricates a minimal ckpt dir on a session-scoped tmp_path by:

- Building a SANA-backed arch from ``configs/model/dual_system_self_attn_sana.yaml``
  with the ``model_path`` swapped to the local asset bundle.
- Saving the arch state_dict to ``<tmp>/checkpoint_step_0.safetensors``.
  ``SanaPipe`` is a ``@dataclass``, so the video DiT and VAE live *outside*
  the ``nn.Module`` tree — their weights are NOT in the safetensors and
  get re-loaded from ``model_path`` every time the arch rebuilds. The
  safetensors only carries ActionDiT + proprio_encoder (~0.5 GB).
- Writing a minimal training-time ``config.yaml`` (model + accelerate).
- Invoking ``load_from_checkpoint_dir(<tmp>, device='cuda:0')`` and
  asserting representative ActionDiT parameters round-trip bitwise
  through the safetensors, and the video DiT is freshly re-loaded.

Skipped unless CUDA + the SANA-Video 2B asset bundle + ``third_party/Sana``
import chain are all available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.gpu


ASSET_PATH = Path(os.environ.get("SANA_ASSET_PATH", "/path/to/assets/SANA-Video_2B_480p"))
REPO_ROOT = Path(__file__).resolve().parents[1]
SANA_CFG_PATH = REPO_ROOT / "configs" / "model" / "dual_system_self_attn_sana.yaml"
DEPLOY_CFG_PATH = REPO_ROOT / "configs" / "deploy.yaml"


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"SANA asset bundle missing at {ASSET_PATH}.")
    if not (ASSET_PATH / "checkpoints").exists():
        pytest.skip(f"SANA checkpoints missing at {ASSET_PATH}/checkpoints.")
    if not SANA_CFG_PATH.exists():
        pytest.skip(f"SANA config missing at {SANA_CFG_PATH}.")
    if not _sana_importable():
        pytest.skip("third_party/Sana not importable.")


def _build_training_cfg_snapshot(model_cfg) -> OmegaConf:
    """Mirror the minimal training-time Hydra snapshot that
    :func:`load_from_checkpoint_dir` expects.

    Includes only what the loader actually reads:
      - ``model`` — feeds ``resolve_architecture_config`` + ``build_architecture``.
      - ``accelerate.mixed_precision`` — dtype dispatch (bf16 → torch.bfloat16).
      - ``dataloader`` — minimal stub so ``_merge_with_training_cfg`` doesn't
        choke on the deploy-side merge step.
    """
    return OmegaConf.create(
        {
            "model": OmegaConf.to_container(model_cfg, resolve=True),
            "accelerate": {"mixed_precision": "bf16"},
            "dataloader": {
                "num_frames": 33,
                "height": 480,
                "width": 832,
                "video_stride": 1,
                "normalize_mode": None,
                "action_mode": "joint",
            },
        }
    )


@pytest.fixture(scope="module")
def fabricated_ckpt_dir(tmp_path_factory):
    """Build a SANA-backed arch, save it as a minimal OpenWAM ckpt dir.

    Module-scoped so the 8 GB SANA load + 8.3 GB safetensors write happen
    once even if multiple tests in this file consume the fixture.
    """
    _skip_unless_runnable()

    from openwam.model import build_architecture, resolve_architecture_config

    # Swap hf:// path for the local asset bundle so build_architecture's
    # video backbone constructor loads pretrained weights from disk.
    cfg = OmegaConf.load(str(SANA_CFG_PATH))
    cfg.video_backbone.model_path = str(ASSET_PATH)

    resolved = resolve_architecture_config(cfg)
    arch = build_architecture(resolved.registry_name, resolved.params)
    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    arch.eval()

    ckpt_dir = tmp_path_factory.mktemp("sana_deploy_ckpt")
    ckpt_path = ckpt_dir / "checkpoint_step_0.safetensors"
    arch.save_checkpoint(str(ckpt_path))

    config_path = ckpt_dir / "config.yaml"
    training_cfg = _build_training_cfg_snapshot(cfg)
    OmegaConf.save(training_cfg, str(config_path))

    # Capture a representative ActionDiT param for the safetensors round-trip
    # check downstream. The SANA video DiT lives outside ``arch`` as an
    # ``@dataclass`` ``SanaPipe`` field, so its weights are NOT in
    # ``arch.state_dict()`` and don't round-trip through safetensors — the
    # rebuild path re-loads them from ``model_path`` via ``from_pretrained``.
    # We verify the video DiT was correctly rebuilt via structural checks
    # (num_layers / head_dim / attn_kernel) in the test bodies, not bitwise.
    ab_param_name = "action_backbone.blocks.0.self_attn.q.weight"
    ab_sig = arch.state_dict()[ab_param_name].clone().cpu()

    del arch
    torch.cuda.empty_cache()

    return {
        "ckpt_dir": ckpt_dir,
        "ab_param_name": ab_param_name,
        "ab_sig": ab_sig,
    }


def test_sana_deploy_load_from_checkpoint_dir(fabricated_ckpt_dir):
    """Fabricated ckpt dir round-trips through ``load_from_checkpoint_dir``.

    This is exactly what ``scripts/deploy.py`` calls on the real path. We
    assert the returned ``(cfg, architecture)`` matches the training
    snapshot on file + the rebuilt arch's representative params bitwise-equal
    the saved ones.
    """
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    ckpt_dir = fabricated_ckpt_dir["ckpt_dir"]

    training_cfg, architecture = load_from_checkpoint_dir(
        ckpt_dir=str(ckpt_dir), device="cuda:0"
    )

    # 1. Returned cfg carries the training snapshot.
    assert OmegaConf.select(training_cfg, "model.video_backbone.name") == "sana_video_2b"
    assert OmegaConf.select(training_cfg, "model.video_backbone.attn_kernel") == "linear_relu"
    assert OmegaConf.select(training_cfg, "accelerate.mixed_precision") == "bf16"

    # 2. Arch shape matches the saved SANA-Video 2B 480p config — confirms
    # the SANA DiT was rebuilt via ``from_pretrained(model_path)`` against
    # the local asset bundle (its weights live outside the safetensors).
    assert architecture.video_backbone.num_layers == 20
    assert architecture.video_backbone.num_heads == 20
    assert architecture.video_backbone.head_dim == 112
    assert architecture.video_backbone.attn_kernel == "linear_relu"
    assert architecture.action_backbone.num_layers == 20

    # 3. ActionDiT params round-trip bitwise through safetensors. The SANA
    # video DiT is NOT in ``arch.state_dict()`` (SanaPipe is a @dataclass,
    # not an nn.Module), so it's freshly re-loaded from ``model_path``
    # every rebuild — bitwise round-trip applies only to ActionDiT +
    # proprio_encoder.
    state = architecture.state_dict()
    ab_loaded = state[fabricated_ckpt_dir["ab_param_name"]].cpu()
    assert torch.equal(ab_loaded, fabricated_ckpt_dir["ab_sig"]), (
        f"Round-trip mismatch on {fabricated_ckpt_dir['ab_param_name']} — "
        "safetensors save/load corrupted the ActionDiT weights."
    )

    # 4. Arch is on the requested device + dtype.
    assert architecture.video_backbone._pipe.dit.blocks[0].attn.qkv.weight.device.type == "cuda"
    assert architecture.video_backbone._pipe.dit.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert architecture.action_backbone.blocks[0].self_attn.q.weight.dtype == torch.bfloat16


def test_sana_deploy_engine_construction(fabricated_ckpt_dir):
    """``JointInferenceEngine(cfg, architecture)`` wraps a SANA-loaded arch.

    Last step before the policy server starts. We don't ``engine.generate(...)``
    — that requires inference inputs the smoke doesn't fabricate — but we
    do verify the engine builds, the architecture is bound, and the
    deploy cfg merge respects the training snapshot's mask/kernel settings.
    """
    from openwam.deploy.joint_engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    ckpt_dir = fabricated_ckpt_dir["ckpt_dir"]
    training_cfg, architecture = load_from_checkpoint_dir(
        ckpt_dir=str(ckpt_dir), device="cuda:0"
    )

    if not DEPLOY_CFG_PATH.exists():
        pytest.skip(f"deploy config missing at {DEPLOY_CFG_PATH}")
    deploy_cfg = OmegaConf.load(str(DEPLOY_CFG_PATH))

    # Mirror scripts/deploy.py:_merge_with_training_cfg.
    cfg = OmegaConf.merge(training_cfg, deploy_cfg)

    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
    assert engine.architecture is architecture
    assert engine.architecture.video_backbone.attn_kernel == "linear_relu"
    # Deploy CFG=1.0 default → no uncond pre-encoded text loaded
    # (Cosmos25-only path; SANA stays on the no-op route).
    assert getattr(engine, "_cfg_scale", 1.0) == 1.0
