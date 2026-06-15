'Public implementation.'

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


ASSET_PATH = Path(os.environ.get("SANA_ASSET_PATH", "/path/to/assets/SANA-Video_2B_480p"))
REPO_ROOT = Path(__file__).resolve().parents[1]
SANA_CFG_PATH = REPO_ROOT / "configs" / "model" / "dual_system_self_attn_sana.yaml"

# Deployed SANA-Video 480p geometry. T_video=81 with VAE temporal 4× → 21
# temporal latents; 480×832 with spatial 8× → 60×104 spatial latents.
_T_VIDEO = 81
_H_VIDEO = 480
_W_VIDEO = 832

_T_LAT = 21
_H_LAT = 60
_W_LAT = 104

_CAPTION_LEN = 8  # short pre_encoded_text — Gemma hidden dim, length doesn't matter for smoke
_CAPTION_DIM = 2304


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
    if not (ASSET_PATH / "vae" / "Wan2.1_VAE.pth").exists():
        pytest.skip(f"Wan2.1 VAE missing at {ASSET_PATH}/vae/Wan2.1_VAE.pth.")
    if not SANA_CFG_PATH.exists():
        pytest.skip(f"SANA config missing at {SANA_CFG_PATH}.")
    if not _sana_importable():
        pytest.skip("third_party/Sana not importable (submodule init or timm version.py missing).")


def _make_stub_sample():
    """81 PIL frames at 480×832, T2V smoke + pre_encoded_text + action.

    SANA is T2V so the auto-injected ``first_frame_image`` from
    ``FirstFrameConditioningTransform`` is intentionally ignored by
    ``SanaVideoBackbone.preprocess_input`` (it falls through ``**kw`` and is
    silently dropped). Verified by the ``num_clean_prefix_frames == 0`` assert
    below.
    """
    import PIL.Image

    return {
        "prompt": "smoke",
        "action": np.zeros((12, 20), dtype=np.float32),
        "proprio": np.zeros((20,), dtype=np.float32),
        "action_mask": np.ones((12,), dtype=bool),
        "video_mask": np.ones((_T_VIDEO,), dtype=bool),
        "video": [PIL.Image.fromarray(np.zeros((_H_VIDEO, _W_VIDEO, 3), dtype=np.uint8)) for _ in range(_T_VIDEO)],
    }


def test_sana_train_smoke_81frame_t2v():
    """End-to-end forward + backward at 81×480×832 against real 2B weights."""
    _skip_unless_runnable()

    from omegaconf import OmegaConf

    from openwam.model import build_architecture, resolve_architecture_config

    cfg = OmegaConf.load(str(SANA_CFG_PATH))
    cfg.video_backbone.model_path = str(ASSET_PATH)
    resolved = resolve_architecture_config(cfg)
    arch = build_architecture(resolved.registry_name, resolved.params)

    assert arch.mot_driver is not None, (
        "DualSystemSelfAttnArchitecture.mot_driver should be wired in __init__ for SANA."
    )
    # SANA-Video 2B is 20 blocks; ActionDiT auto-resolves to the same.
    assert arch.video_backbone.num_layers == arch.action_backbone.num_layers == 20
    assert arch.video_backbone.attn_kernel == "linear_relu"

    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    arch.move_frozen_to_device(torch.device("cuda:0"))
    arch.init_training_schedulers(1000)

    # Keep the video DiT frozen for this smoke — we only need to verify the
    # MoT plumbing + ActionDiT gradient flow. Loading the 8GB 2B ckpt then
    # backpropagating through 20 layers of a 32k-token mixed attention would
    # blow past the test-time budget; freezing the video side keeps the
    # backward graph anchored on ActionDiT params (a few hundred MB).
    arch.video_backbone.requires_grad_(False)

    arch.set_training_runtime(use_gradient_checkpointing=True)

    sample = _make_stub_sample()
    sample["pre_encoded_text"] = torch.randn(_CAPTION_LEN, _CAPTION_DIM)

    inputs = arch.prepare_inputs([sample])

    # T2V invariants — verified at the architecture / backbone boundary.
    assert inputs["input_latents"].shape == (1, 16, _T_LAT, _H_LAT, _W_LAT), (
        f"Expected SANA-Video 81-frame latent shape (1, 16, {_T_LAT}, {_H_LAT}, {_W_LAT}), "
        f"got {tuple(inputs['input_latents'].shape)}"
    )
    assert inputs.get("num_clean_prefix_frames", 0) == 0, (
        "SANA is pure T2V; FirstFrameConditioningTransform's auto-injected "
        "first_frame_image must be dropped by preprocess_input."
    )
    assert inputs.get("first_frame_latents") is None, (
        "SANA T2V path must NOT carry first_frame_latents (no TI2V injection)."
    )

    result = arch.compute_loss(**inputs)
    loss = result["loss"]
    assert torch.isfinite(loss).item(), f"SANA train smoke loss not finite: {loss}"
    assert loss.requires_grad
    loss.backward()

    # ActionDiT must receive gradients through the mixed (linear-relu) attention.
    q_proj = arch.action_backbone.blocks[0].self_attn.q
    assert q_proj.weight.grad is not None, (
        "ActionDiT.blocks[0].self_attn.q got no gradient — MoT join broken on SANA path."
    )
    assert torch.isfinite(q_proj.weight.grad).all(), "ActionDiT.q grad is non-finite"

    # Video DiT stays frozen (we explicitly disabled requires_grad above).
    for n, p in arch.video_backbone.named_parameters():
        assert not p.requires_grad, f"video_backbone.{n} unexpectedly trainable"
