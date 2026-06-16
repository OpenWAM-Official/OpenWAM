"""Construct Wan backbone components without a pipeline container.

``load_wan_components`` replaces the former ``WanVideoPipeline.from_pretrained``:
it loads the Wan modules from a ``ModelConfig`` list and returns a plain holder
(``SimpleNamespace``) that :meth:`WanVideoBackbone.__init__` drains into itself.
No ``BasePipeline`` / ``WanVideoPipeline`` is involved. ``new_components`` builds
the empty holder used by the config-driven (``components``) deploy path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openwam.model.video_backbone.wan.shared.core.device.npu_compatible_device import get_device_type
from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler
from openwam.model.video_backbone.wan.shared.models.model_loader import ModelPool
from openwam.model.video_backbone.wan.text_encoder import HuggingfaceTokenizer

# Names of the module slots a holder carries (Module → backbone named child;
# others stay plain attributes). Mirrors WanVideoPipeline's old attribute set.
_MODULE_SLOTS = (
    "text_encoder",
    "image_encoder",
    "dit",
    "dit2",
    "vae",
    "motion_controller",
    "vace",
    "vace2",
    "vap",
    "animate_adapter",
    "audio_encoder",
)

# .pth → converted-safetensors redirect to avoid re-downloading shared weights.
_REDIRECT_DICT = {
    "models_t5_umt5-xxl-enc-bf16.pth": (
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
    ),
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
    ),
    "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
    "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
}


def new_components(device=get_device_type(), torch_dtype=torch.bfloat16) -> SimpleNamespace:
    """Empty Wan component holder with Wan defaults (scheduler + division factors)."""
    c = SimpleNamespace(
        device=device,
        torch_dtype=torch_dtype,
        scheduler=FlowMatchScheduler("Wan"),
        tokenizer=None,
        audio_processor=None,
        height_division_factor=16,
        width_division_factor=16,
        time_division_factor=4,
        time_division_remainder=1,
    )
    for name in _MODULE_SLOTS:
        setattr(c, name, None)
    return c


def _apply_redirect(model_configs, redirect_common_files=True):
    if not redirect_common_files:
        return
    for mc in model_configs:
        if mc.origin_file_pattern is None or mc.model_id is None:
            continue
        if mc.origin_file_pattern in _REDIRECT_DICT and mc.model_id != _REDIRECT_DICT[mc.origin_file_pattern][0]:
            print(
                f"To avoid repeatedly downloading model files, ({mc.model_id}, {mc.origin_file_pattern}) is "
                f"redirected to {_REDIRECT_DICT[mc.origin_file_pattern]}. You can use "
                f"`redirect_common_files=False` to disable file redirection."
            )
            mc.model_id, mc.origin_file_pattern = _REDIRECT_DICT[mc.origin_file_pattern]


def load_wan_components(
    model_configs,
    tokenizer_config=None,
    *,
    device=get_device_type(),
    torch_dtype=torch.bfloat16,
    vram_limit=None,
    redirect_common_files=True,
) -> SimpleNamespace:
    """Load Wan modules from ``model_configs`` into a holder (was ``WanVideoPipeline.from_pretrained``)."""
    _apply_redirect(model_configs, redirect_common_files)
    c = new_components(device=device, torch_dtype=torch_dtype)

    # (was BasePipeline.download_and_load_models, inlined)
    model_pool = ModelPool()
    for mc in model_configs:
        mc.download_if_necessary()
        vram_config = mc.vram_config()
        vram_config["computation_dtype"] = vram_config["computation_dtype"] or torch_dtype
        vram_config["computation_device"] = vram_config["computation_device"] or device
        model_pool.auto_load_model(
            mc.path,
            vram_config=vram_config,
            vram_limit=vram_limit,
            clear_parameters=mc.clear_parameters,
            state_dict=mc.state_dict,
        )

    c.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    dit = model_pool.fetch_model("wan_video_dit", index=2)
    if isinstance(dit, list):
        c.dit, c.dit2 = dit
    else:
        c.dit = dit
    c.vae = model_pool.fetch_model("wan_video_vae")
    c.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
    c.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
    vace = model_pool.fetch_model("wan_video_vace", index=2)
    if isinstance(vace, list):
        c.vace, c.vace2 = vace
    else:
        c.vace = vace
    c.vap = model_pool.fetch_model("wan_video_vap")
    c.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
    c.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

    if c.vae is not None:
        c.height_division_factor = c.vae.upsampling_factor * 2
        c.width_division_factor = c.vae.upsampling_factor * 2

    if tokenizer_config is not None:
        tokenizer_config.download_if_necessary()
        c.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean="whitespace")

    return c
