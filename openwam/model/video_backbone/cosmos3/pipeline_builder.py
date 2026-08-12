"""Component construction for the Cosmos3-Edge backbone.

Builds the vendored ``Cosmos3OmniTransformer`` (weights from the diffusers-style
bundle's ``transformer/`` sharded safetensors), the ``AutoencoderKLWan`` VAE
(``vae/`` subfolder) and the ``PreTrainedTokenizerFast`` (``text_tokenizer/``),
returning a plain namespace the backbone drains (predict2.5 holder parity).

Config validation happens before any heavy import so CPU CI reproduces the same
errors. The deploy path (``ckpt_dir is not None``) builds empty meta shells from
hardcoded bundle configs — weights arrive later via the architecture's strict
``load_checkpoint`` on the unified safetensors (``dit.*`` / ``vae.*``).

The und (text) pathway + token embedding are frozen at build time
(``freeze_und``, default True): the und tower runs under ``no_grad`` in the
backbone's preprocess, so those parameters never see gradients anyway — this
keeps the optimizer surface honest. ``lm_head`` (268M params, unused by the
diffusion forward) is dropped after load so it never enters checkpoints.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_COSMOS3_INSTALL_HINT = (
    "cosmos3_edge needs `diffusers>=0.38` (for AutoencoderKLWan and the vendored "
    "transformer's support imports) and `transformers` for the tokenizer. "
    "Install with: pip install 'diffusers>=0.38,<1'."
)

# Geometry of the released Cosmos3-Edge generator (transformer/config.json).
# context_dim == hidden_size: the action-stream context is the und final hidden.
_COSMOS3_EDGE_GEOMETRY = dict(dim=2048, num_layers=28, num_heads=16, head_dim=128, context_dim=2048)

# Frame rate attributed to training/inference clips. It feeds two things: the
# vision mRoPE temporal scaling (inert while it equals the model's ``base_fps``
# of 24) and the duration sentence in the prompt templates. Nothing in OpenWAM's
# dataloader stack reports a clip fps — and after ``video_stride`` sub-sampling
# the clip rate is not the source rate anyway — so this is an assumption, not a
# measurement. Left at 24 to keep existing checkpoints reproducible; override
# ``model.video_backbone.fps`` (or drop the duration sentence with
# ``prompt_duration_template=false``) when the real rate is known.
_DEFAULT_CLIP_FPS = 24.0

# Full init kwargs of the released Edge transformer — used verbatim for the
# deploy-time meta shell so a checkpoint deploys without the original bundle.
_COSMOS3_EDGE_NET_KWARGS: dict = dict(
    attention_bias=False,
    attention_dropout=0.0,
    head_dim=128,
    hidden_size=2048,
    intermediate_size=9216,
    base_fps=24,
    enable_fps_modulation=True,
    latent_channel=48,
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    latent_patch_size=2,
    num_attention_heads=16,
    num_hidden_layers=28,
    num_key_value_heads=8,
    patch_latent_dim=192,
    rms_norm_eps=1e-5,
    rope_theta=100000000.0,
    action_dim=64,
    action_gen=True,
    num_embodiment_domains=32,
    sound_gen=False,
    timestep_scale=0.001,
    vocab_size=131072,
    hidden_act="relu2",
    qk_norm_for_text=False,
    use_und_k_norm_for_gen=True,
    rope_axes_dim=[24, 20, 20],
)

# Wan2.2-TI2V VAE config from the bundle's vae/config.json (deploy meta shell +
# the latents_mean/std normalization constants used by encode/decode).
_COSMOS3_VAE_KWARGS: dict = dict(
    base_dim=160,
    decoder_base_dim=256,
    dim_mult=[1, 2, 4, 4],
    dropout=0.0,
    in_channels=12,
    is_residual=True,
    num_res_blocks=2,
    out_channels=12,
    patch_size=2,
    scale_factor_spatial=16,
    scale_factor_temporal=4,
    temperal_downsample=[False, True, True],
    attn_scales=[],
    z_dim=48,
    latents_mean=[
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    latents_std=[
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
)

# Weightless config fields whose values change model behavior without changing
# any tensor shape — asserted against the loaded bundle at build time.
_WEIGHTLESS_CONFIG_KEYS = (
    "intermediate_size",
    "num_key_value_heads",
    "hidden_act",
    "qk_norm_for_text",
    "use_und_k_norm_for_gen",
    "rms_norm_eps",
    "rope_theta",
    "rope_axes_dim",
    "unified_3d_mrope_reset_spatial_ids",
    "unified_3d_mrope_temporal_modality_margin",
    "latent_patch_size",
    "latent_channel",
    "patch_latent_dim",
    "timestep_scale",
    "base_fps",
    "enable_fps_modulation",
)


def _norm_cfg_value(value):
    if isinstance(value, (list, tuple)):
        return [_norm_cfg_value(v) for v in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


_UND_PATHWAY_LAYER_CHILDREN = (
    "input_layernorm",
    "post_attention_layernorm",
    "mlp",
)
_UND_PATHWAY_ATTN_CHILDREN = ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k", "k_norm_und_for_gen")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _video_backbone_cfg(source: Any) -> Any:
    """Normalize the four accepted source shapes (predict2.5 parity)."""
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return {"model_path": str(source)}
    if isinstance(source, dict):
        if "model" in source:
            model = source["model"]
            return model.get("video_backbone", model) if isinstance(model, dict) else model
        return source.get("video_backbone", source)
    model = getattr(source, "model", None)
    if model is not None and getattr(model, "video_backbone", None) is not None:
        return model.video_backbone
    return source


def import_cosmos3_vendor():
    """Lazy-import the vendored transformer module (pulls diffusers)."""
    try:
        from openwam.model.video_backbone.cosmos3._vendor import transformer_cosmos3
    except ImportError as exc:  # diffusers missing or too old
        raise ImportError(_COSMOS3_INSTALL_HINT) from exc
    return transformer_cosmos3


def _freeze_und_pathway(net) -> int:
    """requires_grad_(False) on the und half + shared text embedding. Returns count."""
    frozen = 0
    for module in [net.embed_tokens, net.norm]:
        for p in module.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
    for layer in net.layers:
        for name in _UND_PATHWAY_LAYER_CHILDREN:
            for p in getattr(layer, name).parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        for name in _UND_PATHWAY_ATTN_CHILDREN:
            child = getattr(layer.self_attn, name, None)
            if child is not None:
                for p in child.parameters():
                    p.requires_grad_(False)
                    frozen += p.numel()
    return frozen


def _drop_lm_head(net) -> None:
    if hasattr(net, "lm_head") and net.lm_head is not None:
        delattr(net, "lm_head")
        object.__setattr__(net, "lm_head", None)


def _freeze_unused_native_heads(net) -> int:
    """Freeze the Omni action/sound heads OpenWAM's video-backbone path never
    calls (its action stream is the external ActionDiT). They stay registered
    children — riding the checkpoint unchanged keeps a future native-action
    variant loadable — but frozen they produce uniform zero gradients on every
    rank instead of occupying optimizer state. Returns the frozen param count."""
    frozen = 0
    for name in ("action_proj_in", "action_proj_out", "audio_proj_in", "audio_proj_out"):
        child = getattr(net, name, None)
        if child is not None:
            for p in child.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
    for name in ("action_modality_embed", "audio_modality_embed"):
        param = getattr(net, name, None)
        if param is not None:
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


def _materialize_shell(module, what: str):
    """Give a meta shell real zeroed storage on CPU.

    ``to_empty`` allocates without initializing, so the tensors would hold
    whatever was in memory — including NaN/Inf, which DeepSpeed's ZeRO setup can
    read while flattening parameter groups. Zeroing keeps that deterministic;
    every value is overwritten by ``load_state`` before the first step.
    """
    import torch

    module = module.to_empty(device="cpu")
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        for b in module.buffers():
            if b.is_floating_point():
                b.zero_()
    logger.info("cosmos3_edge: materialized the %s shell for resume (zeroed; load_state overwrites)", what)
    return module


def build_cosmos3_pipeline(
    source: Any,
    *,
    device: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    materialize_weights: bool = False,
    **_unused,
) -> types.SimpleNamespace:
    """Build the Cosmos3-Edge components.

    Train path loads real weights from ``model_path``. A ``ckpt_dir`` build
    produces an empty shell that something else fills: deploy and finetune both
    load a state_dict right after, so the shell can stay on ``meta``.

    ``materialize_weights`` is the resume path's opt-out. ``resume_ckpt_path``
    goes through the same ``ckpt_dir`` construction but deliberately skips the
    safetensors load (``ckpt_model_loader``: accelerate's ``load_state``
    restores a strict superset after ``prepare``) — so nothing materializes the
    shell before ``set_dtype_device`` runs, and a meta shell raises "Cannot copy
    out of meta tensor" there. With this flag the shell gets real (zeroed)
    storage instead, which ``load_state`` then overwrites.
    """
    cfg = _video_backbone_cfg(source)

    # --- Phase A: pure-Python validation before any heavy import -------------
    name = str(_cfg_get(cfg, "name", "cosmos3_edge"))
    if name != "cosmos3_edge":
        raise NotImplementedError(
            f"cosmos3 backbone supports name='cosmos3_edge' only (got {name!r}); "
            "Nano/Super need their own geometry entries."
        )
    model_path_raw = _cfg_get(cfg, "model_path")
    if model_path_raw is None and ckpt_dir is None:
        raise ValueError("cosmos3_edge needs `model_path` (bundle root) unless deploying from a ckpt_dir.")
    shift_video = float(_cfg_get(cfg, "shift_video", 5.0))
    use_system_prompt = bool(_cfg_get(cfg, "use_system_prompt", False))
    prompt_templates = bool(_cfg_get(cfg, "prompt_templates", True))
    duration_template = bool(_cfg_get(cfg, "prompt_duration_template", True))
    clip_fps = float(_cfg_get(cfg, "fps", _DEFAULT_CLIP_FPS))
    if clip_fps <= 0.0:
        raise ValueError(f"cosmos3_edge fps must be > 0, got {clip_fps!r}.")
    freeze_und = bool(_cfg_get(cfg, "freeze_und", True))
    if not freeze_und:
        raise NotImplementedError(
            "cosmos3_edge freeze_und=false is not supported: the und tower always runs under "
            "no_grad (dit_forward.run_und_tower), so its parameters can never receive gradients "
            "— the flag would silently allocate optimizer state for weights that never train. "
            "Remove the override, or implement a grad-enabled und path first."
        )
    max_text_tokens = int(_cfg_get(cfg, "max_text_tokens", 512))
    if max_text_tokens < 8:
        raise ValueError(f"cosmos3_edge max_text_tokens must be >= 8, got {max_text_tokens}.")
    text_dropout = float(_cfg_get(cfg, "text_encoder_dropout", 0.0))
    if not 0.0 <= text_dropout <= 1.0:
        raise ValueError(f"cosmos3_edge text_encoder_dropout must be in [0, 1], got {text_dropout!r}.")
    dropout_seed = _cfg_get(cfg, "text_encoder_dropout_seed")
    if dropout_seed is not None:
        dropout_seed = int(dropout_seed)

    # --- Phase B: heavy imports ---------------------------------------------
    import torch

    vendor = import_cosmos3_vendor()
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise ImportError(_COSMOS3_INSTALL_HINT) from exc
    from transformers import PreTrainedTokenizerFast

    deploy = ckpt_dir is not None
    model_path = Path(str(model_path_raw)) if model_path_raw is not None else None

    # --- Phase C: transformer -----------------------------------------------
    if deploy:
        from accelerate import init_empty_weights

        with init_empty_weights():
            net = vendor.Cosmos3OmniTransformer(**_COSMOS3_EDGE_NET_KWARGS)
        if materialize_weights:
            # Cast while still on meta (free) so materializing allocates bf16 —
            # matching the finetune path's footprint instead of an fp32 peak.
            net = net.to(dtype=torch.bfloat16)
            net = _materialize_shell(net, "transformer")
        # Non-persistent buffers are absent from the state_dict, so the meta
        # shell must materialize them itself or the post-load `.to()` raises
        # "Cannot copy out of meta tensor". The only one is the rotary
        # inv_freq, fully determined by config — recompute it here.
        from openwam.model.video_backbone.cosmos3.dit_forward import rotary_inv_freq

        inv_freq = rotary_inv_freq(
            int(_COSMOS3_EDGE_NET_KWARGS["head_dim"]), float(_COSMOS3_EDGE_NET_KWARGS["rope_theta"])
        )
        net.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        leftover = [n for n, b in net.named_buffers() if b.is_meta]
        if leftover:
            raise RuntimeError(f"cosmos3_edge deploy shell has unmaterialized meta buffers: {leftover}")
    else:
        assert model_path is not None  # guaranteed by Phase A validation
        if not (model_path / "transformer").is_dir():
            raise FileNotFoundError(
                f"cosmos3_edge model_path has no transformer/ subfolder: {model_path} "
                "(expected a diffusers-style Cosmos3-Edge bundle)."
            )
        net = vendor.Cosmos3OmniTransformer.from_pretrained(
            str(model_path), subfolder="transformer", torch_dtype=torch.bfloat16
        )
    _drop_lm_head(net)
    net_cfg = net.config
    geometry_ok = (
        int(net_cfg.hidden_size) == _COSMOS3_EDGE_GEOMETRY["dim"]
        and int(net_cfg.num_hidden_layers) == _COSMOS3_EDGE_GEOMETRY["num_layers"]
        and int(net_cfg.num_attention_heads) == _COSMOS3_EDGE_GEOMETRY["num_heads"]
        and int(net_cfg.head_dim) == _COSMOS3_EDGE_GEOMETRY["head_dim"]
    )
    if not geometry_ok:
        raise ValueError(
            "Loaded Cosmos3 transformer geometry does not match the cosmos3_edge entry "
            f"(hidden={net_cfg.hidden_size}, layers={net_cfg.num_hidden_layers}, "
            f"heads={net_cfg.num_attention_heads}, head_dim={net_cfg.head_dim})."
        )
    # Weightless config fields carry behavior a strict state_dict load cannot
    # catch (rotary phases, timestep scale, mRoPE margins, norm eps, ...). A
    # revised bundle changing any of them must fail loudly, not train wrong.
    mismatches = {
        key: (getattr(net_cfg, key, None), _COSMOS3_EDGE_NET_KWARGS[key])
        for key in _WEIGHTLESS_CONFIG_KEYS
        if _norm_cfg_value(getattr(net_cfg, key, None)) != _norm_cfg_value(_COSMOS3_EDGE_NET_KWARGS[key])
    }
    if mismatches:
        raise ValueError(
            "Cosmos3 bundle config diverges from the cosmos3_edge entry on weightless "
            f"fields {mismatches} (got, expected). Update _COSMOS3_EDGE_NET_KWARGS and port "
            "any behavioral change (text_pack positions / dit_forward) before training."
        )
    # Freeze unconditionally. Gating this on ``deploy`` would be wrong: the
    # training finetune/resume path also sets ``ckpt_dir`` (see
    # train/utils/ckpt_model_loader.py), so a resumed run would put the whole
    # frozen und tower (1.69B with the embeddings and native heads) into the
    # optimizer for parameters that can never receive a gradient — the very
    # thing ``freeze_und=false`` is rejected for above. On the deploy meta-shell
    # this is a harmless no-op: ``load_state_dict`` preserves ``requires_grad``
    # and no optimizer exists there.
    n_heads_frozen = _freeze_unused_native_heads(net)
    if n_heads_frozen:
        logger.info("cosmos3_edge: froze unused native action/sound heads (%.1fM params)", n_heads_frozen / 1e6)
    n_frozen = _freeze_und_pathway(net)
    logger.info("cosmos3_edge: froze und pathway + embeddings (%.1fM params)", n_frozen / 1e6)

    # --- Phase D: VAE --------------------------------------------------------
    if deploy:
        from accelerate import init_empty_weights

        with init_empty_weights():
            vae = AutoencoderKLWan(**_COSMOS3_VAE_KWARGS)
        if materialize_weights:
            vae = _materialize_shell(vae.to(dtype=torch.bfloat16), "vae")
        vae_leftover = [n for n, b in vae.named_buffers() if b.is_meta]
        if vae_leftover:
            raise RuntimeError(f"cosmos3_edge VAE deploy shell has unmaterialized meta buffers: {vae_leftover}")
    else:
        vae = AutoencoderKLWan.from_pretrained(str(model_path), subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latents_mean = torch.tensor(list(vae.config.latents_mean), dtype=torch.float32)
    latents_std = torch.tensor(list(vae.config.latents_std), dtype=torch.float32)

    # --- Phase E: tokenizer ---------------------------------------------------
    tok_root = Path(ckpt_dir) if deploy else Path(str(model_path_raw))
    tok_dir = tok_root / "text_tokenizer"
    if not tok_dir.is_dir():
        raise FileNotFoundError(f"cosmos3_edge tokenizer dir missing: {tok_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))

    # --- Phase F: device ------------------------------------------------------
    if device is not None and not deploy:
        net = net.to(device)
        vae = vae.to(device)

    return types.SimpleNamespace(
        net=net,
        vae=vae,
        tokenizer=tokenizer,
        latents_mean=latents_mean,
        latents_std=latents_std,
        shift_video=shift_video,
        use_system_prompt=use_system_prompt,
        prompt_templates=prompt_templates,
        duration_template=duration_template,
        clip_fps=clip_fps,
        max_text_tokens=max_text_tokens,
        text_dropout_p=text_dropout,
        text_dropout_seed=dropout_seed,
        **_COSMOS3_EDGE_GEOMETRY,
    )
