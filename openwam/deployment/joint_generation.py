"""Package-native joint video-action generation loop."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import torch
from tqdm import tqdm

from openwam.model.action_model.action_repr.base import BaseActionRepresentation
from openwam.model.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)


# Allowlist of pipeline-unit class names that emit text embeddings.
# Substring matching silently miscounts ClipLatentDenormalizer and misses
# renamed encoders, so we match by exact class name or an opt-in flag.
_TEXT_UNIT_CLASS_NAMES = frozenset({"WanVideoUnit_PromptEmbedder"})

_UNKNOWN_UNIT_WARNED: set = set()


def _is_text_unit(unit) -> bool:
    flag = getattr(unit, "is_text_unit", None)
    if flag is not None:
        return bool(flag)
    cls_name = getattr(unit, "__class__", type(unit)).__name__
    if cls_name in _TEXT_UNIT_CLASS_NAMES:
        return True
    lowered = cls_name.lower()
    if cls_name not in _UNKNOWN_UNIT_WARNED and any(kw in lowered for kw in ("text", "prompt")):
        _UNKNOWN_UNIT_WARNED.add(cls_name)
        logger.warning(
            "_is_text_unit: %s looks text-related but is not in the allowlist; treating as non-text",
            cls_name,
        )
    return False


def prepare_pipeline_inputs(
    pipe: Any,
    prompt: str,
    negative_prompt: str = "",
    vace_video=None,
    first_frame_image=None,
    num_frames: int = 49,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    cfg_scale: float = 1.0,
    tiled: bool = True,
    num_inference_steps: int = 50,
    shift: float = 5.0,
    tile_size: tuple = (30, 52),
    tile_stride: tuple = (15, 26),
    vace_cache: Optional[dict] = None,
    prompt_embed_cache: Optional[dict] = None,
):
    """Build the full input dicts required by the Wan pipeline.

    When ``vace_cache`` is provided and populated with the same prompt,
    static inputs (text embeddings, reference image encodings) are reused
    and only the observation-dependent portions are re-computed.

    When ``prompt_embed_cache`` is provided, text embeddings are cached
    per-prompt across episodes.  On prompt change or fresh server start,
    the text encoder runs once and the result is stored; subsequent calls
    with the same prompt skip the text encoder entirely (~290ms saved).
    """
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=shift)

    prompt_key = (prompt, negative_prompt)

    # Fast path: full pipeline cache hit — same prompt, same session.
    if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt_key:
        inputs_shared = vace_cache["inputs_shared"].copy()
        inputs_posi = vace_cache["inputs_posi"].copy()
        inputs_nega = vace_cache["inputs_nega"].copy()

        inputs_shared["seed"] = seed
        inputs_shared["vace_video"] = vace_video
        inputs_shared["num_frames"] = num_frames

        for unit in pipe.units:
            if _is_text_unit(unit):
                continue
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )

        return inputs_shared, inputs_posi, inputs_nega

    # Check prompt_embed_cache: if same prompt was seen before, text encoder can be skipped.
    _text_embed_hit = prompt_embed_cache is not None and prompt_key in prompt_embed_cache
    if _text_embed_hit:
        cached_posi, cached_nega = prompt_embed_cache[prompt_key]
        inputs_posi = dict(cached_posi)
        inputs_nega = dict(cached_nega)
        # Keep dynamic fields fresh
        inputs_posi["num_inference_steps"] = num_inference_steps
        inputs_nega["num_inference_steps"] = num_inference_steps
        logger.debug("[text_cache] HIT — skipping text encoder for prompt: %r", prompt[:60])
    else:
        if prompt_embed_cache is not None:
            logger.info("[text_cache] MISS — running text encoder for prompt: %r", prompt[:60])
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": " ",
            "tea_cache_l1_thresh": None,
            "tea_cache_model_id": "",
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": " ",
            "tea_cache_l1_thresh": None,
            "tea_cache_model_id": "",
            "num_inference_steps": num_inference_steps,
        }
    # Default camera pose matrix — only used when camera_control_direction is
    # not None, so this is effectively inert for WAM inference.
    _DEFAULT_CAMERA_ORIGIN = (
        0,
        0.532139961,
        0.946026558,
        0.5,
        0.5,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    )
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": _DEFAULT_CAMERA_ORIGIN,
        "vace_video": vace_video,
        "vace_video_mask": None,
        # Boundary: OpenWAM first_frame_image -> diffsynth vace_reference_image.
        # On Wan2.2-TI2V this becomes first_frame_latents; on Wan2.1-VACE it
        # becomes the VACE spatial reference. The field name is dictated by
        # the vendored diffsynth pipeline units.
        "vace_reference_image": first_frame_image,
        "vace_scale": 1.0,
        "seed": seed,
        "rand_device": "cpu",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "cfg_merge": False,
        "sigma_shift": shift,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
    }

    _t_text = time.time()
    if _text_embed_hit:
        # Text context tensors are already in inputs_posi/inputs_nega — skip text units.
        for unit in pipe.units:
            if _is_text_unit(unit):
                continue
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
    else:
        # Snapshot AFTER the last text unit runs, so the cache stores the
        # text embedding ("context") but not per-episode observation tensors
        # written by later units (ImageEmbedder, VACE, ...).
        #
        # Assumption on pipe.units[:last_text_idx+1]: every unit in this prefix
        # is either (a) a text unit itself, or (b) writes only to inputs_shared
        # (e.g., ShapeChecker, NoiseInitializer). If a future unit is inserted
        # before last_text_idx and writes inputs_posi / inputs_nega, the
        # snapshot will over-capture per-episode tensors into the cache and
        # the HIT path will replay stale data. Mitigations when that happens:
        # either move the new unit after last_text_idx, mark it as a text unit
        # via is_text_unit=True / _TEXT_UNIT_CLASS_NAMES, or split the cache
        # so only the text-embedding subkeys are snapshotted here.
        last_text_idx = max(
            (i for i, u in enumerate(pipe.units) if _is_text_unit(u)),
            default=-1,
        )
        snapshotted = False
        for i, unit in enumerate(pipe.units):
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
            if i == last_text_idx and prompt_embed_cache is not None:
                prompt_embed_cache[prompt_key] = (inputs_posi.copy(), inputs_nega.copy())
                logger.info(
                    "[text_cache] stored embedding (pipeline_prep=%.3fs) for prompt: %r",
                    time.time() - _t_text,
                    prompt[:60],
                )
                snapshotted = True
        if not snapshotted and prompt_embed_cache is not None:
            prompt_embed_cache[prompt_key] = (inputs_posi.copy(), inputs_nega.copy())
            logger.info(
                "[text_cache] stored embedding (pipeline_prep=%.3fs) for prompt: %r",
                time.time() - _t_text,
                prompt[:60],
            )

    # Populate VACE cache for future closed-loop calls (now prompt-aware)
    if vace_cache is not None:
        vace_cache["inputs_shared"] = inputs_shared.copy()
        vace_cache["inputs_posi"] = inputs_posi.copy()
        vace_cache["inputs_nega"] = inputs_nega.copy()
        vace_cache["populated"] = True
        vace_cache["prompt_key"] = prompt_key

    return inputs_shared, inputs_posi, inputs_nega


def _profile_sync(msg: str, t_start: float, profile: bool):
    """Print profiling message if enabled."""
    if profile:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t_start
        logger.info("[WAM_PROFILE] %s: %.3fs", msg, elapsed)


@torch.no_grad()
def generate_video_and_actions(
    pipe: Any,
    architecture: BaseWAMArchitecture,
    schedule,
    prompt: str,
    negative_prompt: str = "",
    vace_video=None,
    first_frame_image=None,
    num_frames: int = 49,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    cfg_scale: float = 1.0,
    tiled: bool = True,
    input_video_latents: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    shift: float = 5.0,
    tile_size: tuple = (30, 52),
    tile_stride: tuple = (15, 26),
    action_repr: Optional[BaseActionRepresentation] = None,
    # Optimization params
    dit_cache=None,
    cfg_handler=None,
    decode_video: bool = True,
    profile: bool = False,
    vace_cache: Optional[dict] = None,
    prompt_embed_cache: Optional[dict] = None,
):
    """Execute joint video-action denoising driven by a schedule.

    Args:
        pipe: Loaded WanVideoPipeline.
        architecture: WAM architecture implementing BaseWAMArchitecture.
        schedule: List of (video_timestep, action_timestep) pairs.
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt for CFG.
        vace_video: Optional VACE conditioning video (Wan2.1-VACE only).
        first_frame_image: Optional first-frame image, used as the TI2V
            first-frame condition on Wan2.2-TI2V backbones or the VACE
            spatial reference on Wan2.1-VACE backbones. Mapped to diffsynth's
            ``vace_reference_image`` field at the pipeline boundary.
        num_frames: Number of video frames to generate.
        height: Video height in pixels.
        width: Video width in pixels.
        seed: Random seed.
        cfg_scale: Classifier-free guidance scale.
        tiled: Whether to use tiled VAE decoding.
        input_video_latents: Pre-encoded video latents (for action-only mode).
        num_inference_steps: Number of denoising steps for the scheduler.
        shift: Timestep shift parameter for the Wan scheduler.
        tile_size: Spatial tile size for tiled processing.
        tile_stride: Spatial tile stride for tiled processing.
        dit_cache: Optional DiTVelocityCache for skipping redundant video DiT passes.
        cfg_handler: Optional CFGBatchMerger or CFGParallelExecutor.
        decode_video: If False, skip VAE decoding (action-only deployment).
        profile: If True, print per-stage timing info.
        vace_cache: Optional dict for caching static pipeline inputs across calls.

    Returns:
        (video_frames, actions) — list of PIL images (or None) and (T, action_dim) numpy array.
    """
    device = pipe.device
    dtype = pipe.torch_dtype

    t0 = time.time()

    inputs_shared, inputs_posi, inputs_nega = prepare_pipeline_inputs(
        pipe,
        prompt,
        negative_prompt,
        vace_video,
        first_frame_image,
        num_frames,
        height,
        width,
        seed,
        cfg_scale,
        tiled,
        num_inference_steps=num_inference_steps,
        shift=shift,
        tile_size=tile_size,
        tile_stride=tile_stride,
        vace_cache=vace_cache,
        prompt_embed_cache=prompt_embed_cache,
    )

    _profile_sync("pipeline_prep", t0, profile)

    if input_video_latents is not None:
        inputs_shared["latents"] = input_video_latents

    is_ti2v = getattr(pipe.dit, "fuse_vae_embedding_in_latents", False)
    if is_ti2v and first_frame_image is not None:
        inputs_shared["fuse_vae_embedding_in_latents"] = True
        num_clean_prefix = 0
        ref_f = len(first_frame_image) if isinstance(first_frame_image, list) else 1
        num_clean_prefix += ref_f
        inputs_shared["num_clean_prefix_frames"] = num_clean_prefix
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        pipe.load_models_to_device(["vae"])
        ref_tensor = pipe.preprocess_video(ref_frames)
        ref_image_latents = pipe.vae.encode(ref_tensor, device=device, tiled=tiled).to(dtype=dtype, device=device)
        inputs_shared["first_frame_latents"] = ref_image_latents

    action_latents = torch.randn(
        1,
        num_frames - 1,
        architecture.action_dim,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    use_interleaved = architecture.is_interleaved
    bridge_layers_set = set(architecture.bridge_layers)
    cached_bridge = None

    t_loop = time.time()

    for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
        t_v, t_a = schedule[i]
        t_v_next, t_a_next = schedule[i + 1]

        sigma_v = t_v / 1000.0
        sigma_a = t_a / 1000.0
        sigma_v_next = t_v_next / 1000.0
        sigma_a_next = t_a_next / 1000.0

        video_stepping = sigma_v != sigma_v_next
        action_stepping = sigma_a != sigma_a_next

        if not video_stepping and not action_stepping:
            continue

        bridge_features = None
        noise_pred = None
        action_noise_pred = None

        if use_interleaved and action_stepping:
            # Interleaved path: action processing happens inside the video
            # DiT forward pass.
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            action_state = architecture.prepare_action_tokens(action_latents, a_timestep)
            dit_state = action_state.extra.get("dit_state")

            # Required when compile.video_dit=true: marks a new denoising step for
            # the CUDA Graph tree so outputs aren't overwritten mid-step. All direct
            # pipe.model_fn and cfg_handler dispatches below are preceded by this
            # mark; the grep-count invariant is enforced by TestCudagraphMarkStepBegin.
            torch.compiler.cudagraph_mark_step_begin()
            noise_pred_posi = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=v_timestep,
                action_dit_state=dit_state,
            )
            action_noise_pred = dit_state.action_noise_pred if dit_state is not None else None

            if cfg_scale != 1.0:
                if cfg_handler is not None:
                    # cfg_handler dispatches to pipe.model_fn internally.
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred = cfg_handler.forward(
                        pipe.model_fn,
                        models,
                        inputs_shared,
                        inputs_posi,
                        inputs_nega,
                        v_timestep,
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred_nega = pipe.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=v_timestep,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

        elif video_stepping or (action_stepping and cached_bridge is None):
            # Bridge-collection path: run video DiT, collect bridge features.
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            # DiT velocity cache: skip video DiT if prediction is stable
            if dit_cache is not None and video_stepping and not dit_cache.should_recompute(sigma_v):
                noise_pred = dit_cache.get_cached()
                bridge_features = cached_bridge
            else:
                bridge_features = []

                if cfg_handler is not None and cfg_scale != 1.0:
                    # cfg_handler dispatches to pipe.model_fn internally.
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred = cfg_handler.forward(
                        pipe.model_fn,
                        models,
                        inputs_shared,
                        inputs_posi,
                        inputs_nega,
                        v_timestep,
                        bridge_feature_store=bridge_features,
                        bridge_feature_layers=bridge_layers_set,
                        bridge_feature_detach=True,
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred_posi = pipe.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_posi,
                        timestep=v_timestep,
                        bridge_feature_store=bridge_features,
                        bridge_feature_layers=bridge_layers_set,
                        bridge_feature_detach=True,
                    )

                    if cfg_scale != 1.0:
                        torch.compiler.cudagraph_mark_step_begin()
                        noise_pred_nega = pipe.model_fn(
                            **models,
                            **inputs_shared,
                            **inputs_nega,
                            timestep=v_timestep,
                        )
                        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    else:
                        noise_pred = noise_pred_posi

                # Update DiT cache after actual forward pass
                if dit_cache is not None and video_stepping:
                    dit_cache.update(noise_pred, sigma_v)

            cached_bridge = None if video_stepping else bridge_features
        else:
            bridge_features = cached_bridge

        if video_stepping:
            new_latents = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
            if "first_frame_latents" in inputs_shared:
                new_latents = new_latents.clone()
                new_latents[:, :, 0:1] = inputs_shared["first_frame_latents"]
            inputs_shared["latents"] = new_latents

        if action_stepping:
            if action_noise_pred is None:
                a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
                action_state = architecture.prepare_action_tokens(action_latents, a_timestep)
                sorted_layers = sorted(architecture.bridge_layers)
                for layer_idx, layer_id in enumerate(sorted_layers):
                    if bridge_features is not None and layer_idx < len(bridge_features):
                        _, action_state = architecture.on_dit_block(layer_id, bridge_features[layer_idx], action_state)
                action_noise_pred = architecture.extract_action_prediction(action_state)
            action_latents = action_latents + action_noise_pred * (sigma_a_next - sigma_a)

    _profile_sync("denoising_loop", t_loop, profile)

    # VAE decode
    t_vae = time.time()
    if decode_video:
        if first_frame_image is not None:
            ref_count = len(first_frame_image) if isinstance(first_frame_image, list) else 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, ref_count:]

        pipe.load_models_to_device(["vae"])
        video = pipe.vae.decode(inputs_shared["latents"], device=device, tiled=tiled)
        video_frames = pipe.vae_output_to_video(video)
    else:
        video_frames = None

    _profile_sync("vae_decode", t_vae, profile)

    # Action decode
    t_action = time.time()
    if action_repr is not None:
        actions = action_repr.decode(action_latents.float()).squeeze(0).cpu().numpy()
    else:
        actions = action_latents.squeeze(0).float().cpu().numpy()
        denorm = getattr(architecture, "action_denormalizer", None)
        if denorm is not None:
            # ActionNormalizer covers both min-max and z-score correctly
            actions = denorm.unnormalize(actions)
        else:
            # Legacy fallback for checkpoints without a saved action_stats.npy
            # (buffer-based z-score — only correct when training used z-score).
            actions = (
                actions * architecture.action_std.float().cpu().numpy() + architecture.action_mean.float().cpu().numpy()
            )

    _profile_sync("action_decode", t_action, profile)

    pipe.load_models_to_device([])
    return video_frames, actions
