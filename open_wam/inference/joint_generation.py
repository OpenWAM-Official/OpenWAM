"""Package-native joint video-action generation loop."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from open_wam.models.architectures.base import BaseWAMArchitecture


def prepare_pipeline_inputs(
    pipe: Any,
    prompt: str,
    negative_prompt: str = "",
    vace_video=None,
    vace_reference_image=None,
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
):
    """Build the full input dicts required by the Wan pipeline.

    Args:
        pipe: Loaded WanVideoPipeline.
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt for CFG.
        vace_video: Optional VACE conditioning video.
        vace_reference_image: Optional reference image(s).
        num_frames: Number of video frames to generate.
        height: Video height in pixels.
        width: Video width in pixels.
        seed: Random seed.
        cfg_scale: Classifier-free guidance scale.
        tiled: Whether to use tiled VAE decoding.
        num_inference_steps: Number of denoising steps for the scheduler.
        shift: Timestep shift parameter for the Wan scheduler.
        tile_size: Spatial tile size for tiled processing.
        tile_stride: Spatial tile stride for tiled processing.
    """
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=shift)

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
        0, 0.532139961, 0.946026558, 0.5, 0.5,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
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
        "vace_reference_image": vace_reference_image,
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

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    return inputs_shared, inputs_posi, inputs_nega


@torch.no_grad()
def generate_video_and_actions(
    pipe: Any,
    architecture: BaseWAMArchitecture,
    schedule,
    prompt: str,
    negative_prompt: str = "",
    vace_video=None,
    vace_reference_image=None,
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
):
    """Execute joint video-action denoising driven by a schedule.

    Args:
        pipe: Loaded WanVideoPipeline.
        architecture: WAM architecture implementing BaseWAMArchitecture.
        schedule: List of (video_timestep, action_timestep) pairs.
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt for CFG.
        vace_video: Optional VACE conditioning video.
        vace_reference_image: Optional reference image(s).
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

    Returns:
        (video_frames, actions) — list of PIL images and (T, action_dim) numpy array.
    """
    device = pipe.device
    dtype = pipe.torch_dtype

    inputs_shared, inputs_posi, inputs_nega = prepare_pipeline_inputs(
        pipe,
        prompt,
        negative_prompt,
        vace_video,
        vace_reference_image,
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
    )

    if input_video_latents is not None:
        inputs_shared["latents"] = input_video_latents

    is_ti2v = getattr(pipe.dit, "fuse_vae_embedding_in_latents", False)
    if is_ti2v and vace_reference_image is not None:
        inputs_shared["fuse_vae_embedding_in_latents"] = True
        num_clean_prefix = 0
        ref_f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
        num_clean_prefix += ref_f
        inputs_shared["num_clean_prefix_frames"] = num_clean_prefix
        ref_frames = vace_reference_image if isinstance(vace_reference_image, list) else [vace_reference_image]
        pipe.load_models_to_device(["vae"])
        ref_tensor = pipe.preprocess_video(ref_frames)
        ref_image_latents = pipe.vae.encode(ref_tensor, device=device, tiled=tiled).to(dtype=dtype, device=device)
        inputs_shared["first_frame_latents"] = ref_image_latents

    action_latents = torch.randn(
        1,
        num_frames,
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
            # DiT forward pass. Prepare action state through the architecture
            # and pass the internal state to pipe.model_fn.
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            action_state = architecture.prepare_action_tokens(action_latents, a_timestep)
            # The interleaved architecture stores its pipe-compatible state
            # in action_state.extra["dit_state"]
            dit_state = action_state.extra.get("dit_state")

            noise_pred_posi = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=v_timestep,
                action_dit_state=dit_state,
            )
            action_noise_pred = dit_state.action_noise_pred if dit_state is not None else None

            if cfg_scale != 1.0:
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
            # Bridge-collection path: run video DiT, collect bridge features
            # at the architecture's designated layers.
            bridge_features = []
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

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
                noise_pred_nega = pipe.model_fn(
                    **models,
                    **inputs_shared,
                    **inputs_nega,
                    timestep=v_timestep,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            cached_bridge = None if video_stepping else bridge_features
        else:
            bridge_features = cached_bridge

        if video_stepping:
            inputs_shared["latents"] = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        if action_stepping:
            if action_noise_pred is None:
                # Use the architecture interface: prepare → feed bridge
                # features through on_dit_block → extract prediction.
                a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
                action_state = architecture.prepare_action_tokens(action_latents, a_timestep)
                sorted_layers = sorted(architecture.bridge_layers)
                for layer_idx, layer_id in enumerate(sorted_layers):
                    if bridge_features is not None and layer_idx < len(bridge_features):
                        _, action_state = architecture.on_dit_block(
                            layer_id, bridge_features[layer_idx], action_state
                        )
                action_noise_pred = architecture.extract_action_prediction(action_state)
            action_latents = action_latents + action_noise_pred * (sigma_a_next - sigma_a)

    if vace_reference_image is not None:
        ref_count = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
        inputs_shared["latents"] = inputs_shared["latents"][:, :, ref_count:]

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(inputs_shared["latents"], device=device, tiled=tiled)
    video_frames = pipe.vae_output_to_video(video)

    actions = action_latents.squeeze(0).float().cpu().numpy()
    actions = actions * architecture.action_std.float().cpu().numpy() + architecture.action_mean.float().cpu().numpy()

    pipe.load_models_to_device([])
    return video_frames, actions
