"""Package-native joint video-action generation loop."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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
):
    """Replicate the pipeline input setup required by joint generation."""
    pipe.scheduler.set_timesteps(num_inference_steps=50, shift=5.0)

    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": 50,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": 50,
    }
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
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
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": tiled,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
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
    action_dit: Any,
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
):
    """Execute joint video-action denoising driven by a schedule."""
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
        action_dit.action_dim,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    cached_bridge = None
    use_interleaved = action_dit.bridge_type == "joint_self_attn"

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
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            action_dit_state = action_dit.prepare_action_state(action_latents, a_timestep)
            noise_pred_posi = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=v_timestep,
                action_dit_state=action_dit_state,
            )
            action_noise_pred = action_dit_state.action_noise_pred

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
            bridge_features = []
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            noise_pred_posi = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=v_timestep,
                bridge_feature_store=bridge_features,
                bridge_feature_layers=action_dit.bridge_layers_set,
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
                a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
                action_noise_pred = action_dit(
                    action_tokens=action_latents,
                    video_features=bridge_features,
                    timestep=a_timestep,
                )
            action_latents = action_latents + action_noise_pred * (sigma_a_next - sigma_a)

    if vace_reference_image is not None:
        ref_count = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
        inputs_shared["latents"] = inputs_shared["latents"][:, :, ref_count:]

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(inputs_shared["latents"], device=device, tiled=tiled)
    video_frames = pipe.vae_output_to_video(video)

    actions = action_latents.squeeze(0).float().cpu().numpy()
    actions = actions * action_dit.action_std.float().cpu().numpy() + action_dit.action_mean.float().cpu().numpy()

    pipe.load_models_to_device([])
    return video_frames, actions
