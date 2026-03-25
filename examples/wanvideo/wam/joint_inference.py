"""
Synchronized Video-Action Joint Inference with Flexible Schedules.

Replaces the two-stage heuristic (generate video → extract static features → denoise actions)
with a generic denoising loop driven by an explicit schedule of (t_video, t_action) pairs.

A **schedule** is a List[Tuple[float, float]] of length N+1, representing N denoising steps.
At step i, the loop transitions video sigma from schedule[i][0]/1000 → schedule[i+1][0]/1000
and action sigma from schedule[i][1]/1000 → schedule[i+1][1]/1000.

Schedule generators:
    sync           — Both modalities share the same timestep sequence
    video_leading  — Video denoises `lead_steps` ahead of action
    cascade        — Video fully denoises first, then action denoises
    action_only    — Video is clean throughout, only action denoises (UWM policy mode)
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.models.action_dit import ActionDiT, ActionDiTState

# Type alias: list of (t_video, t_action), length N+1 for N denoising steps
Schedule = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Schedule generators
# ---------------------------------------------------------------------------

def _base_timesteps(num_steps: int, shift: float) -> List[float]:
    """Generate Wan-style timesteps (descending from ~1000 toward 0)."""
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(num_steps, shift=shift)
    return scheduler.timesteps.tolist()


def schedule_sync(num_steps: int = 50, shift: float = 5.0) -> Schedule:
    """Both modalities share the same timestep sequence."""
    ts = _base_timesteps(num_steps, shift)
    schedule = [(t, t) for t in ts] + [(0.0, 0.0)]
    return schedule


def schedule_video_leading(num_steps: int = 50, shift: float = 5.0,
                           lead_steps: int = 10) -> Schedule:
    """Video denoises `lead_steps` ahead of action.

    Total denoising steps = num_steps + lead_steps.
    First `lead_steps`: only video denoises, action stays at t_max.
    Next `num_steps`: both denoise together.
    """
    v_ts = _base_timesteps(num_steps + lead_steps, shift)
    a_ts = _base_timesteps(num_steps, shift)
    t_a_max = a_ts[0]

    schedule = []
    for i, v_t in enumerate(v_ts):
        if i < lead_steps:
            schedule.append((v_t, t_a_max))
        else:
            a_idx = i - lead_steps
            schedule.append((v_t, a_ts[a_idx]))
    schedule.append((0.0, 0.0))
    return schedule


def schedule_cascade(video_steps: int = 50, action_steps: int = 50,
                     shift: float = 5.0) -> Schedule:
    """Video fully denoises first, then action denoises.

    Total denoising steps = video_steps + action_steps.
    """
    v_ts = _base_timesteps(video_steps, shift)
    a_ts = _base_timesteps(action_steps, shift)
    t_a_max = a_ts[0]

    schedule = []
    # Phase 1: video denoises, action stays at t_max
    for v_t in v_ts:
        schedule.append((v_t, t_a_max))
    # Phase 2: video stays at 0, action denoises
    for a_t in a_ts:
        schedule.append((0.0, a_t))
    schedule.append((0.0, 0.0))
    return schedule


def schedule_action_only(num_steps: int = 50, shift: float = 5.0) -> Schedule:
    """Video is clean throughout, only action denoises. UWM policy mode."""
    a_ts = _base_timesteps(num_steps, shift)
    schedule = [(0.0, a_t) for a_t in a_ts] + [(0.0, 0.0)]
    return schedule


_SCHEDULE_REGISTRY = {
    "sync": schedule_sync,
    "video_leading": schedule_video_leading,
    "cascade": schedule_cascade,
    "action_only": schedule_action_only,
}


def make_schedule(strategy: str, num_steps: int = 50, shift: float = 5.0,
                  **kwargs) -> Schedule:
    """Dispatch to the appropriate schedule generator.

    Args:
        strategy: One of 'sync', 'video_leading', 'cascade', 'action_only'.
        num_steps: Base number of denoising steps.
        shift: Sigma shift for Wan scheduler.
        **kwargs: Extra args forwarded to specific generators
                  (e.g. lead_steps, video_steps, action_steps).
    """
    if strategy not in _SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule strategy '{strategy}'. "
            f"Choose from: {list(_SCHEDULE_REGISTRY.keys())}"
        )
    fn = _SCHEDULE_REGISTRY[strategy]

    # Build kwargs appropriate for the chosen generator
    call_kwargs = {"shift": shift}
    if strategy == "cascade":
        call_kwargs["video_steps"] = kwargs.get("video_steps", num_steps)
        call_kwargs["action_steps"] = kwargs.get("action_steps", num_steps)
    else:
        call_kwargs["num_steps"] = num_steps
    if strategy == "video_leading":
        call_kwargs["lead_steps"] = kwargs.get("lead_steps", 10)

    return fn(**call_kwargs)


# ---------------------------------------------------------------------------
# Pipeline preprocessing helper
# ---------------------------------------------------------------------------

def prepare_pipeline_inputs(
    pipe: WanVideoPipeline,
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
    """Replicate pipe.__call__'s input setup + unit preprocessing.

    Returns (inputs_shared, inputs_posi, inputs_nega) ready for the denoising loop.
    """
    # Initialize scheduler state before units run
    pipe.scheduler.set_timesteps(num_inference_steps=50, shift=5.0)

    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None, "tea_cache_model_id": "", "num_inference_steps": 50,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None, "tea_cache_model_id": "", "num_inference_steps": 50,
    }
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None,
        "camera_control_direction": None, "camera_control_speed": 1/54, "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": vace_video, "vace_video_mask": None, "vace_reference_image": vace_reference_image, "vace_scale": 1.0,
        "seed": seed, "rand_device": "cpu",
        "height": height, "width": width, "num_frames": num_frames,
        "cfg_scale": cfg_scale, "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": tiled, "tile_size": (30, 52), "tile_stride": (15, 26),
        "sliding_window_size": None, "sliding_window_stride": None,
        "input_audio": None, "audio_sample_rate": 16000, "s2v_pose_video": None, "audio_embeds": None, "s2v_pose_latents": None, "motion_video": None,
        "animate_pose_video": None, "animate_face_video": None, "animate_inpaint_video": None, "animate_mask_video": None,
        "vap_video": None,
    }

    # Run pipeline units (noise init, VACE context encoding, text embedding, etc.)
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    return inputs_shared, inputs_posi, inputs_nega


# ---------------------------------------------------------------------------
# Generic joint inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_video_and_actions(
    pipe: WanVideoPipeline,
    action_dit: ActionDiT,
    schedule: Schedule,
    # Conditioning
    prompt: str,
    negative_prompt: str = "",
    vace_video=None,
    vace_reference_image=None,
    # Generation params
    num_frames: int = 49,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    cfg_scale: float = 1.0,
    tiled: bool = True,
    # For action_only: provide pre-encoded clean video latents
    input_video_latents: Optional[torch.Tensor] = None,
) -> Tuple[List[Image.Image], np.ndarray]:
    """Execute joint video-action denoising driven by a schedule.

    Args:
        pipe: Loaded WanVideoPipeline.
        action_dit: Loaded and eval-mode ActionDiT.
        schedule: List of (t_video, t_action) pairs, length N+1 for N steps.
        prompt: Text prompt for video generation.
        negative_prompt: Negative prompt for CFG.
        vace_video: VACE conditioning video (list of PIL images).
        vace_reference_image: Reference image for VACE.
        num_frames: Number of video frames to generate.
        height: Video height.
        width: Video width.
        seed: Random seed.
        cfg_scale: Classifier-free guidance scale.
        tiled: Whether to use tiled VAE decoding.
        input_video_latents: Pre-encoded clean video latents for action_only mode.

    Returns:
        (video_frames, actions) where video_frames is a list of PIL images
        and actions is a (num_frames, action_dim) numpy array (denormalized).
    """
    device = pipe.device
    dtype = pipe.torch_dtype

    # 1. Preprocess pipeline inputs (text encoding, noise init, VACE, etc.)
    inputs_shared, inputs_posi, inputs_nega = prepare_pipeline_inputs(
        pipe, prompt, negative_prompt, vace_video, vace_reference_image,
        num_frames, height, width, seed, cfg_scale, tiled,
    )

    # Override video latents if provided (action_only with pre-encoded video)
    if input_video_latents is not None:
        inputs_shared["latents"] = input_video_latents

    # Activate TI2V-5B separated timestep conditioning
    is_ti2v = getattr(pipe.dit, 'fuse_vae_embedding_in_latents', False)
    if is_ti2v and vace_reference_image is not None:
        inputs_shared["fuse_vae_embedding_in_latents"] = True
        num_clean_prefix = 0
        ref_f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
        num_clean_prefix += ref_f
        inputs_shared["num_clean_prefix_frames"] = num_clean_prefix
        # Encode ref_image as clean first_frame_latents for TI2V-5B
        ref_frames = vace_reference_image if isinstance(vace_reference_image, list) else [vace_reference_image]
        pipe.load_models_to_device(["vae"])
        ref_tensor = pipe.preprocess_video(ref_frames)
        ref_image_latents = pipe.vae.encode(ref_tensor, device=device, tiled=tiled).to(dtype=dtype, device=device)
        inputs_shared["first_frame_latents"] = ref_image_latents

    # 2. Initialize action latents from noise
    action_latents = torch.randn(
        1, num_frames, action_dit.action_dim,
        device=device, dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # 3. Generic denoising loop driven by schedule
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    cached_bridge = None
    use_interleaved = (action_dit.bridge_type == "joint_self_attn")

    for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
        t_v, t_a = schedule[i]
        t_v_next, t_a_next = schedule[i + 1]

        sigma_v = t_v / 1000.0
        sigma_a = t_a / 1000.0
        sigma_v_next = t_v_next / 1000.0
        sigma_a_next = t_a_next / 1000.0

        video_stepping = (sigma_v != sigma_v_next)
        action_stepping = (sigma_a != sigma_a_next)

        if not video_stepping and not action_stepping:
            continue

        # --- Run video DiT forward ---
        bridge_features = None
        noise_pred = None
        action_noise_pred = None

        if use_interleaved and action_stepping:
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            _action_dit_state = action_dit.prepare_action_state(action_latents, a_timestep)

            noise_pred_posi = pipe.model_fn(
                **models, **inputs_shared, **inputs_posi,
                timestep=v_timestep,
                action_dit_state=_action_dit_state,
            )
            action_noise_pred = _action_dit_state.action_noise_pred

            if cfg_scale != 1.0:
                noise_pred_nega = pipe.model_fn(
                    **models, **inputs_shared, **inputs_nega,
                    timestep=v_timestep,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

        elif video_stepping or (action_stepping and cached_bridge is None):
            bridge_features = []
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            noise_pred_posi = pipe.model_fn(
                **models, **inputs_shared, **inputs_posi,
                timestep=v_timestep,
                bridge_feature_store=bridge_features,
                bridge_feature_layers=action_dit.bridge_layers_set,
                bridge_feature_detach=True,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = pipe.model_fn(
                    **models, **inputs_shared, **inputs_nega,
                    timestep=v_timestep,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            if not video_stepping:
                cached_bridge = bridge_features
            else:
                cached_bridge = None
        else:
            bridge_features = cached_bridge

        # --- Step video latents (Euler) ---
        if video_stepping:
            inputs_shared["latents"] = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # --- Step action latents (Euler) ---
        if action_stepping:
            if action_noise_pred is None:
                a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
                action_noise_pred = action_dit(
                    action_tokens=action_latents,
                    video_features=bridge_features,
                    timestep=a_timestep,
                )
            action_latents = action_latents + action_noise_pred * (sigma_a_next - sigma_a)

    # 4. Strip VACE reference image frames before decode
    if vace_reference_image is not None:
        if isinstance(vace_reference_image, list):
            f = len(vace_reference_image)
        else:
            f = 1
        inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

    # 5. Decode video
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(inputs_shared["latents"], device=device, tiled=tiled)
    video_frames = pipe.vae_output_to_video(video)

    # 6. Denormalize actions
    actions = action_latents.squeeze(0).float().cpu().numpy()
    actions = actions * action_dit.action_std.float().cpu().numpy() + action_dit.action_mean.float().cpu().numpy()

    pipe.load_models_to_device([])
    return video_frames, actions
