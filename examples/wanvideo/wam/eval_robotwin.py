"""
VAM policy adapter for RoboTwin 2.0 SAPIEN evaluation.

Provides the 3-function interface expected by RoboTwin's eval_policy.py:
  - get_model(task_name, ckpt_path, ...) -> model
  - eval_one_step(model, obs_dict) -> action (14/16,)
  - reset_model(model) -> None

The VAM policy generates video + actions jointly. For closed-loop eval:
1. Accumulate past observations as conditioning context
2. Run action-only denoising (no video generation for speed)
3. Return first action of predicted trajectory

Usage:
    # Standalone test (no SAPIEN, loads checkpoint and prints action)
    python eval_robotwin.py \
        --action_checkpoint models/train/robotwin_S3_JointSelfAttn/step-5000.safetensors \
        --task_name adjust_bottle \
        --robot aloha-agilex \
        --test

    # Plug into RoboTwin eval (requires RoboTwin repo installed)
    # In RoboTwin's eval_policy.py, import these three functions:
    #   from eval_robotwin import get_model, eval_one_step, reset_model
"""

import argparse
import glob
import io
import json as _json
import os
import sys
from collections import deque

import h5py
import numpy as np
import torch
from PIL import Image

# Add parent paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if VAM_DIR not in sys.path:
    sys.path.insert(0, VAM_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.action_dit import ActionDiT
from joint_inference import generate_video_and_actions, make_schedule
from video_action_dataset import (
    assemble_multiview_grid, extract_quadrant,
    MULTIVIEW_LAYOUT, MULTIVIEW_CAMERAS,
    _crop_and_resize as _mv_crop_and_resize,
    RoboTwinDataset, MultiTaskRoboTwinDataset,
    ROBOTWIN_TRAIN_TASKS, ROBOTWIN_HOLDOUT_TASKS, ROBOTWIN_ALL_TASKS,
)


class VAMPolicy:
    """VAM-based policy for closed-loop robot control.

    Maintains a history of observations and generates action predictions
    using the ActionDiT conditioned on video features.

    Args:
        pipe: Loaded WanVideoPipeline.
        action_dit: Loaded ActionDiT (eval mode, with action stats).
        task_name: Task description for text prompt.
        num_frames: Number of frames for action generation context.
        height: Frame height for the model.
        width: Frame width for the model.
        history_len: Number of past observations to keep.
        num_denoise_steps: Number of denoising steps for action generation.
        target_camera: Camera key to use from observations.
    """

    def __init__(
        self,
        pipe: WanVideoPipeline,
        action_dit: ActionDiT,
        task_name: str = "manipulation",
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        history_len: int = 49,
        num_denoise_steps: int = 20,
        target_camera: str = "head_camera",
        multiview: bool = False,
    ):
        self.pipe = pipe
        self.action_dit = action_dit
        self.task_name = task_name
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_denoise_steps = num_denoise_steps
        self.target_camera = target_camera
        self.multiview = multiview
        self.cameras = MULTIVIEW_CAMERAS if multiview else [target_camera]
        self.camera_layout = MULTIVIEW_LAYOUT if multiview else None
        self.quadrant_h = height // 2 if multiview else height
        self.quadrant_w = width // 2 if multiview else width

        # Observation history (ring buffer of PIL images)
        self.obs_history = deque(maxlen=history_len)

        # Cached action chunk for temporal action chunking
        self._action_chunk = None
        self._chunk_index = 0

    def reset(self):
        """Reset observation history and cached actions."""
        self.obs_history.clear()
        self._action_chunk = None
        self._chunk_index = 0

    def _obs_to_pil(self, obs_image) -> Image.Image:
        """Convert observation image to PIL Image.

        Handles numpy arrays (H,W,3 uint8) and PIL Images.
        """
        if isinstance(obs_image, Image.Image):
            return obs_image.convert("RGB")
        if isinstance(obs_image, np.ndarray):
            if obs_image.dtype != np.uint8:
                obs_image = obs_image.astype(np.uint8)
            return Image.fromarray(obs_image).convert("RGB")
        raise TypeError(f"Unsupported observation type: {type(obs_image)}")

    def _crop_and_resize(self, image: Image.Image) -> Image.Image:
        """Center crop and resize to model resolution."""
        img_w, img_h = image.size
        scale = max(self.width / img_w, self.height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - self.width) // 2
        top = (new_h - self.height) // 2
        return image.crop((left, top, left + self.width, top + self.height))

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> np.ndarray:
        """Predict a single action from the current observation.

        Args:
            obs_dict: Dictionary with camera observations. Expected keys
                depend on the robot, but typically includes 'head_camera'
                with an RGB image (np.ndarray or PIL.Image).

        Returns:
            action: (action_dim,) numpy array of denormalized joint actions.
        """
        if self.multiview:
            # Assemble multi-camera grid observation
            frames_by_camera = {}
            for cam in self.cameras:
                if cam in obs_dict:
                    img = self._obs_to_pil(obs_dict[cam])
                    img = _mv_crop_and_resize(
                        img, self.quadrant_h, self.quadrant_w
                    )
                    frames_by_camera[cam] = img
                # Missing cameras default to black (handled by assemble_multiview_grid)
            obs_image = assemble_multiview_grid(
                frames_by_camera, self.camera_layout,
                self.quadrant_h, self.quadrant_w,
            )
            self.obs_history.append(obs_image)
        else:
            # Single-camera mode
            obs_key = self.target_camera
            if obs_key not in obs_dict:
                # Try common alternatives
                for alt in ["head_camera", "image", "rgb", "obs"]:
                    if alt in obs_dict:
                        obs_key = alt
                        break
                else:
                    raise KeyError(
                        f"No observation image found in obs_dict. "
                        f"Keys: {list(obs_dict.keys())}"
                    )

            obs_image = self._obs_to_pil(obs_dict[obs_key])
            obs_image = self._crop_and_resize(obs_image)
            self.obs_history.append(obs_image)

        # If we have a cached action chunk with remaining steps, use it
        if self._action_chunk is not None and self._chunk_index < len(self._action_chunk):
            action = self._action_chunk[self._chunk_index]
            self._chunk_index += 1
            return action

        # Build conditioning context from observation history
        context_frames = list(self.obs_history)

        # Pad to num_frames if history is shorter
        while len(context_frames) < self.num_frames:
            context_frames.insert(0, context_frames[0])

        # Truncate if longer
        if len(context_frames) > self.num_frames:
            context_frames = context_frames[-self.num_frames:]

        # Use action-only schedule (no video generation, fast inference)
        schedule = make_schedule(
            "action_only",
            num_steps=self.num_denoise_steps,
            shift=5.0,
        )

        if self.multiview:
            prompt = (
                f"A multi-view video shows that the bimanual robot is performing "
                f"a {self.task_name} task. The video is split into four views: "
                f"head camera (top-left), third-person view (top-right), "
                f"left camera (bottom-left), right camera (bottom-right)."
            )
        else:
            prompt = f"The bimanual robot is performing a {self.task_name} task."

        # Run joint inference with action-only schedule
        # vace_video = context frames (used as conditioning)
        # vace_reference_image = first frame
        _, actions = generate_video_and_actions(
            pipe=self.pipe,
            action_dit=self.action_dit,
            schedule=schedule,
            prompt=prompt,
            vace_video=context_frames,
            vace_reference_image=[context_frames[0]],
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            seed=42,
            cfg_scale=1.0,
            tiled=True,
        )

        # Cache the full action chunk, return first action
        self._action_chunk = actions  # (num_frames, action_dim)
        self._chunk_index = 1  # next call returns index 1
        return actions[0]


# ---------------------------------------------------------------------------
# RoboTwin eval_policy.py interface: 3 functions
# ---------------------------------------------------------------------------

# Global policy instance (set by get_model, used by eval_one_step/reset_model)
_policy: VAMPolicy = None


def get_model(
    task_name: str,
    ckpt_path: str,
    model_paths: str = None,
    tokenizer_path: str = None,
    action_dim: int = 14,
    action_dit_dim: int = 768,
    action_dit_ffn_dim: int = 3072,
    action_dit_num_heads: int = 12,
    action_dit_num_layers: int = 8,
    action_dit_bridge_layers: str = "3,7,11,15,19,23,26,29",
    video_dim: int = 1536,
    bridge_type: str = "joint_self_attn",
    num_frames: int = 49,
    height: int = 480,
    width: int = 832,
    target_camera: str = "head_camera",
    multiview: bool = False,
    device: str = "cuda",
    num_denoise_steps: int = 20,
    **kwargs,
):
    """Load VAM policy model for RoboTwin evaluation.

    Args:
        task_name: Name of the task being evaluated.
        ckpt_path: Path to the trained checkpoint (.safetensors).
        model_paths: JSON list of base model paths (VACE diffusion, T5, VAE).
        tokenizer_path: Path to T5 tokenizer.
        action_dim: Action dimension (14 for most RoboTwin robots, 16 for franka).
        device: Device to load models on.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        VAMPolicy instance.
    """
    global _policy

    # Default model paths
    if model_paths is None:
        model_paths = (
            '["models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",'
            '"models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",'
            '"models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"]'
        )
    if tokenizer_path is None:
        tokenizer_path = "models/Wan-AI/Wan2.1-VACE-1.3B/google/umt5-xxl"

    import json
    model_path_list = json.loads(model_paths) if isinstance(model_paths, str) else model_paths
    model_configs = [ModelConfig(p) for p in model_path_list]
    tokenizer_config = ModelConfig(tokenizer_path)

    # Load video pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # Create ActionDiT
    bridge_layers = tuple(int(x) for x in action_dit_bridge_layers.split(","))
    action_dit = ActionDiT(
        action_dim=action_dim,
        dim=action_dit_dim,
        ffn_dim=action_dit_ffn_dim,
        num_heads=action_dit_num_heads,
        num_layers=action_dit_num_layers,
        video_dim=video_dim,
        bridge_layers=bridge_layers,
        bridge_type=bridge_type,
    ).to(dtype=torch.bfloat16, device=device)

    # Load checkpoint
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    # Extract ActionDiT weights (prefixed with "action_dit.")
    action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_dit.")}
    if action_keys:
        cleaned = {k.removeprefix("action_dit."): v for k, v in action_keys.items()}
        result = action_dit.load_state_dict(cleaned, strict=False)
        if result.missing_keys:
            # Filter out buffers that may not be in checkpoint
            real_missing = [k for k in result.missing_keys
                           if k not in ("action_mean", "action_std")]
            if real_missing:
                print(f"WARNING: Missing ActionDiT keys: {real_missing}")
        print(f"Loaded ActionDiT from {ckpt_path} ({len(action_keys)} keys)")

    # Load VACE weights if present in checkpoint
    # The checkpoint stores VACE keys with prefix stripped (via remove_prefix_in_ckpt),
    # so they should match pipe.vace.state_dict() keys directly.
    candidate_keys = {k: v for k, v in state_dict.items() if not k.startswith("action_dit.")}
    if candidate_keys and hasattr(pipe, "vace"):
        vace_expected = set(pipe.vace.state_dict().keys())
        vace_keys = {k: v for k, v in candidate_keys.items() if k in vace_expected}
        if vace_keys:
            result = pipe.vace.load_state_dict(vace_keys, strict=False)
            matched = len(vace_expected) - len(result.missing_keys)
            print(f"Loaded VACE weights from checkpoint: "
                  f"{matched}/{len(vace_expected)} keys matched")
            if result.missing_keys:
                print(f"  VACE keys not in checkpoint ({len(result.missing_keys)}): "
                      f"{result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
        else:
            ignored = len(candidate_keys)
            print(f"WARNING: Checkpoint has {ignored} non-ActionDiT keys but none "
                  f"match VACE state dict. VACE will use pretrained weights only.")

    action_dit.eval()
    pipe.eval() if hasattr(pipe, "eval") else None

    _policy = VAMPolicy(
        pipe=pipe,
        action_dit=action_dit,
        task_name=task_name,
        num_frames=num_frames,
        height=height,
        width=width,
        num_denoise_steps=num_denoise_steps,
        target_camera=target_camera,
        multiview=multiview,
    )

    return _policy


def eval_one_step(model: VAMPolicy, obs_dict: dict) -> np.ndarray:
    """Predict one action step given the current observation.

    Args:
        model: VAMPolicy instance from get_model().
        obs_dict: Observation dictionary from SAPIEN environment.

    Returns:
        action: (action_dim,) numpy array of joint actions.
    """
    return model.predict_action(obs_dict)


def reset_model(model: VAMPolicy):
    """Reset the policy state between episodes.

    Args:
        model: VAMPolicy instance from get_model().
    """
    model.reset()


# ---------------------------------------------------------------------------
# Video quality metrics
# ---------------------------------------------------------------------------

def compute_video_metrics(gen_frames, gt_frames):
    """Compute PSNR, SSIM, and LPIPS between generated and GT video frames.

    Args:
        gen_frames: List of generated frames (np.ndarray uint8 or PIL.Image).
        gt_frames: List of ground-truth frames (np.ndarray uint8 or PIL.Image).

    Returns:
        dict with keys: video_mse, video_psnr, video_ssim, video_lpips.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    def _to_np(frame):
        if isinstance(frame, Image.Image):
            return np.array(frame.convert("RGB"))
        return frame

    gen_np = [_to_np(f) for f in gen_frames]
    gt_np = [_to_np(f) for f in gt_frames]
    n = min(len(gen_np), len(gt_np))
    gen_np, gt_np = gen_np[:n], gt_np[:n]

    mse_vals, psnr_vals, ssim_vals = [], [], []
    for g, t in zip(gen_np, gt_np):
        # Resize if shapes differ
        if g.shape != t.shape:
            from PIL import Image as _Img
            g = np.array(_Img.fromarray(g).resize((t.shape[1], t.shape[0]), _Img.LANCZOS))
        mse_vals.append(float(np.mean((g.astype(np.float32) - t.astype(np.float32)) ** 2)))
        psnr_vals.append(peak_signal_noise_ratio(t, g, data_range=255))
        ssim_vals.append(structural_similarity(t, g, data_range=255, channel_axis=2))

    # LPIPS (batched)
    lpips_val = _compute_lpips(gen_np, gt_np)

    return {
        "video_mse": float(np.mean(mse_vals)),
        "video_psnr": float(np.mean(psnr_vals)),
        "video_ssim": float(np.mean(ssim_vals)),
        "video_lpips": float(lpips_val),
    }


# Lazy-loaded global LPIPS model (heavy, load once)
_lpips_model = None

def _compute_lpips(gen_np, gt_np):
    """Compute mean LPIPS over frame pairs."""
    global _lpips_model
    import lpips as _lpips_lib

    if _lpips_model is None:
        _lpips_model = _lpips_lib.LPIPS(net="alex").eval()
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()

    device = next(_lpips_model.parameters()).device
    vals = []
    for g, t in zip(gen_np, gt_np):
        # HWC uint8 → CHW float [-1, 1]
        g_t = torch.from_numpy(g).permute(2, 0, 1).float() / 127.5 - 1.0
        t_t = torch.from_numpy(t).permute(2, 0, 1).float() / 127.5 - 1.0
        with torch.no_grad():
            d = _lpips_model(g_t.unsqueeze(0).to(device), t_t.unsqueeze(0).to(device))
        vals.append(d.item())
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Offline evaluation: load held-out episodes, generate actions, compute metrics
# ---------------------------------------------------------------------------

def _action_dim_labels(adim: int) -> list:
    """Return human-readable labels for each action dimension."""
    if adim == 7:
        return ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    if adim == 14:
        return [
            "l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_grip",
            "r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_grip",
        ]
    if adim == 16:
        return [
            "l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_j7", "l_grip",
            "r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_j7", "r_grip",
        ]
    return [f"dim_{j}" for j in range(adim)]


def _save_video(frames, path: str, imageio_mod, fps: int = 15):
    """Save a list of PIL Images (or numpy arrays) as an mp4 file."""
    np_frames = [np.array(f) for f in frames]
    imageio_mod.mimwrite(path, np_frames, fps=fps, quality=6)


def run_offline_eval(policy: VAMPolicy, args):
    """Offline eval: load held-out episodes, generate actions, compute metrics.

    Builds a validation dataset (MultiTaskRoboTwinDataset or RoboTwinDataset),
    runs generate_video_and_actions() on each sample, and computes MSE/MAE
    against ground truth actions.

    Args:
        policy: Loaded VAMPolicy instance.
        args: Parsed CLI arguments.
    """
    # --- Build validation dataset ---
    dataset_kwargs = dict(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        split="val",
        val_ratio=0.1,
        num_val_samples=args.num_eval_samples,
        target_camera=args.target_camera,
        multiview=args.multiview,
        action_stats_path=args.action_stats_path,
    )

    if args.dataset_dir:
        # Resolve task list
        if args.tasks is None:
            tasks = None  # MultiTaskRoboTwinDataset defaults to ROBOTWIN_TRAIN_TASKS
        elif args.tasks == "holdout":
            tasks = ROBOTWIN_HOLDOUT_TASKS
        elif args.tasks == "all":
            tasks = ROBOTWIN_ALL_TASKS
        else:
            tasks = [t.strip() for t in args.tasks.split(",")]

        # MultiTaskRoboTwinDataset takes action_stats_path as its own param,
        # so exclude it from kwargs to avoid duplicate keyword arguments.
        mt_kwargs = {k: v for k, v in dataset_kwargs.items()
                     if k not in ("action_stats_path",)}
        val_dataset = MultiTaskRoboTwinDataset(
            dataset_dir=args.dataset_dir,
            robot=args.robot,
            variant=args.val_variant or args.variant,
            tasks=tasks,
            action_stats_path=args.action_stats_path,
            **mt_kwargs,
        )
    elif args.hdf5_data_root:
        val_dataset = RoboTwinDataset(
            data_root=args.hdf5_data_root,
            **dataset_kwargs,
        )
    else:
        raise ValueError("Offline eval requires --dataset_dir or --hdf5_data_root")

    print(f"\nOffline eval: {len(val_dataset)} val samples")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Metrics accumulators ---
    all_action_mse = []
    all_action_mae = []
    all_video_mse = []
    all_video_psnr = []
    all_video_ssim = []
    all_video_lpips = []

    schedule = make_schedule(
        args.schedule, num_steps=args.num_denoise_steps, shift=5.0,
    )

    action_mean = policy.action_dit.action_mean.float().cpu().numpy()
    action_std = policy.action_dit.action_std.float().cpu().numpy()

    import imageio.v2 as imageio

    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        gt_actions = sample["action_trajectory"]  # (T, action_dim) normalized
        if isinstance(gt_actions, torch.Tensor):
            gt_actions = gt_actions.numpy()

        # Denormalize ground truth
        gt_denorm = gt_actions * action_std + action_mean

        # Build prompt
        prompt = sample["prompt"]
        task_name = sample.get("task_name", "unknown")

        # Run joint inference
        with torch.no_grad():
            gen_video, pred_denorm = generate_video_and_actions(
                pipe=policy.pipe,
                action_dit=policy.action_dit,
                schedule=schedule,
                prompt=prompt,
                vace_video=sample.get("vace_video"),
                vace_reference_image=sample.get("vace_reference_image"),
                num_frames=len(sample["video"]),
                height=sample["video"][0].size[1],
                width=sample["video"][0].size[0],
                seed=42,
                cfg_scale=1.0,
                tiled=True,
            )

        # Align lengths (pred may differ if padding involved)
        min_len = min(len(pred_denorm), len(gt_denorm))
        pred_denorm = pred_denorm[:min_len]
        gt_denorm = gt_denorm[:min_len]

        # Compute action metrics
        action_mse = float(np.mean((pred_denorm - gt_denorm) ** 2))
        action_mae = float(np.mean(np.abs(pred_denorm - gt_denorm)))

        all_action_mse.append(action_mse)
        all_action_mae.append(action_mae)

        # Compute video metrics
        vid_metrics = compute_video_metrics(gen_video, sample["video"])
        all_video_mse.append(vid_metrics["video_mse"])
        all_video_psnr.append(vid_metrics["video_psnr"])
        all_video_ssim.append(vid_metrics["video_ssim"])
        all_video_lpips.append(vid_metrics["video_lpips"])

        # Per-sample metadata
        episode_path = sample.get("episode_path", "")
        start_frame = sample.get("start_frame", -1)
        end_frame = sample.get("end_frame", -1)

        print(f"\n[Sample {i+1}/{len(val_dataset)}] task={task_name}  "
              f"action_MSE={action_mse:.6f}  action_MAE={action_mae:.6f}  "
              f"video_MSE={vid_metrics['video_mse']:.2f}  PSNR={vid_metrics['video_psnr']:.2f}  "
              f"SSIM={vid_metrics['video_ssim']:.4f}  LPIPS={vid_metrics['video_lpips']:.4f}")
        print(f"  episode: {episode_path}  frames [{start_frame}:{end_frame}]")
        print(f"  prompt: {prompt}")

        # --- Save per-sample subfolder ---
        if args.output_dir:
            safe_task = task_name.replace(" ", "_")
            sample_dir = os.path.join(
                args.output_dir, f"sample_{i:03d}_{safe_task}")
            os.makedirs(sample_dir, exist_ok=True)

            # Videos (mp4)
            _save_video(gen_video, os.path.join(sample_dir, "video_generated.mp4"),
                        imageio)
            gt_frames = sample["video"]
            _save_video(gt_frames, os.path.join(sample_dir, "video_gt.mp4"),
                        imageio)

            # Actions (npz) — both normalized and denormalized
            np.savez_compressed(
                os.path.join(sample_dir, "actions_generated.npz"),
                actions=pred_denorm,
            )
            np.savez_compressed(
                os.path.join(sample_dir, "actions_gt.npz"),
                actions=gt_denorm,
                actions_normalized=gt_actions[:min_len],
            )

            # Per-sample metadata
            sample_meta = {
                "sample_index": i,
                "task_name": task_name,
                "prompt": prompt,
                "episode_path": episode_path,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "episode_length": int(sample.get("episode_length", -1)),
                "num_frames": len(gen_video),
                "action_dim": int(pred_denorm.shape[1]),
                "action_mse": action_mse,
                "action_mae": action_mae,
                "video_mse": vid_metrics["video_mse"],
                "video_psnr": vid_metrics["video_psnr"],
                "video_ssim": vid_metrics["video_ssim"],
                "video_lpips": vid_metrics["video_lpips"],
            }
            with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
                _json.dump(sample_meta, f, indent=2)

    # --- Aggregate metrics ---
    print("\n" + "=" * 60)
    print(f"Offline Eval Summary  ({len(val_dataset)} samples, "
          f"{args.num_denoise_steps} denoise steps, schedule={args.schedule})")
    print(f"  Mean action_MSE: {np.mean(all_action_mse):.6f}")
    print(f"  Mean action_MAE: {np.mean(all_action_mae):.6f}")
    print(f"  Mean video_MSE: {np.mean(all_video_mse):.2f}")
    print(f"  Mean video_PSNR: {np.mean(all_video_psnr):.2f}")
    print(f"  Mean video_SSIM: {np.mean(all_video_ssim):.4f}")
    print(f"  Mean video_LPIPS: {np.mean(all_video_lpips):.4f}")
    print("=" * 60)

    # --- Save aggregate results ---
    if args.output_dir:
        results = {
            "num_samples": len(val_dataset),
            "num_denoise_steps": args.num_denoise_steps,
            "schedule": args.schedule,
            "checkpoint": args.action_checkpoint,
            "mean_action_mse": float(np.mean(all_action_mse)),
            "mean_action_mae": float(np.mean(all_action_mae)),
            "mean_video_mse": float(np.mean(all_video_mse)),
            "mean_video_psnr": float(np.mean(all_video_psnr)),
            "mean_video_ssim": float(np.mean(all_video_ssim)),
            "mean_video_lpips": float(np.mean(all_video_lpips)),
            "per_sample_action_mse": [float(m) for m in all_action_mse],
            "per_sample_action_mae": [float(m) for m in all_action_mae],
            "per_sample_video_mse": [float(m) for m in all_video_mse],
            "per_sample_video_psnr": [float(m) for m in all_video_psnr],
            "per_sample_video_ssim": [float(m) for m in all_video_ssim],
            "per_sample_video_lpips": [float(m) for m in all_video_lpips],
        }
        out_path = os.path.join(args.output_dir, "offline_eval_results.json")
        with open(out_path, "w") as f:
            _json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Online (open-loop replay) evaluation
# ---------------------------------------------------------------------------

def _discover_val_episodes(args) -> list:
    """Discover validation episode HDF5 paths for online eval.

    Returns list of (task_name, episode_path) tuples.
    """
    import random

    if args.hdf5_data_root:
        # Single-task: apply same train/val split as RoboTwinDataset
        all_files = sorted(glob.glob(os.path.join(args.hdf5_data_root, "episode*.hdf5")))
        rng = random.Random(42)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_files) * 0.1))
        val_indices = sorted(indices[:n_val])
        task = args.task_name
        return [(task, all_files[i]) for i in val_indices[:args.num_eval_episodes]]

    if args.dataset_dir:
        # Multi-task: resolve task list, then pick val episodes per task
        if args.tasks is None:
            tasks = ROBOTWIN_TRAIN_TASKS
        elif args.tasks == "holdout":
            tasks = ROBOTWIN_HOLDOUT_TASKS
        elif args.tasks == "all":
            tasks = list(ROBOTWIN_ALL_TASKS)
        else:
            tasks = [t.strip() for t in args.tasks.split(",")]

        variant = args.val_variant or args.variant
        episodes = []
        for task_name in tasks:
            data_root = os.path.join(
                args.dataset_dir, task_name,
                f"{args.robot}_{variant}", "data",
            )
            if not os.path.isdir(data_root):
                continue
            all_files = sorted(glob.glob(os.path.join(data_root, "episode*.hdf5")))
            if not all_files:
                continue
            rng = random.Random(42)
            indices = list(range(len(all_files)))
            rng.shuffle(indices)
            n_val = max(1, int(len(all_files) * 0.1))
            val_files = [all_files[i] for i in sorted(indices[:n_val])]
            for ep in val_files[:args.num_eval_episodes]:
                episodes.append((task_name.replace("_", " "), ep))
        return episodes

    raise ValueError("Online eval requires --dataset_dir or --hdf5_data_root")


def run_online_eval(policy: VAMPolicy, args):
    """Online (open-loop replay) evaluation.

    Loads full episodes from HDF5, feeds frames one-by-one to the policy
    via eval_one_step(), and collects the predicted action trajectory.
    Compares to ground truth over the full episode length.

    Saves per-episode: observation video (mp4), predicted/GT actions (npz),
    and metadata (json).
    """
    import imageio.v2 as imageio

    episodes = _discover_val_episodes(args)
    if not episodes:
        print("No validation episodes found.")
        return

    print(f"\nOnline eval: {len(episodes)} episodes")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    action_mean = policy.action_dit.action_mean.float().cpu().numpy()
    action_std = policy.action_dit.action_std.float().cpu().numpy()
    dim_labels = _action_dim_labels(len(action_mean))

    all_mse, all_mae, all_per_dim_mse = [], [], []

    for ep_i, (task_name, episode_path) in enumerate(episodes):
        # --- Load full episode ---
        with h5py.File(episode_path, "r") as f:
            cam = args.target_camera
            raw_rgb = f[f"observation/{cam}/rgb"][:]
            obs_frames = [
                Image.open(io.BytesIO(bytes(raw_rgb[t]))).convert("RGB")
                for t in range(len(raw_rgb))
            ]
            gt_actions_raw = f["joint_action/vector"][:].astype(np.float32)

        episode_len = len(obs_frames)

        # Denormalize GT
        gt_denorm = gt_actions_raw * action_std + action_mean

        # --- Step through episode ---
        reset_model(policy)
        predicted_actions = []

        for t in range(episode_len):
            obs_dict = {cam: obs_frames[t]}
            if policy.multiview:
                # Feed all available cameras for multiview
                with h5py.File(episode_path, "r") as f:
                    for mv_cam in MULTIVIEW_CAMERAS:
                        if mv_cam == "third_view_rgb":
                            if mv_cam in f:
                                raw = f[mv_cam][t]
                                obs_dict[mv_cam] = Image.open(
                                    io.BytesIO(bytes(raw))
                                ).convert("RGB")
                        else:
                            key = f"observation/{mv_cam}/rgb"
                            if key in f:
                                raw = f[key][t]
                                obs_dict[mv_cam] = Image.open(
                                    io.BytesIO(bytes(raw))
                                ).convert("RGB")

            action = eval_one_step(policy, obs_dict)
            predicted_actions.append(action)

        pred_traj = np.stack(predicted_actions, axis=0)  # (T, action_dim)

        # --- Metrics ---
        min_len = min(len(pred_traj), len(gt_denorm))
        pred_traj = pred_traj[:min_len]
        gt_clip = gt_denorm[:min_len]

        mse = float(np.mean((pred_traj - gt_clip) ** 2))
        mae = float(np.mean(np.abs(pred_traj - gt_clip)))
        per_dim_mse = np.mean((pred_traj - gt_clip) ** 2, axis=0)

        all_mse.append(mse)
        all_mae.append(mae)
        all_per_dim_mse.append(per_dim_mse)

        print(f"\n[Episode {ep_i+1}/{len(episodes)}] task={task_name}  "
              f"len={episode_len}  MSE={mse:.6f}  MAE={mae:.6f}")
        print(f"  path: {episode_path}")
        adim = pred_traj.shape[1]
        dim_str = "  ".join(f"{dim_labels[j]}={per_dim_mse[j]:.6f}"
                            for j in range(min(len(dim_labels), adim)))
        print(f"  per-dim MSE: {dim_str}")

        # --- Save per-episode subfolder ---
        if args.output_dir:
            safe_task = task_name.replace(" ", "_")
            ep_dir = os.path.join(
                args.output_dir, f"episode_{ep_i:03d}_{safe_task}")
            os.makedirs(ep_dir, exist_ok=True)

            # Observation video
            _save_video(obs_frames, os.path.join(ep_dir, "video_obs.mp4"),
                        imageio)

            # Actions
            np.savez_compressed(
                os.path.join(ep_dir, "actions_predicted.npz"),
                actions=pred_traj,
            )
            np.savez_compressed(
                os.path.join(ep_dir, "actions_gt.npz"),
                actions=gt_clip,
                actions_raw=gt_actions_raw[:min_len],
            )

            # Metadata
            ep_meta = {
                "episode_index": ep_i,
                "task_name": task_name,
                "episode_path": episode_path,
                "episode_length": episode_len,
                "action_dim": int(adim),
                "num_inference_calls": int(
                    np.ceil(episode_len / policy.num_frames)),
                "mse": mse,
                "mae": mae,
                "per_dim_mse": {dim_labels[j]: float(per_dim_mse[j])
                                for j in range(min(len(dim_labels), adim))},
            }
            with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
                _json.dump(ep_meta, f, indent=2)

    # --- Aggregate ---
    print("\n" + "=" * 60)
    print(f"Online Eval Summary  ({len(episodes)} episodes, "
          f"{args.num_denoise_steps} denoise steps)")
    print(f"  Mean MSE: {np.mean(all_mse):.6f}")
    print(f"  Mean MAE: {np.mean(all_mae):.6f}")
    agg_per_dim = np.mean(all_per_dim_mse, axis=0)
    dim_str = "  ".join(f"{dim_labels[j]}={agg_per_dim[j]:.6f}"
                        for j in range(min(len(dim_labels), len(agg_per_dim))))
    print(f"  Per-dim MSE: {dim_str}")
    print("=" * 60)

    if args.output_dir:
        results = {
            "num_episodes": len(episodes),
            "num_denoise_steps": args.num_denoise_steps,
            "checkpoint": args.action_checkpoint,
            "mean_mse": float(np.mean(all_mse)),
            "mean_mae": float(np.mean(all_mae)),
            "per_dim_mse": {dim_labels[j]: float(agg_per_dim[j])
                           for j in range(min(len(dim_labels), len(agg_per_dim)))},
            "per_episode_mse": [float(m) for m in all_mse],
            "per_episode_mae": [float(m) for m in all_mae],
        }
        out_path = os.path.join(args.output_dir, "online_eval_results.json")
        with open(out_path, "w") as f:
            _json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Standalone evaluation loop (optional, for testing without SAPIEN)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VAM policy adapter for RoboTwin 2.0 SAPIEN evaluation."
    )
    parser.add_argument("--action_checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.safetensors)")
    parser.add_argument("--task_name", type=str, default="adjust_bottle",
                        help="Task name for evaluation")
    parser.add_argument("--robot", type=str, default="aloha-agilex",
                        help="Robot name")
    parser.add_argument("--action_dim", type=int, default=14,
                        help="Action dimension (14 for most robots, 16 for franka)")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="JSON list of base model paths")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to T5 tokenizer")
    parser.add_argument("--bridge_type", type=str, default="joint_self_attn",
                        choices=["cross_attn", "cross_attn_detach", "joint_self_attn"])
    parser.add_argument("--target_camera", type=str, default="head_camera")
    parser.add_argument("--multiview", default=False, action="store_true",
                        help="Use 2x2 multi-view grid (head + third_view + left + right)")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_denoise_steps", type=int, default=20,
                        help="Number of denoising steps (set to 2 for fast verification)")
    parser.add_argument("--test", action="store_true",
                        help="Run a quick test with a dummy observation")
    parser.add_argument("--offline", action="store_true",
                        help="Run offline eval on held-out episodes")
    parser.add_argument("--online", action="store_true",
                        help="Run online (open-loop replay) eval on full episodes")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Top-level RoboTwin dataset dir (for multi-task offline eval)")
    parser.add_argument("--hdf5_data_root", type=str, default=None,
                        help="Single-task HDF5 data root (for single-task offline eval)")
    parser.add_argument("--variant", type=str, default="clean_50",
                        help="Dataset variant (clean_50 or randomized_500)")
    parser.add_argument("--val_variant", type=str, default=None,
                        help="Validation variant (defaults to --variant)")
    parser.add_argument("--action_stats_path", type=str, default=None,
                        help="Path to action normalization stats (.npy)")
    parser.add_argument("--num_eval_samples", type=int, default=2,
                        help="Number of val samples to evaluate in offline mode")
    parser.add_argument("--num_eval_episodes", type=int, default=1,
                        help="Number of val episodes per task in online mode")
    parser.add_argument("--schedule", type=str, default="sync",
                        choices=["sync", "action_only"],
                        help="Denoising schedule for offline eval")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names for offline eval. "
                             "Use 'holdout' for OOD tasks, 'all' for everything, "
                             "or omit for default train tasks (ID).")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes (for SAPIEN eval)")
    parser.add_argument("--output_dir", type=str, default="results/robotwin_eval/",
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    print(f"Loading VAM policy for task '{args.task_name}' on {args.robot}...")

    policy = get_model(
        task_name=args.task_name,
        ckpt_path=args.action_checkpoint,
        model_paths=args.model_paths,
        tokenizer_path=args.tokenizer_path,
        action_dim=args.action_dim,
        bridge_type=args.bridge_type,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        target_camera=args.target_camera,
        multiview=args.multiview,
        device=args.device,
        num_denoise_steps=args.num_denoise_steps,
    )

    if args.test:
        print("\nRunning test with dummy observation...")
        # Create dummy observations for all cameras needed
        dummy_obs = {
            args.target_camera: np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        }
        if args.multiview:
            for cam in MULTIVIEW_CAMERAS:
                dummy_obs[cam] = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        action = eval_one_step(policy, dummy_obs)
        print(f"Predicted action ({len(action)}D): {action}")
        print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")

        reset_model(policy)
        print("Policy reset. Test passed.")
        return

    if args.offline:
        run_offline_eval(policy, args)
        return

    if args.online:
        run_online_eval(policy, args)
        return

    # Full SAPIEN evaluation requires RoboTwin to be installed
    print("\nFor full SAPIEN evaluation, use this adapter with RoboTwin's eval_policy.py:")
    print("  1. Clone RoboTwin: git clone https://github.com/TianxingChen/RoboTwin")
    print("  2. Install: cd RoboTwin && pip install -e .")
    print("  3. In eval_policy.py, import:")
    print("     from eval_robotwin import get_model, eval_one_step, reset_model")
    print(f"\n  Configured for: task={args.task_name}, robot={args.robot}, "
          f"action_dim={args.action_dim}")


if __name__ == "__main__":
    main()
