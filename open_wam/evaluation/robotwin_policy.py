"""Package-native RoboTwin policy compatibility layer."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from open_wam.data.robotwin import (
    MULTIVIEW_CAMERAS,
    MULTIVIEW_LAYOUT,
    _crop_and_resize as _mv_crop_and_resize,
    assemble_multiview_grid,
)
from open_wam.inference import load_wam_models, make_schedule
from open_wam.inference.joint_generation import generate_video_and_actions


class RoboTwinVAMPolicy:
    """Package-native VAM policy for RoboTwin closed-loop evaluation."""

    def __init__(
        self,
        pipe,
        action_dit,
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
        self.obs_history = deque(maxlen=history_len)
        self._action_chunk = None
        self._chunk_index = 0

    def reset(self):
        self.obs_history.clear()
        self._action_chunk = None
        self._chunk_index = 0

    def _obs_to_pil(self, obs_image) -> Image.Image:
        if isinstance(obs_image, Image.Image):
            return obs_image.convert("RGB")
        if isinstance(obs_image, np.ndarray):
            if obs_image.dtype != np.uint8:
                obs_image = obs_image.astype(np.uint8)
            return Image.fromarray(obs_image).convert("RGB")
        raise TypeError(f"Unsupported observation type: {type(obs_image)}")

    def _crop_and_resize(self, image: Image.Image) -> Image.Image:
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
        if self.multiview:
            frames_by_camera = {}
            for cam in self.cameras:
                if cam in obs_dict:
                    img = self._obs_to_pil(obs_dict[cam])
                    img = _mv_crop_and_resize(img, self.quadrant_h, self.quadrant_w)
                    frames_by_camera[cam] = img
            obs_image = assemble_multiview_grid(
                frames_by_camera,
                self.camera_layout,
                self.quadrant_h,
                self.quadrant_w,
            )
            self.obs_history.append(obs_image)
        else:
            obs_key = self.target_camera
            if obs_key not in obs_dict:
                for alt in ["head_camera", "image", "rgb", "obs"]:
                    if alt in obs_dict:
                        obs_key = alt
                        break
                else:
                    raise KeyError(f"No observation image found in obs_dict. Keys: {list(obs_dict.keys())}")
            obs_image = self._obs_to_pil(obs_dict[obs_key])
            obs_image = self._crop_and_resize(obs_image)
            self.obs_history.append(obs_image)

        if self._action_chunk is not None and self._chunk_index < len(self._action_chunk):
            action = self._action_chunk[self._chunk_index]
            self._chunk_index += 1
            return action

        context_frames = list(self.obs_history)
        while len(context_frames) < self.num_frames:
            context_frames.insert(0, context_frames[0])
        if len(context_frames) > self.num_frames:
            context_frames = context_frames[-self.num_frames:]

        schedule = make_schedule("action_only", num_steps=self.num_denoise_steps, shift=5.0)

        if self.multiview:
            prompt = (
                f"A multi-view video shows that the bimanual robot is performing a {self.task_name} task. "
                "The video is split into four views: head camera (top-left), third-person view (top-right), "
                "left camera (bottom-left), right camera (bottom-right)."
            )
        else:
            prompt = f"The bimanual robot is performing a {self.task_name} task."

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

        self._action_chunk = actions
        self._chunk_index = 1
        return actions[0]


def _build_loader_cfg(
    ckpt_path: str,
    model_paths,
    tokenizer_path,
    *,
    action_dim: int,
    action_dit_dim: int,
    action_dit_ffn_dim: int,
    action_dit_num_heads: int,
    action_dit_num_layers: int,
    action_dit_bridge_layers: str,
    video_dim: int,
    bridge_type: str,
):
    bridge_layers = [int(x) for x in action_dit_bridge_layers.split(",") if x]
    return OmegaConf.create(
        {
            "eval": {
                "ckpt_path": ckpt_path,
                "model_paths": model_paths,
                "tokenizer_path": tokenizer_path,
            },
            "model": {
                "action_dim": action_dim,
                "dim": action_dit_dim,
                "ffn_dim": action_dit_ffn_dim,
                "num_heads": action_dit_num_heads,
                "num_layers": action_dit_num_layers,
                "bridge_layers": bridge_layers,
                "bridge_type": bridge_type,
                "backbone": {"video_dim": video_dim},
            },
            "inference": {
                "schedule_type": "action_only",
                "num_steps": 20,
                "shift": 5.0,
                "cfg_scale": 1.0,
            },
        }
    )


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
    """Load a RoboTwin VAM policy using package-native OpenWAM components."""
    cfg = _build_loader_cfg(
        ckpt_path,
        model_paths,
        tokenizer_path,
        action_dim=action_dim,
        action_dit_dim=action_dit_dim,
        action_dit_ffn_dim=action_dit_ffn_dim,
        action_dit_num_heads=action_dit_num_heads,
        action_dit_num_layers=action_dit_num_layers,
        action_dit_bridge_layers=action_dit_bridge_layers,
        video_dim=video_dim,
        bridge_type=bridge_type,
    )
    pipe, action_dit = load_wam_models(cfg, device=device)
    if hasattr(pipe, "eval"):
        pipe.eval()
    return RoboTwinVAMPolicy(
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


def eval_one_step(model: RoboTwinVAMPolicy, obs_dict: dict) -> np.ndarray:
    return model.predict_action(obs_dict)


def reset_model(model: RoboTwinVAMPolicy):
    model.reset()
