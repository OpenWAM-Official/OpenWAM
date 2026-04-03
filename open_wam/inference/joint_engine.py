"""Joint video-action inference engine using package-native generation code."""

from typing import Optional

import torch

from open_wam.inference.base import BaseInferenceEngine
from open_wam.inference.joint_generation import generate_video_and_actions
from open_wam.inference.schedule import make_schedule
from open_wam.models.architectures.base import BaseWAMArchitecture
from open_wam.models.action_repr.base import BaseActionRepresentation


class JointInferenceEngine(BaseInferenceEngine):
    """Joint video-action inference engine.

    Wraps the package-native joint generation loop through the
    :class:`BaseInferenceEngine` interface.

    Args:
        cfg: Hydra config (must contain ``cfg.inference``).
        pipeline: Loaded ``WanVideoPipeline`` instance.
        action_dit: Loaded ``ActionDiT`` instance (legacy, prefer ``architecture``).
        architecture: WAM architecture wrapping the action model. If not
            provided and ``action_dit`` is given, a ``DualSystemArchitecture``
            is constructed automatically for backward compatibility.
    """

    def __init__(
        self, cfg, pipeline, action_dit=None,
        architecture: Optional[BaseWAMArchitecture] = None,
        action_repr: Optional[BaseActionRepresentation] = None,
    ):
        super().__init__(cfg, pipeline, action_dit, architecture)
        self.action_repr = action_repr

        if self.architecture is None and self.action_dit is not None:
            # Backward compat: wrap raw ActionDiT in DualSystemArchitecture
            from open_wam.models.architectures.dual_system import DualSystemArchitecture
            arch = DualSystemArchitecture(cfg=None)
            arch.action_dit = self.action_dit
            self.architecture = arch

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys:
                - prompt (str): text prompt
                - negative_prompt (str, optional): negative prompt for CFG
                - vace_video (list[PIL.Image], optional): VACE conditioning video
                - vace_reference_image (list[PIL.Image], optional): reference image
                - num_frames (int, optional): defaults from cfg
                - height (int, optional): defaults from cfg
                - width (int, optional): defaults from cfg
                - seed (int, optional): random seed, default 42
                - cfg_scale (float, optional): CFG scale, default from cfg
                - tiled (bool, optional): tiled VAE decoding, default True
                - input_video_latents (Tensor, optional): for action_only mode
                - schedule_type (str, optional): override schedule type
                - num_steps (int, optional): override num denoising steps

        Returns:
            dict with ``video`` (list of PIL images) and ``actions`` (numpy array).
        """
        inf_cfg = self.cfg.inference

        # Build schedule
        schedule_type = conditions.get("schedule_type", inf_cfg.schedule_type)
        num_steps = conditions.get("num_steps", inf_cfg.num_steps)
        shift = conditions.get("shift", getattr(inf_cfg, "shift", 5.0))

        schedule_kwargs = {}
        if schedule_type == "video_leading":
            schedule_kwargs["lead_steps"] = getattr(inf_cfg, "lead_steps", 10)
        elif schedule_type == "cascade":
            schedule_kwargs["video_steps"] = getattr(inf_cfg, "video_steps", num_steps)
            schedule_kwargs["action_steps"] = getattr(inf_cfg, "action_steps", num_steps)

        schedule = make_schedule(schedule_type, num_steps=num_steps, shift=shift, **schedule_kwargs)

        # Extract generation params
        video_frames, actions = generate_video_and_actions(
            pipe=self.pipeline,
            architecture=self.architecture,
            schedule=schedule,
            prompt=conditions.get("prompt", ""),
            negative_prompt=conditions.get("negative_prompt", ""),
            vace_video=conditions.get("vace_video", None),
            vace_reference_image=conditions.get("vace_reference_image", None),
            num_frames=conditions.get("num_frames", getattr(inf_cfg, "num_frames", 49)),
            height=conditions.get("height", getattr(inf_cfg, "height", 480)),
            width=conditions.get("width", getattr(inf_cfg, "width", 832)),
            seed=conditions.get("seed", 42),
            cfg_scale=conditions.get("cfg_scale", getattr(inf_cfg, "cfg_scale", 1.0)),
            tiled=conditions.get("tiled", True),
            input_video_latents=conditions.get("input_video_latents", None),
            num_inference_steps=num_steps,
            shift=shift,
            action_repr=self.action_repr,
        )

        return {"video": video_frames, "actions": actions}
