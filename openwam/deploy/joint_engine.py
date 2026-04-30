"""Joint video-action inference engine using package-native generation code."""

import logging
import os
from collections import OrderedDict
from typing import Any, Optional

import torch

from openwam.deploy.base import BaseInferenceEngine
from openwam.deploy.schedule import make_schedule
from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE = 32


class _BoundedPromptEmbedCache(OrderedDict):
    """LRU-bounded dict for ``prompt -> inputs_posi``."""

    def __init__(self, maxsize: int = DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE):
        super().__init__()
        self._maxsize = max(1, int(maxsize))
        self._evict_warned = False

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            evicted_key, _ = self.popitem(last=False)
            if not self._evict_warned:
                self._evict_warned = True
                logger.warning("prompt_embed_cache exceeded maxsize=%d; evicted %r", self._maxsize, evicted_key)


class JointInferenceEngine(BaseInferenceEngine):
    """Joint video-action inference engine.

    Wraps the package-native joint generation loop through the
    :class:`BaseInferenceEngine` interface.

    Args:
        cfg: Hydra config (must contain ``cfg.inference``).
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference retained for base-class storage.
    """

    require_architecture = True

    def __init__(
        self,
        cfg,
        architecture: Optional[BaseWAMArchitecture] = None,
        action_backbone=None,
    ):
        super().__init__(cfg, architecture=architecture, action_backbone=action_backbone)

        self._init_optimizations()

    def _init_optimizations(self):
        """Read cfg.optimization and instantiate optimization components."""
        optimization = getattr(self.cfg, "optimization", None)

        # DiT velocity cache
        self._dit_cache = None
        if optimization and getattr(optimization, "dit_cache", None):
            dc = optimization.dit_cache
            if getattr(dc, "enabled", False):
                from openwam.deploy.optimizations import DiTVelocityCache

                self._dit_cache = DiTVelocityCache(
                    cosine_threshold=getattr(dc, "cosine_threshold", 0.99),
                    max_consecutive_skips=getattr(dc, "max_skips", 3),
                )
                logger.info("DiT velocity cache enabled (threshold=%.3f)", dc.cosine_threshold)

        # torch.compile — ActionDiT, Video DiT, VAE
        if optimization and getattr(optimization, "compile", None):
            try:
                self.architecture.apply_compile_optimizations(optimization.compile)
            except Exception as e:
                logger.warning("torch.compile optimization failed, continuing without: %s", e)

        # VAE decode skip
        self._decode_video = True
        if optimization:
            self._decode_video = getattr(optimization, "decode_video", True)

        # Profiling
        self._profile = os.environ.get("WAM_PROFILE", "0") == "1"

        # VACE context cache for closed-loop reuse
        self._vace_cache: dict = {}

        # Prompt-keyed text embedding cache (bounded LRU).
        cache_maxsize = DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE
        if optimization is not None:
            cache_cfg = getattr(optimization, "prompt_embed_cache", None)
            if cache_cfg is not None:
                cache_maxsize = int(getattr(cache_cfg, "maxsize", cache_maxsize))
        self._prompt_embed_cache = _BoundedPromptEmbedCache(maxsize=cache_maxsize)

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys:
                - prompt (str): text prompt
                - vace_video (list[PIL.Image], optional): VACE conditioning video (Wan2.1-VACE only)
                - first_frame_image (list[PIL.Image], optional): first frame of the
                  observation window; used as TI2V first-frame condition on Wan2.2-TI2V
                  or as VACE spatial reference on Wan2.1-VACE
                - num_frames (int, optional): defaults from cfg
                - height (int, optional): defaults from cfg
                - width (int, optional): defaults from cfg
                - seed (int, optional): random seed, default 42
                - tiled (bool, optional): tiled VAE decoding, default True
                - input_video_latents (Tensor, optional): for action_only mode
                - schedule_type (str, optional): override schedule type
                - denoise_steps (int, optional): override num denoising steps

        Returns:
            dict with ``video`` (list of PIL images or None) and ``actions`` (numpy array).
        """
        inf_cfg = self.cfg.inference
        optimization = getattr(self.cfg, "optimization", None)

        # Build schedule
        schedule_type = conditions.get("schedule_type", inf_cfg.schedule_type)
        # Optimization config can override schedule type; null means "use inference.schedule_type"
        if optimization and getattr(optimization, "schedule", None):
            _optimization_type = getattr(optimization.schedule, "type", None)
            schedule_type = conditions.get("schedule_type", _optimization_type or schedule_type)
        denoise_steps = conditions.get("denoise_steps", inf_cfg.denoise_steps)
        shift = conditions.get("shift", getattr(inf_cfg, "shift", 5.0))

        schedule_kwargs = {}
        if schedule_type == "video_leading":
            schedule_kwargs["lead_steps"] = getattr(inf_cfg, "lead_steps", 10)
        elif schedule_type == "cascade":
            schedule_kwargs["video_steps"] = getattr(inf_cfg, "video_steps", denoise_steps)
            schedule_kwargs["action_steps"] = getattr(inf_cfg, "action_steps", denoise_steps)
        elif schedule_type == "decoupled_flash":
            action_steps = denoise_steps
            if optimization and getattr(optimization, "schedule", None):
                action_steps = getattr(optimization.schedule, "action_steps", denoise_steps)
            schedule_kwargs["action_steps"] = conditions.get("action_steps", action_steps)
        elif schedule_type == "decoupled_asymmetric":
            schedule_kwargs["video_steps"] = getattr(inf_cfg, "video_steps", denoise_steps)
            action_steps = denoise_steps
            if optimization and getattr(optimization, "schedule", None):
                action_steps = getattr(optimization.schedule, "action_steps", denoise_steps)
            schedule_kwargs["action_steps"] = conditions.get("action_steps", action_steps)

        schedule = make_schedule(
            schedule_type,
            video_scheduler=self.architecture.video_backbone.scheduler,
            action_scheduler=self.architecture.action_scheduler,
            num_steps=denoise_steps,
            shift=shift,
            **schedule_kwargs,
        )

        # Reset dit cache for each generation
        if self._dit_cache is not None:
            self._dit_cache.reset()

        # Extract generation params
        result = self.architecture.generate(
            schedule=schedule,
            prompt=conditions.get("prompt", ""),
            vace_video=conditions.get("vace_video", None),
            first_frame_image=conditions.get("first_frame_image", None),
            num_frames=conditions.get("num_frames", getattr(inf_cfg, "num_frames", 49)),
            height=conditions.get("height", getattr(inf_cfg, "height", 480)),
            width=conditions.get("width", getattr(inf_cfg, "width", 832)),
            seed=conditions.get("seed", 42),
            tiled=conditions.get("tiled", True),
            input_video_latents=conditions.get("input_video_latents", None),
            num_inference_steps=denoise_steps,
            shift=shift,
            dit_cache=self._dit_cache,
            decode_video=self._decode_video,
            profile=self._profile,
            vace_cache=self._vace_cache,
            prompt_embed_cache=self._prompt_embed_cache,
        )

        # Attach optimization stats if profiling
        if self._profile and self._dit_cache is not None:
            result["dit_cache_stats"] = self._dit_cache.stats

        return result
