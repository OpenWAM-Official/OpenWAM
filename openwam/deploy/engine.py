"""Deploy-side inference engine: the deployment adapter around ``architecture.generate``.

``BaseInferenceEngine`` declares the engine interface; ``JointInferenceEngine``
is the production implementation. The engine translates deploy config +
per-request conditions into ``architecture.generate(...)`` arguments, builds
the denoise schedule, and owns server-lifetime caches (prompt embeddings,
VACE context). The actual denoising loop lives on the model side
(``BaseWAMArchitecture.generate``).
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional

import torch

from openwam.deploy.denoise_schedule import make_schedule, normalize_denoise_config
from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE = 32


class BaseInferenceEngine(ABC):
    """Inference engine base class.

    Concrete engines implement ``generate`` which accepts observation
    conditions and produces video frames and/or action trajectories.

    Args:
        cfg: Hydra config.
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference for implementations that still expose one.
    """

    require_architecture = False

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        if self.require_architecture and architecture is None:
            raise ValueError("architecture is required")
        self.cfg = cfg
        self.architecture = architecture
        self.action_backbone = action_backbone

    @abstractmethod
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys like ``prompt``, ``reference_image``,
                ``context_video``, ``seed``, etc.

        Returns:
            dict with ``video`` (Tensor) and ``actions`` (Tensor).
        """
        ...


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
        normalize_denoise_config(getattr(cfg, "inference", None))

        self._architecture_generate_accepts_extra_kwargs: Optional[bool] = None
        self._architecture_generate_kwarg_names: Optional[set[str]] = None
        self._architecture_generate_warned_dropped_kwargs: set[tuple[str, tuple[str, ...]]] = set()
        self._init_optimizations()
        self._init_cfg()

    def _filter_architecture_generate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop deploy-only kwargs the architecture's generate() cannot consume."""

        if getattr(self, "_architecture_generate_kwarg_names", None) is None:
            params = inspect.signature(self.architecture.generate).parameters
            self._architecture_generate_accepts_extra_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
            self._architecture_generate_kwarg_names = {
                name
                for name, param in params.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
        if self._architecture_generate_accepts_extra_kwargs:
            return kwargs
        accepted = self._architecture_generate_kwarg_names
        dropped = {name: value for name, value in kwargs.items() if name not in accepted}
        meaningful_dropped = tuple(
            sorted(name for name, value in dropped.items() if not self._is_noop_dropped_generate_kwarg(name, value))
        )
        if meaningful_dropped:
            warned = getattr(self, "_architecture_generate_warned_dropped_kwargs", set())
            architecture_name = type(self.architecture).__name__
            warning_key = (architecture_name, meaningful_dropped)
            if warning_key not in warned:
                warned.add(warning_key)
                self._architecture_generate_warned_dropped_kwargs = warned
                logger.warning(
                    "%s.generate does not accept deploy kwarg(s) %s; dropping them for this request. "
                    "Check deploy config for unsupported features such as CFG on specialized architectures.",
                    architecture_name,
                    ", ".join(meaningful_dropped),
                )
        return {name: value for name, value in kwargs.items() if name in accepted}

    @staticmethod
    def _is_noop_dropped_generate_kwarg(name: str, value: Any) -> bool:
        """Return whether dropping an unsupported deploy kwarg preserves default behavior."""

        if name == "cfg_scale":
            try:
                return float(value) == 1.0
            except (TypeError, ValueError):
                return False
        if name == "cfg_merge":
            return value is False
        if name == "decode_video":
            return value is True
        if name == "profile":
            return value is False
        if name == "vace_cache":
            return not bool(value)
        if name == "prompt_embed_cache":
            # None means the cache was disabled by config — dropping it is a no-op.
            return value is None or (
                isinstance(value, _BoundedPromptEmbedCache)
                and len(value) == 0
                and value._maxsize == DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE
            )
        return value is None

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

        # Prompt-keyed text embedding cache (bounded LRU). enabled: false turns
        # it off entirely: the backbones treat a None cache as "never cache".
        cache_enabled = True
        cache_maxsize = DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE
        if optimization is not None:
            cache_cfg = getattr(optimization, "prompt_embed_cache", None)
            if cache_cfg is not None:
                cache_enabled = bool(getattr(cache_cfg, "enabled", True))
                cache_maxsize = int(getattr(cache_cfg, "maxsize", cache_maxsize))
        if cache_enabled:
            self._prompt_embed_cache = _BoundedPromptEmbedCache(maxsize=cache_maxsize)
        else:
            self._prompt_embed_cache = None
            logger.info("prompt_embed_cache disabled by config")

    def _init_cfg(self):
        """Resolve Classifier-Free Guidance from cfg.inference (CosmosPredict25 only; cfg_scale=1.0 is a no-op).

        The uncond embedding is built by the backbone's live encoder
        (``_build_uncond_context`` → ``text_encoder("")``).
        """
        inf_cfg = getattr(self.cfg, "inference", None)
        self._cfg_scale: float = float(getattr(inf_cfg, "cfg_scale", 1.0)) if inf_cfg else 1.0
        self._cfg_merge: bool = bool(getattr(inf_cfg, "cfg_merge", False)) if inf_cfg else False
        if self._cfg_scale < 1.0:
            raise ValueError(f"inference.cfg_scale must be >= 1.0; got {self._cfg_scale}.")
        if self._cfg_scale > 1.0:
            logger.info("CFG enabled: cfg_scale=%.3f cfg_merge=%s", self._cfg_scale, self._cfg_merge)

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
                - num_frames (int, optional): raw state/action window length;
                  generated action chunk length is ``num_frames - 1``
                - video_num_frames (int, optional): Wan video length after any
                  training-time video_stride sub-sampling; defaults from cfg,
                  then falls back to ``num_frames``
                - height (int, optional): defaults from cfg
                - width (int, optional): defaults from cfg
                - seed (int, optional): random seed, default 42
                - tiled (bool, optional): tiled VAE decoding, default True
                - input_video_latents (Tensor, optional): precomputed video latents
                - denoise_mode (str, optional): "sync" or "async"; "sync" ignores
                  the three async controls below, including any inherited from cfg,
                  so a request can downgrade to the sync trajectory on its own
                - lead_modality (str, optional): "async" only; "action" or "video"
                - variance_shift_alpha (float, optional): "async" only; lead curve shift
                - linear_offset (float, optional): "async" only; lag start delay
                - denoise_steps (int, optional): override num denoising steps

        Returns:
            dict with ``video`` (list of PIL images or None) and ``actions`` (numpy array).
        """
        inf_cfg = self.cfg.inference

        denoise_mode = conditions.get("denoise_mode", getattr(inf_cfg, "denoise_mode", "sync"))
        denoise_steps = conditions.get("denoise_steps", inf_cfg.denoise_steps)
        lead_modality = conditions.get("lead_modality", getattr(inf_cfg, "lead_modality", "video"))
        variance_shift_alpha = conditions.get("variance_shift_alpha", getattr(inf_cfg, "variance_shift_alpha", 1.0))
        linear_offset = conditions.get("linear_offset", getattr(inf_cfg, "linear_offset", 0.0))
        # Single source of truth for each stream's α-shift is the backbone
        # property — ``action_backbone.shift_action`` and
        # ``video_backbone.shift_video`` — set via the model yaml and saved in
        # the checkpoint, so the training sigma buffer (set by
        # ``init_training_schedulers`` via the same properties) and the
        # inference trajectory match. Deploy carries no independent shift knob.
        # ``conditions[...]`` allows a per-request override (smoke tests /
        # ablations). ``getattr`` guards lightweight stub architectures.
        _ab = getattr(self.architecture, "action_backbone", None)
        shift = conditions.get("shift", getattr(_ab, "shift_action", None) if _ab is not None else None)
        if shift is None:
            shift = 5.0
        _vb = getattr(self.architecture, "video_backbone", None)
        shift_video = conditions.get(
            "shift_video",
            getattr(_vb, "shift_video", None) if _vb is not None else None,
        )

        schedule = make_schedule(
            denoise_mode,
            video_scheduler=self.architecture.video_scheduler,
            action_scheduler=self.architecture.action_scheduler,
            num_steps=denoise_steps,
            shift=shift,
            shift_video=shift_video,
            lead=lead_modality,
            alpha=variance_shift_alpha,
            offset=linear_offset,
        )

        # Reset dit cache for each generation
        if self._dit_cache is not None:
            self._dit_cache.reset()

        # Deploy proprio: array-like in, normalized model-space tensor out.
        proprio = conditions.get("proprio")
        if proprio is None:
            observation = conditions.get("observation") or {}
            proprio = observation.get("state") if isinstance(observation, dict) else None
        if proprio is not None:
            proprio = self.architecture.normalize_deploy_proprio(proprio)

        action_num_frames = int(conditions.get("num_frames", getattr(inf_cfg, "num_frames", 49)))
        video_num_frames = int(
            conditions.get(
                "video_num_frames",
                getattr(inf_cfg, "video_num_frames", action_num_frames),
            )
        )

        generate_kwargs = self._filter_architecture_generate_kwargs(
            {
                "schedule": schedule,
                "prompt": conditions.get("prompt", ""),
                "vace_video": conditions.get("vace_video", None),
                "first_frame_image": conditions.get("first_frame_image", None),
                "num_frames": video_num_frames,
                "action_num_frames": action_num_frames,
                "height": conditions.get("height", getattr(inf_cfg, "height", 384)),
                "width": conditions.get("width", getattr(inf_cfg, "width", 320)),
                "seed": conditions.get("seed", 42),
                "tiled": conditions.get("tiled", True),
                "input_video_latents": conditions.get("input_video_latents", None),
                "num_inference_steps": denoise_steps,
                "shift": shift,
                "dit_cache": self._dit_cache,
                "decode_video": self._decode_video,
                "profile": self._profile,
                "vace_cache": self._vace_cache,
                "prompt_embed_cache": self._prompt_embed_cache,
                "proprio": proprio,
                "cfg_scale": self._cfg_scale,
                "cfg_merge": self._cfg_merge,
            }
        )
        result = self.architecture.generate(**generate_kwargs)

        # Attach optimization stats if profiling
        if self._profile and self._dit_cache is not None:
            result["dit_cache_stats"] = self._dit_cache.stats

        return result
