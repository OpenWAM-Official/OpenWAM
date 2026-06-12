"""Deploy-side inference engine: the deployment adapter around ``architecture.generate``.

``BaseInferenceEngine`` declares the engine interface; ``JointInferenceEngine``
is the production implementation. The engine translates deploy config +
per-request conditions into ``architecture.generate(...)`` arguments, builds
the denoise schedule, and owns server-lifetime caches (prompt embeddings,
VACE context, CFG uncond embedding). The actual denoising loop lives on the
model side (``BaseWAMArchitecture.generate``).
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from openwam.dataloader.transforms.text_embedding_cache import resolve_cache_path_for_sha, sha256_for_prompt
from openwam.deploy.denoise_schedule import make_schedule
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

        self._architecture_generate_accepts_extra_kwargs: Optional[bool] = None
        self._architecture_generate_kwarg_names: Optional[set[str]] = None
        self._architecture_generate_warned_dropped_kwargs: set[tuple[str, tuple[str, ...]]] = set()
        self._init_optimizations()

    def _filter_architecture_generate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop deploy-only kwargs that a legacy/specialized architecture cannot consume."""

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
            return (
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

        # Prompt-keyed text embedding cache (bounded LRU).
        cache_maxsize = DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE
        if optimization is not None:
            cache_cfg = getattr(optimization, "prompt_embed_cache", None)
            if cache_cfg is not None:
                cache_maxsize = int(getattr(cache_cfg, "maxsize", cache_maxsize))
        self._prompt_embed_cache = _BoundedPromptEmbedCache(maxsize=cache_maxsize)

        # §15 — Classifier-Free Guidance (Cosmos25 only today; Wan keeps its
        # own pipeline-internal CFG path). When `cfg_scale > 1.0` the engine
        # resolves an uncond embedding once at init time (offline
        # `empty.safetensors` if `text_embedding_cache_dir` is set; otherwise
        # falls back to the backbone's live encoder via the adapter).
        inf_cfg = getattr(self.cfg, "inference", None)
        self._cfg_scale: float = float(getattr(inf_cfg, "cfg_scale", 1.0)) if inf_cfg else 1.0
        self._cfg_merge: bool = bool(getattr(inf_cfg, "cfg_merge", False)) if inf_cfg else False
        cache_dir_raw = getattr(inf_cfg, "text_embedding_cache_dir", None) if inf_cfg else None
        self._text_embedding_cache_dir: Optional[Path] = Path(str(cache_dir_raw)) if cache_dir_raw is not None else None
        self._uncond_pre_encoded_text: Optional[torch.Tensor] = None
        if self._cfg_scale < 1.0:
            raise ValueError(f"inference.cfg_scale must be >= 1.0; got {self._cfg_scale}.")
        if self._cfg_scale > 1.0:
            self._uncond_pre_encoded_text = self._resolve_uncond_pre_encoded_text()
            logger.info(
                "CFG enabled: cfg_scale=%.3f cfg_merge=%s uncond_source=%s",
                self._cfg_scale,
                self._cfg_merge,
                "empty.safetensors" if self._uncond_pre_encoded_text is not None else "live encoder",
            )

    def _arch_dtype_device(self) -> tuple:
        """Return (dtype, device) of the wrapped architecture, with sensible defaults."""
        dtype = getattr(self.architecture, "dtype", torch.bfloat16)
        device = getattr(self.architecture, "device", torch.device("cuda:0"))
        if isinstance(device, str):
            device = torch.device(device)
        return dtype, device

    def _load_pre_encoded_text_safetensors(self, path: Path) -> torch.Tensor:
        """Load a single ``.safetensors`` cache file into a tensor on arch dtype/device.

        Accepts either ``"pre_encoded_text"`` or the first stored tensor as the
        payload (older precompute outputs used different key names; mirror that
        tolerance so a re-run isn't required when keys drift).
        """
        from safetensors.torch import load_file

        sf = load_file(str(path))
        tensor = sf.get("pre_encoded_text")
        if tensor is None and sf:
            tensor = next(iter(sf.values()))
        if tensor is None:
            raise ValueError(f"{path} contains no readable tensor (expected key 'pre_encoded_text').")
        # The precompute writer (`reason1_embedding_computation._project_to_postproj`)
        # `.squeeze(0)`s a `(1, L, D)` tensor to 2D `(L, D)` before saving. Every
        # downstream consumer (pipeline_wrapper preprocess, _build_uncond_context,
        # _append_proprio_context_token) assumes 3D `(B, L, D)`. The training path
        # `base.py:_compute_proprio_state_and_context` already normalises via a
        # `ndim==2 → stack` step (base.py:687-706); deploy has no equivalent until
        # here. Restore the batch axis at the cache/inference boundary so existing
        # 2D safetensors caches stay readable without a rewrite.
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        dtype, device = self._arch_dtype_device()
        return tensor.to(device=device, dtype=dtype)

    def _backbone_has_live_text_encoder(self) -> bool:
        vb = getattr(self.architecture, "video_backbone", None)
        pipe = getattr(vb, "_pipe", None)
        return getattr(pipe, "text_encoder", None) is not None

    def _resolve_uncond_pre_encoded_text(self) -> Optional[torch.Tensor]:
        """Find the unconditional embedding once at engine init.

        Mirrors the cond precedence: ``empty.safetensors`` in
        ``text_embedding_cache_dir`` wins; otherwise return ``None`` and rely
        on the backbone's live encoder via the adapter (``_build_uncond_context``
        in cosmos25 adapter falls through to ``text_encoder("")``).
        """
        if self._text_embedding_cache_dir is not None:
            empty_path = self._text_embedding_cache_dir / "empty.safetensors"
            if not empty_path.exists():
                raise FileNotFoundError(
                    f"inference.cfg_scale={self._cfg_scale} > 1.0 with "
                    f"text_embedding_cache_dir={self._text_embedding_cache_dir} but "
                    f"{empty_path} does not exist. Re-run precompute "
                    f"(docs/cosmos25_backbone.md §10.4 step ②) so the empty embedding "
                    f"lands alongside per-prompt caches, or unset "
                    f"inference.text_embedding_cache_dir to fall back to the live encoder."
                )
            return self._load_pre_encoded_text_safetensors(empty_path)
        if not self._backbone_has_live_text_encoder():
            raise RuntimeError(
                f"inference.cfg_scale={self._cfg_scale} > 1.0 but no uncond source available. "
                "Either set `inference.text_embedding_cache_dir` to a directory containing "
                "`empty.safetensors`, or build the checkpoint with "
                "`video_backbone.text_encoder=reason1_live` so the adapter can use the live "
                "encoder on the empty prompt."
            )
        return None

    def _load_pre_encoded_text_for_prompt(self, prompt: str) -> Optional[torch.Tensor]:
        """Resolve a bucketed or legacy flat cache file for ``prompt``.

        Returns ``None`` when the cache dir is unset OR the prompt has no
        per-prompt file (engine falls back to live encoder via adapter, if
        configured). When the cache dir IS set but the file is missing, we
        deliberately don't fall back silently — re-raise instead so the user
        notices the mismatch before model output drift goes undetected.
        """
        if self._text_embedding_cache_dir is None:
            return None
        sha = sha256_for_prompt(prompt)
        cache_path = Path(resolve_cache_path_for_sha(str(self._text_embedding_cache_dir), sha))
        if not cache_path.exists():
            if self._backbone_has_live_text_encoder():
                # cache_dir set but file missing AND live encoder available —
                # treat as a hard miss so the user knows to extend their cache,
                # rather than silently mixing cache+live sources per prompt
                # (which would defeat the precompute determinism guarantee).
                raise FileNotFoundError(
                    f"text_embedding_cache_dir is set but no cache exists for prompt "
                    f"sha256={sha[:12]}... ({cache_path}). Add this prompt to the "
                    f"precompute set or unset text_embedding_cache_dir to use the "
                    f"live encoder."
                )
            # No live fallback either; return None and let the backbone
            # raise a clear error downstream.
            return None
        return self._load_pre_encoded_text_safetensors(cache_path)

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
                - input_video_latents (Tensor, optional): for action_only mode
                - schedule_type (str, optional): override schedule type
                - denoise_steps (int, optional): override num denoising steps

        Returns:
            dict with ``video`` (list of PIL images or None) and ``actions`` (numpy array).
        """
        inf_cfg = self.cfg.inference

        # Build schedule (only "sync" is supported; make_schedule raises on anything else)
        schedule_type = conditions.get("schedule_type", inf_cfg.schedule_type)
        denoise_steps = conditions.get("denoise_steps", inf_cfg.denoise_steps)
        shift = conditions.get("shift", getattr(inf_cfg, "shift", 5.0))

        # Single source of truth: ``arch.video_backbone.shift_video`` is the
        # ONLY place the video α-shift is configured (set via
        # ``cfg.model.video_backbone.shift_video`` at yaml time). Reading
        # here — rather than from ``cfg.inference.*`` — guarantees that the
        # discrete training sigma buffer (set by
        # ``init_training_schedulers`` via the same property) and the
        # inference denoising trajectory are sampled from the identical
        # shifted schedule. ``conditions["shift_video"]`` lets callers
        # override per-request (deploy smoke tests / ablations) without
        # editing the cfg tree.
        # Nested ``getattr`` — some lightweight test doubles
        # (e.g. ``_CaptureDeployArchitecture`` in
        # ``tests/test_action_normalization.py``) construct a stub
        # architecture without a ``video_backbone`` attribute at all;
        # production code always has it.
        _vb = getattr(self.architecture, "video_backbone", None)
        shift_video = conditions.get(
            "shift_video",
            getattr(_vb, "shift_video", None) if _vb is not None else None,
        )

        schedule = make_schedule(
            schedule_type,
            video_scheduler=self.architecture.video_scheduler,
            action_scheduler=self.architecture.action_scheduler,
            num_steps=denoise_steps,
            shift=shift,
            shift_video=shift_video,
        )

        # Reset dit cache for each generation
        if self._dit_cache is not None:
            self._dit_cache.reset()

        # Extract generation params
        proprio_state = conditions.get("proprio_state", None)
        if proprio_state is None:
            observation = conditions.get("observation") or {}
            proprio_state = observation.get("state") if isinstance(observation, dict) else None
        if proprio_state is not None and not isinstance(proprio_state, torch.Tensor):
            if isinstance(proprio_state, np.ndarray):
                proprio_state = torch.from_numpy(proprio_state)
            else:
                proprio_state = torch.tensor(proprio_state, dtype=torch.float32)
        if proprio_state is not None:
            proprio_state = self.architecture.normalize_deploy_proprio(proprio_state)

        action_num_frames = int(conditions.get("num_frames", getattr(inf_cfg, "num_frames", 49)))
        video_num_frames = int(
            conditions.get(
                "video_num_frames",
                getattr(inf_cfg, "video_num_frames", action_num_frames),
            )
        )

        # §15 — Cosmos25 cache-mode pre_encoded_text resolution. Wan never
        # reads this kwarg (its prepare_inputs_for_inference signature has no
        # `pre_encoded_text`); the architecture-level `generate()` only forwards
        # the kwarg to the backbone when it is non-None, so Wan stays untouched.
        prompt = conditions.get("prompt", "")
        cached_pre_encoded_text = self._load_pre_encoded_text_for_prompt(prompt)

        generate_kwargs = self._filter_architecture_generate_kwargs(
            {
                "schedule": schedule,
                "prompt": prompt,
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
                "proprio_state": proprio_state,
                "cfg_scale": self._cfg_scale,
                "cfg_merge": self._cfg_merge,
                "pre_encoded_text": cached_pre_encoded_text,
                "uncond_pre_encoded_text": self._uncond_pre_encoded_text,
            }
        )
        result = self.architecture.generate(**generate_kwargs)

        # Attach optimization stats if profiling
        if self._profile and self._dit_cache is not None:
            result["dit_cache_stats"] = self._dit_cache.stats

        return result
