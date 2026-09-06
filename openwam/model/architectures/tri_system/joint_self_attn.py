"""Tri-system joint self-attention architecture.

OpenWAM-native trimodal MoT: Wan video DiT + shared ActionDiT + Understanding
Expert + frozen Qwen3-VL. Loss = video + action only; understanding is trained
through those supervised streams.
"""

# Source: https://github.com/thu-ml/Motus.
# Upstream revision: UNKNOWN (the original internal import did not record a commit SHA).
# License: Apache-2.0; see the repository-level LICENSE (Apache-2.0).
# Modified by OpenWAM contributors: the tri-expert architecture was restructured
# for OpenWAM registries, modular backbones, configuration, and lifecycle rules.

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.tri_system.mot_driver import TriSystemMoTDriver
from openwam.model.architectures.tri_system.und_expert import (
    UnderstandingExpert,
    UnderstandingExpertConfig,
)
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    torch_compile_kwargs,
    tri_system_compile_cfg,
)
from openwam.model.vlm_backbone import build_vlm_backbone

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_has(cfg, key) -> bool:
    if cfg is None:
        return False
    try:
        return key in cfg
    except TypeError:
        return hasattr(cfg, key)


@register_architecture(
    "tri_system_joint_self_attn",
    status="supported",
    note="Tri-system OpenWAM-style: video + shared ActionDiT + understanding via "
    "trimodal joint attention with frozen Qwen3-VL.",
    framework="tri_system",
    variant="joint_self_attn",
)
class TriSystemJointSelfAttnArchitecture(BaseWAMArchitecture):
    """Three-stream architecture: video + shared action DiT + understanding via MoT.

    ``understanding_expert`` is an architecture-level **trainable** expert, NOT
    a backbone. It does not appear in ``backbones`` because it has no
    independent scheduler or dtype/device lifecycle — the architecture manages
    it directly via ``set_dtype_device`` override. This is intentional:
    backbones own their own initialization and checkpoint loading, while the
    understanding expert's weights are part of the architecture's flat
    state_dict.

    **Trainability contract**: ``understanding_expert`` MUST remain trainable.
    It receives gradients indirectly through the trimodal joint attention
    (video and action loss back-propagate through the shared Q/K/V
    projection). ``freeze_modules`` will raise if asked to freeze it.
    """

    # Modules that must stay trainable — freeze_modules rejects these.
    _NEVER_FREEZE = frozenset({"understanding_expert"})

    def __init__(self, cfg=None):
        vlm_cfg = _cfg_get(cfg, "vlm_backbone", {}) or {}
        if _cfg_has(vlm_cfg, "freeze"):
            raise ValueError(
                "model.architecture.vlm_backbone.freeze has moved to the model `freeze:` list. "
                "Remove it from configs/model/tri_system.yaml's vlm_backbone block and add "
                "'vlm_backbone.vlm_model' to the top-level `freeze:` list instead."
            )
        super().__init__(cfg)
        self.vlm_backbone = None
        self.understanding_expert = None
        self._mot_driver = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object, object]]] = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
            # Default to "every video layer participates" if neither bridge_layers nor
            # bridge_interval is supplied. resolve_bridge_layers ignores bridge_interval
            # when bridge_layers is non-null, so this is safe even if the user later
            # passes an explicit bridge_layers list.
            cfg.setdefault("bridge_interval", 1)

        self.vlm_backbone = build_vlm_backbone(
            _cfg_get(vlm_cfg, "name", "qwen3_vl_2b"),
            checkpoint_path=_cfg_get(vlm_cfg, "checkpoint_path"),
            dtype=self.dtype,
            load_pretrained=bool(_cfg_get(vlm_cfg, "load_pretrained", True)),
            max_length=int(_cfg_get(vlm_cfg, "max_length", 512)),
        )

        und_cfg_dict = _cfg_get(cfg, "understanding_expert", {}) or {}
        und_dim = int(_cfg_get(und_cfg_dict, "dim", 512))
        if _cfg_has(und_cfg_dict, "ffn_dim_multiplier"):
            raise ValueError(
                "model.architecture.understanding_expert.ffn_dim_multiplier has been removed. "
                "Use model.architecture.understanding_expert.ffn_dim instead."
            )
        und_ffn_dim = int(_cfg_get(und_cfg_dict, "ffn_dim", 2048))
        und_cfg = UnderstandingExpertConfig(
            dim=und_dim,
            ffn_dim=und_ffn_dim,
            num_layers=self.video_backbone.num_layers,
            vlm_input_dim=self.vlm_backbone.hidden_size,
            vlm_projector_type=str(_cfg_get(und_cfg_dict, "vlm_projector_type", "mlp3x_silu")),
            eps=float(_cfg_get(und_cfg_dict, "eps", 1e-5)),
        )
        self.understanding_expert = UnderstandingExpert(
            und_cfg,
            wan_dim=self.video_backbone.dim,
            wan_num_heads=self.video_backbone.num_heads,
        )

        action_dim_hidden = int(_cfg_get(cfg, "dim", 1024))
        num_heads = int(_cfg_get(cfg, "num_heads", self.video_backbone.num_heads))
        attn_head_dim = int(_cfg_get(cfg, "attn_head_dim", self.video_backbone.head_dim))
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        text_dim = int(_cfg_get(cfg, "text_dim", getattr(self.video_backbone, "text_dim", None) or 4096))

        # Bridge layers — which video DiT layers participate in joint attention.
        # For ``joint_self_attn`` variant the MoT driver requires
        # ``len(bl) == vb.num_layers`` (1:1 mapping); ``TriSystemMoTDriver.__init__``
        # validates this. If the cfg supplies a sparse pattern, the driver raises
        # with a clear error. We accept dual_system's defaults (``bridge_layers`` or
        # ``bridge_interval``) so existing yaml conventions transfer.
        bl = resolve_bridge_layers(cfg, num_layers=self.video_backbone.num_layers)

        self._init_proprio_context(cfg, text_dim=text_dim)
        self.action_backbone = ActionDiT(
            action_dim=int(_cfg_get(cfg, "action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(_cfg_get(cfg, "ffn_dim", action_dim_hidden * int(_cfg_get(cfg, "ffn_dim_multiplier", 4)))),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=self.video_backbone.dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=_cfg_get(cfg, "shift_action"),
        )
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(_cfg_get(cfg, "mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(_cfg_get(cfg, "attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(_cfg_get(cfg, "video_attention_mask_mode", "first_frame_causal")),
        }
        self.build_mot_driver()

    def build_mot_driver(self) -> TriSystemMoTDriver:
        """Construct the trimodal MoT driver from the current components."""

        if self.video_backbone is None:
            raise RuntimeError("TriSystemJointSelfAttnArchitecture.build_mot_driver: video_backbone is not set.")
        if self.action_backbone is None:
            raise RuntimeError("TriSystemJointSelfAttnArchitecture.build_mot_driver: action_backbone is not set.")
        if self.understanding_expert is None:
            raise RuntimeError("TriSystemJointSelfAttnArchitecture.build_mot_driver: understanding_expert is not set.")
        self._mot_driver = TriSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            self.understanding_expert,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    @property
    def backbones(self) -> dict[str, nn.Module]:
        result = super().backbones
        if self.vlm_backbone is not None:
            result["vlm_backbone"] = self.vlm_backbone
        return result

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time trimodal MoT loop when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_mot_run_joint_loop = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = tri_system_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("tri-system MoT compile disabled by config; running eager.")
            return
        if self.video_backbone is None or self.action_backbone is None or self.understanding_expert is None:
            logger.warning("tri-system MoT compile requested before backbones are ready; running eager.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_joint_loop(vstate, astate, ustate, attn_mask):
            return driver.run_joint_loop_for_compile(
                vstate,
                astate,
                ustate,
                attn_mask=attn_mask,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        try:
            self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_mot_run_joint_loop = None
            logger.warning("tri-system MoT torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled tri-system MoT compile with torch.compile kwargs=%s", kwargs)

    def freeze_modules(self, names: list[str]) -> list[str]:
        rejected = self._NEVER_FREEZE & set(names)
        if rejected:
            raise ValueError(
                f"tri_system: refusing to freeze {rejected}. "
                f"These modules must stay trainable — they receive gradients "
                f"through trimodal joint attention. To freeze the VLM backbone, "
                f"use 'vlm_backbone.vlm_model' instead."
            )
        return super().freeze_modules(names)

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if self.understanding_expert is not None:
            self.understanding_expert.to(dtype=dtype, device=device)

    def _extract_first_image(self, sample: dict):
        first_frame_image = sample.get("first_frame_image")
        if isinstance(first_frame_image, list) and first_frame_image:
            return first_frame_image[0]
        if first_frame_image is not None:
            return first_frame_image
        video = sample.get("video")
        if video is not None and len(video) > 0:
            return video[0]
        return None

    def _collate_vlm_inputs(self, items: list[dict]) -> dict[str, torch.Tensor]:
        return self.vlm_backbone.batch_vlm_inputs(items)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        if isinstance(batch, dict):
            batch = [batch]
        if not batch:
            raise ValueError("tri_system.prepare_inputs: empty batch")

        # FirstFrameConditioningTransform is applied by super().prepare_inputs() in place
        # via its own pipeline transform instance; we read sample['first_frame_image']
        # afterward.
        inputs = super().prepare_inputs(batch)
        samples = batch  # super() mutates the dicts in place — same references.

        provided = [sample.get("vlm_inputs") for sample in samples]
        if all(item is not None for item in provided):
            inputs["vlm_inputs"] = self._collate_vlm_inputs(provided)
            return inputs
        if any(item is not None for item in provided):
            raise ValueError("Mixed vlm_inputs in tri-system batch: provide vlm_inputs for all samples or none.")

        prompts = [sample["prompt"] for sample in samples]
        images = [self._extract_first_image(sample) for sample in samples]
        if any(image is None for image in images):
            raise ValueError("tri_system requires sample['vlm_inputs'] or a first frame image/video[0].")
        inputs["vlm_inputs"] = self.vlm_backbone.prepare_vlm_inputs(prompts, images)
        return inputs

    def _prepare_generation_vlm_inputs(self, prompt: str, first_frame_image):
        if first_frame_image is None:
            raise ValueError("tri_system generation requires first_frame_image to build Qwen3-VL inputs.")
        image = first_frame_image[0] if isinstance(first_frame_image, list) else first_frame_image
        return self.vlm_backbone.prepare_vlm_inputs([prompt], [image])

    @torch.no_grad()
    def generate(self, schedule, prompt: str, *, first_frame_image=None, **kwargs) -> dict:
        """Run trimodal denoising with one-shot VLM forward shared across all steps.

        Within one call: ``vlm_hidden`` is computed once from ``prompt`` +
        ``first_frame_image`` and reused across every denoising step via
        ``inputs_shared``. Across calls there is no implicit cache — each
        invocation pops ``vlm_hidden`` from ``kwargs`` (starts ``None``) and
        re-runs the VLM if absent. So changing ``prompt`` / ``first_frame_image``
        between calls correctly recomputes the hidden state.

        **Caller-managed cache contract**: if you bypass the recompute by passing
        a pre-computed ``vlm_hidden=`` (and optionally ``vlm_attention_mask=``)
        through ``kwargs``, YOU OWN invalidation when the underlying prompt /
        image changes. This method does NOT cross-check the explicit
        ``vlm_hidden`` against the supplied ``prompt`` / ``first_frame_image``.
        Reusing a stale ``vlm_hidden`` across different inputs silently produces
        wrong outputs — the trimodal joint attention will use the old VLM
        context regardless of what ``prompt`` actually was.
        """
        vlm_inputs = kwargs.pop("vlm_inputs", None)
        vlm_hidden = kwargs.pop("vlm_hidden", None)
        vlm_attention_mask = kwargs.pop("vlm_attention_mask", None)
        if vlm_inputs is None and self.action_backbone is not None:
            vlm_inputs = self._prepare_generation_vlm_inputs(prompt, first_frame_image)
        if vlm_attention_mask is None and isinstance(vlm_inputs, dict):
            vlm_attention_mask = vlm_inputs.get("attention_mask")
        if vlm_hidden is None and vlm_inputs is not None:
            vlm_hidden = self.vlm_backbone.extract_features(vlm_inputs)
        return super().generate(
            schedule,
            prompt,
            first_frame_image=first_frame_image,
            vlm_hidden=vlm_hidden,
            vlm_attention_mask=vlm_attention_mask,
            **kwargs,
        )

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        ub = self.understanding_expert
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        vlm_inputs = pipeline_inputs.pop("vlm_inputs", None)
        vlm_hidden = pipeline_inputs.pop("vlm_hidden", None)
        vlm_attention_mask = pipeline_inputs.pop("vlm_attention_mask", None)
        if vlm_attention_mask is None and isinstance(vlm_inputs, dict):
            vlm_attention_mask = vlm_inputs.get("attention_mask")
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        # Same 4D + clean-prefix-aligned t_mod opt-in as the dual_system /
        # single_system forwards. TI2V fires its own branch first so these
        # kwargs are inert there. VACE and I2V both work here: VACE routes its
        # condition through ``vace_context`` → per-video-block ``vace_hints``
        # (applied in ``post_attn_at_layer`` → ``apply_post_block_residuals``,
        # the same path dual_system uses; the video vstate is video-only so the
        # hint spans the full video slice); I2V rides the ``y`` channel + CLIP
        # context built in ``vb.prepare()``. Neither changes the video token
        # count, so the trimodal mask is unaffected. ``first_frame_latents`` is
        # absent for both, so the clean-prefix zeroing is a no-op.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        if vlm_hidden is None and vlm_inputs is None:
            raise ValueError("tri_system forward with actions requires `vlm_inputs` or cached `vlm_hidden`.")

        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if vlm_hidden is None:
            vlm_hidden = self.vlm_backbone.extract_features(vlm_inputs)
        vlm_hidden = vlm_hidden.to(device=self.device)
        if vlm_hidden.shape[0] != vstate.hidden_states.shape[0]:
            raise ValueError(
                f"vlm_hidden batch size ({vlm_hidden.shape[0]}) does not match "
                f"video state batch size ({vstate.hidden_states.shape[0]}). If using a cached "
                f"vlm_hidden, ensure it was computed for the same batch."
            )
        ustate = ub.prepare_state(vlm_hidden, dtype=vstate.hidden_states.dtype, vlm_attention_mask=vlm_attention_mask)
        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()
        compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
        if compiled_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            try:
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                video_tokens_per_frame = driver._video_tokens_per_frame(vstate)
                s_video = int(vstate.grid_frames) * video_tokens_per_frame
                s_action = driver._get_action_tokens(astate).shape[1]
                s_understanding = ustate.und_tokens.shape[1]
                attn_mask = driver._build_attention_mask(
                    s_video=s_video,
                    s_action=s_action,
                    s_understanding=s_understanding,
                    video_tokens_per_frame=video_tokens_per_frame,
                    device=vstate.hidden_states.device,
                    und_mask=getattr(ustate, "und_mask", None),
                )
                vstate, astate, ustate = compiled_loop(vstate, astate, ustate, attn_mask)
            except Exception as exc:
                self._compiled_mot_run_joint_loop = None
                logger.warning("tri-system compiled MoT loop failed; falling back to eager: %s", exc)
                vstate, astate, ustate = driver.run_joint_loop(
                    vstate,
                    astate,
                    ustate,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        else:
            vstate, astate, ustate = driver.run_joint_loop(
                vstate,
                astate,
                ustate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        return vb.finalize(vstate), ab.extract_prediction(astate)


__all__ = ["TriSystemJointSelfAttnArchitecture"]
