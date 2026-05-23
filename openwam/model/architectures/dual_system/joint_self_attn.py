"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`MoTJointDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.joint_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.mot_compile import CompiledMoTLoop
from openwam.model.architectures.dual_system.mot_driver import MoTJointDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.compile_options import compile_mode, section_enabled, self_attn_compile_cfg
from openwam.utils import resolve_bridge_layers


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: MoT-style mixed attention at every layer.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mot_driver: MoTJointDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_loop: CompiledMoTLoop | None = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
            # Propagate the video backbone's attention kernel down to the
            # action backbone so the SDPA / linear-relu choice is made in one
            # place. SanaMoTJointDriver requires both sides to match; this
            # avoids the user having to set the kernel twice.
            cfg.setdefault("attn_kernel", getattr(self.video_backbone, "attn_kernel", "softmax"))
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        text_dim = self._resolve_text_dim(cfg)
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at MoTJointDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            attn_kernel=str(cfg.get("attn_kernel", "softmax")),
        )

        # MoT driver is built once both backbones are available. The video
        # backbone is normally constructed in ``BaseWAMArchitecture._init_video_backbone``
        # (already done by ``super().__init__``), so we can build the driver
        # right here. Tests that swap in a mock video backbone after init call
        # :meth:`build_mot_driver` directly.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(cfg.get("attention_mask_mode", "joint")),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "first_frame_causal")),
        }
        if self.video_backbone is not None:
            self.build_mot_driver()

    def build_mot_driver(self) -> MoTJointDriver:
        """Construct the :class:`MoTJointDriver` from the current backbones.

        Re-callable; raises if either backbone is missing. Tests that swap in
        a mock video backbone after ``__init__`` should call this method to
        wire up the driver afterwards.

        Dispatches on ``video_backbone.attn_kernel``:

        - ``"softmax"`` (default for Wan/Cosmos25): plain
          :class:`MoTJointDriver` with SDPA.
        - ``"linear_relu"`` (SANA): :class:`SanaMoTJointDriver` with cumsum
          linear-attention. The action backbone must also be configured for
          ``linear_relu`` — that's enforced inside the driver's constructor.
        - Anything else: :class:`ValueError`. Silent fallback would hide a
          config typo behind correct-looking but mathematically wrong output.
        """
        if self.video_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: video_backbone is not "
                "set. Construct the architecture with a video_backbone config or attach "
                "one before calling this method."
            )
        if self.action_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: action_backbone is not "
                "set. Architecture must be built from a non-None cfg."
            )

        kernel = getattr(self.video_backbone, "attn_kernel", "softmax")
        if kernel == "linear_relu":
            from openwam.model.architectures.dual_system.sana_mot_driver import (
                SanaMoTJointDriver,
            )
            self._mot_driver = SanaMoTJointDriver(
                self.video_backbone,
                self.action_backbone,
                **self._mot_driver_kwargs,
            )
        elif kernel == "softmax":
            self._mot_driver = MoTJointDriver(
                self.video_backbone,
                self.action_backbone,
                **self._mot_driver_kwargs,
            )
        else:
            raise ValueError(
                f"DualSystemSelfAttnArchitecture: unsupported video_backbone.attn_kernel='{kernel}'. "
                "Expected 'softmax' or 'linear_relu'."
            )
        return self._mot_driver

    @property
    def mot_driver(self) -> MoTJointDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply the self-attention compile mode through the MoT-loop helper."""
        mode = compile_mode(compile_cfg, default="none", strict=True)
        if mode != "auto":
            super().apply_compile_optimizations(compile_cfg)
            self._compiled_mot_loop = None
            return

        self_attn_cfg = self_attn_compile_cfg(compile_cfg)
        if section_enabled(self_attn_cfg, default=False):
            driver = self._mot_driver
            if driver is None:
                driver = self.build_mot_driver()
            self._compiled_mot_loop = CompiledMoTLoop(driver, self_attn_cfg)
        else:
            self._compiled_mot_loop = None

    def _iter_zero3_external_params(self):
        """Raw-access leaves read by the MoT driver outside the owners' ``__call__``.

        - ``vb._dit.blocks[i].modulation`` (Wan / Cosmos25) is read inside
          ``pre_attn_at_layer_for_compile`` (``wan_adapter.py:536``)
        - ``vb._dit.blocks[i].scale_shift_table`` (SANA) is read inside
          ``SanaMSVideoSplit.block_pre_attn`` (``blocks_split.py:271``) — the
          AdaLN parameter on SANA blocks (analog of Wan ``modulation``); under
          ZeRO-3 this would otherwise be partitioned at read time.
        - ``ab.blocks[i].modulation`` is read inside
          ``ActionDiT.pre_attn_at_layer_for_compile`` (``joint_action_dit.py:782``)
        """
        vb = self.video_backbone
        dit = getattr(vb, "_dit", None) if vb is not None else None
        if dit is not None:
            for block in getattr(dit, "blocks", ()):
                for attr in ("modulation", "scale_shift_table"):
                    p = getattr(block, attr, None)
                    if p is not None:
                        yield p
        ab = self.action_backbone
        if ab is not None:
            for block in getattr(ab, "blocks", ()):
                p = getattr(block, "modulation", None)
                if p is not None:
                    yield p

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio_state)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
        # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
        # path; TI2V itself fires that path first so these kwargs are inert for
        # it). VACE and I2V do NOT emit ``first_frame_latents`` (VACE routes
        # its first-frame condition through ``vace_context``; I2V uses the
        # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
        # structurally inert — kept on only so the joint MoT driver gets the
        # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
        # pass False.
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

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        compiled_loop = self._compiled_mot_loop
        if compiled_loop is not None and compiled_loop.can_run(
            vstate,
            astate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        ):
            vstate, astate = compiled_loop.run(vstate, astate)
        else:
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vb.finalize(vstate), ab.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
