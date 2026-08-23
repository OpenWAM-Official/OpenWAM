"""DualSystem IDM (Inverse Dynamics Model) architecture.

Two-stage denoising variant inspired by FastWAM-IDM:

**Training**: three parallel branches flow through the MoT in one forward:
  - Branch A: noisy video (denoising target)
  - Branch B: teacher-forcing cond video (optionally noised for robustness)
  - Branch C: noisy action (denoising target)

A teacher-forcing attention mask ensures:
  - noisy video ↔ noisy video (v2v sub-mask)
  - cond video ↔ cond video (v2v sub-mask)
  - action → cond video + action (full connection)
  - all other cross-branch paths are blocked

**Inference**: two sequential stages:
  Stage 1 — denoise video independently (standard video DiT loop)
  Stage 2 — freeze denoised video as condition, denoise action with KV cache
"""

from __future__ import annotations

import logging
import types
from typing import Callable, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor

from openwam.model.action_backbone.components import rope_apply_1d
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import widen_mask_for_prefix_kv
from openwam.model.compile_options import (
    compile_enabled,
    idm_compile_cfg,
    section_enabled,
    torch_compile_kwargs,
)

logger = logging.getLogger(__name__)


class IDMMoTDriver(DualSystemMoTDriver):
    """Extended MoT driver for IDM teacher-forcing training.

    At training time the video sequence is doubled: [noisy_video, cond_video].
    The driver must build a teacher-forcing mask instead of the standard joint
    mask, and split the merged video output back into noisy/cond halves.

    At inference time (action-only stage 2), only cond_video tokens are present
    and the driver operates in a standard joint mode.
    """

    def _build_teacher_forcing_mask(
        self,
        s_noisy_video: int,
        s_cond_video: int,
        s_action: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the IDM teacher-forcing attention mask.

        Layout (rows=queries, cols=keys; True=attend):

            [noisy_video, cond_video, action]

                          noisy_video  cond_video  action
            noisy_video   v2v_mask     False       False
            cond_video    False        v2v_mask    False
            action        False        True        True
        """
        noisy_end = s_noisy_video
        cond_end = noisy_end + s_cond_video
        total = cond_end + s_action
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        # noisy_video ↔ noisy_video
        mask[:noisy_end, :noisy_end] = self.vb.build_video_to_video_mask(
            video_seq_len=s_noisy_video,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # cond_video ↔ cond_video
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.vb.build_video_to_video_mask(
            video_seq_len=s_cond_video,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action ↔ action
        mask[cond_end:, cond_end:] = True
        # action → cond_video
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def run_idm_training_loop(
        self,
        vstate_noisy,
        vstate_cond,
        astate,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        """Run the IDM training loop with merged [noisy_video || cond_video] as video.

        The backbone owns the branch concatenation (``merge_idm_video_branches``)
        and split (``split_idm_video_branches``) since the layout differs by
        backbone: Wan merges flat ``(B, L, D)`` fields, while CosmosPredict25
        concatenates a 5D grid plus its ``extras`` (per-frame ``t_embedding`` and
        per-token ``rope_emb``) along the right axes. The MoT joint loop then
        runs as if there is one large video sequence + action.
        """
        # Merge noisy + cond video sequences. IDM teacher forcing needs two
        # different video timesteps inside one video-expert sequence; the
        # backbone returns the per-branch TOKEN counts (``T·H·W``) so the
        # teacher-forcing mask never depends on ``hidden_states.shape[1]`` (which
        # is ``T`` for 5D-grid backbones like CosmosPredict25).
        merged_vstate, s_noisy, s_cond = self.vb.merge_idm_video_branches(vstate_noisy, vstate_cond)
        s_action = astate.payload.x_action.shape[1]

        # Build IDM teacher-forcing mask (token granularity).
        video_tokens_per_frame = self._video_tokens_per_frame(vstate_noisy)
        attn_mask = self._build_teacher_forcing_mask(
            s_noisy_video=s_noisy,
            s_cond_video=s_cond,
            s_action=s_action,
            video_tokens_per_frame=video_tokens_per_frame,
            device=merged_vstate.hidden_states.device,
        )
        # Backbones whose per-layer keys carry a prefix with no matching query
        # rows (Cosmos3's cached und text stream) need the extra key columns.
        attn_mask = widen_mask_for_prefix_kv(attn_mask, merged_vstate)

        # Run the standard joint loop with the merged video state
        for layer_id in range(self.num_layers):
            merged_vstate, astate = self.step(
                layer_id,
                merged_vstate,
                astate,
                attn_mask=attn_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        # Split merged video back into noisy + cond
        vstate_noisy, vstate_cond = self.vb.split_idm_video_branches(merged_vstate, vstate_noisy, vstate_cond)

        return vstate_noisy, vstate_cond, astate

    def _build_video_only_attention_mask(
        self,
        *,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        return self.vb.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    @torch.no_grad()
    def prefill_video_cache(self, vstate):
        """Run the frozen video branch once and cache per-layer K/V for IDM inference.

        Returns ``(kv_cache, video_key_len, prefix_kv_mask)``:

        - ``video_key_len`` is the cached per-layer key length callers need to
          size the action mask — so they need not re-derive it via the private
          ``_video_tokens_per_frame``. For backbones without a prefix K/V block
          this equals the video token count (T·H·W); with one (Cosmos3's und text
          stream) it additionally covers the prefix.
        - ``prefix_kv_mask`` is the per-sample gate for those leading prefix
          columns. It is ``None`` in exactly two cases: the backbone declares no
          prefix at all, or it declares one that carries no padding (backbones
          signal that by leaving ``prefix_kv_mask`` unset — see
          :func:`widen_mask_for_prefix_kv`). ``None`` therefore means "no column
          needs closing", never "there is no prefix" on its own.
          Stage 2 MUST honor it: an all-ones action mask over ``video_key_len``
          would open padded und slots that the joint loop masks out, so a batch
          with unequal prompt lengths would silently attend padding. Pass it to
          :meth:`build_cached_action_mask`.
        """
        # Token count (T·H·W), not hidden_states.shape[1] — that is T for 5D-grid
        # backbones (CosmosPredict25). Mirrors run_joint_loop's s_video formula.
        video_tokens_per_frame = self._video_tokens_per_frame(vstate)
        video_seq_len = int(vstate.grid_frames) * video_tokens_per_frame
        attn_mask = self._build_video_only_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=vstate.hidden_states.device,
        )
        attn_mask = widen_mask_for_prefix_kv(attn_mask, vstate)
        kv_cache: list[dict[str, Tensor]] = []
        for layer_id in range(self.num_layers):
            q_v, k_v, v_v, vpost = self.vb.pre_attn_at_layer(layer_id, vstate)
            mixed_v = self._mixed_attention(q_v, k_v, v_v, attn_mask)
            vstate = self.vb.post_attn_at_layer(layer_id, vstate, mixed_v.contiguous(), vpost)
            kv_cache.append({"k": k_v, "v": v_v})
        # The cached keys include any backbone prefix (Cosmos3 und stream), so
        # report the actual cached key length — Stage 2 sizes its action mask
        # from this, and for prefix-free backbones it equals video_seq_len. The
        # prefix gate travels with it so Stage 2 can reproduce the joint loop's
        # masking of padded und columns.
        key_len = int(kv_cache[0]["k"].shape[1]) if kv_cache else video_seq_len
        prefix_mask = getattr(vstate, "prefix_kv_mask", None)
        if int(getattr(vstate, "prefix_kv_len", 0) or 0) <= 0:
            prefix_mask = None
        return kv_cache, key_len, prefix_mask

    def build_cached_action_mask(
        self,
        *,
        s_action: int,
        video_key_len: int,
        device: torch.device,
        prefix_kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Stage-2 action mask over ``[cached video keys ; action keys]``.

        Action attends every cached video key and every action key — the
        action-query row slice of the ``action_sees_video`` joint mask, which for
        dual_system (no readonly tail) reduces to all-ones. The one exception is
        the prefix block: padded und columns must stay closed, exactly as
        :func:`widen_mask_for_prefix_kv` does in the joint loop, so the cached
        and joint paths agree on what the action stream can see.
        """
        prefix = 0 if prefix_kv_mask is None else int(prefix_kv_mask.shape[1])
        base = torch.ones(
            (int(s_action), int(video_key_len) - prefix + int(s_action)),
            dtype=torch.bool,
            device=device,
        )
        if prefix == 0:
            return base
        gate = types.SimpleNamespace(prefix_kv_len=prefix, prefix_kv_mask=prefix_kv_mask)
        return widen_mask_for_prefix_kv(base, gate)

    def run_action_with_video_cache(
        self,
        astate,
        *,
        video_kv_cache: list[dict[str, Tensor]],
        video_seq_len: int,
        prefix_kv_mask: Optional[Tensor] = None,
    ):
        """Run only the action branch, attending to cached frozen-video K/V.

        ``video_seq_len`` is the cached per-layer KEY length returned by
        :meth:`prefill_video_cache` — for backbones with a prefix K/V block
        (Cosmos3's und text stream) that covers the prefix too. Pass that call's
        ``prefix_kv_mask`` alongside it so padded und columns stay closed; the
        action rows then see exactly what the joint loop lets them see.
        """
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(f"video_kv_cache must contain {self.num_layers} layers, got {len(video_kv_cache)}.")
        payload = astate.payload
        if payload is None or not hasattr(payload, "x_action"):
            raise RuntimeError("IDM cached action path requires ActionDiT.prepare_state payload.")

        s_action = int(payload.x_action.shape[1])
        action_mask = self.build_cached_action_mask(
            s_action=s_action,
            video_key_len=int(video_seq_len),
            device=payload.x_action.device,
            prefix_kv_mask=prefix_kv_mask,
        )

        for layer_id in range(self.num_layers):
            q_a, k_a, v_a, apost = self.ab.pre_attn_at_layer(layer_id, astate)
            cache = video_kv_cache[layer_id]
            k_video = cache["k"]
            v_video = cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"video_kv_cache[{layer_id}] seq length mismatch: "
                    f"expected {video_seq_len}, got {k_video.shape[1]} and {v_video.shape[1]}."
                )
            k_cat = torch.cat([k_video, k_a], dim=1)
            v_cat = torch.cat([v_video, v_a], dim=1)
            mixed_a = self._mixed_attention(q_a, k_cat, v_cat, action_mask)
            astate = self.ab.post_attn_at_layer(layer_id, astate, mixed_a.contiguous(), apost)
        return astate

    def video_kv_cache_to_tuples(
        self, video_kv_cache: list[dict[str, Tensor]]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Convert the legacy list/dict K/V cache into compile-friendly tuples."""

        if len(video_kv_cache) != self.num_layers:
            raise ValueError(f"video_kv_cache must contain {self.num_layers} layers, got {len(video_kv_cache)}.")
        return tuple(cache["k"] for cache in video_kv_cache), tuple(cache["v"] for cache in video_kv_cache)

    def run_action_with_video_cache_tensor_loop(
        self,
        x_action: Tensor,
        t_mod: Tensor,
        action_freqs: Tensor,
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
        action_mask: Tensor,
        video_k_tuple: tuple[Tensor, ...],
        video_v_tuple: tuple[Tensor, ...],
    ) -> Tensor:
        """Tensor-only IDM action-cache loop for deploy-time compile.

        The eager public path keeps the old ``ActionState`` and list/dict cache
        contract. This helper mirrors the same block math with only tensors and
        tuples at the boundary, which lets ``torch.compile`` capture the fixed
        action-with-video-cache hot loop without specializing on mutable Python
        state containers.
        """

        if len(video_k_tuple) != self.num_layers or len(video_v_tuple) != self.num_layers:
            raise ValueError(
                "video_k_tuple/video_v_tuple must contain one tensor per IDM layer, "
                f"got {len(video_k_tuple)} and {len(video_v_tuple)} for {self.num_layers} layers."
            )
        if context is None and context_mask is not None:
            raise ValueError("context_mask was provided but context is None.")

        x = x_action
        ab = self.ab

        for layer_id, block in enumerate(ab.blocks):
            chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

            residual_x = x
            attn_input = block.self_attn_norm(residual_x) * (1 + scale_msa) + shift_msa

            sa = block.self_attn
            q = sa.norm_q(sa.q(attn_input))
            k = sa.norm_k(sa.k(attn_input))
            v = sa.v(attn_input)

            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            q = rope_apply_1d(q, action_freqs)
            k = rope_apply_1d(k, action_freqs)

            q_a = rearrange(q, "b n s d -> b s (n d)", n=self.num_heads)
            k_a = rearrange(k, "b n s d -> b s (n d)", n=self.num_heads)
            v_a = rearrange(v, "b n s d -> b s (n d)", n=self.num_heads)

            k_video = video_k_tuple[layer_id]
            v_video = video_v_tuple[layer_id]
            if k_video.shape[0] != x.shape[0] or v_video.shape[0] != x.shape[0]:
                raise ValueError(
                    f"video K/V cache batch mismatch at layer {layer_id}: "
                    f"x={x.shape[0]}, k={k_video.shape[0]}, v={v_video.shape[0]}."
                )
            k_cat = torch.cat([k_video, k_a], dim=1)
            v_cat = torch.cat([v_video, v_a], dim=1)

            mixed_a = self._mixed_attention(q_a, k_cat, v_cat, action_mask)
            x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_a.contiguous()))

            if context is not None:
                text_mask = context_mask
                if text_mask is not None:
                    if text_mask.dim() == 2:
                        text_mask = text_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
                    elif text_mask.dim() not in (3, 4):
                        raise ValueError(
                            "IDM action-cache context_mask must be [B, L], [B, T_action, L], "
                            f"or broadcastable [B, heads, T_action, L], got {tuple(text_mask.shape)}"
                        )
                x = x + block.cross_attn(block.context_attn_norm(x), context, ctx_mask=text_mask)

            mlp_input = block.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
            x = block.gate(x, gate_mlp, block.ffn(mlp_input))

        return x


@register_architecture(
    "dual_system_idm",
    status="supported",
    note="DualSystem IDM: two-stage denoising — video first, then action with frozen video KV.",
    framework="dual_system",
    variant="idm",
)
class DualSystemIDMArchitecture(BaseWAMArchitecture):
    """DualSystem with Inverse Dynamics Model (IDM) two-stage denoising.

    Inherits the same MoT-based joint attention pattern as
    ``DualSystemSelfAttnArchitecture`` but overrides training loss and
    inference to implement the two-stage approach from FastWAM-IDM.
    """

    # Probability of adding noise to cond-video during training
    video_cond_noise_prob: float = 0.5

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mot_driver: IDMMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_idm_action_cache_loop: Optional[Callable[..., Tensor]] = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

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
            variant="idm",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )

        # IDM ignores attention_mask_mode: it never depends on the cross-modal
        # mode. Stage-2 cached-action attention is all-ones (built in
        # run_action_with_video_cache); teacher-forcing/prefill build their own
        # submasks. So attention_mask_mode is not forwarded to the driver.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "first_frame_causal")),
        }

        # IDM-specific config
        self.video_cond_noise_prob = float(cfg.get("idm_video_cond_noise_prob", cfg.get("video_cond_noise_prob", 0.5)))
        if not (0.0 <= self.video_cond_noise_prob <= 1.0):
            raise ValueError(
                "idm_video_cond_noise_prob must be in [0, 1] "
                f"(it gates a Bernoulli mask over the cond-video branch), got {self.video_cond_noise_prob}."
            )

        if self.video_backbone is not None:
            self.build_mot_driver()

    def build_mot_driver(self) -> IDMMoTDriver:
        """Construct the IDM-extended MoT driver.

        IDM training needs per-branch video timesteps so the noisy + cond
        branches can be concatenated along the sequence dim. On Wan this is
        token-wise (4D) ``t_mod``: TI2V provides it natively
        (``seperated_timestep + fuse_vae_embedding_in_latents``); other Wan
        backbones (VACE, I2V) get the same 4D shape via
        ``force_per_token_t_mod=True`` + ``zero_clean_prefix_t_mod=True`` in
        :meth:`_forward_idm_training`. On CosmosPredict25 the analogue is a
        per-FRAME ``t_embedding`` (Cosmos modulation is intrinsically per-frame);
        the same ``force_per_token_t_mod=True`` flag drives
        ``prepare_block_loop`` to expand the per-sample timestep to ``(B, T)`` so
        both branches concatenate along the frame axis (see
        ``cosmos_predict25.idm_merge``). The backbone owns the merge/split and the
        runtime shape checks, so the driver stays backbone-agnostic.
        """
        if self.video_backbone is None:
            raise RuntimeError("DualSystemIDMArchitecture.build_mot_driver: video_backbone is not set.")
        if self.action_backbone is None:
            raise RuntimeError("DualSystemIDMArchitecture.build_mot_driver: action_backbone is not set.")
        self._mot_driver = IDMMoTDriver(
            self.video_backbone,
            self.action_backbone,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    @property
    def mot_driver(self) -> IDMMoTDriver | None:
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile IDM's deploy-time action-cache hot loop when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_idm_action_cache_loop = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = idm_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("IDM compile disabled by config; running eager.")
            return
        action_cache_section = section.action_cache
        if not section_enabled(action_cache_section, default=True):
            logger.info("IDM action-cache compile disabled by config; running eager action stage.")
            return
        if self.video_backbone is None or self.action_backbone is None:
            logger.warning("IDM action-cache compile requested before backbones are ready; running eager.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_action_cache_loop(
            x_action,
            t_mod,
            action_freqs,
            context,
            context_mask,
            action_mask,
            video_k_tuple,
            video_v_tuple,
        ):
            return driver.run_action_with_video_cache_tensor_loop(
                x_action,
                t_mod,
                action_freqs,
                context,
                context_mask,
                action_mask,
                video_k_tuple,
                video_v_tuple,
            )

        kwargs = torch_compile_kwargs(action_cache_section, default_mode="reduce-overhead")
        try:
            self._compiled_idm_action_cache_loop = torch.compile(_run_action_cache_loop, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_idm_action_cache_loop = None
            logger.warning("IDM action-cache torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled IDM action-cache compile with torch.compile kwargs=%s", kwargs)

    # ------------------------------------------------------------------
    # Forward: dispatches between standard joint (inference fallback) and
    # the IDM 3-branch training path.
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        cond_video_latents: Optional[Tensor] = None,
        cond_video_timestep: Optional[Tensor] = None,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward dispatch.

        - ``cond_video_latents is None``: standard joint forward (inference
          stage 1 / fallback). Mirrors :class:`DualSystemSelfAttnArchitecture`.
        - ``cond_video_latents is not None``: IDM 3-branch training forward
          (noisy + cond video + optional action through the teacher-forcing
          mask). This is the path :meth:`compute_loss` drives.

        Hard contract: ``cond_video_latents`` and ``cond_video_timestep`` are
        a paired unit — both must be passed (training branch) or both must
        be ``None`` (joint inference branch). A half-passed pair would
        otherwise propagate ``timestep=None`` into ``video_backbone.prepare``
        and crash deep in the backbone with an opaque trace.
        """
        if (cond_video_latents is None) != (cond_video_timestep is None):
            raise ValueError(
                "DualSystemIDMArchitecture.forward: cond_video_latents and "
                "cond_video_timestep must be passed together (both or neither). "
                f"Got cond_video_latents={'<tensor>' if cond_video_latents is not None else 'None'}, "
                f"cond_video_timestep={'<tensor>' if cond_video_timestep is not None else 'None'}."
            )
        if cond_video_latents is not None:
            return self._forward_idm_training(
                noisy_actions=noisy_actions,
                action_timestep=action_timestep,
                cond_video_latents=cond_video_latents,
                cond_video_timestep=cond_video_timestep,
                proprio=proprio,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                **pipeline_inputs,
            )

        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None")

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        if noisy_actions is None or ab is None:
            return self._run_video_only_backbone(
                pipeline_inputs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            ), None

        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        # Joint forward with standard MoT (inference stage 2 or fallback)
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
        vstate, astate = driver.run_joint_loop(
            vstate,
            astate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return vb.finalize(vstate), ab.extract_prediction(astate)

    def _forward_idm_training(
        self,
        *,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        cond_video_latents: Tensor,
        cond_video_timestep: Tensor,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the IDM 3-branch training pass.

        Called only via :meth:`forward` (and hence ``self.__call__``) so any
        architecture-level forward-pre-hooks fire before the MoT driver loop.

        ``pipeline_inputs`` carries the noisy-branch ``latents`` and
        ``timestep``; the cond branch overrides them with
        ``cond_video_latents`` / ``cond_video_timestep`` while sharing the
        rest of the pipeline state.
        """
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None")

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        # IDM structurally requires 4D ``t_mod`` (the driver concatenates the
        # noisy + cond branches along the sequence dim with per-branch
        # timesteps). For TI2V this is native; for VACE / I2V we go through
        # the broadcast path. ``zero_clean_prefix_t_mod`` is kept on for
        # symmetry — it only fires when ``first_frame_latents`` is set, which
        # holds for TI2V but NOT VACE (post the native-VACE refactor VACE
        # routes its first-frame condition through ``vace_context`` instead).
        # So the flag is load-bearing only for TI2V here; for VACE and I2V it
        # is structurally inert. Forced assignment (not setdefault) — callers
        # cannot disable: doing so would re-raise the
        # ``IDMMoTDriver.run_idm_training_loop`` 4D-shape assertion.
        pipeline_inputs["force_per_token_t_mod"] = True
        pipeline_inputs["zero_clean_prefix_t_mod"] = True

        # Noisy branch: ``latents`` and ``timestep`` already in pipeline_inputs.
        vstate_noisy = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        # Cond branch: same pipeline state, overridden latents + timestep.
        cond_inputs = dict(pipeline_inputs)
        cond_inputs["latents"] = cond_video_latents
        cond_inputs["timestep"] = cond_video_timestep
        vstate_cond = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **cond_inputs,
        )

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        if noisy_actions is not None and ab is not None:
            astate = ab.prepare_state(
                noisy_actions,
                action_timestep,
                context=action_context,
                context_mask=action_context_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            vstate_noisy, vstate_cond, astate = driver.run_idm_training_loop(
                vstate_noisy,
                vstate_cond,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            action_noise_pred = ab.extract_prediction(astate)
        else:
            # Video-only IDM (lambda_action == 0): run both branches independently.
            # TODO(perf): the cond branch is computed but never read — only
            # ``vstate_noisy`` feeds ``vb.finalize`` below. With no action
            # backbone in play, cond exists solely as the teacher-forcing
            # signal for the action loss; when that loss is disabled the
            # cond forward is dead work. Safe to skip if a future caller
            # actually runs lambda_action==0 (OpenWAM training does not today).
            for block_id in range(vb.num_layers):
                vstate_noisy = vb.run_block(block_id, vstate_noisy)
            for block_id in range(vb.num_layers):
                vstate_cond = vb.run_block(block_id, vstate_cond)
            action_noise_pred = None

        video_noise_pred = vb.finalize(vstate_noisy)
        return video_noise_pred, action_noise_pred

    def _run_video_only_backbone(
        self,
        pipeline_inputs: dict,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tensor:
        """Run the video backbone once using already-prepared pipeline inputs.

        The caller owns any architecture-level context augmentation (such as
        proprio-as-context). This keeps IDM generate's two-stage path from
        re-entering ``forward()`` after proprio has already been appended.
        The caller also owns ``torch.compiler.cudagraph_mark_step_begin()`` for
        inference dispatches that may hit fixed-shape compiled loops.
        """
        vb = self.video_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None")
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
        return vb.finalize(vstate)

    # ------------------------------------------------------------------
    # Training: IDM compute_loss with 3 branches
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        *,
        actions: Optional[Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        **inputs,
    ) -> dict:
        """IDM training loss with three branches and teacher-forcing mask.

        Branch A: noisy video (denoising target)
        Branch B: noisy action (denoising target)
        Branch C: teacher-forcing cond video (condition for action, optionally noised)

        Branch A/B timesteps are drawn independently with ``torch.randint``.
        """
        vb = self.video_backbone
        ab = self.action_backbone
        action_scheduler = ab.scheduler
        _dtype = self.dtype
        _device = self.device

        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)

        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        input_latents = inputs["input_latents"]
        B = input_latents.shape[0]

        # ---- Branch A: noisy video (denoising target) ----
        video_timestep_ids = torch.randint(min_tb, max_tb, (B,))

        video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
        video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        video_noise = torch.randn_like(input_latents)
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        latents_noisy = (1 - sigma_bc) * input_latents + sigma_bc * video_noise
        video_target = video_noise - input_latents

        if inputs.get("first_frame_latents") is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        # ---- Branch B: noisy action (denoising target) ----
        noisy_actions = None
        action_target = None
        action_timesteps = None
        action_timestep_ids = None
        if lambda_action > 0 and actions is not None:
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)

            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)

            action_noise = torch.randn_like(actions)
            a_sigma_bc = action_sigmas.view(B, 1, 1)
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # ---- Branch C: teacher-forcing cond video ----
        cond_noise_mask = torch.rand((B,), device=_device) < self.video_cond_noise_prob
        latents_cond = input_latents.clone()

        # Cond video timestep: 0 for clean, sampled for noised
        cond_video_timesteps = torch.zeros((B,), dtype=_dtype, device=_device)
        if bool(cond_noise_mask.any()):
            cond_sampled_ids = torch.randint(min_tb, max_tb, (B,))
            cond_sigmas = vb.scheduler.sigmas[cond_sampled_ids].to(dtype=_dtype, device=_device)
            cond_sampled_timesteps = vb.scheduler.timesteps[cond_sampled_ids].to(dtype=_dtype, device=_device)
            cond_video_timesteps = torch.where(cond_noise_mask, cond_sampled_timesteps, cond_video_timesteps)
            noise_cond = torch.randn_like(input_latents)
            cond_sigma_bc = cond_sigmas.view(B, 1, 1, 1, 1)
            latents_cond_noisy = (1 - cond_sigma_bc) * input_latents + cond_sigma_bc * noise_cond
            selector = cond_noise_mask.view(B, 1, 1, 1, 1)
            latents_cond = torch.where(selector, latents_cond_noisy, latents_cond)

        if inputs.get("first_frame_latents") is not None:
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        # ---- Prepare forward inputs ----
        forward_inputs = dict(inputs)
        proprio = forward_inputs.pop("proprio", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)
        # ``latents`` carries the noisy-branch video latents into the forward;
        # the cond branch is passed alongside as ``cond_video_latents``.
        forward_inputs["latents"] = latents_noisy

        # ---- Run forward through ``self.__call__`` ----
        # Routing through ``self(...)`` (not ``self.forward(...)``) makes
        # ``nn.Module.__call__`` invoke any architecture-level forward-pre-hooks.
        # Mirrors the base ``compute_loss`` pattern.
        video_noise_pred, action_noise_pred = self(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio=proprio,
            use_gradient_checkpointing=use_grad_ckpt,
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            cond_video_latents=latents_cond,
            cond_video_timestep=cond_video_timesteps,
            timestep=video_timesteps,
            **forward_inputs,
        )

        # ---- Video loss (same as base) ----
        loss_video = self._compute_video_loss(
            video_noise_pred,
            video_target,
            video_timestep_ids,
            inputs,
            _device,
        )

        if lambda_action == 0 or action_noise_pred is None:
            return {
                "loss": lambda_video * loss_video,
                "loss_video": lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # ---- Action loss (same as base) ----
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            inputs,
            _device,
        )

        if lambda_video == 0:
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        return {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

    # ------------------------------------------------------------------
    # Inference: two-stage generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        schedule,
        prompt: str,
        *,
        vace_video=None,
        first_frame_image=None,
        num_frames: int = 49,
        action_num_frames: Optional[int] = None,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        tiled: bool = True,
        input_video_latents: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = None,
        tile_stride: tuple = None,
        dit_cache=None,
        decode_video: bool = True,
        profile: bool = False,
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
        proprio: Optional[Tensor] = None,
        active_action_mask: Optional[Tensor] = None,
    ) -> dict:
        """Two-stage IDM generation.

        Stage 1: Denoise video independently using the video DiT (no action).
        Stage 2: Freeze denoised video latents, denoise action with standard
                 MoT joint loop using the frozen video as condition.
        """
        import time

        from tqdm import tqdm

        vb = self.video_backbone
        ab = self.action_backbone
        device = self.device
        dtype = self.dtype

        t0 = time.time()

        action_num_frames = int(action_num_frames if action_num_frames is not None else num_frames)

        # Wan uses its own native tiling grid; tile_size/tile_stride are not forwarded.
        inputs_shared = vb.preprocess_input_for_inference(
            prompt=prompt,
            vace_video=vace_video,
            first_frame_image=first_frame_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=num_inference_steps,
            shift=shift,
            tiled=tiled,
            vace_cache=vace_cache,
            prompt_embed_cache=prompt_embed_cache,
        )

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[IDM_PROFILE] pipeline_prep: %.3fs", time.time() - t0)

        if input_video_latents is not None:
            inputs_shared["latents"] = input_video_latents
        ref_latents = inputs_shared.get("first_frame_latents")
        if ref_latents is not None:
            latents = inputs_shared["latents"].clone()
            latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared["latents"] = latents
        if self.uses_proprioception:
            if proprio is None:
                raise ValueError("use_proprioception=True requires `proprio` during generation.")
            inputs_shared["proprio"] = proprio.to(device=device, dtype=dtype)

        proprio_arg = inputs_shared.pop("proprio", None)
        inputs_shared_with_proprio = self._append_proprio_context_token(dict(inputs_shared), proprio_arg)

        # Initialize latents
        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        # Same inactive-dim pinning as BaseWAMArchitecture.generate: unsupervised
        # unified-action dims must stay on their analytic sigma * eps0 noise path.
        inactive_action_noise = None
        inactive_action_dims = self._resolve_inactive_action_dims(active_action_mask, device)

        num_train_ts_v = float(self.video_scheduler.num_train_timesteps)
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)

        # ---- Stage 1: Denoise video independently ----
        t_stage1 = time.time()
        did_video_step = False
        for i in tqdm(range(len(schedule) - 1), desc="IDM Stage 1: Video"):
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            sigma_v = t_v / num_train_ts_v
            sigma_v_next = t_v_next / num_train_ts_v

            video_stepping = sigma_v != sigma_v_next
            if not video_stepping:
                continue
            did_video_step = True

            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)

            if dit_cache is not None and not dit_cache.should_recompute(sigma_v):
                noise_pred = dit_cache.get_cached()
            else:
                # Video-only forward. We already appended proprio to the context
                # at generate() scope because stage 2 bypasses forward(), so
                # stage 1 must drive the video backbone directly instead of
                # re-entering self.forward(), which would try to append proprio a
                # second time.
                torch.compiler.cudagraph_mark_step_begin()
                noise_pred = self._run_video_only_backbone(
                    {
                        **inputs_shared_with_proprio,
                        "timestep": v_timestep,
                    },
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                )
                if dit_cache is not None:
                    dit_cache.update(noise_pred, sigma_v)

            new_latents = inputs_shared_with_proprio["latents"] + noise_pred * (sigma_v_next - sigma_v)
            ref_latents = inputs_shared_with_proprio.get("first_frame_latents")
            if ref_latents is not None:
                new_latents = new_latents.clone()
                new_latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared_with_proprio["latents"] = new_latents

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[IDM_PROFILE] stage1_video: %.3fs", time.time() - t_stage1)

        # ---- Stage 2: Denoise action with frozen video ----
        t_stage2 = time.time()

        # IDM Stage 2 prefills frozen video K/V from `inputs_shared_with_proprio["latents"]`
        # and treats it as t=0 clean video. That's only valid when those latents
        # are an actual video condition — either Stage 1 denoised them, or the
        # caller supplied `input_video_latents`. With a schedule that has no
        # video step (e.g. action_only) and no input_video_latents,
        # we'd silently prefill from the random initial latents and the action
        # denoiser would condition on pure noise. Fail loudly instead.
        if not did_video_step and input_video_latents is None:
            raise ValueError(
                "IDM generate(): Stage 1 produced no video denoising step (schedule has no "
                "video timestep deltas) and no `input_video_latents` were provided. "
                "Stage 2 would prefill frozen video K/V from the random initial latents, "
                "feeding pure noise to the action denoiser. Provide a schedule with at "
                "least one video step, or pass `input_video_latents` precomputed elsewhere."
            )

        # Use denoised video latents as condition (t=0 means clean).
        # The video branch is independent from action (v→a is masked), so
        # prefill its layer-wise K/V cache once and reuse it across all action
        # denoising steps, matching FastWAM-IDM's stage-2 path.
        cond_inputs = dict(inputs_shared_with_proprio)
        cond_timestep = torch.zeros(1, dtype=dtype, device=device)
        action_context = cond_inputs.get("context")
        action_context_mask = cond_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and cond_inputs.get("seq_lens") is not None:
            seq_lens = cond_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        cond_vstate = vb.prepare(
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            timestep=cond_timestep,
            **cond_inputs,
        )
        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()
        # prefill returns the cached per-layer KEY length (which covers a backbone
        # prefix K/V block, if any) plus the gate for that prefix — no need to
        # re-derive either via the private _video_tokens_per_frame.
        video_kv_cache, video_seq_len, video_prefix_mask = driver.prefill_video_cache(cond_vstate)
        compiled_action_cache_inputs = None
        if getattr(self, "_compiled_idm_action_cache_loop", None) is not None:
            video_k_tuple, video_v_tuple = driver.video_kv_cache_to_tuples(video_kv_cache)
            action_seq_len = int(action_latents.shape[1])
            action_mask = driver.build_cached_action_mask(
                s_action=action_seq_len,
                video_key_len=video_seq_len,
                device=action_latents.device,
                prefix_kv_mask=video_prefix_mask,
            )
            action_context_dtype = ab._embed_actions(action_latents).dtype
            action_context_emb, action_context_attn_mask = ab._prepare_context(
                action_context,
                action_context_mask,
                batch_size=action_latents.shape[0],
                seq_len=action_seq_len,
                dtype=action_context_dtype,
                device=action_latents.device,
            )
            action_freqs = ab._get_rope_freqs(action_seq_len).to(device=action_latents.device)
            compiled_action_cache_inputs = (
                video_k_tuple,
                video_v_tuple,
                action_mask,
                action_context_emb,
                action_context_attn_mask,
                action_freqs,
            )

        for i in tqdm(range(len(schedule) - 1), desc="IDM Stage 2: Action"):
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            sigma_a = t_a / num_train_ts_a
            sigma_a_next = t_a_next / num_train_ts_a

            action_stepping = sigma_a != sigma_a_next
            if not action_stepping:
                continue

            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)

            action_noise_pred = None
            compiled_action_cache_loop = getattr(self, "_compiled_idm_action_cache_loop", None)
            if compiled_action_cache_loop is not None and compiled_action_cache_inputs is not None:
                (
                    video_k_tuple,
                    video_v_tuple,
                    action_mask,
                    action_context_emb,
                    action_context_attn_mask,
                    action_freqs,
                ) = compiled_action_cache_inputs
                try:
                    x_action = ab._embed_actions(action_latents)
                    prepared_timestep = ab._prepare_timestep(a_timestep, action_latents.shape[0])
                    t_embed = ab.time_embedding(prepared_timestep)
                    t_mod = ab.time_projection(t_embed)
                    torch.compiler.cudagraph_mark_step_begin()
                    x_action = compiled_action_cache_loop(
                        x_action,
                        t_mod,
                        action_freqs,
                        action_context_emb,
                        action_context_attn_mask,
                        action_mask,
                        video_k_tuple,
                        video_v_tuple,
                    ).clone()
                    action_noise_pred = ab.action_decoder(x_action)
                except Exception as exc:
                    self._compiled_idm_action_cache_loop = None
                    logger.warning("IDM compiled action-cache loop failed; falling back to eager: %s", exc)

            if action_noise_pred is None:
                astate = ab.prepare_state(
                    action_latents,
                    a_timestep,
                    context=action_context,
                    context_mask=action_context_mask,
                )
                astate = driver.run_action_with_video_cache(
                    astate,
                    video_kv_cache=video_kv_cache,
                    video_seq_len=video_seq_len,
                    prefix_kv_mask=video_prefix_mask,
                )
                action_noise_pred = ab.extract_prediction(astate)

            if action_noise_pred is not None:
                if inactive_action_dims is not None and inactive_action_noise is None:
                    sigma_a_f = float(sigma_a)
                    if sigma_a_f <= 0.0:
                        raise ValueError("Cannot initialize inactive action noise from a non-positive sigma.")
                    inactive_action_noise = action_latents[..., inactive_action_dims].detach().clone() / sigma_a_f
                action_latents = self.action_scheduler.flow_step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )
                if inactive_action_dims is not None:
                    action_latents[..., inactive_action_dims] = inactive_action_noise * float(sigma_a_next)

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[IDM_PROFILE] stage2_action: %.3fs", time.time() - t_stage2)

        # VAE decode
        if decode_video:
            video_frames = vb.decode_video(inputs_shared_with_proprio["latents"], tiled=tiled)
        else:
            video_frames = None
        actions_out = action_latents.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            actions_out = normalizer.unnormalize(actions_out)

        return {"video": video_frames, "actions": actions_out}


__all__ = ["DualSystemIDMArchitecture"]
