"""Stateless Cosmos DiT block-loop helpers.

Mirrors ``openwam/model/video_backbone/wan/dit_forward.py``: the per-block
orchestration of the upstream Cosmos DiT (``MinimalV1LVGDiT`` / ``MiniTrainDIT``)
lives here as free functions taking ``(net, ...)``, so ``CosmosPredict25VideoBackbone``
stays a thin flat adapter (no wrapper nn.Module). Each function replicates a
slice of ``MiniTrainDIT.forward`` that the backbone's ``prepare`` / ``run_block``
/ ``finalize`` / ``pre_attn_at_layer`` / ``post_attn_at_layer`` delegate to.

BlockLoopState contract (Cosmos-specific; field names follow ``base.py``):
- ``state.hidden_states`` : ``(B, T, H, W, D)`` patchified hidden state.
- ``state.context``       : ``(B, L, context_dim)`` cross-attn key/value source.
- ``state.grid_frames`` / ``grid_height`` / ``grid_width`` : patch grid ``(T,H,W)``.
- ``state.extras``        : Cosmos-specific tensors (t_embedding_B_T_D,
  adaln_lora_B_T_3D, rope_emb_L_1_1_D, extra_per_block_pos_emb).
- ``state.time_mod`` / ``state.rope_freqs`` : Wan-style placeholders (unused on
  joint_cross_attn).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import (
    gradient_checkpoint_forward,
)


def prepare_block_loop(
    net: Any,
    *,
    latents: Optional[Tensor] = None,
    input_latents: Optional[Tensor] = None,
    context: Tensor,
    timestep: Tensor,
    context_mask: Optional[Tensor] = None,
    condition_mask: Optional[Tensor] = None,
    padding_mask: Optional[Tensor] = None,
    fps: Optional[Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    force_per_token_t_mod: bool = False,
    **_ignored: Any,
) -> BlockLoopState:
    """Replicate ``MiniTrainDIT.forward`` up to (but excluding) the block loop.

    ``latents`` (preferred, noised at training time) or ``input_latents`` is the
    ``(B, C_z, T, H, W)`` DiT input. ``context`` is projected here when its last
    dim matches ``net.crossattn_proj_in_channels``. See the original wrapper
    docstring for the full per-arg contract.
    """
    x_in = latents if latents is not None else input_latents
    if x_in is None:
        raise ValueError(
            "cosmos_predict25.dit_forward.prepare_block_loop requires either `latents` "
            "(preferred, noised at training time) or `input_latents`."
        )
    B, _C, T_lat, H_lat, W_lat = x_in.shape
    device = x_in.device
    dtype = x_in.dtype

    # MinimalV1LVGDiT prepends a condition-mask channel and scales timesteps
    # before delegating to MiniTrainDIT.prepare_embedded_sequence.
    if hasattr(net, "timestep_scale"):
        if condition_mask is None:
            condition_mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, dtype=dtype, device=device)
        x_B_C_T_H_W = torch.cat([x_in, condition_mask.to(dtype=dtype)], dim=1)
        timesteps_eff = timestep * float(net.timestep_scale)
    else:
        x_B_C_T_H_W = x_in
        timesteps_eff = timestep

    if getattr(net, "concat_padding_mask", False) and padding_mask is None:
        padding_mask = torch.zeros(B, 1, H_lat, W_lat, dtype=dtype, device=device)

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_per_block_pos_emb = net.prepare_embedded_sequence(
        x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
    )

    # Optional cross-attn projection (Stage-c 2B uses 100352 → 1024).
    if getattr(net, "use_crossattn_projection", False):
        proj_in = int(getattr(net, "crossattn_proj_in_channels", -1))
        if context.shape[-1] == proj_in:
            context = net.crossattn_proj(context)

    # TI2V per-token timestep: when ``condition_mask`` marks any frames as clean
    # prefix (``mask == 1``), broadcast the per-sample timestep to ``(B, T_lat)``
    # and zero out the prefix positions (the DiT sees ``t=0`` on those frames).
    # T2V keeps the legacy ``(B, 1)`` broadcast shape (pure no-op).
    ti2v_active = timesteps_eff.ndim == 1 and condition_mask is not None and bool(condition_mask.any())
    if timesteps_eff.ndim == 1:
        if ti2v_active:
            timesteps_eff = timesteps_eff.unsqueeze(1).expand(-1, T_lat).clone()
            frame_mask = condition_mask[:, 0, :, 0, 0]  # (B, T_lat)
            timesteps_eff = torch.where(frame_mask > 0, torch.zeros_like(timesteps_eff), timesteps_eff)
        elif force_per_token_t_mod:
            # IDM teacher-forcing: expand a uniform (or t=0) per-sample timestep to
            # per-frame ``(B, T_lat)`` so this branch yields ``t_embedding_B_T_D`` of
            # shape ``(B, T_lat, D)``. The noisy and cond branches then concatenate
            # along the frame axis with their own per-branch timesteps (see
            # ``cosmos_predict25.idm_merge``). Cosmos's per-FRAME timestep is the
            # analogue of Wan's per-token ``t_mod`` here: Cosmos modulation is
            # already per-frame (``_compute_modulation`` broadcasts ``(B,T,1,1,D)``
            # over H,W), so frame-granular timesteps are all IDM needs.
            # ``T_lat == f`` because the Cosmos temporal patch size is 1.
            timesteps_eff = timesteps_eff.unsqueeze(1).expand(-1, T_lat).clone()
        else:
            timesteps_eff = timesteps_eff.unsqueeze(1)
    t_embedding_B_T_D, adaln_lora_B_T_3D = net.t_embedder(timesteps_eff)
    t_embedding_B_T_D = net.t_embedding_norm(t_embedding_B_T_D)

    f = x_B_T_H_W_D.shape[1]
    h = x_B_T_H_W_D.shape[2]
    w = x_B_T_H_W_D.shape[3]

    return BlockLoopState(
        hidden_states=x_B_T_H_W_D,
        time_mod=torch.zeros((), dtype=dtype, device=device),  # placeholder; unused on joint_cross_attn
        rope_freqs=torch.zeros((), dtype=torch.complex64, device=device),  # placeholder
        context=context,
        context_mask=context_mask,
        grid_frames=f,
        grid_height=h,
        grid_width=w,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        extras={
            "t_embedding_B_T_D": t_embedding_B_T_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "extra_per_block_pos_emb": extra_per_block_pos_emb,
        },
    )


def run_block(net: Any, block_id: int, state: BlockLoopState) -> BlockLoopState:
    block = net.blocks[block_id]
    # Wrap the block in the shared activation-checkpoint helper (mirrors
    # ``wan_backbone.run_block``). When both flags are False this is a plain
    # ``block(*args, **kwargs)``, so the non-checkpointed path is byte-identical;
    # when ``training.use_gradient_checkpointing`` is on (the train.yaml default)
    # the 28-block 2B DiT actually trades compute for activation memory instead
    # of silently materialising every block's activations.
    state.hidden_states = gradient_checkpoint_forward(
        block,
        state.use_gradient_checkpointing,
        state.use_gradient_checkpointing_offload,
        state.hidden_states,
        state.extras["t_embedding_B_T_D"],
        state.context,
        rope_emb_L_1_1_D=state.extras["rope_emb_L_1_1_D"],
        adaln_lora_B_T_3D=state.extras["adaln_lora_B_T_3D"],
        extra_per_block_pos_emb=state.extras["extra_per_block_pos_emb"],
    )
    return state


def pre_attn_at_layer(net: Any, block_id: int, state: BlockLoopState):
    """Pre half of block ``block_id`` for the MoT joint-attention driver.

    Returns ``(q, k, v, post_state)`` with Q/K/V at ``(B, T·H·W, H·D)``.
    """
    from openwam.model.video_backbone.cosmos_predict25.block_split import pre_self_attn

    block = net.blocks[block_id]
    return pre_self_attn(
        block,
        state.hidden_states,
        state.extras["t_embedding_B_T_D"],
        state.extras["adaln_lora_B_T_3D"],
        state.extras["rope_emb_L_1_1_D"],
        state.extras["extra_per_block_pos_emb"],
    )


def post_attn_at_layer(
    block_id: int,
    state: BlockLoopState,
    attn_out: Tensor,
    post_state: dict,
) -> BlockLoopState:
    """Post half of block ``block_id``: output_proj → residual → cross-attn → MLP.

    The ``block_id`` arg is part of the Wan-compatible hook signature but unused
    here (the block ref lives in ``post_state``).
    """
    from openwam.model.video_backbone.cosmos_predict25.block_split import post_self_attn

    _ = block_id
    state.hidden_states = post_self_attn(attn_out, state.context, post_state)
    return state


def finalize_block_loop(net: Any, state: BlockLoopState) -> Tensor:
    x_B_T_H_W_O = net.final_layer(
        state.hidden_states,
        state.extras["t_embedding_B_T_D"],
        adaln_lora_B_T_3D=state.extras["adaln_lora_B_T_3D"],
    )
    return net.unpatchify(x_B_T_H_W_O)


__all__ = [
    "prepare_block_loop",
    "run_block",
    "pre_attn_at_layer",
    "post_attn_at_layer",
    "finalize_block_loop",
]
