"""Wan DiT forward-path helpers: time modulation + post-block VACE residual.

Free functions, dependency-injected, no backbone import. ``build_time_modulation``
takes the DiT / timestep / latents / patch_size explicitly; ``apply_post_block_residuals``
reads everything it needs off the passed ``state`` (``extras["vace"]`` / ``vace_hints``),
so neither touches the backbone instance.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d


def build_time_modulation(
    dit,
    timestep,
    latents,
    *,
    patch_size,
    fuse_vae_embedding_in_latents: bool,
    force_per_token_t_mod: bool,
    num_clean_prefix_frames: int,
    zero_clean_prefix_t_mod: bool,
    has_first_frame_latents: bool,
) -> Tuple[Tensor, Tensor]:
    """Build ``(time_embed, time_modulation)`` for ``prepare``. TI2V uses
    per-token t=0 on the clean prefix; ``force_per_token_t_mod`` broadcasts a
    single (B,) embedding to (B, L, dim); else the plain (B,) path.
    """
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        batch_size = latents.shape[0]
        num_clean = max(num_clean_prefix_frames, 1)
        f_lat = latents.shape[2]
        tokens_per_frame = latents.shape[3] * latents.shape[4] // (patch_size[1] * patch_size[2])
        token_timesteps = torch.ones(
            batch_size, f_lat, tokens_per_frame, dtype=latents.dtype, device=latents.device
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, :num_clean, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        time_sinusoid = sinusoidal_embedding_1d(dit.freq_dim, token_timesteps.reshape(-1))
        time_embed = dit.time_embedding(time_sinusoid.to(latents.dtype)).reshape(batch_size, -1, dit.dim)
        time_modulation = dit.time_projection(time_embed).unflatten(2, (6, dit.dim))
    elif force_per_token_t_mod:
        # Non-TI2V backbones under joint-attention / single_system need 4D
        # t_mod. Compute the time embedding once on (B,) and broadcast to
        # (B, L, dim) — a per-token MLP would repeat the stack L times.
        batch_size = latents.shape[0]
        f_lat = latents.shape[2]
        tokens_per_frame = latents.shape[3] * latents.shape[4] // (patch_size[1] * patch_size[2])
        L = f_lat * tokens_per_frame
        time_embed_base = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype)
        )  # (B, dim)
        time_embed = time_embed_base.unsqueeze(1).expand(batch_size, L, -1).contiguous()

        # Optional clean-prefix alignment: when opted in AND a clean prefix
        # is present, overwrite the first ``num_clean`` frames' time embedding
        # with ``time_embedding(0)`` — mirrors TI2V's per-token t=0 pin but
        # at the embedding layer, keeping the MLP a single (B, dim) call.
        has_clean_ref = num_clean_prefix_frames > 0 or has_first_frame_latents
        if zero_clean_prefix_t_mod and has_clean_ref:
            num_clean = max(num_clean_prefix_frames, 1)
            zero_ts = torch.zeros_like(timestep)
            time_embed_zero = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, zero_ts).to(latents.dtype)
            )  # (B, dim)
            time_embed = time_embed.view(batch_size, f_lat, tokens_per_frame, -1)
            time_embed[:, :num_clean] = time_embed_zero.view(batch_size, 1, 1, -1)
            time_embed = time_embed.reshape(batch_size, L, -1)

        time_modulation = dit.time_projection(time_embed).unflatten(2, (6, dit.dim))
    else:
        time_embed = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        time_modulation = dit.time_projection(time_embed).unflatten(1, (6, dit.dim))
    return time_embed, time_modulation


def apply_post_block_residuals(block_id: int, state) -> None:
    """Apply the VACE hint residual to ``state.hidden_states`` in place.
    Shared by run_block and post_attn_at_layer.
    """
    vace = state.extras.get("vace")

    if state.vace_hints is not None and vace is not None and block_id in vace.vace_layers_mapping:
        current_vace_hint = state.vace_hints[vace.vace_layers_mapping[block_id]]
        vace_len = current_vace_hint.shape[1]
        if state.hidden_states.shape[1] == vace_len:
            # dual_system / video-only: hint spans the full sequence.
            state.hidden_states = state.hidden_states + current_vace_hint
        elif state.hidden_states.shape[1] > vace_len:
            # single_system: residual applies only to the leading video
            # slice; action/state tokens get VACE via self-attention.
            video_slice = state.hidden_states[:, :vace_len] + current_vace_hint
            state.hidden_states = torch.cat([video_slice, state.hidden_states[:, vace_len:]], dim=1)
        else:
            # Defensive: hidden_states shorter than the hint breaks the
            # video-token-count invariant — investigate before patching.
            raise ValueError(
                f"apply_post_block_residuals: state.hidden_states.shape[1]={state.hidden_states.shape[1]} "
                f"< vace_hint.shape[1]={vace_len} at block {block_id}; "
                "this is unreachable under dual_system or single_system today—"
                "investigate the upstream caller before patching this branch."
            )
