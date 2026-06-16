from typing import Optional

import torch
from einops import rearrange

from openwam.model.video_backbone.wan.models.dit import WanModel, sinusoidal_embedding_1d
from openwam.model.video_backbone.wan.models.vace import VaceWanModel
from openwam.model.video_backbone.wan.shared.core import gradient_checkpoint_forward


def model_fn_wan_video(
    dit: WanModel,
    motion_controller=None,
    vace: VaceWanModel = None,
    vap=None,
    animate_adapter=None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames: bool = True,
    tea_cache=None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    num_clean_prefix_frames: int = 0,
    on_before_blocks=None,
    on_after_block=None,
    on_after_blocks=None,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        # Per-token timestep: frame 0..num_clean-1 get t=0, rest get sampled t.
        # Aligned with wan_video_dit.py forward() — supports batched timestep (B,).
        batch_size = latents.shape[0]
        num_clean = max(num_clean_prefix_frames, 1)  # at least 1 (original I2V behavior)
        f = latents.shape[2]
        tokens_per_frame = latents.shape[3] * latents.shape[4] // 4
        token_timesteps = torch.ones(
            batch_size, f, tokens_per_frame, dtype=latents.dtype, device=latents.device
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, :num_clean, :] = 0  # clean conditioning frames
        token_timesteps = token_timesteps.reshape(batch_size, -1)  # (B, f*tokens_per_frame)
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, token_timesteps.reshape(-1))
        t = dit.time_embedding(t_emb.to(latents.dtype)).reshape(batch_size, -1, dit.dim)
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Camera control
    x = dit.patchify(x, control_camera_latents_input)

    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)

    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    # VAP
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, "b c f h w -> b (f h w) c").contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f, h, w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if vace_context is not None:
        vace_hints = vace(
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    # Architecture callback: on_before_blocks. Architectures use this hook to
    # append action tokens, extend RoPE freqs with identity rotations, and
    # extend t_mod for the appended positions when the backbone uses
    # per-token AdaLN (Wan2.2-TI2V-5B fuse_vae_embedding_in_latents path).
    if on_before_blocks is not None:
        x, t_mod, freqs = on_before_blocks(x, t_mod, freqs)

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in chunks
            ]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:

        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)

            return custom_forward

        for block_id, block in enumerate(dit.blocks):
            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x,
                            context,
                            t_mod,
                            freqs,
                            x_vap,
                            context_vap,
                            t_mod_vap,
                            freqs_vap,
                            block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_vap,
                        context_vap,
                        t_mod_vap,
                        freqs_vap,
                        block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                x = gradient_checkpoint_forward(
                    block, use_gradient_checkpointing, use_gradient_checkpointing_offload, x, context, t_mod, freqs
                )

            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[
                        get_sequence_parallel_rank()
                    ]
                    current_vace_hint = torch.nn.functional.pad(
                        current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0
                    )
                x = x + current_vace_hint * vace_scale

            # Architecture callback: on_after_block. Architectures use this
            # hook for bridge feature capture, MoE expert FFN injection, or
            # interleaved joint ActionDiT execution. Runs after VACE hint
            # addition and before Animate.
            if on_after_block is not None:
                x = on_after_block(block_id, x)

            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)

    # Architecture callback: on_after_blocks. Architectures use this hook to
    # slice out appended action tokens before the head sees x, and to record
    # the final hidden state for action output projection.
    if on_after_blocks is not None:
        x = on_after_blocks(x)

    # CRITICAL (batched training): Head.forward has a 2D path and a 3D path.
    # The 2D path (t shape = (B, dim)) is BROKEN for B>1: PyTorch broadcasting
    # mixes sample 0's timestep into shift and sample 1's into scale when B=2,
    # and crashes outright for B>2.  Unsqueezing to 3D (B, 1, dim) routes to
    # the correct per-sample modulation path.  For B=1 both paths are identical.
    # When seperated_timestep is active, t is already 3D (B, seq_len, dim).
    t_head = t if t.dim() == 3 else t.unsqueeze(1)
    x = dit.head(x, t_head)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x
