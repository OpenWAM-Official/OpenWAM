"""Prompt tokenization + 3D-mRoPE position packing for the Cosmos3-Edge backbone.

Faithful port of the upstream ``Cosmos3OmniPipeline`` text/vision sequence
assembly (``_tokenize`` / ``_add_special_tokens`` / ``get_3d_mrope_ids_*`` /
``_prepare_text_segment`` / ``_prepare_vision_segment`` at diffusers @6ad35739),
reshaped for OpenWAM's uniform-shape batches:

- One prompt per batch item; und token lists are right-padded to the batch max
  and masked (the und stream is causal, so right padding never contaminates
  real tokens, and padded keys are masked out of both attention pathways).
- The vision temporal offset is uniform across the batch:
  ``max(und_len) + unified_3d_mrope_temporal_modality_margin``. The native
  pipeline uses each sample's own ``und_len``; with the 15000-token margin the
  difference is a batch-constant shift of a few tokens on the T axis and it
  makes the gen rotary shareable across the batch. For B=1 (deploy and the
  golden-parity replay) this is exactly the native offset.

Pure torch + stdlib — the tokenizer object is passed in, so importing this
module stays CPU-CI safe.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch

# Upstream constants (pipeline_cosmos3_omni.py @6ad35739; typos are upstream's).
SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
VIDEO_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
INVERSE_IMAGE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."
INVERSE_VIDEO_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."

START_OF_GENERATION_TOKEN = "<|vision_start|>"


def apply_prompt_templates(
    text: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    negative: bool = False,
    add_duration_template: bool = True,
    add_resolution_template: bool = True,
) -> str:
    """Append the duration/resolution metadata sentences the model was trained with."""
    is_image = num_frames == 1

    def _append(base: str, addition: str) -> str:
        base = base.rstrip(".")
        return f"{base}. {addition}" if base else addition

    if not is_image and add_duration_template:
        template = INVERSE_DURATION_TEMPLATE if negative else DURATION_TEMPLATE
        text = _append(text, template.format(duration=num_frames / fps, fps=fps))
    if add_resolution_template:
        if is_image:
            template = INVERSE_IMAGE_RESOLUTION_TEMPLATE if negative else IMAGE_RESOLUTION_TEMPLATE
        else:
            template = INVERSE_VIDEO_RESOLUTION_TEMPLATE if negative else VIDEO_RESOLUTION_TEMPLATE
        text = _append(text, template.format(height=height, width=width))
    return text


def _templated_ids(tokenizer: Any, conversations: list, *, add_generation_prompt: bool) -> List[int]:
    ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    if not isinstance(ids, list):  # some transformers versions return BatchEncoding-likes
        ids = list(ids["input_ids"] if hasattr(ids, "__getitem__") else ids)
    return list(ids)


def _common_suffix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[len(a) - 1 - n] == b[len(b) - 1 - n]:
        n += 1
    return n


def tokenize_prompt(
    tokenizer: Any,
    text: str,
    *,
    use_system_prompt: bool = False,
    is_image: bool = False,
    max_length: Optional[int] = None,
) -> List[int]:
    """Chat-template tokenize one prompt and append ``[eos, <|vision_start|>]``.

    Mirrors the pipeline's ``_tokenize`` + ``_add_special_tokens``. ``max_length``
    caps the templated ids before the two special tokens are appended.

    Truncation cuts the prompt *body*, never the tail. The whole constant tail
    matters, not just the assistant header: a chat template emits
    ``<user content> <|im_end|> <assistant header>``, so a plain right-cut
    deletes the turn terminator *and* the header and hands the model a
    mid-sentence stop followed by ``[eos, <|vision_start|>]`` — a shape it never
    saw in training. The tail is recovered as the longest common suffix against
    the same template rendered with empty content, which is exactly the part
    that does not depend on the prompt.
    """
    conversations = []
    if use_system_prompt:
        conversations.append({"role": "system", "content": SYSTEM_PROMPT_IMAGE if is_image else SYSTEM_PROMPT_VIDEO})
    conversations.append({"role": "user", "content": text})
    ids = _templated_ids(tokenizer, conversations, add_generation_prompt=True)
    if max_length is not None and len(ids) > max_length - 2:
        budget = max_length - 2
        empty = list(conversations[:-1]) + [{"role": conversations[-1]["role"], "content": ""}]
        tail_len = _common_suffix_len(ids, _templated_ids(tokenizer, empty, add_generation_prompt=True))
        # Clamp the result too: a tail longer than the budget would otherwise
        # push the final length back over max_length.
        ids = (ids[: max(0, budget - tail_len)] + ids[len(ids) - tail_len :])[:budget] if tail_len else ids[:budget]
    eos = tokenizer.eos_token_id
    start_of_generation = tokenizer.convert_tokens_to_ids(START_OF_GENERATION_TOKEN)
    # Fast tokenizers map unknown tokens to unk_token_id instead of None, so a
    # wrong/partial tokenizer dir would otherwise slip a silent [eos, unk] tail
    # into every prompt instead of the generation marker.
    unk = getattr(tokenizer, "unk_token_id", None)
    if eos is None or start_of_generation is None or (unk is not None and start_of_generation == unk):
        raise ValueError(
            "Cosmos3 tokenizer must define eos_token_id and the "
            f"'{START_OF_GENERATION_TOKEN}' token; got eos={eos!r}, start={start_of_generation!r} "
            f"(unk={unk!r}). Check that text_tokenizer/ is complete and from the Cosmos3 bundle."
        )
    return list(ids) + [int(eos), int(start_of_generation)]


def text_mrope_positions(num_tokens: int, *, float_positions: bool) -> torch.Tensor:
    """``[3, num_tokens]`` position ids for text: all three axes share arange, offset 0."""
    dtype = torch.float32 if float_positions else torch.long
    ids = torch.arange(num_tokens, dtype=dtype)
    return ids.unsqueeze(0).expand(3, -1).contiguous()


def vision_mrope_positions(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    *,
    temporal_offset: float,
    fps: Optional[float],
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: Optional[int] = None,
    start_frame_offset: int = 0,
    reset_spatial_indices: bool = True,
) -> torch.Tensor:
    """``[3, T·H·W]`` position ids for vision patches — port of
    ``get_3d_mrope_ids_vae_tokens`` (T-major, then H, then W raster)."""
    fps_modulation = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation:
        assert fps is not None  # narrowed by fps_modulation
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation:
        return torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    return torch.stack([t_index, h_index, w_index], dim=0)


def pad_und_batch(
    ids_per_sample: Sequence[Sequence[int]], *, pad_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad per-sample und token lists to the batch max.

    Returns ``(input_ids (B, L) long, und_mask (B, L) bool — True at real tokens,
    seq_lens (B,) long)``.
    """
    if not ids_per_sample:
        raise ValueError("pad_und_batch needs at least one sample.")
    lens = [len(ids) for ids in ids_per_sample]
    max_len = max(lens)
    batch = torch.full((len(ids_per_sample), max_len), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(ids_per_sample), max_len), dtype=torch.bool)
    for i, ids in enumerate(ids_per_sample):
        batch[i, : len(ids)] = torch.tensor(list(ids), dtype=torch.long)
        mask[i, : len(ids)] = True
    return batch, mask, torch.tensor(lens, dtype=torch.long)


def build_joint_positions(
    und_len: int,
    grid: Tuple[int, int, int],
    *,
    modality_margin: int,
    fps: Optional[float],
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    float_positions: bool = True,
    reset_spatial_indices: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Positions for the padded und block and the vision block.

    ``und_len`` here is the *padded* batch length; the vision temporal offset is
    ``und_len + modality_margin`` (native semantics for B=1, uniform batch shift
    otherwise — see module docstring). Returns ``(text_pos [3, und_len],
    vision_pos [3, T·H·W])``.
    """
    text_pos = text_mrope_positions(und_len, float_positions=float_positions)
    grid_t, grid_h, grid_w = grid
    vision_pos = vision_mrope_positions(
        grid_t,
        grid_h,
        grid_w,
        temporal_offset=float(und_len + modality_margin),
        fps=fps if float_positions else None,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        reset_spatial_indices=reset_spatial_indices,
    )
    if float_positions:
        vision_pos = vision_pos.to(torch.float32)
        text_pos = text_pos.to(torch.float32)
    return text_pos, vision_pos


def patch_grid(latent_t: int, latent_h: int, latent_w: int, patch_size: int) -> Tuple[int, int, int]:
    """Latent grid → patch grid, with ceil-padding on H/W (upstream zero-pads)."""
    return latent_t, math.ceil(latent_h / patch_size), math.ceil(latent_w / patch_size)
