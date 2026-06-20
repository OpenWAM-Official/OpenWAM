"""Standalone LAPA-DINOv3 model pieces used for online target extraction.

This module is a minimal in-tree implementation of the LAPA-DINOv3 inference
path needed by OpenWAM. It intentionally avoids importing the reference
``ref_code/LARYBench`` checkout at runtime.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from torch import einsum, nn

logger = logging.getLogger(__name__)

_ALLOWED_MISSING_PREFIXES = ("dino_tokenizer.",)
_ALLOWED_UNEXPECTED_PREFIXES = ("module.", "dino_tokenizer.")


def _only_allowed(keys: Iterable[str], allowed_prefixes: tuple[str, ...]) -> bool:
    return all(any(str(key).startswith(prefix) for prefix in allowed_prefixes) for key in keys)


def exists(val) -> bool:
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pair(val) -> tuple[int, int]:
    ret = (val, val) if not isinstance(val, tuple) else val
    if len(ret) != 2:
        raise ValueError(f"Expected pair, got {ret}")
    return int(ret[0]), int(ret[1])


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def feed_forward(dim: int, mult: int = 4, dropout: float = 0.0) -> nn.Sequential:
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PEG(nn.Module):
    def __init__(self, dim: int, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    def forward(self, x: torch.Tensor, shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        needs_shape = x.ndim == 3
        if needs_shape and not exists(shape):
            raise ValueError("PEG requires video shape when input is flattened.")

        orig_shape = x.shape
        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)
        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")
        return x.reshape(orig_shape)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @staticmethod
    def _get_slopes(heads: int) -> list[float]:
        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
        )

    def get_bias(self, i: int, j: int, device: torch.device) -> torch.Tensor:
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        return -torch.abs(rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1"))

    def forward(self, sim: torch.Tensor) -> torch.Tensor:
        h, i, j, device = *sim.shape[-3:], sim.device
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer("bias", bias, persistent=False)
        return self.bias


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_context: int | None = None,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        num_null_kv: int = 0,
        norm_context: bool = True,
        dropout: float = 0.0,
        scale: int = 8,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, device = x.shape[0], x.device
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ContinuousPositionBias(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        num_dims: int = 2,
        layers: int = 2,
        log_dist: bool = True,
        cache_rel_pos: bool = False,
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.LeakyReLU(0.1)))
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1)))
        self.net.append(nn.Linear(dim, heads))
        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer("rel_pos", rel_pos, persistent=False)

        rel_pos = self.rel_pos.float()
        for layer in self.net:
            rel_pos = layer(rel_pos)
        return rearrange(rel_pos, "i j h -> h i j")


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_context: int | None = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        peg: bool = False,
        peg_causal: bool = False,
        attn_num_null_kv: int = 2,
        has_cross_attn: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            dim_context=dim_context,
                            heads=heads,
                            causal=False,
                            num_null_kv=attn_num_null_kv,
                            dropout=attn_dropout,
                        )
                        if has_cross_attn
                        else None,
                        feed_forward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )
        self.norm_out = LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        video_shape: tuple[int, int, int, int] | None = None,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=video_shape) + x
            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x
            x = ff(x) + x
        return self.norm_out(x)


class NSVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str = torch.device("cpu"),
        discarding_threshold: float = 0.1,
        initialization: str = "normal",
        code_seq_len: int = 1,
        patch_size: int = 32,
        image_size: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim
        self.patch_size = patch_size

        if initialization == "normal":
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=self.device)
        elif initialization == "uniform":
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample(
                [self.num_embeddings, self.embedding_dim]
            )
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        self.codebooks = nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=self.device)
        self.project_in = nn.Linear(dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, dim)

        if code_seq_len == 1:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=1, padding=0),
            )
        elif code_seq_len == 2:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(3, 4), stride=1, padding=0),
            )
        elif code_seq_len == 4:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=0),
            )
        elif code_seq_len == 16:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            )
        elif code_seq_len == 49:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            )
        elif code_seq_len == 64:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2, padding=1),
            )
        elif code_seq_len == 256:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError("Unsupported code_seq_len")

    def encode(self, input_data: torch.Tensor, batch_size: int) -> torch.Tensor:
        input_data = self.project_in(input_data)
        input_data = input_data.permute(0, 2, 1).contiguous()
        size = int(self.image_size / self.patch_size)
        input_data = input_data.reshape(batch_size, self.embedding_dim, size, size)
        input_data = self.cnn_encoder(input_data)
        input_data = input_data.reshape(batch_size, self.embedding_dim, -1)
        input_data = input_data.permute(0, 2, 1).contiguous()
        return input_data.reshape(-1, self.embedding_dim)

    def decode(self, quantized_input: torch.Tensor, batch_size: int) -> torch.Tensor:
        quantized_input = quantized_input.reshape(batch_size, self.embedding_dim, -1)
        quantized_input = quantized_input.permute(0, 2, 1).contiguous()
        return self.project_out(quantized_input)

    def forward(
        self,
        input_data_first: torch.Tensor,
        input_data_last: torch.Tensor,
        codebook_training_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_data_first.shape[0]
        input_data_first = self.encode(input_data_first.contiguous(), batch_size)
        input_data_last = self.encode(input_data_last, batch_size)
        input_data = input_data_last - input_data_first

        distances = (
            torch.sum(input_data**2, dim=1, keepdim=True)
            - 2 * torch.matmul(input_data, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        random_vector = normal_dist.Normal(0, 1).sample(input_data.shape).to(input_data.device)

        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = (norm_quantization_residual / norm_random_vector + self.eps) * random_vector
        quantized_input = hard_quantized_input if codebook_training_only else input_data + vq_error

        encodings = torch.zeros(input_data.shape[0], self.num_embeddings, device=input_data.device)
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            self.codebooks_used[min_indices.detach().cpu()] += 1

        quantized_input = self.decode(quantized_input, batch_size)
        return quantized_input, perplexity, self.codebooks_used.cpu(), min_indices.reshape(batch_size, -1)


def load_dinov3_tokenizer(model_path: str | Path, *, device: torch.device) -> nn.Module:
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def get_dinov3_reps(input_tensor: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    # LAPA-DINOv3 is trained for ViT-L/16 at 224px: skip 1 CLS + 4 register
    # tokens, then reshape the remaining patches into the fixed 14x14 grid.
    input_tensor = input_tensor.squeeze(2)
    input_tensor = input_tensor * 2 - 1
    outputs = encoder(input_tensor)
    last_hidden_states = outputs.last_hidden_state[:, 5:, :]
    x = last_hidden_states.detach().unsqueeze(1)
    batch_size, _, _, dim = x.shape
    return x.reshape(batch_size, 1, 14, 14, dim)


class LatentActionQuantizationDinov3Feature(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        quant_dim: int,
        codebook_size: int,
        image_size: int,
        patch_size: int,
        spatial_depth: int,
        temporal_depth: int,
        dinov3_model_dir: str | Path,
        device: torch.device,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 3,  # kept for checkpoint/config compatibility
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        code_seq_len: int = 1,
    ):
        super().__init__()
        del channels
        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.dino_tokenizer = load_dinov3_tokenizer(dinov3_model_dir, device=device)

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        image_height, image_width = self.image_size
        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError("image_size must be divisible by patch_size")

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        transformer_with_action_kwargs = dict(
            **transformer_kwargs,
            has_cross_attn=True,
            dim_context=dim,
        )
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            device=device,
            code_seq_len=code_seq_len,
            patch_size=patch_size,
            image_size=image_size,
        )
        self.dec_spatial_transformer = Transformer(depth=spatial_depth, **transformer_with_action_kwargs)

    @property
    def patch_height_width(self) -> tuple[int, int]:
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def load_state_dict(self, *args, **kwargs):
        kwargs["strict"] = False
        incompatible = super().load_state_dict(*args, **kwargs)
        state = args[0] if args and isinstance(args[0], dict) else {}
        matched = len(set(state.keys()) & set(self.state_dict().keys()))
        logger.info(
            "LAPA-DINOv3 load_state_dict matched=%d missing=%d unexpected=%d",
            matched,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if matched == 0:
            raise RuntimeError(
                "LAPA-DINOv3 checkpoint did not match any model parameters; "
                "check that laq_dinov3.pt matches this model definition."
            )
        if not _only_allowed(incompatible.missing_keys, _ALLOWED_MISSING_PREFIXES) or not _only_allowed(
            incompatible.unexpected_keys, _ALLOWED_UNEXPECTED_PREFIXES
        ):
            raise RuntimeError(
                "LAPA-DINOv3 checkpoint key mismatch: "
                f"missing={list(incompatible.missing_keys)[:10]}, "
                f"unexpected={list(incompatible.unexpected_keys)[:10]}"
            )
        return incompatible

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tokens.shape[0]
        h, w = self.patch_height_width
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)
        return tokens[:, :1], tokens[:, 1:]

    def forward(
        self,
        video: torch.Tensor,
        step: int = 0,
        mask: torch.Tensor | None = None,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ):
        del return_recons_only
        if video.ndim not in {4, 5}:
            raise ValueError(f"video must be 4D or 5D, got {tuple(video.shape)}")
        if video.ndim == 4:
            video = rearrange(video, "b c h w -> b c 1 h w")
            if exists(mask):
                raise ValueError("mask is not supported for image inputs")

        _b, _c, f, *image_dims = video.shape
        if tuple(image_dims) != self.image_size:
            raise ValueError(f"Expected image dims {self.image_size}, got {tuple(image_dims)}")
        if exists(mask) and mask.shape[-1] != f:
            raise ValueError(f"mask length {mask.shape[-1]} does not match frames {f}")

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
        first_frame_tokens = get_dinov3_reps(first_frame, self.dino_tokenizer)
        rest_frames_tokens = get_dinov3_reps(rest_frames, self.dino_tokenizer)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)
        first_tokens, last_tokens = self.encode(tokens)
        first_tokens, _ = pack([first_tokens], "b * d")
        last_tokens, _ = pack([last_tokens], "b * d")
        tokens, _perplexity, _codebook_usage, indices = self.vq(
            first_tokens,
            last_tokens,
            codebook_training_only=return_only_codebook_ids,
        )
        if (
            not return_only_codebook_ids
            and step != 0
            and ((step % 10 == 0 and step < 100) or (step % 100 == 0 and step < 1000))
        ):
            self.vq.codebooks_used[:] = 0
        if return_only_codebook_ids:
            return tokens, indices
        return tokens, indices
