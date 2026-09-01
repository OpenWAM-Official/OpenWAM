"""Joint self-attention support for the CosmosPredict25 video backbone (§17).

Two layers of coverage:

* **CPU** — a hand-rolled ``_RichCosmosBlock`` mirrors upstream ``Block.forward``
  (modulation → norm → flatten 5D→3D → ``compute_qkv`` → q_norm/k_norm/v_norm
  → joint attention → ``output_proj`` → residual → cross-attn → MLP). The split
  helpers are validated against this monolithic forward at ``allclose(atol=1e-5,
  rtol=1e-5)`` (float32), then the wrapper + adapter + MoT driver are exercised
  end-to-end on a 5D state to lock in the dispatch wiring.

* **GPU** — ``test_real_block_split_matches_monolithic`` loads the real
  CosmosPredict25-2B post-trained DiT, runs one block both ways (monolithic vs
  pre + attn_op + post), and asserts ``allclose(atol=1e-3)`` at bf16. This is
  the parity canary for any submodule pin bump — if upstream renames any
  attribute the test goes red.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------------------------------------------------
# Rich CPU fake — mirrors upstream Block.forward submodule-for-submodule.
# ----------------------------------------------------------------------


class _RichSelfAttn(nn.Module):
    """Mirror upstream ``Attention`` with ``is_selfattn=True`` (no RoPE here)."""

    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner, dim, bias=False)
        self.output_dropout = nn.Identity()

    def compute_qkv(self, x, context=None, rope_emb=None):
        # rope_emb is intentionally ignored on CPU fakes — we test the dispatch
        # contract; the real RoPE is gated by the GPU parity test.
        _ = rope_emb
        ctx = x if context is None else context
        q = rearrange(self.q_proj(x), "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
        k = rearrange(self.k_proj(ctx), "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
        v = rearrange(self.v_proj(ctx), "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)

    def attn_op(self, q_4d, k_4d, v_4d):
        """Mirror TE ``DotProductAttention`` output shape: ``(B, S, H·D)`` flat."""
        q = rearrange(q_4d, "b s h d -> b h s d")
        k = rearrange(k_4d, "b s h d -> b h s d")
        v = rearrange(v_4d, "b s h d -> b h s d")
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b h s d -> b s (h d)")

    def forward(self, x, context=None, rope_emb=None, video_size=None, kv_cache_cfg=None):
        # Reproduce upstream Attention.forward: compute_qkv → attn_op → output_proj.
        _ = video_size  # TE backend ignores
        _ = kv_cache_cfg  # not used at training time
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.output_dropout(self.output_proj(self.attn_op(q, k, v)))


class _RichCrossAttn(nn.Module):
    """Same submodules as upstream cross-attn Attention — no RoPE applied to cross-attn."""

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.output_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x, context, rope_emb=None):
        _ = rope_emb  # cross-attn never applies RoPE upstream
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads, d=self.head_dim)
        k = rearrange(self.k_proj(context), "b s (h d) -> b h s d", h=self.n_heads, d=self.head_dim)
        v = rearrange(self.v_proj(context), "b s (h d) -> b h s d", h=self.n_heads, d=self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.output_proj(out)


class _RichCosmosBlock(nn.Module):
    """Faithful CPU mirror of upstream ``minimal_v4_dit.Block`` — same submodule
    names, same forward call order. Used to validate the split helpers without
    requiring CUDA / TransformerEngine.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        n_heads: int,
        adaln_lora_dim: int = 32,
        use_adaln_lora: bool = True,
    ) -> None:
        super().__init__()
        head_dim = x_dim // n_heads
        self.x_dim = x_dim
        self.use_adaln_lora = use_adaln_lora

        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _RichSelfAttn(x_dim, n_heads, head_dim)
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = _RichCrossAttn(x_dim, context_dim, n_heads, head_dim)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(x_dim, 4 * x_dim), nn.GELU(), nn.Linear(4 * x_dim, x_dim))

        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def forward(
        self,
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        rope_emb_L_1_1_D=None,
        adaln_lora_B_T_3D=None,
        extra_per_block_pos_emb=None,
    ):
        # Faithful port of upstream ``Block.forward`` (use_wan_fp32_strategy=False).
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _x5d(t):
            return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_sa_5d, scale_sa_5d, gate_sa_5d = _x5d(shift_sa), _x5d(scale_sa), _x5d(gate_sa)
        shift_ca_5d, scale_ca_5d, gate_ca_5d = _x5d(shift_ca), _x5d(scale_ca), _x5d(gate_ca)
        shift_mlp_5d, scale_mlp_5d, gate_mlp_5d = _x5d(shift_mlp), _x5d(scale_mlp), _x5d(gate_mlp)

        _, T, H, W, _ = x_B_T_H_W_D.shape

        # Self-attn sublayer
        normed = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_sa_5d) + shift_sa_5d
        sa_flat = rearrange(normed, "b t h w d -> b (t h w) d")
        sa_out = self.self_attn(sa_flat, None, rope_emb=rope_emb_L_1_1_D)
        sa_out_5d = rearrange(sa_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa_5d * sa_out_5d

        # Cross-attn sublayer
        normed = self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_ca_5d) + shift_ca_5d
        ca_flat = rearrange(normed, "b t h w d -> b (t h w) d")
        ca_out = self.cross_attn(ca_flat, crossattn_emb, rope_emb=rope_emb_L_1_1_D)
        ca_out_5d = rearrange(ca_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = ca_out_5d * gate_ca_5d + x_B_T_H_W_D  # upstream order: result*gate + x

        # MLP sublayer
        normed = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp_5d) + shift_mlp_5d
        mlp_out = self.mlp(normed)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_5d * mlp_out
        return x_B_T_H_W_D


# ----------------------------------------------------------------------
# CPU: split helpers numerically reproduce the monolithic block.
# ----------------------------------------------------------------------


def _make_rich_block_inputs(*, B=1, T=2, H=2, W=3, D=16, ctx_dim=12, ctx_L=4):
    torch.manual_seed(123)
    x = torch.randn(B, T, H, W, D)
    emb = torch.randn(B, T, D)
    adaln_lora = torch.randn(B, T, 3 * D)
    ctx = torch.randn(B, ctx_L, ctx_dim)
    return x, emb, adaln_lora, ctx


def test_block_split_matches_monolithic_cpu():
    """``pre + (block's own attn_op) + post`` must reproduce ``block(...)`` exactly.

    Validates that the split helper extracts modulation, applies the right
    norm/scale/shift, and that ``post_self_attn`` correctly applies
    ``output_proj`` + residual + cross-attn + MLP. Run in float32 with
    ``atol=1e-5`` so the test catches arithmetic drift, not bf16 noise.
    """
    from openwam.model.video_backbone.cosmos_predict25.block_split import post_self_attn, pre_self_attn

    block = _RichCosmosBlock(x_dim=16, context_dim=12, n_heads=4)
    block.eval()
    x, emb, adaln_lora, ctx = _make_rich_block_inputs()

    # Monolithic
    y_mono = block(x, emb, ctx, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora)

    # Split
    q, k, v, post_state = pre_self_attn(block, x, emb, adaln_lora, rope_emb_L_1_1_D=None, extra_per_block_pos_emb=None)
    # Run the block's own attn_op (mirrors what compute_attention's interior does)
    q_4d = rearrange(q, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    k_4d = rearrange(k, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    v_4d = rearrange(v, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    attn_out = block.self_attn.attn_op(q_4d, k_4d, v_4d)  # (B, S, H·D)
    y_split = post_self_attn(attn_out, ctx, post_state)

    assert torch.allclose(y_mono, y_split, atol=1e-5, rtol=1e-5), (
        f"pre+attn_op+post must reproduce monolithic block (max abs diff = "
        f"{(y_mono - y_split).abs().max().item():.2e})."
    )


def test_block_split_pre_returns_3d_qkv_with_heads_inlined():
    """Pre half must return Q/K/V as ``(B, S, H·D)`` flat — that's the MoT contract."""
    from openwam.model.video_backbone.cosmos_predict25.block_split import pre_self_attn

    block = _RichCosmosBlock(x_dim=16, context_dim=12, n_heads=4)
    x, emb, adaln_lora, _ = _make_rich_block_inputs(B=2, T=2, H=2, W=3, D=16)

    q, k, v, post_state = pre_self_attn(block, x, emb, adaln_lora, rope_emb_L_1_1_D=None, extra_per_block_pos_emb=None)
    B = x.shape[0]
    S = 2 * 2 * 3  # T·H·W
    assert q.shape == (B, S, 16) and k.shape == (B, S, 16) and v.shape == (B, S, 16)
    assert post_state["T"] == 2 and post_state["H"] == 2 and post_state["W"] == 3
    # gate / shift / scale carry 5D broadcast layout
    assert post_state["gate_self_attn"].shape == (B, 2, 1, 1, 16)


# ----------------------------------------------------------------------
# CPU: wrapper + adapter dispatch through the new hooks.
# ----------------------------------------------------------------------


class _RichFakeMiniDIT(nn.Module):
    """Like ``_FakeMiniDIT`` in ``test_cosmos_predict25_pipeline_wrapper.py`` but with
    ``_RichCosmosBlock`` blocks so the split hooks have real submodules to call.
    """

    def __init__(self, *, dim=16, num_blocks=2, ctx_dim_post=12, ctx_dim_pre=24, n_heads=4):
        super().__init__()
        self.patch_spatial = 2
        self.patch_temporal = 1
        self.timestep_scale = 1.0
        self.concat_padding_mask = True
        self.use_crossattn_projection = True
        self.crossattn_proj_in_channels = ctx_dim_pre
        self.crossattn_proj = nn.Sequential(nn.Linear(ctx_dim_pre, ctx_dim_post, bias=True))
        self.t_embedder = _FakeTEmbedder(dim=dim, lora_dim=dim * 3)
        self.t_embedding_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [_RichCosmosBlock(x_dim=dim, context_dim=ctx_dim_post, n_heads=n_heads) for _ in range(num_blocks)]
        )
        self.final_layer = _FakeFinalLayer(dim=dim, out_per_patch=self.patch_spatial * self.patch_spatial * 16)
        self._patched_dim = dim
        self._out_channels = 16

    def prepare_embedded_sequence(self, x_B_C_T_H_W, *, fps=None, padding_mask=None):
        if self.concat_padding_mask:
            assert padding_mask is not None
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        B, _C, T, H, W = x_B_C_T_H_W.shape
        f = T // self.patch_temporal
        h = H // self.patch_spatial
        w = W // self.patch_spatial
        x_B_T_H_W_D = torch.zeros(B, f, h, w, self._patched_dim, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D[..., 0] = x_B_C_T_H_W[:, 0, :: self.patch_temporal, :: self.patch_spatial, :: self.patch_spatial]
        rope = torch.zeros(f * h * w, 1, 1, self._patched_dim, dtype=torch.float32, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, rope, None

    def unpatchify(self, x_B_T_H_W_O):
        B, f, h, w, _O = x_B_T_H_W_O.shape
        out = x_B_T_H_W_O.reshape(B, f, h, w, self._out_channels, self.patch_spatial, self.patch_spatial, 1)
        out = out.permute(0, 4, 1, 7, 2, 5, 3, 6).contiguous()
        out = out.reshape(B, self._out_channels, f, h * self.patch_spatial, w * self.patch_spatial)
        return out


class _FakeTEmbedder(nn.Module):
    def __init__(self, *, dim: int, lora_dim: int) -> None:
        super().__init__()
        self.t_proj = nn.Linear(1, dim, bias=False)
        self.adaln_proj = nn.Linear(1, lora_dim, bias=False)

    def forward(self, timesteps_B_T):
        t_f = timesteps_B_T.to(torch.float32).unsqueeze(-1)
        return self.t_proj(t_f), self.adaln_proj(t_f)


class _FakeFinalLayer(nn.Module):
    def __init__(self, *, dim: int, out_per_patch: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, out_per_patch, bias=False)

    def forward(self, x_B_T_H_W_D, t_embedding_B_T_D, *, adaln_lora_B_T_3D=None):
        _ = t_embedding_B_T_D
        _ = adaln_lora_B_T_3D
        return self.linear(x_B_T_H_W_D)


def _build_rich_wrapper(num_blocks=2):
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    net = _RichFakeMiniDIT(dim=16, num_blocks=num_blocks, n_heads=4, ctx_dim_post=12, ctx_dim_pre=24)
    return CosmosPredict25VideoBackbone(
        net=net,
        vae=None,
        text_encoder=None,
        dim=16,
        num_layers=num_blocks,
        num_heads=4,
        head_dim=4,
        context_dim=12,
        shift_video=5.0,
    )


def test_wrapper_pre_post_attn_dispatch_round_trip():
    """``pre_attn_at_layer → block.self_attn.attn_op → post_attn_at_layer`` must
    produce the same 5D state as ``run_block`` (monolithic).
    """
    wrapper = _build_rich_wrapper(num_blocks=1)
    latents = torch.randn(1, 16, 2, 4, 4)
    context = torch.randn(1, 4, 12)
    timestep = torch.randint(0, 1000, (1,))

    state_mono = wrapper.prepare(input_latents=latents, context=context, timestep=timestep)
    state_split = wrapper.prepare(input_latents=latents, context=context, timestep=timestep)

    state_mono = wrapper.run_block(0, state_mono)

    q, k, v, post_state = wrapper.pre_attn_at_layer(0, state_split)
    block = wrapper.dit.blocks[0]
    q_4d = rearrange(q, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    k_4d = rearrange(k, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    v_4d = rearrange(v, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    attn_out = block.self_attn.attn_op(q_4d, k_4d, v_4d)
    state_split = wrapper.post_attn_at_layer(0, state_split, attn_out, post_state)

    assert torch.allclose(state_mono.hidden_states, state_split.hidden_states, atol=1e-5, rtol=1e-5)


def test_adapter_pre_post_attn_delegate_to_wrapper():
    """``CosmosPredict25VideoBackbone.pre_attn_at_layer`` must round-trip through the wrapper."""
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    wrapper = _build_rich_wrapper(num_blocks=1)
    backbone = CosmosPredict25VideoBackbone(
        net=wrapper.dit, dim=16, num_layers=1, num_heads=4, head_dim=4, context_dim=12, freeze=False
    )
    state = wrapper.prepare(
        input_latents=torch.randn(1, 16, 2, 4, 4),
        context=torch.randn(1, 4, 12),
        timestep=torch.randint(0, 1000, (1,)),
    )
    q, k, v, post_state = backbone.pre_attn_at_layer(0, state)
    # Q/K/V shape matches the driver contract: (B, T·H·W, H·D)
    assert q.shape == (1, 2 * 2 * 2, 16) and k.shape == q.shape and v.shape == q.shape
    block = wrapper.dit.blocks[0]
    q_4d = rearrange(q, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    k_4d = rearrange(k, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    v_4d = rearrange(v, "b s (h d) -> b s h d", h=block.self_attn.n_heads, d=block.self_attn.head_dim)
    attn_out = block.self_attn.attn_op(q_4d, k_4d, v_4d)
    state = backbone.post_attn_at_layer(0, state, attn_out, post_state)
    assert state.hidden_states.shape == (1, 2, 2, 2, 16)
    assert torch.isfinite(state.hidden_states).all()


# ----------------------------------------------------------------------
# CPU: v↔v mask modes on CosmosPredict25VideoBackbone.
# ----------------------------------------------------------------------


def test_v2v_mask_bidirectional():
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    backbone = CosmosPredict25VideoBackbone(
        net=_build_rich_wrapper(num_blocks=1).dit,
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
        freeze=False,
    )
    backbone.video_attention_mask_mode = "bidirectional"
    mask = backbone.build_video_to_video_mask(video_seq_len=8, video_tokens_per_frame=4, device=torch.device("cpu"))
    assert mask.shape == (8, 8)
    assert mask.all()


def test_v2v_mask_per_frame_causal():
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    backbone = CosmosPredict25VideoBackbone(
        net=_build_rich_wrapper(num_blocks=1).dit,
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
        freeze=False,
    )
    backbone.video_attention_mask_mode = "per_frame_causal"
    # 2 frames × 4 tokens-per-frame = 8 total. Frame 0 sees only itself,
    # frame 1 sees both — block-diagonal lower triangular at frame granularity.
    mask = backbone.build_video_to_video_mask(video_seq_len=8, video_tokens_per_frame=4, device=torch.device("cpu"))
    assert mask.shape == (8, 8)
    # First-frame queries (rows 0-3) attend only to first-frame keys (cols 0-3)
    assert mask[:4, :4].all() and not mask[:4, 4:].any()
    # Second-frame queries (rows 4-7) attend to both
    assert mask[4:, :].all()


def test_v2v_mask_first_frame_causal():
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    backbone = CosmosPredict25VideoBackbone(
        net=_build_rich_wrapper(num_blocks=1).dit,
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
        freeze=False,
    )
    backbone.video_attention_mask_mode = "first_frame_causal"
    mask = backbone.build_video_to_video_mask(video_seq_len=8, video_tokens_per_frame=4, device=torch.device("cpu"))
    # First-frame rows only see first-frame cols
    assert mask[:4, :4].all() and not mask[:4, 4:].any()
    # Later rows see everything
    assert mask[4:, :].all()


def test_v2v_mask_default_is_bidirectional():
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    backbone = CosmosPredict25VideoBackbone(
        net=_build_rich_wrapper(num_blocks=1).dit,
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
        freeze=False,
    )
    # Property defaults to bidirectional without explicit setter.
    assert backbone.video_attention_mask_mode == "bidirectional"


def test_v2v_mask_rejects_unknown_mode():
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    backbone = CosmosPredict25VideoBackbone(
        net=_build_rich_wrapper(num_blocks=1).dit,
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
        freeze=False,
    )
    backbone.video_attention_mask_mode = "bogus_mode"
    with pytest.raises(ValueError, match="bogus_mode"):
        backbone.build_video_to_video_mask(video_seq_len=4, video_tokens_per_frame=2, device=torch.device("cpu"))


# ----------------------------------------------------------------------
# CPU: MoT driver runs end-to-end through CosmosPredict25VideoBackbone with the
# `_RichCosmosBlock` fakes — locks in the s_video = f*h*w fix on the driver
# (state.hidden_states is 5D for CosmosPredict25, so the old `vstate.hidden_states.shape[1]` would have
# returned T instead of T*H*W).
# ----------------------------------------------------------------------


def test_mot_driver_runs_through_cosmos_predict25_5d_state():
    """End-to-end: build DualSystemSelfAttnArchitecture pointing at a
    CosmosPredict25VideoBackbone backed by ``_RichCosmosBlock``, and run one full
    layer of joint mixed attention. This regression-tests the
    ``mot_driver.py::run_joint_loop`` s_video fix (formerly
    ``vstate.hidden_states.shape[1]`` which gives ``T`` for Cosmos's 5D state).
    """
    from openwam.model import build_architecture
    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    wrapper = _build_rich_wrapper(num_blocks=2)
    backbone = CosmosPredict25VideoBackbone(
        net=wrapper.dit, dim=16, num_layers=2, num_heads=4, head_dim=4, context_dim=12, freeze=False
    )

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 16,
        "ffn_dim": 32,
        "num_heads": 4,
        "attn_head_dim": 4,
        "video_dim": 16,
        "text_dim": 12,
        "bridge_layers": (0, 1),
        # This repo's cross-modal taxonomy is mutual/action_sees_video/... —
        # `mutual` is the bidirectional-equivalent (the reference used the
        # now-renamed 'bidirectional').
        "attention_mask_mode": "mutual",
    }
    arch = build_architecture("dual_system_self_attn", cfg)
    arch.video_backbone = backbone
    driver = arch.build_mot_driver()
    arch.eval()

    state = wrapper.prepare(
        input_latents=torch.randn(1, 16, 2, 4, 4),
        context=torch.randn(1, 4, 12),
        timestep=torch.randint(0, 1000, (1,)),
    )
    # state.hidden_states is 5D: (1, 2, 2, 2, 16); f*h*w = 2*2*2 = 8.
    actions = torch.randn(1, 3, 7)
    astate = arch.action_backbone.prepare_state(
        actions, torch.tensor([0.5]), context=torch.randn(1, 4, 12), context_mask=torch.ones(1, 4, dtype=torch.bool)
    )

    vstate, astate = driver.run_joint_loop(state, astate)

    # After the joint loop state.hidden_states should still be a finite 5D tensor of the
    # original shape — invariant for the post-attn unflatten.
    assert vstate.hidden_states.shape == (1, 2, 2, 2, 16)
    assert torch.isfinite(vstate.hidden_states).all()


@pytest.mark.gpu
def test_real_block_split_matches_monolithic(stub_reason1):
    """Numerical parity on a real CosmosPredict25-2B block: ``pre + attn_op + post``
    must reproduce ``block(...)`` within bf16 noise.

    This is the canary for any submodule pin bump: if upstream renames any
    Block submodule (``adaln_modulation_*``, ``self_attn.{compute_qkv,
    output_proj, output_dropout, attn_op}``, ``cross_attn``, ``mlp``,
    ``layer_norm_*``) the assertion goes red — that's the signal to update
    ``block_split.py`` for the new layout.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA + CosmosPredict25 install")
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError:
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos_predict25.sh first.")

    from openwam.model.video_backbone.cosmos_predict25.block_split import post_self_attn, pre_self_attn
    from openwam.model.video_backbone.cosmos_predict25.pipeline_builder import build_cosmos_predict25_pipeline

    cfg = {
        "video_backbone": {
            "model_path": "/path/to/assets/Cosmos-Predict2.5-2B",
            "model_variant": "base/post-trained",
            "text_encoder_path": "/stub",
            "vae": "none",  # parity test doesn't need VAE
            "shift_video": 5.0,
            "sac_mode": "none",
        }
    }
    try:
        wrapper = build_cosmos_predict25_pipeline(cfg, device="cuda:0")
    except FileNotFoundError as exc:
        pytest.skip(f"Cosmos checkpoint missing: {exc}")

    # ``build_cosmos_predict25_pipeline`` returns a lightweight SimpleNamespace
    # holder exposing the DiT as ``.net`` (not ``.dit`` — that attribute only
    # exists on the drained CosmosPredict25VideoBackbone).
    block = wrapper.net.blocks[0].eval()
    device = next(block.parameters()).device
    dtype = next(block.parameters()).dtype

    B, T, H, W = 1, 2, 4, 4
    D = wrapper.dim
    torch.manual_seed(42)
    x = torch.randn(B, T, H, W, D, dtype=dtype, device=device)
    emb = torch.randn(B, T, D, dtype=dtype, device=device)
    adaln_lora = torch.randn(B, T, 3 * D, dtype=dtype, device=device)
    ctx = torch.randn(B, 8, wrapper.context_dim, dtype=dtype, device=device)
    # RoPE: pull it from the actual pos_embedder by running prepare_embedded_sequence
    # over a fake latent of matching geometry.
    latent = torch.randn(B, 16, T, H * 2, W * 2, dtype=dtype, device=device)  # net adds +1 channel
    cond_mask = torch.zeros(B, 1, T, H * 2, W * 2, dtype=dtype, device=device)
    pad_mask = torch.zeros(B, 1, H * 2, W * 2, dtype=dtype, device=device)
    x_in = torch.cat([latent, cond_mask], dim=1)
    _x5d, rope_emb_L_1_1_D, _extra = wrapper.net.prepare_embedded_sequence(x_in, fps=None, padding_mask=pad_mask)

    with torch.no_grad():
        y_mono = block(
            x.clone(),
            emb,
            ctx,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora,
            extra_per_block_pos_emb=None,
        )

        q, k, v, post_state = pre_self_attn(
            block, x.clone(), emb, adaln_lora, rope_emb_L_1_1_D=rope_emb_L_1_1_D, extra_per_block_pos_emb=None
        )
        n_heads = block.self_attn.n_heads
        head_dim = block.self_attn.head_dim
        q_4d = rearrange(q, "b s (h d) -> b s h d", h=n_heads, d=head_dim)
        k_4d = rearrange(k, "b s (h d) -> b s h d", h=n_heads, d=head_dim)
        v_4d = rearrange(v, "b s (h d) -> b s h d", h=n_heads, d=head_dim)
        attn_out = block.self_attn.attn_op(q_4d, k_4d, v_4d)  # (B, S, H·D)
        y_split = post_self_attn(attn_out, ctx, post_state)

    diff = (y_mono - y_split).abs().max().item()
    assert torch.allclose(y_mono, y_split, atol=1e-3, rtol=5e-3), (
        f"Real Cosmos block split must match monolithic within bf16 noise; max abs diff = {diff:.3e}. "
        "If upstream `Block.forward` changed, update `block_split.py` and re-run."
    )
