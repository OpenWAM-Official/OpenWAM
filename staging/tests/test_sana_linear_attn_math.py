"""Phase 1 + Phase 2 math equivalence tests for SANA-style linear attention.

Covers acceptance criteria in plans/sana_mot_integration_plan.md §1 + §2:

Phase 1 (``_expanded_linear_attn`` — the ``O(N²)`` reference):
- (1) expanded form ≡ SANA's fused ``O(N · d²)`` form when ``mask is None``.
- (2) ``mask=all-1s`` is equivalent to ``mask=None``.
- (3) Under a lower-triangular (causal) mask, perturbing the LAST key/value
      does not change the FIRST query's output.
- (4) Numerical agreement with SANA's actual upstream ``LiteLAReLURope``
      forward — gated on ``third_party/Sana`` importable AND CUDA available.

Phase 2 (``_mask_to_chunk_index`` + ``_chunked_linear_attn`` — the ``O(N · d²)``
fast path for monotonic block-causal masks):
- (5) ``_chunked_linear_attn`` output equals ``_expanded_linear_attn`` output
      under random monotonic-block masks.
- (6) ``_mask_to_chunk_index`` recognizes the three OpenWAM joint-mask
      topologies (``bidirectional``, ``first_frame_causal``, ``per_frame_causal``).
- (7) Non-monotonic / non-rectangular masks fall back to ``None`` so the
      driver re-routes to the expanded path instead of producing wrong output
      silently.

CPU-only tests have no SANA dependency — they pin the OpenWAM-side math
independently of upstream so a SANA pin-bump cannot mask a regression.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.architectures.dual_system.sana_linear_attn import (
    _chunked_linear_attn,
    _chunked_linear_attn_checkpointed,
    _expanded_linear_attn,
    _mask_to_chunk_index,
)

# ---------------------------------------------------------------------------
# SANA availability probe (mirrors tests/test_sana_backbone_smoke.py)
# ---------------------------------------------------------------------------


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


SANA_AVAILABLE = _sana_importable()
requires_sana = pytest.mark.skipif(
    not SANA_AVAILABLE,
    reason="third_party/Sana not importable (submodule init or timm version.py missing)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_qkv(B: int, H: int, N: int, d: int, *, dtype=torch.float32, device="cpu", seed: int = 0):
    """Synthetic ReLU'd Q/K/V — emulates ``block_pre_attn`` outputs.

    Returns ``(tilde_q, tilde_k, v, phi_q, phi_k)`` all shaped ``(B, H, N, d)``.

    ``tilde_*`` are the "post-RoPE" Q/K used in the numerator. ``phi_*`` are
    the "pre-RoPE" Q/K used in the denominator (still ReLU'd, hence
    non-negative — which is what keeps the denominator strictly positive,
    see plans/sana_mot_math_conflict.md §1). For the math equivalence tests
    we don't actually need RoPE — we just need the numerator-vs-denominator
    inputs to be **different but consistent**.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    phi_q = torch.randn(B, H, N, d, dtype=dtype, device=device, generator=g).relu_()
    phi_k = torch.randn(B, H, N, d, dtype=dtype, device=device, generator=g).relu_()
    v = torch.randn(B, H, N, d, dtype=dtype, device=device, generator=g)
    # Simulate RoPE by mixing the last two dims with a deterministic rotation.
    # The exact form doesn't matter for the math — we just need ``tilde_*`` to
    # differ from ``phi_*`` so a bug that confuses the two tracks would be
    # caught.
    theta = torch.linspace(0, 3.14159, d, device=device, dtype=dtype)
    cos, sin = theta.cos(), theta.sin()
    tilde_q = phi_q * cos + phi_q.roll(1, dims=-1) * sin
    tilde_k = phi_k * cos + phi_k.roll(1, dims=-1) * sin
    return tilde_q, tilde_k, v, phi_q, phi_k


def _fused_no_mask(
    tilde_q: torch.Tensor,
    tilde_k: torch.Tensor,
    v: torch.Tensor,
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``O(N · d²)`` fused form — mirrors ``LiteLAReLURope.forward`` math.

    Inputs in ``(B, H, N, d)``. See the docstring of ``_expanded_linear_attn``
    for the identity that makes this equal to the expanded form when
    ``mask is None``.
    """
    s_mat = v.transpose(-1, -2) @ tilde_k  # (B, H, d, d): Σ_j v_j ⊗ tilde_k_j
    num = tilde_q @ s_mat.transpose(-1, -2)  # (B, H, N, d): Σ_j (tilde_q_i · tilde_k_j) v_j
    # Σ_j phi_q_i · phi_k_j = phi_q_i · (Σ_j phi_k_j)
    z_vec = phi_k.sum(dim=-2, keepdim=True)  # (B, H, 1, d)
    denom = (phi_q @ z_vec.transpose(-1, -2)) + eps  # (B, H, N, 1)
    return num / denom


# ---------------------------------------------------------------------------
# CPU-only — no SANA dependency
# ---------------------------------------------------------------------------


def test_expanded_eq_fused_no_mask():
    """Expanded ``O(N²)`` form matches the fused ``O(N · d²)`` form (mask=None)."""
    B, H, N, d = 2, 4, 32, 16
    eps = 1e-15
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d, dtype=torch.float64, seed=42)

    out_expanded = _expanded_linear_attn(tq, tk, v, pq, pk, mask=None, eps=eps)
    out_fused = _fused_no_mask(tq, tk, v, pq, pk, eps=eps)

    torch.testing.assert_close(out_expanded, out_fused, rtol=1e-10, atol=1e-12)


def test_expanded_all_ones_mask_eq_no_mask():
    """``mask=ones`` and ``mask=None`` must yield identical output."""
    B, H, N, d = 1, 2, 16, 8
    eps = 1e-15
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d, dtype=torch.float64, seed=7)
    mask_ones = torch.ones(N, N, dtype=torch.bool)

    out_none = _expanded_linear_attn(tq, tk, v, pq, pk, mask=None, eps=eps)
    out_ones = _expanded_linear_attn(tq, tk, v, pq, pk, mask=mask_ones, eps=eps)

    torch.testing.assert_close(out_none, out_ones, rtol=1e-10, atol=1e-12)


def test_expanded_diagonal_mask_has_closed_form():
    """Identity mask reduces to a per-token closed form ``(tq·tk)·v / (pq·pk + eps)``.

    Pins the meaning of the mask: ``mask[i, j]`` gates the ``(i, j)`` pair
    in BOTH the numerator and the denominator. If the mask were applied to
    only one of the two tracks, this assertion would fail.
    """
    B, H, N, d = 1, 1, 12, 8
    eps = 1e-15
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d, dtype=torch.float64, seed=3)
    eye = torch.eye(N, dtype=torch.bool)

    out = _expanded_linear_attn(tq, tk, v, pq, pk, mask=eye, eps=eps)

    score = (tq * tk).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
    norm = (pq * pk).sum(dim=-1, keepdim=True) + eps  # (B, H, N, 1)
    expected = score * v / norm

    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-12)


def test_expanded_lower_triangular_causality():
    """Under a causal mask, perturbing the LAST K/V doesn't change the FIRST query's output.

    Verifies the row-wise locality property that Phase 2's cumsum
    chunked implementation depends on: ``out_i`` depends only on keys
    ``j ≤ i`` (when the mask is lower-triangular).
    """
    B, H, N, d = 1, 1, 16, 8
    eps = 1e-15
    causal = torch.tril(torch.ones(N, N, dtype=torch.bool))
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d, dtype=torch.float64, seed=11)

    out1 = _expanded_linear_attn(tq, tk, v, pq, pk, mask=causal, eps=eps)

    # Perturb token N-1 only in v / phi_k / tilde_k (it's the "future" — first
    # N-1 queries cannot see it under causal mask).
    v2 = v.clone()
    pk2 = pk.clone()
    tk2 = tk.clone()
    v2[..., -1, :] += torch.randn(d, dtype=v.dtype, generator=torch.Generator().manual_seed(99))
    pk2[..., -1, :] += torch.randn(d, dtype=v.dtype, generator=torch.Generator().manual_seed(100)).abs()
    tk2[..., -1, :] += torch.randn(d, dtype=v.dtype, generator=torch.Generator().manual_seed(101))

    out2 = _expanded_linear_attn(tq, tk2, v2, pq, pk2, mask=causal, eps=eps)

    # First N-1 queries unchanged.
    torch.testing.assert_close(out1[..., :-1, :], out2[..., :-1, :], rtol=1e-12, atol=1e-14)
    # Last query: not asserted — it CAN see the perturbed last token, so it
    # legitimately changes. This is a positive assertion that the mask
    # actually plumbs the data through somewhere.
    assert not torch.allclose(out1[..., -1, :], out2[..., -1, :])


def test_expanded_strict_lower_triangular_forbids_self_attention():
    """Strict lower-tri ``j < i`` mask: query 0 has empty key set ⇒ output = 0 / eps = 0.

    Sanity check that the denominator's ``+ eps`` keeps the math finite when
    no key is visible — and that the result is well-defined (not NaN / Inf).
    """
    B, H, N, d = 1, 1, 8, 4
    eps = 1e-15
    strict = torch.tril(torch.ones(N, N, dtype=torch.bool), diagonal=-1)
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d, dtype=torch.float64, seed=5)

    out = _expanded_linear_attn(tq, tk, v, pq, pk, mask=strict, eps=eps)

    # Query 0 has no visible keys — numerator is 0, denominator is eps, output is 0.
    torch.testing.assert_close(out[..., 0, :], torch.zeros_like(out[..., 0, :]))
    assert torch.isfinite(out).all()


def test_expanded_accepts_float_mask():
    """Soft masks (float in [0, 1]) are accepted and behave as expected.

    Not strictly part of OpenWAM's contract (the joint mask is bool), but
    the helper should not silently reject a soft mask — Phase 5 may use
    fractional gating for ablations.
    """
    B, H, N, d = 1, 1, 8, 4
    eps = 1e-15
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d, dtype=torch.float64, seed=2)

    out_bool = _expanded_linear_attn(tq, tk, v, pq, pk, mask=torch.ones(N, N, dtype=torch.bool), eps=eps)
    out_float = _expanded_linear_attn(tq, tk, v, pq, pk, mask=torch.ones(N, N, dtype=torch.float64), eps=eps)
    torch.testing.assert_close(out_bool, out_float, rtol=1e-12, atol=1e-14)


def test_expanded_shape_validation():
    """Wrong-rank inputs raise ``ValueError`` — guards against (B, S, H*D) being passed."""
    B, H, N, d = 1, 2, 8, 4
    tq, tk, v, pq, pk = _make_random_qkv(B, H, N, d)
    # Collapse to MoT layout (B, S, H*D) — driver layout, not expanded layout.
    tq_mot = tq.transpose(1, 2).reshape(B, N, H * d)
    with pytest.raises(ValueError, match="expects .B, H, N, d. inputs"):
        _expanded_linear_attn(tq_mot, tk, v, pq, pk)


# ---------------------------------------------------------------------------
# SANA + CUDA — numerical agreement with upstream
# ---------------------------------------------------------------------------


@requires_sana
def test_expanded_eq_sana_native_attn():
    """``_expanded_linear_attn`` ≡ ``SanaMSVideoSplit.native_attn`` when mask=None.

    This is the cross-check between OpenWAM's reference implementation and
    SANA's actual fused forward — pinning that the math derivation in
    plans/sana_mot_math_conflict.md §1 is the right one to expand.

    Mask is ``None`` here: the fused form is only equivalent to the
    expanded form for the un-masked case. Once a non-trivial joint mask is
    in play, ``native_attn`` is wrong by construction and the driver must
    use the expanded / cumsum path.
    """
    from einops import rearrange

    from openwam.model.video_backbone.sana import SanaVideoBackbone

    if not torch.cuda.is_available():
        pytest.skip(
            "Numerical equivalence requires CUDA — fp32 CPU paths in SANA's "
            "depth_conv can disagree from the cuda kernel by >rtol."
        )

    f, h, w = 4, 8, 8
    bb = SanaVideoBackbone.from_mini_config(
        depth=1,
        hidden_size=128,
        num_heads=4,
        linear_head_dim=32,
        f=f,
        h=h,
        w=w,
        device="cuda",
        dtype=torch.float32,
    )
    B = 1
    x = torch.randn(B, 16, f, h, w, device="cuda", dtype=torch.float32)
    timestep = torch.tensor([100], device="cuda")
    y = torch.randn(B, 1, 8, 64, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, 1, 1, 8, dtype=torch.int16, device="cuda")

    state = bb.prepare(x=x, timestep=timestep, y=y, mask=mask)
    split = state.extras["split"]

    with torch.no_grad():
        # Native fused path — uses ``attn.eps`` internally.
        q_mot, k_mot, v_mot, post = split.block_pre_attn(0, state.x, state.t_mod, state.freqs)
        out_native_mot = split.native_attn(post)  # (B, S, H*D)

        # Expanded path — pivot MoT layout to (B, H, N, d).
        H = bb.num_heads
        tq = rearrange(q_mot, "b s (h d) -> b h s d", h=H)
        tk = rearrange(k_mot, "b s (h d) -> b h s d", h=H)
        vv = rearrange(v_mot, "b s (h d) -> b h s d", h=H)
        pq = rearrange(post["q_unrot"], "b s (h d) -> b h s d", h=H)
        pk = rearrange(post["k_unrot"], "b s (h d) -> b h s d", h=H)

        attn_eps = bb._dit.blocks[0].attn.eps
        out_expanded_bhnd = _expanded_linear_attn(tq, tk, vv, pq, pk, mask=None, eps=attn_eps)
        out_expanded_mot = rearrange(out_expanded_bhnd, "b h s d -> b s (h d)")

    torch.testing.assert_close(out_expanded_mot, out_native_mot, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Phase 2 — _mask_to_chunk_index + _chunked_linear_attn
# ---------------------------------------------------------------------------


def _build_monotonic_mask(chunk_index: list[int]) -> torch.Tensor:
    """Construct the ``(N, N)`` bool mask defined by ``chunk_index``.

    ``M[i, j] = True`` iff ``chunk(j) <= chunk(i)``. The inverse of
    :func:`_mask_to_chunk_index` — used by tests to feed a known-good mask
    into the translator and check round-trip identity.
    """
    n = chunk_index[-1]
    mask = torch.zeros(n, n, dtype=torch.bool)
    for c in range(len(chunk_index) - 1):
        i_start, i_end = chunk_index[c], chunk_index[c + 1]
        # Queries in chunk c see chunks 0..c, i.e. cols [0, i_end).
        mask[i_start:i_end, :i_end] = True
    return mask


def _build_joint_mask_synthetic(s_video: int, s_action: int, video_tokens_per_frame: int, mode: str) -> torch.Tensor:
    """Inline the FastWAM-Joint mask construction for testing.

    Mirrors ``MoTJointDriver._build_joint_mask`` composed with
    ``WanVideoBackbone.build_video_to_video_mask`` for each ``mode``.
    Kept here as a test-only helper so the chunk-translator tests don't
    depend on instantiating a full backbone + driver.
    """
    n = s_video + s_action
    mask = torch.zeros(n, n, dtype=torch.bool)

    if mode == "bidirectional":
        mask[:s_video, :s_video] = True
    elif mode == "first_frame_causal":
        mask[:s_video, :s_video] = True
        first_frame = min(video_tokens_per_frame, s_video)
        # First-frame rows do NOT see beyond the first frame's keys.
        mask[:first_frame, first_frame:s_video] = False
    elif mode == "per_frame_causal":
        if s_video % video_tokens_per_frame != 0:
            raise ValueError("s_video must be divisible by video_tokens_per_frame for per_frame_causal")
        num_frames = s_video // video_tokens_per_frame
        frame_causal = torch.tril(torch.ones(num_frames, num_frames, dtype=torch.bool))
        mask[:s_video, :s_video] = frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
            video_tokens_per_frame, dim=1
        )
    else:
        raise ValueError(f"unknown mode '{mode}'")

    mask[s_video:, s_video:] = True
    mask[s_video:, :s_video] = True
    # video → action stays False (the FastWAM-Joint "no peek" rule).
    return mask


def test_mask_to_chunk_index_bidirectional():
    """``bidirectional`` joint mask → 2 chunks: ``[0, Sv, Sv+Sa]``."""
    s_video, s_action, vt = 12, 5, 4
    mask = _build_joint_mask_synthetic(s_video, s_action, vt, "bidirectional")
    ci = _mask_to_chunk_index(mask)
    assert ci == [0, s_video, s_video + s_action]


def test_mask_to_chunk_index_first_frame_causal():
    """``first_frame_causal`` joint mask → 3 chunks: ``[0, s, Sv, Sv+Sa]``.

    The OpenWAM default and the topology the SANA path is most likely to
    train under (see plans/sana_mot_integration_plan.md §1 closing example).
    """
    vt = 4  # video_tokens_per_frame
    f = 3  # num video frames
    s_video = vt * f  # 12
    s_action = 5
    mask = _build_joint_mask_synthetic(s_video, s_action, vt, "first_frame_causal")
    ci = _mask_to_chunk_index(mask)
    assert ci == [0, vt, s_video, s_video + s_action]


def test_mask_to_chunk_index_per_frame_causal():
    """``per_frame_causal`` joint mask → ``F + 1`` chunks (per-frame + action)."""
    vt = 4
    f = 3
    s_video = vt * f  # 12
    s_action = 5
    mask = _build_joint_mask_synthetic(s_video, s_action, vt, "per_frame_causal")
    ci = _mask_to_chunk_index(mask)
    # Chunks: each frame is a chunk (size vt), action is the final chunk.
    expected = [0, vt, 2 * vt, 3 * vt, s_video + s_action]
    assert ci == expected


def test_mask_to_chunk_index_round_trip():
    """``_build_monotonic_mask`` ∘ ``_mask_to_chunk_index`` is the identity on
    well-formed chunk lists."""
    for chunk_index in [
        [0, 5],
        [0, 3, 8],
        [0, 4, 10, 15],
        [0, 1, 2, 3, 4, 5],
    ]:
        mask = _build_monotonic_mask(chunk_index)
        ci = _mask_to_chunk_index(mask)
        assert ci == chunk_index, f"round-trip failed for {chunk_index}: got {ci}"


def test_mask_to_chunk_index_rejects_video_to_action_leak():
    """If video→action becomes True, the mask is no longer block-causal.

    Verifies the dispatcher's safety net: a malformed joint mask (one that
    silently grows a v→a edge) is rejected, so the chunked path doesn't
    silently miscompute.
    """
    s_video, s_action, vt = 8, 4, 4
    mask = _build_joint_mask_synthetic(s_video, s_action, vt, "first_frame_causal")
    # Inject a video → action edge that breaks rectangularity at row 0.
    mask[0, s_video] = True
    assert _mask_to_chunk_index(mask) is None


def test_mask_to_chunk_index_rejects_non_monotonic():
    """Random non-monotonic mask returns ``None`` — the driver fallback signal."""
    n = 8
    rng = torch.Generator().manual_seed(0)
    mask = (torch.rand(n, n, generator=rng) > 0.5).bool()
    # With high probability this is neither rectangular nor monotonic.
    assert _mask_to_chunk_index(mask) is None


def test_mask_to_chunk_index_rejects_non_square_shape():
    """Non-square or non-2D masks return ``None`` instead of asserting."""
    assert _mask_to_chunk_index(torch.ones(4, 5, dtype=torch.bool)) is None
    assert _mask_to_chunk_index(torch.ones(4, dtype=torch.bool)) is None
    assert _mask_to_chunk_index(torch.ones(2, 4, 4, dtype=torch.bool)) is None


def test_mask_to_chunk_index_rejects_empty_row():
    """A row with no visible keys is rejected — the cumsum path can't represent it."""
    mask = _build_monotonic_mask([0, 4, 8])
    mask[0, :] = False  # row 0 is now all-False
    assert _mask_to_chunk_index(mask) is None


def test_mask_to_chunk_index_accepts_full_mask():
    """``mask == ones`` collapses to a single chunk covering all tokens."""
    n = 6
    ci = _mask_to_chunk_index(torch.ones(n, n, dtype=torch.bool))
    assert ci == [0, n]


def test_chunked_eq_expanded_random_monotonic():
    """``_chunked_linear_attn`` matches ``_expanded_linear_attn`` on random
    monotonic masks — Phase 2's headline equivalence.

    Tests several chunk shapes that mirror realistic joint-attention layouts
    (bidirectional → 2 chunks; first_frame_causal → 3; per_frame_causal → many).
    """
    B, H, d = 2, 3, 16
    eps = 1e-15
    for chunk_index in [
        [0, 10, 22],  # bidirectional-ish
        [0, 4, 16, 22],  # first_frame_causal-ish
        [0, 4, 8, 12, 16, 20, 22],  # per_frame_causal-ish
        [0, 22],  # all-one chunk
    ]:
        n = chunk_index[-1]
        mask = _build_monotonic_mask(chunk_index)
        tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d, dtype=torch.float64, seed=sum(chunk_index))

        out_expanded = _expanded_linear_attn(tq, tk, v, pq, pk, mask=mask, eps=eps)
        out_chunked = _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=chunk_index, eps=eps)

        torch.testing.assert_close(
            out_chunked,
            out_expanded,
            rtol=1e-10,
            atol=1e-12,
            msg=lambda m, ci=chunk_index: f"chunk_index={ci}: {m}",
        )


def test_chunked_eq_expanded_on_joint_mask_first_frame_causal():
    """End-to-end Phase 2: joint mask → chunk_index → cumsum, vs expanded.

    Verifies the path the driver will take: take a realistic
    ``first_frame_causal`` joint mask, translate it, and confirm the
    chunked output equals the expanded output. This is the strongest
    integration-style check that the math primitives + the translator
    compose correctly.
    """
    vt = 4
    s_video = vt * 4  # 16 video tokens
    s_action = 6
    n = s_video + s_action
    mask = _build_joint_mask_synthetic(s_video, s_action, vt, "first_frame_causal")

    ci = _mask_to_chunk_index(mask)
    assert ci is not None, "first_frame_causal mask should factorize"

    B, H, d = 1, 4, 8
    eps = 1e-15
    tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d, dtype=torch.float64, seed=2026)

    out_expanded = _expanded_linear_attn(tq, tk, v, pq, pk, mask=mask, eps=eps)
    out_chunked = _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=ci, eps=eps)

    torch.testing.assert_close(out_chunked, out_expanded, rtol=1e-10, atol=1e-12)


def test_chunked_rejects_bad_chunk_index():
    """``_chunked_linear_attn`` validates its ``chunk_index`` argument."""
    B, H, n, d = 1, 1, 8, 4
    tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d)

    with pytest.raises(ValueError, match="at least 2 entries"):
        _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=[0])
    with pytest.raises(ValueError, match="must start at 0 and end at N"):
        _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=[1, 8])
    with pytest.raises(ValueError, match="must start at 0 and end at N"):
        _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=[0, 7])


def test_chunked_checkpointed_eq_chunked_forward():
    """``_chunked_linear_attn_checkpointed`` forward equals the non-checkpointed
    version across realistic chunk layouts — Phase 3 per-chunk gradient
    checkpoint variant must not change numerical output.
    """
    B, H, d = 2, 3, 16
    eps = 1e-15
    for chunk_index in [
        [0, 10, 22],
        [0, 4, 16, 22],
        [0, 4, 8, 12, 16, 20, 22],
        [0, 22],
        [0, 3, 3, 8],  # with an empty chunk in the middle
    ]:
        n = chunk_index[-1]
        tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d, dtype=torch.float64, seed=sum(chunk_index))

        out_plain = _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=chunk_index, eps=eps)
        out_ckpt = _chunked_linear_attn_checkpointed(tq, tk, v, pq, pk, chunk_index=chunk_index, eps=eps)

        torch.testing.assert_close(
            out_ckpt,
            out_plain,
            rtol=1e-12,
            atol=1e-14,
            msg=lambda m, ci=chunk_index: f"chunk_index={ci}: {m}",
        )


def test_chunked_checkpointed_grad_eq_chunked():
    """Backward through the checkpointed variant produces gradients identical
    to the non-checkpointed variant.

    This is the load-bearing correctness check — a per-chunk checkpoint that
    drops the wrong activations would silently corrupt gradients. We compute
    grads for both paths on a fresh-leaves copy of the inputs and require
    elementwise equality.
    """
    B, H, d = 1, 2, 8
    chunk_index = [0, 4, 10, 14]
    n = chunk_index[-1]
    eps = 1e-15

    tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d, dtype=torch.float64, seed=2026)

    inputs_plain = [t.detach().clone().requires_grad_(True) for t in (tq, tk, v, pq, pk)]
    inputs_ckpt = [t.detach().clone().requires_grad_(True) for t in (tq, tk, v, pq, pk)]

    out_plain = _chunked_linear_attn(*inputs_plain, chunk_index=chunk_index, eps=eps)
    out_ckpt = _chunked_linear_attn_checkpointed(*inputs_ckpt, chunk_index=chunk_index, eps=eps)

    torch.testing.assert_close(out_ckpt, out_plain, rtol=1e-12, atol=1e-14)

    # Use a deterministic upstream grad so the comparison is reproducible.
    grad_gen = torch.Generator().manual_seed(7)
    grad_out = torch.randn(out_plain.shape, dtype=out_plain.dtype, generator=grad_gen)
    out_plain.backward(grad_out)
    out_ckpt.backward(grad_out)

    for name, ip, ic in zip(("tq", "tk", "v", "pq", "pk"), inputs_plain, inputs_ckpt):
        torch.testing.assert_close(
            ic.grad,
            ip.grad,
            rtol=1e-10,
            atol=1e-12,
            msg=lambda m, n=name: f"grad mismatch on {n}: {m}",
        )


def test_chunked_checkpointed_rejects_bad_chunk_index():
    """The checkpointed variant inherits the same chunk_index validation."""
    B, H, n, d = 1, 1, 8, 4
    tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d)

    with pytest.raises(ValueError, match="at least 2 entries"):
        _chunked_linear_attn_checkpointed(tq, tk, v, pq, pk, chunk_index=[0])
    with pytest.raises(ValueError, match="must start at 0 and end at N"):
        _chunked_linear_attn_checkpointed(tq, tk, v, pq, pk, chunk_index=[1, 8])


def test_chunked_handles_empty_chunks():
    """An empty chunk (``chunk_index[c] == chunk_index[c+1]``) is skipped cleanly.

    Lets the driver compose chunk lists from variable-length subsequences
    without special-casing the boundary where some subsequence is empty
    (e.g. action-free inference batches).
    """
    chunk_index = [0, 3, 3, 8]  # middle chunk is empty
    B, H, n, d = 1, 2, 8, 4
    tq, tk, v, pq, pk = _make_random_qkv(B, H, n, d, dtype=torch.float64, seed=1)
    mask = _build_monotonic_mask(chunk_index)

    out_chunked = _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index=chunk_index, eps=1e-15)
    out_expanded = _expanded_linear_attn(tq, tk, v, pq, pk, mask=mask, eps=1e-15)
    torch.testing.assert_close(out_chunked, out_expanded, rtol=1e-10, atol=1e-12)
