"""Pure-math primitives for SANA-style linear attention under arbitrary masks.

This module hosts the math that a future :class:`SanaMoTJointDriver`
(plans/sana_mot_integration_plan.md §3) substitutes for SDPA's softmax in
:meth:`MoTJointDriver._mixed_attention`. It is intentionally **driver-agnostic**
and free of OpenWAM type imports so the same primitive can be reused by
inference paths, unit tests, and ablation scripts.

Layout convention — matches plan §1.1 / §2.2:

- ``tilde_q``, ``tilde_k``: ``(B, H, N, d)`` — rotated, kernel-applied (ReLU).
- ``phi_q``, ``phi_k``:     ``(B, H, N, d)`` — kernel-applied (ReLU), **NOT**
  rotated. They carry the denominator's row-sum and must keep the original
  (un-RoPE'd) values so the dual-track normalization that SANA pretrained
  with stays intact (see plans/sana_mot_math_conflict.md §1).
- ``v``:                    ``(B, H, N, d)``.
- Output:                   ``(B, H, N, d)``.

SANA's fused per-block forward uses ``(B, h, h_d, N)`` storage and the
identity

    out_i = (v @ tilde_k.T @ tilde_q)_i / (phi_k.sum(N).T @ phi_q + eps)_i

to compute the result in ``O(N · d²)`` time. :func:`_expanded_linear_attn`
below computes the same quantity in the ``O(N²)`` mask-aware form — the
slow but unambiguous reference that Phase 2's cumsum implementation will
be checked against. The two are mathematically identical when ``mask`` is
``None`` (Phase 1.2 test); for masked sequences only the expanded form
is correct because the fused form folds N out of existence before the
mask is applied.

The default ``eps=1e-15`` matches ``LiteLAReLURope.__init__`` upstream
(third_party/Sana/diffusion/model/nets/sana_blocks.py:317). The driver
should pass through ``self_attn.eps`` rather than rely on this default,
so a future pin-bump that changes upstream's default doesn't silently
shift the numerical contract.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.utils.checkpoint
from torch import Tensor


def _expanded_linear_attn(
    tilde_q: Tensor,
    tilde_k: Tensor,
    v: Tensor,
    phi_q: Tensor,
    phi_k: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-15,
) -> Tensor:
    """``O(N²)`` mask-aware reference for SANA's dual-track linear attention.

    Computes, per (batch, head, query position ``i``):

        out_i = (Σ_j  M_{ij} · (tilde_q_i · tilde_k_j) · v_j)
              / (Σ_j  M_{ij} · (phi_q_i  · phi_k_j) + eps)

    All inputs must be ``(B, H, N, d)`` and share the same dtype / device.

    Parameters
    ----------
    tilde_q, tilde_k
        Rotated, kernel-applied Q/K. Used for the numerator's per-pair score.
    v
        Values to be aggregated.
    phi_q, phi_k
        Kernel-applied (ReLU'd) Q/K **without** RoPE, used for the denominator.
        SANA's pretrained "two-track" formulation diverges from a standard
        linear attention here: it needs the un-rotated K to keep the
        normalization positive (rotated K can be negative after the complex
        rotation), see plans/sana_mot_math_conflict.md §1.
    mask
        Optional ``(N, N)`` or broadcastable bool/float tensor. ``True`` /
        nonzero means "the query at row ``i`` is allowed to see the key at
        column ``j``". ``None`` is equivalent to a fully-True mask.
    eps
        Denominator epsilon. The driver should pass ``LiteLAReLURope.eps``
        (currently ``1e-15`` upstream).

    Returns
    -------
    Tensor
        ``(B, H, N, d)``, same dtype as inputs.

    Notes
    -----
    Allocates two ``(B, H, N, N)`` matrices — quadratic in sequence length.
    Intended for tests and as a fallback when :func:`_mask_to_chunk_index`
    (Phase 2) cannot reduce the mask to a monotonic block layout.
    """
    if tilde_q.dim() != 4 or tilde_k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "_expanded_linear_attn expects (B, H, N, d) inputs; got "
            f"tilde_q={tuple(tilde_q.shape)}, tilde_k={tuple(tilde_k.shape)}, v={tuple(v.shape)}."
        )
    if phi_q.shape != tilde_q.shape or phi_k.shape != tilde_k.shape:
        raise ValueError(
            "phi_q/phi_k must match tilde_q/tilde_k shapes; got "
            f"phi_q={tuple(phi_q.shape)}, tilde_q={tuple(tilde_q.shape)}, "
            f"phi_k={tuple(phi_k.shape)}, tilde_k={tuple(tilde_k.shape)}."
        )

    a = tilde_q @ tilde_k.transpose(-1, -2)  # (B, H, N, N)
    b = phi_q @ phi_k.transpose(-1, -2)  # (B, H, N, N)

    if mask is not None:
        m = mask.to(a.dtype) if mask.dtype != a.dtype else mask
        a = a * m
        b = b * m

    num = a @ v  # (B, H, N, d)
    denom = b.sum(dim=-1, keepdim=True) + eps  # (B, H, N, 1)
    return num / denom


def _mask_to_chunk_index(mask: Tensor) -> Optional[List[int]]:
    """Translate an ``(N, N)`` bool mask into a monotonic block-causal chunk list.

    A *monotonic block-causal* mask of ``N`` tokens with chunks of sizes
    ``[s_0, s_1, ..., s_{C-1}]`` satisfies

        M[i, j] == 1   iff   chunk(j) <= chunk(i)

    where ``chunk(i)`` is the chunk that token ``i`` belongs to. All three
    ``video_attention_mask_mode`` values that OpenWAM ships
    (``bidirectional`` / ``first_frame_causal`` / ``per_frame_causal``, see
    :meth:`WanVideoBackbone.build_video_to_video_mask`) compose with the
    FastWAM-Joint ``[Sv+Sa, Sv+Sa]`` topology — action↔action True,
    action→video True, video→action False — to yield such a layout, so the
    chunked path covers the realistic joint-attention training surface.

    Parameters
    ----------
    mask
        Bool / integer / float ``(N, N)`` tensor. Nonzero = "attend to".

    Returns
    -------
    List[int]
        Ascending chunk boundaries ``[0, s_0, s_0 + s_1, ..., N]`` of length
        ``C + 1`` if ``mask`` factorizes; ``None`` if it doesn't. ``None``
        is the driver's signal to fall back to the ``O(N²)`` expanded path.

    Notes
    -----
    Validation is strict by design — a malformed mask must not silently route
    to the cumsum implementation, since the cumsum result would be wrong
    only at masked positions and almost certainly slip past coarse-grained
    integration tests. Concretely we check:

    1. Each row is exactly ``[0, j_max[i]]`` True and the rest False
       (rectangularity).
    2. ``j_max`` is non-decreasing in row order (monotonicity).
    3. Run-length encoding of ``j_max`` produces consistent square chunks
       — for the unified-chunk cumsum to be correct, ``j_max + 1`` for the
       run of rows in chunk ``c`` must equal the cumulative chunk size up
       to and including ``c``.

    Empty rows (no visible key) are rejected: they'd require a degenerate
    chunk and the cumsum path can't represent them. The caller should
    handle them on the expanded path.
    """
    if not isinstance(mask, Tensor):
        return None
    if mask.dim() != 2:
        return None
    n_rows, n_cols = mask.shape
    if n_rows != n_cols:
        return None
    if n_rows == 0:
        return [0]
    n = int(n_rows)

    mask_bool = mask.bool() if mask.dtype != torch.bool else mask
    arange = torch.arange(n, device=mask_bool.device, dtype=torch.long)

    # j_max[i] = last True col in row i, or -1 if no True col. Process in
    # row chunks to bound peak memory — at N ≈ 32k (81 frames × 480p joint
    # mask) the naive ``arange.unsqueeze(0).expand(n, n)`` would allocate
    # an 8 GB int64 intermediate. CHUNK_ROWS=4096 caps peak at ~1 GB.
    CHUNK_ROWS = 4096
    j_max = torch.empty(n, dtype=torch.long, device=mask_bool.device)
    rectangular = True
    for start in range(0, n, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n)
        rows = mask_bool[start:end]  # (chunk, N)
        # last True col: flip+argmax. argmax returns first occurrence; on
        # the flipped row that's the last True col in the original.
        flipped_int = rows.flip(-1).to(torch.int8)
        first_true_in_flipped = flipped_int.argmax(dim=-1).to(torch.long)
        any_true = rows.any(dim=-1)
        chunk_j_max = torch.where(any_true, n - 1 - first_true_in_flipped, torch.full_like(first_true_in_flipped, -1))
        j_max[start:end] = chunk_j_max
        # Rectangularity check on this row slice.
        if rectangular:
            expected = arange.unsqueeze(0) <= chunk_j_max.unsqueeze(-1)
            if not torch.equal(rows, expected):
                rectangular = False

    if not rectangular:
        return None
    if (j_max < 0).any():
        return None

    # Monotonicity of j_max in row order.
    if not bool((j_max[1:] >= j_max[:-1]).all().item()):
        return None

    # Run-length encode j_max and validate consistency with square-chunk layout.
    j_max_cpu = j_max.cpu().tolist()
    chunk_index: List[int] = [0]
    run_start = 0
    while run_start < n:
        run_val = j_max_cpu[run_start]
        run_end = run_start
        while run_end < n and j_max_cpu[run_end] == run_val:
            run_end += 1
        run_len = run_end - run_start
        expected_chunk_end = run_val + 1  # because rows in chunk c see cols [0, V_c] ⇒ chunk_index[c+1] = V_c + 1
        expected_run_len = expected_chunk_end - chunk_index[-1]
        if run_len != expected_run_len:
            return None
        chunk_index.append(expected_chunk_end)
        run_start = run_end

    if chunk_index[-1] != n:
        return None

    return chunk_index


def _chunked_linear_attn(
    tilde_q: Tensor,
    tilde_k: Tensor,
    v: Tensor,
    phi_q: Tensor,
    phi_k: Tensor,
    chunk_index: List[int],
    eps: float = 1e-15,
) -> Tensor:
    """``O(N · d²)`` cumsum implementation for monotonic block-causal masks.

    Equivalent to :func:`_expanded_linear_attn` when ``mask`` factorizes into
    ``chunk_index``. The "state" matrices

        S = Σ_{j in chunks 0..c} v_j ⊗ tilde_k_j      ∈ R^{d×d}
        z = Σ_{j in chunks 0..c} phi_k_j              ∈ R^{d}

    are accumulated across chunks; each chunk's queries are applied to the
    state *after* this chunk's keys have been added (so queries see chunks
    ``0..c``, matching the block-causal semantics). Storage stays at
    ``O(B · H · d²)`` regardless of ``N`` — that's the whole point of the
    cumsum form versus the expanded ``(B, H, N, N)`` matrix.

    Parameters
    ----------
    chunk_index
        Ascending boundaries from :func:`_mask_to_chunk_index`. Length
        ``C + 1``; ``chunk_index[0] == 0`` and ``chunk_index[-1] == N``.
    eps
        Denominator epsilon, matching :func:`_expanded_linear_attn`.

    Returns
    -------
    Tensor
        ``(B, H, N, d)``, same dtype / device as inputs.

    Notes
    -----
    The per-chunk loop is in Python — vectorizing across chunks loses the
    monotonic-block savings (cumsum cannot start "from the future"). For
    OpenWAM's typical 3-chunk first-frame-causal mask this loop runs three
    times per layer per forward; for the 81-frame per-frame-causal fallback
    it runs 82 times. The plan's risk register §7 notes the per-frame mode
    as a likely future fused-kernel target.
    """
    if tilde_q.dim() != 4 or tilde_k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "_chunked_linear_attn expects (B, H, N, d) inputs; got "
            f"tilde_q={tuple(tilde_q.shape)}, tilde_k={tuple(tilde_k.shape)}, v={tuple(v.shape)}."
        )
    if len(chunk_index) < 2:
        raise ValueError(f"chunk_index needs at least 2 entries (got {chunk_index}).")
    if chunk_index[0] != 0 or chunk_index[-1] != tilde_q.shape[-2]:
        raise ValueError(f"chunk_index must start at 0 and end at N={tilde_q.shape[-2]}; got {chunk_index}.")

    B, H, N, d = tilde_q.shape
    out = torch.empty_like(v)

    s_mat = torch.zeros(B, H, d, d, dtype=v.dtype, device=v.device)
    z_vec = torch.zeros(B, H, 1, d, dtype=v.dtype, device=v.device)

    for c in range(len(chunk_index) - 1):
        i_start, i_end = int(chunk_index[c]), int(chunk_index[c + 1])
        if i_start == i_end:
            continue  # empty chunk — degenerate but harmless
        if i_start > i_end:
            raise ValueError(f"chunk_index not ascending: chunk_index[{c}]={i_start} > chunk_index[{c + 1}]={i_end}.")

        tilde_k_c = tilde_k[:, :, i_start:i_end, :]
        v_c = v[:, :, i_start:i_end, :]
        phi_k_c = phi_k[:, :, i_start:i_end, :]
        tilde_q_c = tilde_q[:, :, i_start:i_end, :]
        phi_q_c = phi_q[:, :, i_start:i_end, :]

        # S_ab += Σ_j v_c[j, a] tilde_k_c[j, b]
        s_mat = s_mat + v_c.transpose(-1, -2) @ tilde_k_c  # (B, H, d, d)
        z_vec = z_vec + phi_k_c.sum(dim=-2, keepdim=True)  # (B, H, 1, d)

        # num[i, a] = Σ_b S[a, b] tilde_q_c[i, b]  =  (tilde_q_c @ S.T)[i, a]
        num = tilde_q_c @ s_mat.transpose(-1, -2)  # (B, H, s_c, d)
        denom = (phi_q_c @ z_vec.transpose(-1, -2)) + eps  # (B, H, s_c, 1)
        out[:, :, i_start:i_end, :] = num / denom

    return out


def _chunked_linear_attn_checkpointed(
    tilde_q: Tensor,
    tilde_k: Tensor,
    v: Tensor,
    phi_q: Tensor,
    phi_k: Tensor,
    chunk_index: List[int],
    eps: float = 1e-15,
) -> Tensor:
    """Per-chunk gradient-checkpointed variant of :func:`_chunked_linear_attn`.

    Numerically identical to :func:`_chunked_linear_attn` (same dtype, same eps),
    but each chunk's forward is wrapped in
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` so the
    ``(B, H, s_c, d)`` per-chunk activations (``num``, ``denom``, intermediate
    matmul outputs) are not kept around for backward — instead they are
    recomputed per-chunk.

    The cumulative state matrices ``S`` (``B, H, d, d``) and ``z``
    (``B, H, 1, d``) are passed through checkpoint boundaries as outputs.
    They are far smaller than ``(B, H, N, d)`` so holding them across the
    forward loop is what allows backward to recompute each chunk
    independently of the others.

    Parameters
    ----------
    Same as :func:`_chunked_linear_attn`.

    Returns
    -------
    Tensor
        ``(B, H, N, d)``, same dtype / device as inputs.

    Notes
    -----
    This is the path the SanaMoTJointDriver takes when
    ``mot_checkpoint_mixed_attn=True`` and the mask factorizes
    (plan/sana_mot_integration_plan.md §A, user decision: per-chunk checkpoint).
    The non-factorizing fallback wraps :func:`_expanded_linear_attn` at the
    full-tensor granularity in the driver instead — there's no useful sub-chunk
    structure to checkpoint over.

    ``eps`` is converted to a 0-dim tensor before crossing the checkpoint
    boundary so that ``use_reentrant=False`` does not emit a warning about
    non-tensor inputs being passed through autograd.
    """
    if tilde_q.dim() != 4 or tilde_k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "_chunked_linear_attn_checkpointed expects (B, H, N, d) inputs; got "
            f"tilde_q={tuple(tilde_q.shape)}, tilde_k={tuple(tilde_k.shape)}, v={tuple(v.shape)}."
        )
    if len(chunk_index) < 2:
        raise ValueError(f"chunk_index needs at least 2 entries (got {chunk_index}).")
    if chunk_index[0] != 0 or chunk_index[-1] != tilde_q.shape[-2]:
        raise ValueError(
            f"chunk_index must start at 0 and end at N={tilde_q.shape[-2]}; got {chunk_index}."
        )

    B, H, N, d = tilde_q.shape
    out_chunks: List[Tensor] = []

    s_mat = torch.zeros(B, H, d, d, dtype=v.dtype, device=v.device)
    z_vec = torch.zeros(B, H, 1, d, dtype=v.dtype, device=v.device)
    eps_t = torch.tensor(eps, dtype=v.dtype, device=v.device)

    def _chunk_step(tq_c, pq_c, tk_c, v_c, pk_c, s_in, z_in, eps_in):
        s_new = s_in + v_c.transpose(-1, -2) @ tk_c
        z_new = z_in + pk_c.sum(dim=-2, keepdim=True)
        num = tq_c @ s_new.transpose(-1, -2)
        denom = (pq_c @ z_new.transpose(-1, -2)) + eps_in
        return num / denom, s_new, z_new

    for c in range(len(chunk_index) - 1):
        i_start, i_end = int(chunk_index[c]), int(chunk_index[c + 1])
        if i_start == i_end:
            out_chunks.append(v.new_empty(B, H, 0, d))
            continue
        if i_start > i_end:
            raise ValueError(
                f"chunk_index not ascending: chunk_index[{c}]={i_start} > chunk_index[{c + 1}]={i_end}."
            )

        args = (
            tilde_q[:, :, i_start:i_end, :],
            phi_q[:, :, i_start:i_end, :],
            tilde_k[:, :, i_start:i_end, :],
            v[:, :, i_start:i_end, :],
            phi_k[:, :, i_start:i_end, :],
            s_mat,
            z_vec,
            eps_t,
        )
        out_c, s_mat, z_vec = torch.utils.checkpoint.checkpoint(
            _chunk_step, *args, use_reentrant=False
        )
        out_chunks.append(out_c)

    return torch.cat(out_chunks, dim=2)


__all__ = [
    "_expanded_linear_attn",
    "_mask_to_chunk_index",
    "_chunked_linear_attn",
    "_chunked_linear_attn_checkpointed",
]
