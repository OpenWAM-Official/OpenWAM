"""Map heterogeneous per-robot action vectors into a unified action space.

Different robots emit different raw action layouts (Wuji 54-D joint, OXE
single-arm EEF, ...). To train one model over a
shared action head, each dataset declares — in its yaml — which raw dims map to
which slots of a single unified space of width ``UNIFY_DIM``. This module is the
pure-mechanism layer for that:

  * :func:`parse_unify_spec`   — yaml spec → a ``dst_index`` array
  * :func:`map_to_unify`       — scatter raw action into the unified vector + mask
  * :func:`unmap_from_unify`   — gather the unified vector back to raw dims (deploy)

It is intentionally **schema-agnostic**: it does NOT know what each unified slot
*means* (that semantic layout lives elsewhere). ``unify_dim`` is passed in.
Everything is numpy with no side effects and no torch dependency, so it is
reusable both in the dataloader and in deployment post-processing.

Spec forms (both accepted by :func:`parse_unify_spec`)
------------------------------------------------------
* **single list** — source dims are implicitly ``0..N-1`` in order; each token
  is the destination (unified) slot(s) for the next source dim(s)::

      ["0-8", 31, "33-40"]
      # raw 0..8 -> unified 0..8 ; raw 9 -> unified 31 ; raw 10..17 -> unified 33..40
      # dst_index = [0,1,2,3,4,5,6,7,8, 31, 33,34,35,36,37,38,39,40]   (N = 18)

* **paired list** — explicit ``[src_range, dst_range]`` pairs; does not assume
  contiguous source dims::

      [["0-8", "0-8"], [9, 31], ["10-17", "33-40"]]
      # same dst_index as above, but src dims are stated explicitly

A token is a closed range string ``"a-b"``, a single int, or a list mixing
those. The two forms are told apart by whether the top-level elements are
lists/tuples (paired) or not (single).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

# Width of the unified action space. The *semantic* layout of these slots
# (which slot is which robot's which joint/EEF dim) is defined separately by the
# project; this module treats the width purely as a size and takes ``unify_dim``
# as an explicit argument everywhere. This constant is just the default.
UNIFY_DIM = 80

# A single spec token: "a-b" range, a bare int, or a (possibly nested) list of those.
Token = Union[int, str, Sequence]


def _expand_ranges(tokens: Token) -> List[int]:
    """Expand a token (or list of tokens) into a flat list of int indices.

    Accepts ``"a-b"`` closed ranges, single ints, and lists mixing them::

        _expand_ranges("0-8")            -> [0,1,2,3,4,5,6,7,8]
        _expand_ranges(31)               -> [31]
        _expand_ranges(["0-2", 5, "7-8"])-> [0,1,2,5,7,8]
    """
    # Normalize to a flat iterable of "atoms" (int or "a-b" str).
    if isinstance(tokens, (list, tuple)):
        atoms = tokens
    else:
        atoms = [tokens]

    out: List[int] = []
    for atom in atoms:
        if isinstance(atom, (int, np.integer)):
            out.append(int(atom))
        elif isinstance(atom, str):
            s = atom.strip()
            if "-" in s:
                a_str, b_str = s.split("-", 1)
                try:
                    a, b = int(a_str), int(b_str)
                except ValueError as e:
                    raise ValueError(f"unify spec: malformed range token {atom!r}") from e
                if b < a:
                    raise ValueError(f"unify spec: range {atom!r} has end < start")
                out.extend(range(a, b + 1))  # closed range [a, b]
            else:
                try:
                    out.append(int(s))
                except ValueError as e:
                    raise ValueError(f"unify spec: malformed index token {atom!r}") from e
        else:
            raise ValueError(f"unify spec: unsupported token type {type(atom).__name__}: {atom!r}")
    return out


def _coerce_spec(spec):
    """Normalize an OmegaConf ``ListConfig`` to a plain (possibly nested) list.

    Dataloader configs reach the readers as OmegaConf containers —
    ``scripts/train.py`` passes ``cfg.dataloader`` straight into
    ``build_dataset`` without ``to_container`` — so ``unify_action_map`` arrives
    as a ``ListConfig``, which is **not** a ``list``/``tuple`` and would be
    rejected below. Convert recursively (also unwraps the nested paired form).
    No-op for plain lists or when OmegaConf isn't installed.
    """
    try:
        from omegaconf import ListConfig, OmegaConf
    except Exception:
        return spec
    if isinstance(spec, ListConfig):
        return OmegaConf.to_container(spec, resolve=True)
    return spec


def _is_paired(spec: Sequence) -> bool:
    """Paired form iff every top-level element is itself a list/tuple."""
    return all(isinstance(el, (list, tuple)) for el in spec)


def parse_unify_spec(spec: Sequence, unify_dim: int = UNIFY_DIM) -> np.ndarray:
    """Parse a yaml unify spec into a ``dst_index`` array.

    Returns ``dst_index`` of shape ``(N,)`` int, where ``dst_index[i]`` is the
    unified slot that raw action dim ``i`` maps to (``N`` = raw action width).
    Both :func:`map_to_unify` and :func:`unmap_from_unify` are driven by this
    array, which guarantees they are exact inverses.

    Args:
        spec: single-list or paired-list spec (see module docstring).
        unify_dim: width of the unified space (dst indices must be in range).

    Raises:
        ValueError: on any malformed / out-of-range / overlapping / non-covering
            spec, with a message pointing at the offending part.
    """
    spec = _coerce_spec(spec)  # OmegaConf ListConfig (Hydra-loaded yaml) -> plain list
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError(f"unify spec must be a non-empty list, got {spec!r}")

    if _is_paired(spec):
        # Paired: [[src_range, dst_range], ...] — explicit src->dst.
        src_to_dst: dict[int, int] = {}
        for pair in spec:
            if len(pair) != 2:
                raise ValueError(f"unify spec: paired entry must be [src, dst], got {pair!r}")
            src = _expand_ranges(pair[0])
            dst = _expand_ranges(pair[1])
            if len(src) != len(dst):
                raise ValueError(
                    f"unify spec: src/dst length mismatch in {pair!r} (src has {len(src)}, dst has {len(dst)})"
                )
            for s, d in zip(src, dst):
                if s in src_to_dst:
                    raise ValueError(f"unify spec: source dim {s} mapped more than once")
                src_to_dst[s] = d
        src_dims = sorted(src_to_dst)
        # src must cover 0..N-1 with no holes.
        expected = list(range(len(src_dims)))
        if src_dims != expected:
            raise ValueError(
                f"unify spec: source dims must cover 0..N-1 with no gaps; got {src_dims} (expected {expected})"
            )
        dst_index = np.array([src_to_dst[s] for s in src_dims], dtype=np.int64)
    else:
        # Single: source dims implied 0..N-1, in token order.
        dst_index = np.array(_expand_ranges(list(spec)), dtype=np.int64)

    # Shared validation: dst in range, no duplicate destinations.
    if dst_index.size == 0:
        raise ValueError("unify spec: parsed to an empty mapping")
    if dst_index.min() < 0 or dst_index.max() >= unify_dim:
        bad = dst_index[(dst_index < 0) | (dst_index >= unify_dim)]
        raise ValueError(
            f"unify spec: destination index/indices {bad.tolist()} out of range "
            f"[0, {unify_dim}); check unify_dim or the spec."
        )
    uniq, counts = np.unique(dst_index, return_counts=True)
    if np.any(counts > 1):
        dup = uniq[counts > 1].tolist()
        raise ValueError(f"unify spec: destination slot(s) {dup} mapped from more than one source dim")

    return dst_index


def map_to_unify(
    action: np.ndarray, dst_index: np.ndarray, unify_dim: int = UNIFY_DIM
) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter a raw action into the unified space and build its dim mask.

    Args:
        action: ``(..., N)`` raw action; ``N`` must equal ``len(dst_index)``.
            Any leading dims (time, batch) are preserved.
        dst_index: ``(N,)`` from :func:`parse_unify_spec`.
        unify_dim: width of the unified space.

    Returns:
        ``(unified, dim_mask)`` where:
          * ``unified`` is ``(..., unify_dim)`` with ``unified[..., dst_index] =
            action`` and all other slots 0 (same dtype as ``action``).
          * ``dim_mask`` is ``(unify_dim,)`` bool, True exactly at ``dst_index``.
            This is the per-dim validity mask; the time axis is handled
            elsewhere (e.g. ``build_action_mask_2d``).
    """
    action = np.asarray(action)
    n = dst_index.shape[0]
    if action.shape[-1] != n:
        raise ValueError(
            f"map_to_unify: action last dim {action.shape[-1]} != mapping size {n} (action shape {action.shape})"
        )
    unified = np.zeros((*action.shape[:-1], unify_dim), dtype=action.dtype)
    unified[..., dst_index] = action  # scatter along the last axis
    dim_mask = np.zeros(unify_dim, dtype=bool)
    dim_mask[dst_index] = True
    return unified, dim_mask


def unmap_from_unify(unified: np.ndarray, dst_index: np.ndarray) -> np.ndarray:
    """Gather the unified vector back to the raw action dims (inverse of map).

    Args:
        unified: ``(..., unify_dim)``.
        dst_index: ``(N,)`` from :func:`parse_unify_spec`.

    Returns:
        ``action`` of shape ``(..., N)``, where ``action[..., i] =
        unified[..., dst_index[i]]``. Exact inverse of :func:`map_to_unify`'s
        first return:  ``unmap_from_unify(map_to_unify(a, idx, U)[0], idx) == a``.
    """
    unified = np.asarray(unified)
    return unified[..., dst_index]  # gather along the last axis


__all__ = ["UNIFY_DIM", "parse_unify_spec", "map_to_unify", "unmap_from_unify"]
