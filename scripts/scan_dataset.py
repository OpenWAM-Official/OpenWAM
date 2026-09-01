#!/usr/bin/env python3
"""Full-load verification / bad-episode scanner for OpenWAM dataloaders.

Motivation
----------
Training reads samples through ``LeRobotV3Reader._safe_get``, which retries a
failing window up to 64× by advancing the index +1. With ``window_stride=1`` the
adjacent windows come from the *same* video file, so a single truncated /
undecodable clip that backs ≥64 consecutive windows defeats every retry, the
DataLoader worker dies, and training crashes (``MixtureDataset.__getitem__`` has
no guard). The sanctioned remedy is the reader's episode blacklist (normally
``meta/excluded_episodes.json``), but that needs a full end-to-end scan — which
this tool performs for active public LeRobot v3 readers.

Two phases
----------
Phase 1 — ``scan`` (per dataset): call the leaf reader's ``_getitem_impl``
directly, bypassing ``_safe_get`` so failures are recorded rather than retried
away. The default ``--mode episode`` checks the last/highest-frame window of
every episode. A literal every-window pass requires the explicit
``--mode window --sample-stride 1``; it is the deepest check, but is normally a
long-running on large datasets. Failures are grouped by episode and can be
published with the separate ``emit`` command.

Phase 2 — ``verify-mixed``: after exclusions are written, build the real
``mixture`` and iterate the whole mixed index through the *real* path (with
``_safe_get``) under a tolerant wrapper, confirming zero unrecoverable failures
remain — i.e. training will not crash on a full mixed load.

Parallelism / robustness
-------------------------
The load path is **CPU-decode + disk-IO bound** (PyAV), not GPU-bound. Parallelize
with ``--num-workers`` (a torch DataLoader forks that many decode workers sharing
the dataset copy-on-write). For multi-process / multi-rank fan-out, run several
copies with ``--shard i --num-shards N`` (disjoint window ranges); merge with
``--emit-exclusions`` afterwards. Failures + progress are written incrementally
so a disconnect (run under **tmux**) never loses work.

Examples
--------
  # Phase 1 fast pass: one last/highest-frame window per episode (the default)
  python scripts/scan_dataset.py scan --config agibotworld  --num-workers 64 --out-dir scan_out
  python scripts/scan_dataset.py scan --config robocoin      --num-workers 64 --out-dir scan_out
  python scripts/scan_dataset.py scan --config interndata_a1 --num-workers 64 --out-dir scan_out

  # Strict pass: every original window, with no _safe_get retry masking
  python scripts/scan_dataset.py scan --config robocoin --mode window \
      --sample-stride 1 --num-workers 64 --out-dir scan_out_strict

  # Merge failures -> write exclusion files (meta/excluded_episodes.json etc.)
  python scripts/scan_dataset.py emit --config agibotworld  --out-dir scan_out
  python scripts/scan_dataset.py emit --config robocoin      --out-dir scan_out
  python scripts/scan_dataset.py emit --config interndata_a1 --out-dir scan_out

  # Phase 2: confirm the mixed load is clean after exclusions
  python scripts/scan_dataset.py verify-mixed --config mixture --num-workers 64 --out-dir scan_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openwam.dataloader.utils.exclusion_io import atomic_publish_text, locked_exclusion_files  # noqa: E402

# ---------------------------------------------------------------------------
# Config + build
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The exclusion files are consumed by the training-path readers with a bare
    ``json.load`` — an interrupted plain write would leave a truncated JSON that
    crashes dataset construction on every later run. tmp+replace makes the
    reader always see either the old or the fully-written new content. Keep this
    wrapper for ``gpu_decode_scan`` and other importers; exclusion transactions
    additionally take the shared sidecar lock in :func:`cmd_emit`.
    """
    atomic_publish_text(path, text)


def _build_dataset(config_name: str):
    """Compose ``configs/dataloader/<config_name>.yaml`` and build the dataset."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from openwam.dataloader.registry import build_dataset

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=f"dataloader/{config_name}")
    dl = cfg.dataloader
    if OmegaConf.select(dl, "seed", default=None) is None:
        OmegaConf.update(dl, "seed", 42, force_add=True)
    return build_dataset(dl, split="train")


def _peek_config_type(config_name: str) -> Optional[str]:
    """Return the ``type`` of a dataloader config WITHOUT building it.

    ``scan`` is a per-dataset operation — it scans one dataset's leaves and
    writes THAT dataset's exclusion file — so a ``mixture`` config is rejected
    up front (scan each source separately; ``verify-mixed`` covers the mixture).
    Compose is cheap; building the full mixture is not — this lets ``cmd_scan``
    reject in milliseconds instead of after the (very slow) full build.
    (``_enumerate_leaves`` itself CAN unpack a mixture — the emit/re-probe path
    relies on that — the restriction here is scan's per-dataset contract, not a
    mechanical limitation.)
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=f"dataloader/{config_name}")
    return OmegaConf.select(cfg.dataloader, "type", default=None)


# ---------------------------------------------------------------------------
# Leaf adapters (episode key + exclusion target per reader kind)
# ---------------------------------------------------------------------------


def _enumerate_leaves(ds) -> List[Any]:
    from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
    from openwam.dataloader.mixture import MixtureDataset

    # A mixture wraps several sub-datasets (each possibly a multibucket wrapper);
    # recurse so callers always see real LeRobotV3Reader leaves.
    if isinstance(ds, MixtureDataset):
        leaves: List[Any] = []
        for sub in ds._datasets:
            leaves.extend(_enumerate_leaves(sub))
        return leaves
    if isinstance(ds, MultiLeRobotV3Reader):
        return list(ds.buckets)
    return [ds]


def _leaf_info(leaf) -> Tuple[str, str]:
    """Return ``(kind, exclusion_target_path)`` for a leaf reader."""
    from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader

    if isinstance(leaf, LeRobotV3Reader):
        return "lerobot", str(Path(leaf._dataset_dir) / "meta" / "excluded_episodes.json")
    raise TypeError(f"scan_dataset: unsupported leaf type {type(leaf).__name__}")


def _leaf_episode_key(leaf, local_idx: int):
    """Map a window index within a leaf to its episode key (for exclusion)."""
    ep_local = int(np.searchsorted(leaf._cum_n_starts, local_idx, side="right") - 1)
    return int(leaf._eps_df.iloc[ep_local]["episode_index"])


def _leaf_locate_episode(leaf, key) -> Optional[int]:
    """Reverse of ``_leaf_episode_key``: episode key → its LAST window's global
    index in the leaf's CURRENT index, or None if the episode is absent / has no
    windows now. The last window reaches the highest frames, so it reproduces the
    truncated/corrupt-video class the same way the scanner's episode mode does.

    Used by the re-probe to locate an episode by identity rather than by the
    scan-time positional window index (which shifts if the exclusion set changed
    between scan and emit).
    """
    matches = np.where(leaf._eps_df["episode_index"].to_numpy().astype(np.int64) == int(key))[0]
    if matches.size == 0:
        return None
    ep_local = int(matches[0])
    start = int(leaf._cum_n_starts[ep_local])
    end = int(leaf._cum_n_starts[ep_local + 1])
    if end <= start:
        return None  # episode contributes no windows now — nothing to decode
    return end - 1


# ---------------------------------------------------------------------------
# Lazy shard/stride plan over all leaves (no giant index array materialized)
# ---------------------------------------------------------------------------


class _ScanPlan:
    """Maps a dense scan position -> (leaf_i, local_window_idx) lazily.

    For each leaf we scan the strided windows ``j*sample_stride`` for
    ``j in [0, ceil(n/sample_stride))`` that fall in this shard
    (``j % num_shards == shard``). Nothing is materialized: position -> local is
    pure arithmetic + one searchsorted over per-leaf scan counts.
    """

    def __init__(self, leaves: List[Any], sample_stride: int, shard: int, num_shards: int, limit: Optional[int]):
        self.sample_stride = max(1, int(sample_stride))
        self.shard = int(shard)
        self.num_shards = max(1, int(num_shards))
        counts = []
        for leaf in leaves:
            n = len(leaf)
            n_strided = (n + self.sample_stride - 1) // self.sample_stride  # j = 0..n_strided-1
            n_shard = len(range(self.shard, n_strided, self.num_shards))
            counts.append(n_shard)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        self.total = int(self.cum[-1])
        if limit is not None:
            self.total = min(self.total, int(limit))

    def __len__(self) -> int:
        return self.total

    def resolve(self, pos: int) -> Tuple[int, int]:
        leaf_i = int(np.searchsorted(self.cum, pos, side="right") - 1)
        k = pos - int(self.cum[leaf_i])  # k-th scanned window in this leaf
        j = self.shard + k * self.num_shards
        local = j * self.sample_stride
        return leaf_i, local


class _EpisodePlan:
    """One entry per episode: its LAST window (highest frame reach).

    The last window of an episode requests the highest frame indices, so a
    truncated / short video fails there first — this is the strongest, cheapest
    signal for the bad-video class that crashes training. Covers every episode
    in one decode each instead of every overlapping window.
    """

    def __init__(self, leaves: List[Any], shard: int, num_shards: int, limit: Optional[int]):
        entries = []
        for leaf_i, leaf in enumerate(leaves):
            cns = np.asarray(leaf._cum_n_starts, dtype=np.int64)
            starts, ends = cns[:-1], cns[1:]
            has = ends > starts  # episodes with >= 1 window
            last_local = ends[has] - 1
            if last_local.size:
                entries.append(np.stack([np.full(last_local.shape, leaf_i, dtype=np.int64), last_local], axis=1))
        plan = np.concatenate(entries, axis=0) if entries else np.zeros((0, 2), dtype=np.int64)
        plan = plan[int(shard) :: max(1, int(num_shards))]
        if limit is not None:
            plan = plan[: int(limit)]
        self._plan = plan

    def __len__(self) -> int:
        return int(self._plan.shape[0])

    def resolve(self, pos: int) -> Tuple[int, int]:
        row = self._plan[pos]
        return int(row[0]), int(row[1])


class _ScanDataset(Dataset):
    """Decodes one window per index (discarding the sample) and returns only the
    outcome, so no PIL images cross the worker IPC boundary."""

    def __init__(self, leaves: List[Any], plan: _ScanPlan):
        self.leaves = leaves
        self.plan = plan

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, pos: int) -> dict:
        leaf_i, local = self.plan.resolve(pos)
        leaf = self.leaves[leaf_i]
        try:
            leaf._getitem_impl(local)  # decode + assemble; result discarded
            return {"leaf_i": leaf_i, "local": local, "ok": True}
        except Exception as e:  # noqa: BLE001 — record every failure kind
            return {
                "leaf_i": leaf_i,
                "local": local,
                "ok": False,
                "err": f"{type(e).__name__}: {e}",
                "key": _leaf_episode_key(leaf, local),
            }


def _identity_collate(batch):
    return batch


# ---------------------------------------------------------------------------
# Phase 1: scan
# ---------------------------------------------------------------------------


def _validate_shard(args) -> None:
    """Reject an out-of-range shard before the (slow) build. Shards are 0-based —
    a 1-based misuse (``--shard 1 --num-shards 1``) would otherwise silently scan
    an empty stride and leave every window covered by no shard (treated as clean).

    Reads ``args.shard`` / ``args.num_shards`` directly: argparse always sets both
    (each has a default), so a missing attribute is a malformed Namespace that
    should raise, not silently pass — this function's whole point is fail-loud.
    """
    ns = int(args.num_shards)
    sh = int(args.shard)
    if ns < 1 or not (0 <= sh < ns):
        raise SystemExit(
            f"[shard] --shard must satisfy 0 <= shard < num_shards; "
            f"got shard={sh} num_shards={ns} (shards are 0-based: --shard 0..N-1 with --num-shards N)."
        )


def cmd_scan(args) -> int:
    # 'scan' is per-dataset by contract — it scans one dataset's leaves and writes
    # THAT dataset's exclusion file — so a mixture config is rejected up front,
    # before the multi-minute build (see _peek_config_type; scan each source
    # separately, verify-mixed covers the mixture).
    if _peek_config_type(args.config) == "mixture":
        print(
            f"[scan] --config {args.config!r} is a mixture, which 'scan' does not support. "
            "Scan each source separately (for example `robocoin`, `oxe_droid`, or `interndata_a1`), "
            "then confirm the mixed load with `verify-mixed --config mixture`.",
            flush=True,
        )
        return 2
    _validate_shard(args)
    ds = _build_dataset(args.config)
    leaves = _enumerate_leaves(ds)
    leaf_meta = [_leaf_info(leaf) for leaf in leaves]  # (kind, target)
    if args.mode == "episode":
        plan = _EpisodePlan(leaves, args.shard, args.num_shards, args.limit)
    else:
        plan = _ScanPlan(leaves, args.sample_stride, args.shard, args.num_shards, args.limit)
    total = len(plan)
    n_windows = sum(len(leaf) for leaf in leaves)
    print(
        f"[scan {args.config}] mode={args.mode} leaves={len(leaves)} total_windows={n_windows} "
        f"scanning={total} (stride={args.sample_stride}, shard={args.shard}/{args.num_shards}) "
        f"workers={args.num_workers}",
        flush=True,
    )
    if total == 0:
        print("[scan] nothing to scan for this shard.", flush=True)
        return 0

    out_dir = Path(args.out_dir) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_path = out_dir / f"failures.shard{args.shard}-of-{args.num_shards}.jsonl"
    prog_path = out_dir / f"progress.shard{args.shard}-of-{args.num_shards}.json"
    fail_f = open(fail_path, "a", buffering=1, encoding="utf-8")  # line-buffered; records ensure_ascii=False

    loader = DataLoader(
        _ScanDataset(leaves, plan),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
        shuffle=False,
        drop_last=False,
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )

    t0 = time.time()
    scanned = 0
    n_fail = 0
    try:
        for batch in loader:
            for r in batch:
                scanned += 1
                if not r["ok"]:
                    n_fail += 1
                    kind, target = leaf_meta[r["leaf_i"]]
                    rec = {
                        "kind": kind,
                        "target": target,
                        "key": r["key"],
                        "local": r["local"],
                        "err": r["err"],
                    }
                    fail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if scanned % args.log_every < args.batch_size:
                dt = time.time() - t0
                rate = scanned / max(dt, 1e-6)
                eta = (total - scanned) / max(rate, 1e-6)
                prog = {
                    "scanned": scanned,
                    "total": total,
                    "failures": n_fail,
                    "rate_per_s": round(rate, 1),
                    "eta_s": int(eta),
                    "elapsed_s": int(dt),
                }
                prog_path.write_text(json.dumps(prog))
                print(
                    f"[scan {args.config}] {scanned}/{total} ({100 * scanned / total:.1f}%) "
                    f"fails={n_fail} rate={rate:.0f}/s eta={eta / 3600:.2f}h",
                    flush=True,
                )
    finally:
        fail_f.close()

    dt = time.time() - t0
    (out_dir / f"done.shard{args.shard}-of-{args.num_shards}.json").write_text(
        json.dumps({"scanned": scanned, "failures": n_fail, "elapsed_s": int(dt)})
    )
    print(
        f"[scan {args.config}] DONE shard {args.shard}/{args.num_shards}: scanned={scanned} failures={n_fail} in {dt / 3600:.2f}h",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Merge failures -> exclusion files
# ---------------------------------------------------------------------------


def _read_all_failures(out_dir: Path) -> List[dict]:
    recs: List[dict] = []
    n_bad = 0
    for p in sorted(out_dir.glob("failures.shard*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    # A shard killed mid-write (the scanner appends line-buffered)
                    # can leave a truncated final line. Skip it rather than abort
                    # the whole emit — the tool is built to survive interruption.
                    n_bad += 1
    if n_bad:
        print(f"[emit] skipped {n_bad} malformed/truncated failure line(s)", flush=True)
    return recs


def _group_failures(recs: List[dict]) -> Tuple[dict, dict]:
    """Group failure records by reader kind.

    Returns ``(lerobot, unknown)`` where ``lerobot`` is
    ``{target: set(episode_index)}`` and ``unknown`` is ``{kind: count}`` for any
    record whose ``kind`` has no emit branch. The caller must surface unknown
    kinds rather than silently treating them as a clean scan.
    """
    lerobot: dict = {}
    unknown: dict = {}
    for r in recs:
        kind, target, key = r["kind"], r["target"], r["key"]
        if kind == "lerobot":
            lerobot.setdefault(target, set()).add(int(key))
        else:
            unknown[kind] = unknown.get(kind, 0) + 1
    return lerobot, unknown


# An existing exclusion file that fails to parse must NOT be silently treated as
# empty: emit unions the new failures with what it reads and then atomically
# OVERWRITES the file, so an empty read would wipe hundreds of already-recorded
# exclusions and reload those bad episodes into training (violating the documented
# only-add union). That covers valid-JSON-wrong-shape too — INCLUDING a dict
# missing the canonical key (e.g. a hand-written unwrapped {dir: [eps]} or a
# misspelled "episode_indices"): emit's own writes always carry the key, so its
# absence means a hand-edit whose entries an empty read would silently drop.
# These files are machine-written atomically — a parse/shape failure means real
# corruption or a hand-edit; fail loud and let a human resolve it. (The
# training-path readers, by contrast, tolerate a corrupt file and over-include to
# avoid crashing construction — a different, non-destructive context.)
def _read_existing_exclusions_or_die(target: str, extract):
    p = Path(target)
    if not p.exists():
        return extract(None)
    try:
        return extract(json.loads(p.read_text()))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[emit] existing exclusion file {target} is unreadable/malformed ({type(e).__name__}: {e}). "
            "Refusing to emit: overwriting it would silently drop its existing exclusions and reload those "
            "bad episodes into training. Inspect/repair (or delete) it, then re-run emit."
        )


def _parse_lerobot_episode_indices(value, field: str) -> set[int]:
    if not isinstance(value, list) or any(type(i) is not int or i < 0 for i in value):
        raise ValueError(f"{field} must be a list of non-negative integers")
    return set(value)


def _read_lerobot_payload(target: str) -> dict:
    def _extract(doc):
        if doc is None:
            return {"episode_indices": []}
        if not isinstance(doc, dict):
            raise TypeError("top-level value must be an object")
        # Indexing (not .get): a missing canonical key dies via the wrapper
        # rather than reading as empty (see the module comment above).
        _parse_lerobot_episode_indices(doc["episode_indices"], "episode_indices")
        return dict(doc)

    return _read_existing_exclusions_or_die(target, _extract)


def _read_lerobot_excluded(target: str) -> set:
    return _parse_lerobot_episode_indices(_read_lerobot_payload(target)["episode_indices"], "episode_indices")


def _merge_lerobot_payload(payload: dict, failures: set[int]) -> tuple[dict, set[int], set[int]]:
    """Merge generic failures without destroying DROID's per-source ownership."""
    from openwam.dataloader.oxe_droid import (
        DROID_PROMPT_EXCLUSION_KEY,
        DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY,
    )

    existing = _parse_lerobot_episode_indices(payload["episode_indices"], "episode_indices")
    out = dict(payload)
    if DROID_PROMPT_EXCLUSION_KEY in out:
        prompt = out[DROID_PROMPT_EXCLUSION_KEY]
        if not isinstance(prompt, dict):
            raise ValueError(f"{DROID_PROMPT_EXCLUSION_KEY} must be an object")
        prompt_owned = _parse_lerobot_episode_indices(
            prompt["episode_indices"],
            f"{DROID_PROMPT_EXCLUSION_KEY}.episode_indices",
        )
        independently_owned = _parse_lerobot_episode_indices(
            prompt[DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY],
            f"{DROID_PROMPT_EXCLUSION_KEY}.{DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY}",
        )
        if existing != prompt_owned | independently_owned:
            raise ValueError("episode_indices must equal the union of prompt-owned and independently-owned exclusions")
        prompt = dict(prompt)
        # Record every failure this scanner owns, including IDs already owned
        # by the prompt scanner. Without the overlap, a later prompt repair
        # could incorrectly re-enable an independently bad episode.
        prompt[DROID_PROMPT_INDEPENDENT_EXCLUSIONS_KEY] = sorted(independently_owned | failures)
        out[DROID_PROMPT_EXCLUSION_KEY] = prompt
    out["episode_indices"] = sorted(existing | failures)
    return out, existing, failures - existing


def _build_leaves(config_name: str) -> Optional[List[Any]]:
    """Build the dataset for ``config_name`` ONCE and return its leaf readers, or
    ``None`` if it cannot be built here (e.g. emitting on a machine without the
    data). Shared by emit's re-probe and its over-exclusion guardrail so the (for
    lerobot configs, "very slow") full build is paid once, not twice."""
    try:
        ds = _build_dataset(config_name)
        return _enumerate_leaves(ds)
    except Exception as e:  # noqa: BLE001
        print(
            f"[emit] could not build {config_name!r} ({type(e).__name__}: {e}); "
            "re-probe skipped (all recorded failures union as-is) and "
            "over-exclusion guardrail disabled for this run.",
            flush=True,
        )
        return None


def _target_episode_totals(leaves: List[Any]) -> dict:
    """Map each exclusion target to its pre-exclusion episode population.

    The readers capture this count immediately before applying their exclusion
    artifact. Unlike ``len(_eps_df)`` / ``len(_dirs)``, it is independent of the
    artifact snapshot that happened to exist while the reader was constructed,
    so a concurrent writer cannot inflate the guardrail denominator.
    """
    totals: dict = {}
    for leaf in leaves:
        try:
            _, target = _leaf_info(leaf)
        except TypeError:
            continue
        n = getattr(leaf, "_n_episodes_before_exclusions", None)
        if n is None:
            continue
        totals[target] = totals.get(target, 0) + int(n)
    return totals


def _exclusion_ratio_violations(planned: List[dict], totals: dict, max_frac: float) -> List[tuple]:
    """Return ``[(target, n_final, universe, frac), ...]`` for targets whose
    post-emit exclusion fraction exceeds ``max_frac`` (only when this emit adds
    something new and the total is known).

    ``totals[target]`` is the stable population captured before the reader
    applied any exclusions. It must not be reconstructed as ``loaded + current
    exclusions`` because the current artifact may have changed since build.
    """
    viol: List[tuple] = []
    for pl in planned:
        universe = totals.get(pl["target"])
        if universe is None:  # unknown total → cannot judge, skip
            continue
        if pl["n_new"] > 0 and universe > 0:
            frac = pl["n_final"] / universe
            if frac > max_frac:
                viol.append((pl["target"], pl["n_final"], universe, frac))
    return viol


def _filter_reprobe_against_leaves(recs: List[dict], leaves: List[Any], tries: int) -> List[dict]:
    """Re-probe each recorded failure against already-built ``leaves``, keeping only
    the ones that STILL fail every time (deterministic corruption); returned records
    are the survivors.

    The scan probes each window once with no retry and records *any* exception, so a
    single transient hiccup (flaky NFS / momentary IO error) would blacklist a
    healthy episode forever — ``emit`` unions into the exclusion file only-adds, and
    Phase-2 ``verify-mixed`` rebuilds without the excluded episode, so the
    misclassification is never rediscovered. This pass, run at emit time (well after
    the scan, so a transient will have cleared), decodes each recorded failure up to
    ``tries`` times and drops it if it now succeeds.

    Each record is probed by EPISODE IDENTITY, not by the scan-time positional
    window index: if the record's ``local`` still maps to the same episode key it
    is probed directly (exact window — matters for a ``window``-mode scan that hit
    a mid-clip frame), otherwise the episode is relocated by key and its last
    window is probed. This is robust to the exclusion set changing between scan
    and emit (which shifts positional indices) — a stale ``local`` can no longer
    silently probe a different, healthy episode and drop a real bad one.

    File-level records (``local == -1``, written by ``gpu_decode_scan``) are the
    exception: they are kept WITHOUT re-probing. The GPU sweep already decodes the
    whole file end-to-end and does its own transient retry, so its verdict is
    deterministic; a CPU last-window re-probe here cannot reproduce a mid-file
    corruption and would wrongly "clear" it.
    """
    # The exclusion-target path is unique per leaf → map each record back to its
    # leaf. If two leaves ever shared a target (they don't in shipped configs),
    # the mapping would be ambiguous, so such targets are kept unprobed.
    by_target: dict = {}
    dup_targets: set = set()
    for leaf in leaves:
        try:
            _, target = _leaf_info(leaf)
        except TypeError:
            continue
        if target in by_target:
            dup_targets.add(target)
        by_target[target] = leaf

    tries = max(1, int(tries))
    confirmed: List[dict] = []
    n_cleared = 0
    n_unmapped = 0
    n_filelevel = 0
    for r in recs:
        target = r.get("target")
        leaf = by_target.get(target)
        key = r.get("key")
        local = r.get("local")
        # File-level records (local == -1) come from gpu_decode_scan's full-file
        # NVDEC decode, which ALREADY did its own transient classification + a
        # cross-GPU retry, so its decode_error:/truncated:/missing_file: verdicts
        # are deterministic. Re-probing them here would only CPU-decode this
        # episode's LAST window — which cannot reproduce a MID-file corruption the
        # full-file decode caught, so a mid-file break would be silently "cleared"
        # and stay in the training set (the GPU sweep's whole point defeated). Keep
        # such records as-is without re-probing.
        if local == -1:
            n_filelevel += 1
            confirmed.append(r)
            continue
        if leaf is None or target in dup_targets:
            # Unmappable (config changed / ambiguous target): keep — excluding is safe.
            n_unmapped += 1
            confirmed.append(r)
            continue
        # Prefer the exact recorded window when it still belongs to this episode;
        # otherwise (index drift between scan and emit) relocate by episode key.
        probe_idx: Optional[int] = None
        if isinstance(local, int) and 0 <= local < len(leaf):
            try:
                if _leaf_episode_key(leaf, local) == key:
                    probe_idx = local
            except Exception:  # noqa: BLE001
                probe_idx = None
        if probe_idx is None:
            probe_idx = _leaf_locate_episode(leaf, key)
        if probe_idx is None:
            # Episode not in the current dataset (already excluded / absent): keep.
            n_unmapped += 1
            confirmed.append(r)
            continue
        still_bad = True
        for _ in range(tries):
            try:
                leaf._getitem_impl(int(probe_idx))
                still_bad = False
                break
            except Exception:  # noqa: BLE001
                continue
        if still_bad:
            confirmed.append(r)
        else:
            n_cleared += 1
    print(
        f"[emit] re-probe ({tries}x): {len(confirmed)} deterministic kept, "
        f"{n_cleared} transient cleared, {n_unmapped} unmappable kept, "
        f"{n_filelevel} file-level (gpu) kept unprobed",
        flush=True,
    )
    return confirmed


def cmd_emit(args) -> int:
    out_dir = Path(args.out_dir) / args.config
    if not out_dir.exists():
        print(f"[emit] no scan output at {out_dir}", flush=True)
        return 1
    recs = _read_all_failures(out_dir)
    print(f"[emit {args.config}] {len(recs)} failure records", flush=True)

    # Build the dataset ONCE (slow for lerobot configs) and share it between the
    # re-probe and the over-exclusion guardrail. Skipped entirely when there are
    # no failure records — there is then nothing to re-probe or measure. On a
    # failed build _build_leaves already prints the one skip notice (re-probe
    # skipped + guardrail disabled; failures union as-is) — no second message here.
    leaves = _build_leaves(args.config) if recs else None
    if recs and leaves is not None and getattr(args, "reprobe", True):
        recs = _filter_reprobe_against_leaves(recs, leaves, getattr(args, "reprobe_tries", 3))

    lerobot, unknown = _group_failures(recs)
    if unknown:
        # A failure record with a kind no emit branch handles used to be dropped
        # silently. Fail loudly instead: a reader kind was added without matching
        # emit handling, so its bad episodes would never be excluded.
        raise SystemExit(
            f"[emit] {sum(unknown.values())} failure record(s) have unknown kind(s) "
            f"{dict(sorted(unknown.items()))} with no emit branch — refusing to emit. "
            "Add handling in scan_dataset._group_failures / cmd_emit."
        )

    totals = _target_episode_totals(leaves) if leaves is not None else {}
    targets = list(lerobot)
    with locked_exclusion_files(targets):
        # Every value below is derived from a fresh read while all targets are
        # locked. A plan made before acquiring the lock can lose exclusions from
        # another prompt/generic writer between its read and replace.
        lerobot_plan = []  # (target, merged_payload, existing_set, new_set)
        for target, eps in lerobot.items():
            payload = _read_lerobot_payload(target)
            try:
                merged_payload, existing, new = _merge_lerobot_payload(payload, eps)
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"[emit] existing exclusion provenance in {target} is malformed or unsafe to merge "
                    f"({exc}); refusing to overwrite it"
                ) from exc
            lerobot_plan.append((target, merged_payload, existing, new))

        # An environmental scan failure (ffmpeg without NVDEC, no visible GPU,
        # mass timeouts) can mark nearly every clip bad. Recompute this guardrail
        # from the locked, freshly merged plans so concurrent individually-small
        # emits cannot cumulatively cross the threshold.
        planned = [
            {
                "target": target,
                "n_final": len(merged["episode_indices"]),
                "existing": len(existing),
                "n_new": len(new),
            }
            for target, merged, existing, new in lerobot_plan
        ]
        violations = _exclusion_ratio_violations(planned, totals, args.max_exclude_frac)
        if violations and not args.force:
            print(
                f"[emit {args.config}] ABORT: would exclude more than "
                f"--max-exclude-frac={args.max_exclude_frac:.3f} of a dataset — refusing to write:",
                flush=True,
            )
            for target, n_final, universe, frac in violations:
                print(f"    {target}: {n_final}/{universe} = {frac:.1%} excluded", flush=True)
            print(
                "  This usually means an ENVIRONMENTAL scan failure (ffmpeg without NVDEC, no visible GPU, "
                "mass timeouts) recorded healthy clips as bad — inspect the scan_out failures before excluding. "
                "Re-run with --force to override.",
                flush=True,
            )
            return 3

        n_targets = 0
        n_eps = 0
        # lerobot: {"episode_indices": [...]} unioned with existing.
        for target, merged, _existing, new in lerobot_plan:
            if not args.dry_run:
                atomic_publish_text(target, json.dumps(merged))
            n_targets += 1
            n_eps += len(new)
            print(f"  [lerobot] {target}: +{len(new)} new (total {len(merged['episode_indices'])})", flush=True)

    what = "DRY-RUN (no files written)" if args.dry_run else "wrote"
    print(f"[emit {args.config}] {what} exclusions for {n_targets} target(s), {n_eps} bad episodes total", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Phase 2: verify mixed load through the real (retry) path
# ---------------------------------------------------------------------------


class _ShardedIndices:
    """Lazy ``range(shard, size, num_shards)`` with an optional prefix limit.

    Materializing ``np.arange`` for a large mixture wastes memory before the
    first sample is loaded, even when a small ``--limit`` smoke test was
    requested. This range-like adapter keeps the same ordering and sharding
    contract with O(1) memory.
    """

    def __init__(
        self,
        size: int,
        *,
        shard: int,
        num_shards: int,
        limit: Optional[int] = None,
    ) -> None:
        self._shard = int(shard)
        self._num_shards = int(num_shards)
        total = len(range(self._shard, int(size), self._num_shards))
        self._length = total if limit is None else min(total, max(0, int(limit)))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> int:
        i = int(i)
        if i < 0:
            i += self._length
        if not 0 <= i < self._length:
            raise IndexError(i)
        return self._shard + i * self._num_shards


class _TolerantDataset(Dataset):
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        try:
            self.ds[idx]  # real path: MixtureDataset -> sub._safe_get (retries)
            return {"idx": idx, "ok": True}
        except Exception as e:  # noqa: BLE001
            return {"idx": idx, "ok": False, "err": f"{type(e).__name__}: {e}"}


def cmd_verify_mixed(args) -> int:
    _validate_shard(args)
    ds = _build_dataset(args.config)
    n = len(ds)
    idx = _ShardedIndices(
        n,
        shard=args.shard,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    print(
        f"[verify {args.config}] len={n} verifying={len(idx)} (shard={args.shard}/{args.num_shards}) workers={args.num_workers}",
        flush=True,
    )

    out_dir = Path(args.out_dir) / f"{args.config}-verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_path = out_dir / f"unrecoverable.shard{args.shard}-of-{args.num_shards}.jsonl"
    fail_f = open(fail_path, "a", buffering=1, encoding="utf-8")  # records ensure_ascii=False

    loader = DataLoader(
        _TolerantDataset(ds, idx),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
        shuffle=False,
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    t0 = time.time()
    scanned = 0
    n_fail = 0
    try:
        for batch in loader:
            for r in batch:
                scanned += 1
                if not r["ok"]:
                    n_fail += 1
                    fail_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if scanned % args.log_every < args.batch_size:
                dt = time.time() - t0
                rate = scanned / max(dt, 1e-6)
                print(
                    f"[verify {args.config}] {scanned}/{len(idx)} ({100 * scanned / len(idx):.1f}%) "
                    f"unrecoverable={n_fail} rate={rate:.0f}/s eta={(len(idx) - scanned) / max(rate, 1e-6) / 3600:.2f}h",
                    flush=True,
                )
    finally:
        fail_f.close()
    dt = time.time() - t0
    status = "CLEAN ✅" if n_fail == 0 else f"{n_fail} UNRECOVERABLE ❌"
    print(f"[verify {args.config}] DONE: {status} (scanned {scanned} in {dt / 3600:.2f}h)", flush=True)
    return 0 if n_fail == 0 else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, config_help):
        sp.add_argument("--config", required=True, help=config_help)
        sp.add_argument("--out-dir", default="scan_out")
        sp.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
        sp.add_argument("--batch-size", type=int, default=64)
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)
        sp.add_argument("--limit", type=int, default=None, help="scan at most N windows (smoke test)")
        sp.add_argument("--log-every", type=int, default=5000)

    sp_scan = sub.add_parser("scan", help="Phase 1: direct episode/window scan (bypass _safe_get)")
    # scan is per-dataset only — NOT mixture (see cmd_scan fast-fail).
    add_common(
        sp_scan,
        config_help="active public LeRobot v3 config group (for example agibotworld, robocoin, "
        "oxe_droid, interndata_a1); "
        "NOT mixture",
    )
    sp_scan.add_argument(
        "--mode",
        choices=["episode", "window"],
        default="episode",
        help="episode = one high-frame window per episode (fast, finds truncated/corrupt videos); "
        "window = every S-th window (exhaustive, decode-bound, ~days for full datasets)",
    )
    sp_scan.add_argument(
        "--sample-stride", type=int, default=1, help="[window mode] scan every S-th window (1 = every window)"
    )
    sp_scan.set_defaults(func=cmd_scan)

    sp_emit = sub.add_parser("emit", help="Merge failure jsonl -> write/union exclusion files")
    sp_emit.add_argument("--config", required=True)
    sp_emit.add_argument("--out-dir", default="scan_out")
    sp_emit.add_argument("--dry-run", action="store_true")
    sp_emit.add_argument(
        "--max-exclude-frac",
        type=float,
        default=0.05,
        help="abort if any dataset would end up with more than this fraction of its episodes excluded "
        "(guards against an environmental scan failure zeroing a dataset); override with --force",
    )
    sp_emit.add_argument("--force", action="store_true", help="write exclusions even if --max-exclude-frac is exceeded")
    sp_emit.add_argument(
        "--no-reprobe",
        dest="reprobe",
        action="store_false",
        help="skip the serial re-probe and union every recorded failure as-is "
        "(use when the dataset is not available on this machine)",
    )
    sp_emit.set_defaults(reprobe=True)
    sp_emit.add_argument(
        "--reprobe-tries",
        type=int,
        default=3,
        help="re-probe each recorded failure up to N times; only failures that "
        "reproduce every time are excluded (default 3; transient one-offs are dropped)",
    )
    sp_emit.set_defaults(func=cmd_emit)

    sp_ver = sub.add_parser("verify-mixed", help="Phase 2: tolerant full iterate through the real path")
    add_common(sp_ver, config_help="active dataloader config group or mixture")
    sp_ver.set_defaults(func=cmd_verify_mixed)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
