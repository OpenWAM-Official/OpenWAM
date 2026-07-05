"""Ego4D dataloader for lerobot v3 standard datasets.

Reads ``/path/to/Ego4D`` (12 ``group_XX`` sub-datasets, egocentric
human-hand LeRobot v3 conversion, ``robot_type: "ego_hand"``) through the
shared :class:`~openwam.dataloader.bases.lerobot_v3_reader.LeRobotV3Reader`
machinery, emitting the canonical sample-dict shape (video / vace_video /
first_frame_image / action / action_mask / video_mask / proprio / proprio_mask
/ prompt) so it batch-collates cleanly with the other sources in
``configs/dataloader/mixture.yaml``.

Design contract (mirrors EgoDex — video-only supervised):

  - Action loss is **disabled**. ``_action_20d`` / ``_proprio_20d`` inherit the
    base ``None`` default, so the finalizer emits ``action = zeros(T-1, dim)``
    with an all-False mask and ``proprio = zeros(1, dim)`` all-False. Ego4D's
    per-frame payload is bimanual *human* MANO hand pose (not a robot action)
    and has no materialized ``action`` column, so only video / image-prediction
    loss is supervised. With ``unify_action: true`` the emitted width is the
    shared 80-D ``UNIFY_DIM`` (all zeros, all masked) so it collates with the
    other 80-D-head sources.

  - **Bilingual prompt → English only.** Each ``task_index`` in
    ``meta/tasks.parquet`` maps to a single combined string of the form
    ``中文:<zh>。英文:<en>``. We keep only the English half, split on the
    ``英文[:：]`` marker (both the ASCII ``:`` and the full-width ``：`` occur in
    the data). See :func:`extract_english_prompt`.

  - **Null-prompt episodes are dropped.** A small number of tasks (12 across the
    12 groups, ~0.06% of episodes) carry the literal string ``"null"`` instead
    of a bilingual prompt. Those episodes have no usable instruction and are
    excluded via :meth:`_filter_episodes` (using the per-episode ``tasks``
    column of ``meta/episodes/*.parquet``) so they never reach training.

  - **Per-group task_index namespaces.** Each ``group_XX`` restarts
    ``task_index`` at 0, so prompt lookup MUST stay within a group. That is
    automatic here: root mode builds one reader (bucket) per group, and each
    bucket's :meth:`_load_prompts` reads only its own group's ``tasks.parquet``.

  - Videos are LeRobot-v3 concatenated per-view mp4 (~93 episodes per file);
    the base reader seeks each episode's ``from_timestamp`` window via the
    per-episode ``videos/.../{chunk,file}_index`` offsets — nothing Ego4D
    specific is needed for that.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bilingual-prompt English extraction (pure function, independently testable)
# ---------------------------------------------------------------------------

# The combined string is normally ``中文:<zh>。英文:<en>``. A few tasks (~5/189689)
# carry a stray/duplicated ``英文:`` inside the Chinese half, or the reversed
# order ``英文:<en>。中文:<zh>``. To be robust:
#   * greedy ``^.*英文`` anchors to the LAST ``英文`` marker (so a duplicated
#     marker doesn't leak the Chinese clause before it), and
#   * any trailing ``中文:…`` tail (reversed order) is stripped.
# Both the ASCII colon ``:`` and the full-width colon ``：`` (U+FF1A) appear after
# 英文/中文, so the character classes cover both.
_EN_MARKER_RE = re.compile(r"(?s)^.*英文[:：]\s*(.*)$")
_ZH_MARKER_RE = re.compile(r"中文[:：]")
# CJK Unified Ideographs (+ extension A). Used as a final guard: an "English"
# clause that still contains Han characters is a malformed / mixed annotation.
_CJK_RE = re.compile(r"[㐀-鿿]")


def extract_english_prompt(combined: Optional[str]) -> Optional[str]:
    """Return the English half of a combined ``中文:…英文:…`` prompt, or None.

    Returns None when the input is missing, the literal ``"null"`` placeholder,
    carries no ``英文`` marker, the English half is blank after stripping, or the
    extracted text still contains Han characters (a malformed / mixed annotation
    — e.g. a stray Han glyph inside the English, or a duplicated ``English:``
    block). A None result marks the episode as having no usable English
    instruction so the caller drops it (:meth:`Ego4DDataset._filter_episodes`),
    guaranteeing no Chinese ever reaches training. Such malformed tasks are
    ~13/189689 (0.007%), on top of the ~12 ``null`` placeholders.
    """
    if combined is None:
        return None
    s = str(combined).strip()
    if not s or s == "null":
        return None
    m = _EN_MARKER_RE.search(s)
    if m is None:
        return None
    # Strip a trailing ``中文:…`` clause (reversed-order annotations); the greedy
    # ``^.*英文`` already discarded any Chinese before the final 英文 marker.
    en = _ZH_MARKER_RE.split(m.group(1))[0].strip()
    if not en or _CJK_RE.search(en):
        return None
    return en


def _episode_tasks_string(tasks_cell) -> Optional[str]:
    """Extract the single prompt string from an episodes-parquet ``tasks`` cell.

    ``meta/episodes/*.parquet`` stores ``tasks`` as a ``list<string>`` of length
    1 (verified: 0 episodes carry >1 task string across all 12 groups). Handle
    list / ndarray / scalar forms defensively.
    """
    if tasks_cell is None:
        return None
    if isinstance(tasks_cell, (list, tuple, np.ndarray)):
        return str(tasks_cell[0]) if len(tasks_cell) else None
    return str(tasks_cell)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Ego4DDataset(LeRobotV3Reader):
    """lerobot v3 reader for Ego4D with action loss disabled (English prompts).

    Video-only supervised (like EgoDex): ``_action_20d`` / ``_proprio_20d``
    inherit the base ``None`` default → zero action/proprio with all-False
    masks. See module docstring.
    """

    DATASET_NAME = "Ego4D"
    # Reader-emit dims; matches the EEF mixture schema (action_dim=20 pre-unify).
    _ACTION_DIM = 20
    _STATE_DIM = 20
    ACTION_DIM = 20
    # Only task_index is read from the per-frame parquet (for the inherited
    # task_index prompt path); the MANO hand columns are ignored (video-only).
    NEEDED_COLS = ("task_index",)
    # Prompt convention: task_index → text via meta/tasks.parquet, English-only.
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True

    def _resolve_cameras(self, info: dict):
        # Single ego camera; no wrist cameras.
        return (self._target_camera or "observation.images.ego", None, None)

    def _train_min_window_len(self) -> int:
        # video_stride+1 guarantees >= 2 real video frames per train window
        # (t0 reference + first prediction step). Same floor as EgoDex.
        return self._video_stride + 1

    def _load_prompts(self) -> None:
        """Build ``{task_index: english_prompt}`` from ``meta/tasks.parquet``.

        The parquet's row index holds the combined ``中文:…英文:…`` string; we
        keep only the English half. A task with no usable English (the ``"null"``
        placeholder) maps to ``""`` so the inherited :meth:`_resolve_prompt`
        raises loudly if such an episode ever slips past :meth:`_filter_episodes`
        (it should not — those episodes are dropped at init).
        """
        tasks_path = self._dataset_dir / "meta" / "tasks.parquet"
        tasks_df = pd.read_parquet(tasks_path)  # FileNotFoundError if missing
        task_idx = tasks_df["task_index"].to_numpy()
        task_str = tasks_df.index.to_numpy()
        self._task_idx_to_text: Dict[int, str] = {
            int(i): (extract_english_prompt(s) or "") for i, s in zip(task_idx.tolist(), task_str.tolist())
        }

    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """Drop episodes whose bilingual prompt has no usable English half.

        Uses the per-episode ``tasks`` column of ``meta/episodes/*.parquet``
        (a ``list<string>`` of length 1) so filtering does not depend on the
        task_index join. Runs before the window index is built, so dropped
        episodes never produce a sample.
        """
        if "tasks" not in eps_df.columns:
            # Defensive: every Ego4D group carries the tasks column. If a future
            # conversion drops it, fail loud rather than silently keep nulls.
            raise KeyError(
                f"Ego4D({self._dataset_id}): meta/episodes parquet has no 'tasks' column; "
                "cannot filter null-prompt episodes."
            )
        keep = eps_df["tasks"].apply(lambda t: extract_english_prompt(_episode_tasks_string(t)) is not None)
        n_before = len(eps_df)
        n_drop = int((~keep).to_numpy().sum())
        if n_drop:
            logger.info(
                "Ego4D(%s): dropped %d/%d episodes with no usable English prompt",
                self._dataset_id,
                n_drop,
                n_before,
            )
        return eps_df[keep.to_numpy()].reset_index(drop=True)

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiBucketEgo4DDataset


# ---------------------------------------------------------------------------
# Multi-bucket wrapper (root mode)
# ---------------------------------------------------------------------------


class MultiBucketEgo4DDataset(MultiLeRobotV3Reader):
    """Aggregate of the 12 Ego4D ``group_XX`` buckets.

    Used in root mode: ``Ego4DDataset.from_config({dataset_dir: <root>})`` where
    ``<root>`` contains ``group_00 … group_11`` subdirs, each with
    ``meta/info.json``. All buckets share ``action_dim`` and
    ``normalization_stats=None`` by design; the base defaults match directly.
    """

    def __init__(self, buckets: List["Ego4DDataset"]):
        super().__init__(buckets)
        logger.info(
            "MultiBucketEgo4DDataset: %d buckets, %d total windows (action_dim=%d)",
            len(self._buckets),
            len(self),
            self.action_dim,
        )

    @property
    def buckets(self) -> List["Ego4DDataset"]:
        return self._buckets


__all__ = ["Ego4DDataset", "MultiBucketEgo4DDataset", "extract_english_prompt"]
