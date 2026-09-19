"""RoboTwin benchmark RNG isolation and deterministic instruction selection."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils.rng_domain import GlobalRngDomain  # noqa: E402

BENCHMARK = "robotwin"
PROTOCOL_DESCRIPTION_LIMIT = 100


def choose_instructions(
    generator: Callable[[str, list[dict], int], list[dict]],
    *,
    task: str,
    candidate_seed: int,
    episode_info: dict,
) -> dict[str, str]:
    """Run RoboTwin's native prompt path without changing caller RNG state."""

    domain = GlobalRngDomain.from_seed(candidate_seed)
    with domain.activate():
        descriptions = generator(task, [episode_info], PROTOCOL_DESCRIPTION_LIMIT)
        selection_state = np.random.get_state()
        selected: dict[str, str] = {}
        for instruction_type in ("seen", "unseen"):
            np.random.set_state(selection_state)
            options: Any = descriptions[0][instruction_type]
            selected[instruction_type] = str(np.random.choice(options))
    return selected
