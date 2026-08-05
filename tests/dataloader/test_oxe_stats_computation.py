from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from openwam.dataloader.oxe_droid import (
    DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
    DROID_PROMPT_INPUTS_DIGEST_KEY,
    OxeDroidDataset,
    compute_droid_prompt_inputs_digest,
)
from openwam.dataloader.utils.stats_computation.oxe_stats_computation import compute_dataset_stats


def _write_exclusions(root, canonical, prompt_owned=None, independently_owned=None):
    prompt_owned = prompt_owned or []
    independently_owned = independently_owned if independently_owned is not None else canonical
    fallback_chain = list(OxeDroidDataset.PROMPT_FALLBACK_COLS)
    (root / "meta" / "excluded_episodes.json").write_text(
        json.dumps(
            {
                "episode_indices": canonical,
                "droid_prompt_exclusions": {
                    "schema_version": DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
                    "fallback_chain": fallback_chain,
                    "episode_indices": prompt_owned,
                    "independently_owned_episode_indices": independently_owned,
                    "latest_scan": {
                        "episode_indices": prompt_owned,
                        "stats": {
                            "rows_scanned": 120,
                            "unresolved_rows": 0,
                            "episodes_all_unresolved": len(prompt_owned),
                            "episodes_partially_unresolved": 0,
                            "task_index_missing_from_tasks_parquet": 0,
                            "fallback_chain": fallback_chain,
                        },
                        DROID_PROMPT_INPUTS_DIGEST_KEY: compute_droid_prompt_inputs_digest(root),
                    },
                },
            }
        )
    )


def test_droid_stats_exclude_reader_blacklist_population(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    n_rows = 60
    kept = np.zeros((n_rows, 7), dtype=np.float32)
    kept[:, 0] = 1.0
    # Deliberately malformed 2-D poses: filtering must happen in Arrow before
    # numpy conversion, matching the reader's episode-level exclusion behavior.
    excluded = np.zeros((n_rows, 2), dtype=np.float32)
    excluded[:, 0] = 100.0
    values = list(kept) + list(excluded)
    pd.DataFrame(
        {
            "episode_index": [0] * n_rows + [1] * n_rows,
            "task_index": [0] * (n_rows * 2),
            "state": values,
            "other_information.action_tcp_pose": values,
            **{column: [""] * (n_rows * 2) for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_exclusions(root, canonical=[1])

    stats, n_state, n_action = compute_dataset_stats(root, "DROID", rot6d_identity=False)

    assert n_state == n_rows
    assert n_action == n_rows
    assert stats["n_samples"] == n_rows * 2
    assert stats["max"][0] == 1.0
    assert stats["excluded_episode_indices"] == [1]


def test_droid_stats_fail_clearly_when_every_row_is_excluded(tmp_path):
    root = tmp_path / "Droid"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    values = list(np.zeros((120, 7), dtype=np.float32))
    pd.DataFrame(
        {
            "episode_index": [0] * 120,
            "task_index": [0] * 120,
            "state": values,
            "other_information.action_tcp_pose": values,
            **{column: [""] * 120 for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    _write_exclusions(root, canonical=[0])

    with pytest.raises(ValueError, match="every parquet row is excluded"):
        compute_dataset_stats(root, "DROID", rot6d_identity=False)
