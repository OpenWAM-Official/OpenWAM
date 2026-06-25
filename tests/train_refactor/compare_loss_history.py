"""Compare two seeded debug-run loss-history CSVs (the train-refactor golden gate).

"Behaves like main" criterion: with a fixed ``cfg.project.seed`` and
``training.debug=true``, the ``debug_loss_history.csv`` produced before and after
the refactor must match value-for-value on the deterministic columns.

Only deterministic columns are compared; ``steps_per_sec`` (wall-clock dependent,
never bit-identical) is excluded. Comparison is on the raw CSV strings (the trainer
writes ``%.10g``, byte-identical on the deterministic path), so no atol is needed.
Latent mode renames the loss_action / loss_decoder columns, but both runs share a
config, so intersecting on column name realigns them automatically.

Usage:
    python tests/train_refactor/compare_loss_history.py baseline.csv candidate.csv
Exit 0 = identical; 1 = mismatch (first one printed).
"""

import csv
import sys

NON_DETERMINISTIC = {
    "steps_per_sec",
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(main_csv, refactor_csv):
    a = _read(main_csv)
    b = _read(refactor_csv)
    if len(a) != len(b):
        print(f"FAIL: row count differs main={len(a)} refactor={len(b)}")
        return 1
    if not a:
        print("FAIL: empty CSV, no steps to compare")
        return 1

    # Compare only deterministic columns present in both CSVs. Mode differences
    # (latent vs non-latent) change whether loss_action/loss_decoder exist; the
    # intersection realigns them, and asymmetric columns are printed to avoid a
    # silent skip.
    only_a = [c for c in a[0] if c not in b[0] and c not in NON_DETERMINISTIC]
    only_b = [c for c in b[0] if c not in a[0] and c not in NON_DETERMINISTIC]
    if only_a or only_b:
        print(
            f"NOTE: asymmetric columns (mode diff), comparing intersection. main-only={only_a} refactor-only={only_b}"
        )
    cols = [c for c in a[0].keys() if c not in NON_DETERMINISTIC and c in b[0]]
    for i, (ra, rb) in enumerate(zip(a, b)):
        for c in cols:
            if ra.get(c) != rb.get(c):
                print(f"FAIL: row {i} col '{c}' differs: main={ra.get(c)!r} refactor={rb.get(c)!r}")
                return 1
    print(f"OK: {len(a)} steps identical across {len(cols)} deterministic columns ({cols})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
