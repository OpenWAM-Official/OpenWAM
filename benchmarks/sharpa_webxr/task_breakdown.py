"""Per-task and per-category judge success (k/n) for Sharpa sim-eval score files, side by side across models.

    python benchmarks/sharpa_webxr/task_breakdown.py motus=path/scores_gpt-6-luna_r2.json xwam=... [--tasks]

Each argument is <label>=<scores_*.json written by score.py>. Categories group the 58 suite scenes by the kind of
manipulation the instruction asks for (keyed on the 5-digit scene prefix).
"""

import json
import sys

CATEGORIES = {
    "pick / grasp only": "00177 00178 00375 01116 01174 01304 01337",
    "pick and place": "00001 00019 00286 00381 00391 00457 00636 00646 00807 00921 01012 01083 01085 01089 01242 "
                      "01440 self-",
    "articulated open/close": "00005 00079 00082 00337 00626 01000 01018 01747",
    "cap / lid": "00023 00075 00077 00894",
    "pour": "00175 00475 00487 01480 01498",
    "tool use": "00051 00749 00787 00863 01477 01587 01640",
    "reorient / flip / rotate": "01032 01041 01246 01541 01542 01564",
    "shake / toss": "00779 01592 01605 01652",
}
CAT_OF = {p: c for c, ps in CATEGORIES.items() for p in ps.split()}


def load(path: str) -> dict[str, tuple[int, int, str]]:
    """scene prefix -> (successes, scored rollouts, task text)."""
    out = {}
    for scene, t in json.load(open(path))["per_task"].items():
        ok = [s for s in t["seeds"] if s.get("judge") and "success" in s["judge"]]
        out[scene[:5]] = (sum(bool(s["judge"]["success"]) for s in ok), len(ok), t["task"])
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = {lab: load(p) for lab, p in (a.split("=", 1) for a in args)}
    labels = list(runs)
    tasks = {k: v[2] for r in runs.values() for k, v in r.items()}
    missing = set(tasks) - set(CAT_OF)
    if missing:
        sys.exit(f"uncategorized scenes: {sorted(missing)}")

    w = 13
    print(f"{'category':26s} {'#':>3s} " + " ".join(f"{lab:>{w}s}" for lab in labels))
    for cat in CATEGORIES:
        keys = [k for k in tasks if CAT_OF[k] == cat]
        cells = []
        for lab in labels:
            k = sum(runs[lab].get(s, (0, 0))[0] for s in keys)
            n = sum(runs[lab].get(s, (0, 0))[1] for s in keys)
            cells.append(f"{k:>3d}/{n:<3d}{100 * k / max(n, 1):4.0f}%")
        print(f"{cat:26s} {len(keys):3d} " + " ".join(f"{c:>{w}s}" for c in cells))
    tot = []
    for lab in labels:
        k = sum(v[0] for v in runs[lab].values())
        n = sum(v[1] for v in runs[lab].values())
        tot.append(f"{k:>3d}/{n:<3d}{100 * k / max(n, 1):4.0f}%")
    print(f"{'ALL':26s} {len(tasks):3d} " + " ".join(f"{c:>{w}s}" for c in tot))

    if "--tasks" in sys.argv:
        print()
        print(f"{'scene':5s} {'category':24s} " + " ".join(f"{lab[:7]:>7s}" for lab in labels) + "  task")
        for cat in CATEGORIES:
            for s in sorted(k for k in tasks if CAT_OF[k] == cat):
                cells = [f"{runs[lab][s][0]}/{runs[lab][s][1]}" if s in runs[lab] else "-" for lab in labels]
                print(f"{s:5s} {cat[:24]:24s} " + " ".join(f"{c:>7s}" for c in cells) + f"  {tasks[s][:60]}")


if __name__ == "__main__":
    main()
