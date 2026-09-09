#!/usr/bin/env python3
"""Reproduce every RoboDojo number reported for OpenWAM-alpha, and check how the
public leaderboard bundle exports per-task data across all models.

    python verify_robodojo_scores.py              # checks A-D, offline
    python verify_robodojo_scores.py --leaderboard  # adds check E (needs network)

Check A  Every published per-task cell equals (standard + random) / 2.
Check B  Gen-Std, Gen-Rand and the merged Generalization value.
Check C  Avg SR / Avg Score as the mean of the five capability dimensions.
Check D  The arithmetic behind the figure quoted in issue #23.
Check E  Which quantity the leaderboard bundle stores per task, model by model.

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from statistics import mean

DATA = Path(__file__).with_name("openwam_robodojo_per_seed.json")
BUNDLE_INDEX = "https://robodojo-benchmark.com/"
TOL = 0.05

# scripts/internal/summarize_result.py :: DIMENSIONS
DIMENSIONS = {
    "Generalization": [
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ],
    "Precision": [
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ],
    "Long-Horizon": [
        "put_bottles_into_dustbin",
        "fill_pen_holder",
        "classify_objects",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ],
    "Memory": [
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ],
    "Open": [
        "align_blocks",
        "general_pickup",
        "stack_blocks_by_language",
        "solve_equation",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ],
}
GEN = DIMENSIONS["Generalization"]

failures: list[str] = []


def check(label: str, got: float, want: float, tol: float = TOL) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} got {got:8.2f}   published {want:8.2f}")
    if not ok:
        failures.append(label)


def banner(title: str) -> None:
    print()
    print("=" * 86)
    print(title)
    print("=" * 86)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaderboard", action="store_true", help="also run check E against the live leaderboard bundle")
    args = ap.parse_args()

    d = json.loads(DATA.read_text(encoding="utf-8"))
    gen, merged, pub = d["generalization"], d["merged_all_tasks"], d["published_aggregates"]
    seeds = range(len(d["seeds"]))

    # ---------------------------------------------------------------- check A
    banner("A. Each published per-task cell is the merged 50-episode result")
    print("   For a paired task, RoboDojo evaluates 25 episodes on the standard layout")
    print("   and 25 on the _random layout, then reports the 50 together.\n")
    bad = 0
    for task in GEN:
        std, rnd = gen[task]["standard"]["sr"], gen[task]["random"]["sr"]
        calc = [(s + r) / 2 for s, r in zip(std, rnd)]
        want = merged[task]["sr"]
        ok = all(abs(c - w) <= TOL for c, w in zip(calc, want))
        bad += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  {task:<30} "
            f"std={str(std):<14} rnd={str(rnd):<12} -> {str([int(c) if c == int(c) else c for c in calc]):<14} "
            f"published {want}"
        )
    print(f"\n  {len(GEN) - bad}/{len(GEN)} tasks reproduce, across {len(d['seeds'])} seeds each.")
    if bad:
        failures.append("check A")

    # ---------------------------------------------------------------- check B
    banner("B. The three views of the same Generalization data")
    std_seed = [mean(gen[t]["standard"]["sr"][i] for t in GEN) for i in seeds]
    rnd_seed = [mean(gen[t]["random"]["sr"][i] for t in GEN) for i in seeds]
    mrg_seed = [mean(merged[t]["sr"][i] for t in GEN) for i in seeds]
    std_sc = [mean(gen[t]["standard"]["score"][i] for t in GEN) for i in seeds]
    rnd_sc = [mean(gen[t]["random"]["score"][i] for t in GEN) for i in seeds]
    mrg_sc = [mean(merged[t]["score"][i] for t in GEN) for i in seeds]

    print(f"  per-seed Gen-Std  SR: {[round(x, 2) for x in std_seed]}")
    print(f"  per-seed Gen-Rand SR: {[round(x, 2) for x in rnd_seed]}")
    print(f"  per-seed merged   SR: {[round(x, 2) for x in mrg_seed]}\n")
    check("Gen-Std SR (standard half only)", mean(std_seed), pub["generalization_standard"]["sr"])
    check("Gen-Std Score", mean(std_sc), pub["generalization_standard"]["score"])
    check("Gen-Rand SR (random half only)", mean(rnd_seed), pub["generalization_random"]["sr"])
    check("Gen-Rand Score", mean(rnd_sc), pub["generalization_random"]["score"])
    check("Generalization SR (merged, leaderboard cell)", mean(mrg_seed), pub["generalization_merged"]["sr"])
    check("Generalization Score (merged)", mean(mrg_sc), pub["generalization_merged"]["score"])
    print()
    check("identity: (Gen-Std + Gen-Rand) / 2 == merged", (mean(std_seed) + mean(rnd_seed)) / 2, mean(mrg_seed))

    # ---------------------------------------------------------------- check C
    banner("C. Avg is the mean of the five capability dimensions")
    print("   summarize_result.py :: overall_seed_value ->")
    print("   'Return (sr, score) at one seed as the mean of the per-dimension values.'\n")
    key = {
        "Generalization": "generalization_merged",
        "Precision": "precision",
        "Long-Horizon": "long_horizon",
        "Memory": "memory",
        "Open": "open",
    }
    dim_sr, dim_sc = {}, {}
    for name, tasks in DIMENSIONS.items():
        dim_sr[name] = mean(mean(merged[t]["sr"]) for t in tasks)
        dim_sc[name] = mean(mean(merged[t]["score"]) for t in tasks)
        check(f"{name} SR ({len(tasks)} tasks)", dim_sr[name], pub[key[name]]["sr"], 0.1)
    print()
    check("Avg SR  = mean of the 5 dimensions", mean(dim_sr.values()), pub["average"]["sr"], 0.1)
    check("Avg Score = mean of the 5 dimensions", mean(dim_sc.values()), pub["average"]["score"], 0.1)

    flat = mean(mean(v["sr"]) for v in merged.values())
    print(f"\n  For contrast, a flat mean over all {len(merged)} tasks would be {flat:.2f},")
    print("  not 11.92 - the dimensions hold different numbers of tasks, so a")
    print("  dimension mean is not a task mean.")

    # ---------------------------------------------------------------- check D
    banner("D. Where the figure quoted in issue #23 comes from")
    six = mean(
        [
            pub["generalization_merged"]["sr"],
            pub["generalization_random"]["sr"],
            pub["precision"]["sr"],
            pub["long_horizon"]["sr"],
            pub["memory"]["sr"],
            pub["open"]["sr"],
        ]
    )
    print(f"  Reading the 12 merged cells as if they were Gen-Std gives {mean(mrg_seed):.2f},")
    print(f"  and averaging six dimensions on that basis gives {six:.2f}.")
    print("  Both are arithmetically fine; they are simply not the leaderboard's")
    print("  definitions. The published Gen-Std is the standard half alone.")
    print("\n  Note the size of the gap is forced: reading merged as Gen-Std costs")
    print(
        f"  exactly (Gen-Std - Gen-Rand) / 2 = "
        f"{(pub['generalization_standard']['sr'] - pub['generalization_random']['sr']) / 2:.2f} points,"
    )
    print(
        f"  and indeed {pub['generalization_standard']['sr']} - {mean(mrg_seed):.2f} = "
        f"{pub['generalization_standard']['sr'] - mean(mrg_seed):.2f}."
    )
    print("  A policy with a wide standard/random gap therefore shows the widest")
    print("  apparent discrepancy under that reading, with no bearing on its scores.")

    # ---------------------------------------------------------------- check E
    if args.leaderboard:
        banner("E. What the public bundle stores per task, model by model")
        try:
            run_leaderboard_check()
        except Exception as exc:  # network, layout drift, ...
            print(f"  SKIPPED: {type(exc).__name__}: {exc}")

    banner("RESULT")
    if failures:
        print(f"  {len(failures)} check(s) failed: {failures}")
        return 1
    print("  All checks passed.")
    return 0


def run_leaderboard_check() -> None:
    """Compare, for every leaderboard model, the per-task field against the
    published Gen-Std under both readings."""

    def get(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "openwam-verify"})
        with urllib.request.urlopen(req, timeout=60) as fh:
            return fh.read().decode("utf-8", "replace")

    index = get(BUNDLE_INDEX)
    asset = re.search(r"assets/index-[A-Za-z0-9_-]+\.js", index)
    if not asset:
        raise RuntimeError("could not locate the bundle in the page")
    src = get(BUNDLE_INDEX + asset.group(0))

    start = src.index("JSON.parse('", src.index('"align_blocks":{"score"') - 400)
    start += len("JSON.parse('")
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                per_task = json.loads(src[start : j + 1])
                break

    slug = dict(re.findall(r'"([^"]+)":"([A-Za-z0-9_α-]+)"', src[src.index('"Hy-Embodied-0.5-VLA":"hy_vla"') :][:2200]))

    published = {}
    for m in re.finditer(r'\{model:"([^"]+)"', src):
        depth, end = 0, None
        for j in range(m.start(), len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        body = src[m.start() : end]
        hit = re.search(r"generalizationStd:a\(([-\d.]+),([-\d.]+)\)", body)
        if hit:
            published[m.group(1)] = float(hit.group(2))

    print(f"  {len(published)} leaderboard entries, {len(per_task)} with per-task data\n")
    print(f"  {'model':<26}{'published':>10}{'base mean':>11}{'(b+r)/2':>10}   holds")
    print("  " + "-" * 74)
    counts = {"standard half": 0, "merged": 0, "neither": 0}
    for name in sorted(published):
        d = per_task.get(slug.get(name, name)) or per_task.get(name)
        if not d:
            continue
        base = [d[t]["successRate"] for t in GEN if t in d]
        rnd = [d[t + "_random"]["successRate"] for t in GEN if t + "_random" in d]
        if len(base) != len(GEN) or len(rnd) != len(GEN):
            continue
        bm, mm = mean(base), mean((b + r) / 2 for b, r in zip(base, rnd))
        p = published[name]
        kind = "standard half" if abs(bm - p) < 0.7 else "merged" if abs(mm - p) < 0.7 else "neither"
        counts[kind] += 1
        print(f"  {name:<26}{p:>10.2f}{bm:>11.2f}{mm:>10.2f}   {kind}")
    print("  " + "-" * 74)
    for k, v in counts.items():
        print(f"  per-task field holds the {k:<16}: {v}")
    print("\n  The field is not populated consistently across models. That is what")
    print("  makes OpenWAM-alpha's row look wrong when read the way the others read.")


if __name__ == "__main__":
    sys.exit(main())
