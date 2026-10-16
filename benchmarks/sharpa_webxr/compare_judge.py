"""Verifier vs VLM judge on policy rollouts: per category / scene confusion between verifier.json (success_any or
sim_prefix_ok) and judge_<model>_r2.json (success), with the disagreeing rollout dirs listed for review.

usage: compare_judge.py <folder> [--model gpt-6-luna] [--show N] [--tsv out.tsv]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = json.loads((HERE / "task_categories.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--show", type=int, default=3, help="disagreements to list per scene")
    p.add_argument("--tsv", type=Path)
    a = p.parse_args()
    rows = []
    for vf in a.folder.rglob("verifier.json"):
        d = vf.parent
        jf = d / f"judge_{a.model}_r2.json"
        if not jf.exists():
            continue
        v, j = json.loads(vf.read_text()), json.loads(jf.read_text())
        if "error" in v or "error" in j:
            continue
        rows.append({"dir": str(d), "scene": v["scene_id"], "cat": CATS.get(v["scene_id"], "?"), "needs_judge": v["needs_judge"],
                     "ver": bool(v["sim_prefix_ok"]), "ver_any": bool(v.get("sim_prefix_any_time", v["sim_prefix_ok"])),
                     "ver_end": bool(v.get("sim_prefix_end_state", v["sim_prefix_ok"])), "judge": bool(j["success"]), "judge_conf": j.get("confidence"),
                     "stages": [(s["name"], s["achieved"]) for s in v["stages"]], "rationale": j.get("rationale", "")[:160],
                     "j_stages": {k: j.get(k) for k in ("reach", "grasp", "move", "complete")}})
    by_cat, by_scene = defaultdict(list), defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
        by_scene[r["scene"]].append(r)

    def conf(rs):
        n = len(rs)
        both = sum(r["ver"] and r["judge"] for r in rs)
        vonly = sum(r["ver"] and not r["judge"] for r in rs)
        jonly = sum(r["judge"] and not r["ver"] for r in rs)
        neither = n - both - vonly - jonly
        return n, both, vonly, jonly, neither

    print(f"{'category':26s} {'n':>4s} {'ver_SR':>7s} {'judge_SR':>8s} {'agree':>6s} {'both':>5s} {'ver_only':>8s} {'judge_only':>10s}")
    out = []
    for cat, rs in sorted(by_cat.items()):
        n, both, vo, jo, ne = conf(rs)
        rec = {"category": cat, "n": n, "ver_sr": round(sum(r["ver"] for r in rs) / n, 3), "ver_any_sr": round(sum(r["ver_any"] for r in rs) / n, 3),
               "ver_end_sr": round(sum(r["ver_end"] for r in rs) / n, 3), "judge_sr": round(sum(r["judge"] for r in rs) / n, 3),
               "agree": round((both + ne) / n, 3), "both": both, "ver_only": vo, "judge_only": jo}
        out.append(rec)
        print(f"{cat:26s} {n:4d} {rec['ver_sr']:7.3f} {rec['judge_sr']:8.3f} {rec['agree']:6.3f} {both:5d} {vo:8d} {jo:10d}   any={rec['ver_any_sr']:.3f} end={rec['ver_end_sr']:.3f}")
    n, both, vo, jo, ne = conf(rows)
    print(f"{'ALL':26s} {n:4d} {sum(r['ver'] for r in rows) / n:7.3f} {sum(r['judge'] for r in rows) / n:8.3f} {(both + ne) / n:6.3f} {both:5d} {vo:8d} {jo:10d}")
    sim_rows = [r for r in rows if not r["needs_judge"]]
    n, both, vo, jo, ne = conf(sim_rows)
    print(f"{'SIM-ONLY scenes':26s} {n:4d} {sum(r['ver'] for r in sim_rows) / n:7.3f} {sum(r['judge'] for r in sim_rows) / n:8.3f} {(both + ne) / n:6.3f} {both:5d} {vo:8d} {jo:10d}")
    # judge-ending scenes: how often the sim prefix gate would have skipped a judge call, and judge successes it would have blocked
    jr = [r for r in rows if r["needs_judge"]]
    if jr:
        blocked = sum(not r["ver"] for r in jr)
        blocked_js = sum((not r["ver"]) and r["judge"] for r in jr)
        print(f"\njudge-ending scenes: {len(jr)} rollouts; sim prefix fails {blocked} ({blocked / len(jr):.0%}) -> judge calls saved; "
              f"of those the judge said success in {blocked_js}")

    print(f"\n{'scene':8s} {'category':24s} {'n':>3s} {'ver':>4s} {'jdg':>4s} {'both':>4s} {'v!j':>4s} {'j!v':>4s}")
    for sc, rs in sorted(by_scene.items()):
        n, both, vo, jo, ne = conf(rs)
        print(f"{sc[:8]:8s} {rs[0]['cat'][:24]:24s} {n:3d} {sum(r['ver'] for r in rs):4d} {sum(r['judge'] for r in rs):4d} {both:4d} {vo:4d} {jo:4d}")
        for r in [r for r in rs if r["ver"] != r["judge"]][: a.show]:
            tag = "VER-only" if r["ver"] else "JUDGE-only"
            print(f"    {tag:10s} {r['dir']}  stages={r['stages']}  judge={r['j_stages']} ({r['judge_conf']}) {r['rationale']}")
    if a.tsv:
        cols = list(out[0].keys())
        with open(a.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in out:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
