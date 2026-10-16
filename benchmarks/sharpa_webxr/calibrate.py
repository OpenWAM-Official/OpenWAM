"""Verifier vs human labels on data/webxr_episodes (success / recover = completed, failure = not completed).

usage: calibrate.py [--root data/webxr_episodes] [--tsv out.tsv] [--show-errors N]
Per scene: n by label, recall on completed (success_any true), specificity on failures, per-stage achieved rate
split by label, and the worst-offending stage. Scenes whose chain ends in a judge stage are scored on sim_prefix_ok.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((Path(__file__).resolve().parent / "tasks.json").read_text())


def load(root: Path):
    rows = []
    for f in root.glob("*/*/verifier.json"):
        v = json.loads(f.read_text())
        if "error" in v:
            rows.append({"scene_id": v["scene_id"], "dir": str(f.parent), "error": v["error"]})
            continue
        meta = json.loads((f.parent / "rollout.json").read_text())
        v["label"] = meta["outcome"]
        v["completed"] = meta["outcome"] != "failure"
        v["dir"] = str(f.parent)
        rows.append(v)
    return rows


def rate(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "data/webxr_episodes")
    p.add_argument("--tsv", type=Path)
    p.add_argument("--show-errors", type=int, default=0, help="print N false negatives / positives per scene")
    p.add_argument("--scenes", nargs="*")
    a = p.parse_args()
    rows = load(a.root)
    by = defaultdict(list)
    for r in rows:
        if not a.scenes or any(r["scene_id"].startswith(s) for s in a.scenes):
            by[r["scene_id"]].append(r)
    out = []
    print(f"{'scene':8s} {'category':24s} {'n_s/r/f':>9s} {'recall':>7s} {'spec':>6s} {'rec_s':>6s} {'rec_r':>6s}  stage any/final rate: completed|failure")
    tot = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    tot_sim = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}  # scenes whose chain is fully sim-verifiable
    for sid in sorted(by):
        rs = [r for r in by[sid] if "error" not in r]
        errs = [r for r in by[sid] if "error" in r]
        if not rs:
            print(sid[:8], "all errors:", errs[0]["error"][:80] if errs else "")
            continue
        pos = [r for r in rs if r["completed"]]
        neg = [r for r in rs if not r["completed"]]
        key = "sim_prefix_ok"
        tp = sum(r[key] for r in pos); fn = len(pos) - tp
        fp = sum(r[key] for r in neg); tn = len(neg) - fp
        for k, v in zip(("tp", "fn", "tn", "fp"), (tp, fn, tn, fp)):
            tot[k] += v
            if not rs[0]["needs_judge"]:
                tot_sim[k] += v
        stages = [s["name"] for s in rs[0]["stages"]]
        st = []
        for i, name in enumerate(stages):
            if rs[0]["stages"][i]["check"] == "judge":
                st.append(f"{name}:J")
                continue
            ap = rate([r["stages"][i]["achieved"] for r in pos]); an = rate([r["stages"][i]["achieved"] for r in neg])
            fp_ = rate([bool(r["stages"][i]["final"]) for r in pos]); fn_ = rate([bool(r["stages"][i]["final"]) for r in neg])
            st.append(f"{name}:{ap}/{fp_}|{an}/{fn_}")
        n_s = sum(r["label"] == "success" for r in rs); n_r = sum(r["label"] == "recover" for r in rs)
        rec = {"scene": sid[:8], "category": rs[0].get("category"), "n_success": n_s, "n_recover": n_r, "n_failure": len(neg),
               "recall": rate([r[key] for r in pos]), "specificity": rate([not r[key] for r in neg]),
               "recall_success": rate([r[key] for r in pos if r["label"] == "success"]),
               "recall_recover": rate([r[key] for r in pos if r["label"] == "recover"]),
               "fallen_rate": rate([any(r["fallen"].values()) for r in rs]), "n_errors": len(errs), "stages": " ".join(st)}
        out.append(rec)
        print(f"{rec['scene']:8s} {str(rec['category'])[:24]:24s} {n_s:3d}/{n_r:2d}/{len(neg):2d} {str(rec['recall']):>7s} {str(rec['specificity']):>6s} "
              f"{str(rec['recall_success']):>6s} {str(rec['recall_recover']):>6s}  {rec['stages']}" + (f"  ERR={len(errs)}" if errs else ""))
        if a.show_errors:
            for r in [r for r in pos if not r[key]][: a.show_errors]:
                print("   FN", r["dir"].split("/")[-1], [(s["name"], s["achieved"]) for s in r["stages"]], {s["name"]: s["metrics"] for s in r["stages"] if not s["achieved"]})
            for r in [r for r in neg if r[key]][: a.show_errors]:
                print("   FP", r["dir"].split("/")[-1], [(s["name"], round(s["t_first_s"], 1) if s["t_first_s"] is not None else None) for s in r["stages"]])
    # same totals under pure any-time and pure end-state semantics (sim-only scenes)
    def any_time(r):
        return all(s["any_time"] for s in r["stages"]) and not any(r["fallen"].values())

    def end_state(r):  # earlier stages any time, the goal stage still true at the end of the episode
        return all(s["any_time"] for s in r["stages"][:-1]) and r["stages"][-1]["end_state"] and not any(r["fallen"].values())

    for fn_, label in ((any_time, "SIM-ONLY, any-time semantics"), (end_state, "SIM-ONLY, end-state semantics")):
        rs = [r for sid in by for r in by[sid] if "error" not in r and not r["needs_judge"] and "any_time" in r["stages"][0]]
        if rs:
            tp = sum(fn_(r) for r in rs if r["completed"]); fn = sum(not fn_(r) for r in rs if r["completed"])
            fp = sum(fn_(r) for r in rs if not r["completed"]); tn = sum(not fn_(r) for r in rs if not r["completed"])
            print(f"{label}: recall={tp / max(1, tp + fn):.3f} specificity={tn / max(1, tn + fp):.3f} accuracy={(tp + tn) / len(rs):.3f}")
    for name, t in (("ALL (judge scenes scored on sim prefix)", tot), ("SIM-ONLY scenes", tot_sim)):
        n = sum(t.values())
        if n:
            print(f"\n{name}: n={n} recall={t['tp'] / max(1, t['tp'] + t['fn']):.3f} specificity={t['tn'] / max(1, t['tn'] + t['fp']):.3f} "
                  f"accuracy={(t['tp'] + t['tn']) / n:.3f}  (tp={t['tp']} fn={t['fn']} tn={t['tn']} fp={t['fp']})")
    if a.tsv:
        cols = list(out[0].keys())
        with open(a.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in out:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
