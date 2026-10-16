"""Pull human-labeled teleop episodes (success / recover / failure) into the rollout format for verifier calibration.

Source: gs://foundational-research/webxr-teleop/live/indexes/recordings.json (one entry per episode: outcome,
scene_id, scene_commit, trajectory_key, ...). Labels: success = done; recover = done after a retry (suboptimal);
failure = not done. 20 Hz rows = initial_state qpos + states/qpos[2::3], as in make_gt_rollouts.py.

Output: <out>/<scene_id>/<outcome>_<uuid8>/{trajectory.npz, rollout.json}; scene.xml is saved too when the
episode's scene revision differs from data/webxr_scenes/<scene_id> (replay needs the recorded scene then).
usage: pull_episodes.py --out data/webxr_episodes [--scenes ...] [--outcomes success recover failure] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

LIVE = "gs://foundational-research/webxr-teleop/live"
ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((Path(__file__).resolve().parent / "tasks.json").read_text())
SCENES_ROOT = ROOT / "data/webxr_scenes"


def local_revision(scene_id: str) -> str | None:
    st = SCENES_ROOT / scene_id / ".revision.json"
    return json.loads(st.read_text()).get("revision") if st.exists() else None


def pull_one(e: dict, out: Path) -> str:
    sid = e["scene_id"]
    d = out / sid / f"{e['outcome']}_{e['episode_uuid'][:8]}"
    if (d / "rollout.json").exists():
        return "skip"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["gsutil", "-q", "cp", f"{LIVE}/{e['trajectory_key']}", tmp], check=True)
        with h5py.File(Path(tmp) / "trajectory.hdf5") as f:
            g = f["data/demo_0"]
            q = np.concatenate([g["initial_state/qpos"][:][None], g["states/qpos"][2::3]])
            xml = bytes(g["source_files/scene.xml"][:]).decode() if "source_files/scene.xml" in g else None
    rev_match = e.get("scene_commit") == local_revision(sid)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "trajectory.npz", qpos=q)
    if not rev_match and xml:
        (d / "scene.xml").write_text(xml)
    meta = {"scene_id": sid, "seed": d.name, "gt": True, "outcome": e["outcome"], "completed": e["outcome"] != "failure",
            "episode_uuid": e["episode_uuid"], "user_id": e.get("user_id"), "upload_date": e.get("upload_date"),
            "scene_commit": e.get("scene_commit"), "revision_match": rev_match, "source_num_steps": int(e["num_steps"]),
            "task": TASKS[sid]["task"], "steps": len(q) - 1}
    (d / "rollout.json").write_text(json.dumps(meta, indent=1))
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data/webxr_episodes")
    p.add_argument("--scenes", nargs="*", default=None, help="default: every scene in tasks.json")
    p.add_argument("--outcomes", nargs="*", default=["success", "recover", "failure", "suboptimal"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--index", type=Path, default=None, help="local copy of recordings.json (default: fetch)")
    a = p.parse_args()
    if a.index:
        idx = json.loads(a.index.read_text())["entries"]
    else:
        idx = json.loads(subprocess.run(["gsutil", "-q", "cat", f"{LIVE}/indexes/recordings.json"], check=True,
                                        capture_output=True).stdout)["entries"]
    scenes = set(a.scenes or TASKS)
    eps = sorted((e for e in idx.values() if e["scene_id"] in scenes and e["outcome"] in a.outcomes),
                 key=lambda e: (e["scene_id"], e["outcome"], e["episode_uuid"]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "recordings_index.json").write_text(json.dumps({"entries": {e["episode_uuid"]: e for e in eps}}, indent=0))
    print(f"{len(eps)} episodes, {len(scenes)} scenes -> {a.out}", flush=True)
    done = {"ok": 0, "skip": 0, "err": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        def run(e):
            try:
                return e, pull_one(e, a.out)
            except Exception as ex_:  # noqa: BLE001  one bad episode must not stop the pull
                return e, f"err {type(ex_).__name__}: {ex_}"
        for i, (e, r) in enumerate(ex.map(run, eps), 1):
            done["err" if r.startswith("err") else r] += 1
            if r.startswith("err") or i % 100 == 0:
                print(f"[{i}/{len(eps)}] {e['scene_id'][:30]} {e['outcome']} {r}", flush=True)
    print("DONE", done, flush=True)


if __name__ == "__main__":
    main()
