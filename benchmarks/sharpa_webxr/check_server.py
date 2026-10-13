"""Conformance check for a policy server (policies/common.py contract). Runs in .venv_webxr, like rollout.py.

Calls "meta" and validates it, then sends real observations from one scene (reset + one follow-up request, at
every row in meta obs_offsets) and validates each response: shapes, finiteness, rotations, frame, distance from
the current pose, latency. A server must pass this before it is evaluated.

usage: check_server.py --port 11600 [--scene <scene_id>] [--max-jump 0.30]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from rollout import TASKS, PolicyClient, _pack, make_env, seed_for

META_KEYS = {"model": str, "codec": str, "chunk": int, "execute_steps": int, "obs_offsets": (list, tuple),
             "control_hz": (int, float), "upstream": str, "sampler": dict, "ckpt": str}
HAND = 22


def check_meta(meta: dict, fail: list[str]) -> None:
    for k, t in META_KEYS.items():
        if k not in meta:
            fail.append(f"meta: missing {k!r}")
        elif not isinstance(meta[k], t):
            fail.append(f"meta: {k!r} is {type(meta[k]).__name__}, expected {t}")
    if fail:
        return
    if not 0 < meta["execute_steps"] <= meta["chunk"]:
        fail.append(f"meta: execute_steps {meta['execute_steps']} not in 1..chunk ({meta['chunk']})")
    if not meta["obs_offsets"] or any(int(o) > 0 for o in meta["obs_offsets"]) or 0 not in map(int, meta["obs_offsets"]):
        fail.append(f"meta: obs_offsets {meta['obs_offsets']} must be <= 0 and include 0")
    for name, hw in (meta.get("cameras") or {}).items():
        if len(hw) != 2:
            fail.append(f"meta: cameras[{name!r}] = {hw}, expected [H, W]")


def check_response(r: dict, meta: dict, state: dict, max_jump: float, tag: str, fail: list[str]) -> None:
    h = int(meta["chunk"])
    for k, shape in (("pos", (h, 2, 3)), ("rot", (h, 2, 3, 3)), ("hand", (h, 2, HAND))):
        if k not in r:
            fail.append(f"{tag}: missing {k!r}")
            continue
        a = np.asarray(r[k])
        if a.shape != shape:
            fail.append(f"{tag}: {k} shape {a.shape}, expected {shape} (chunk={h})")
        elif not np.isfinite(a).all():
            fail.append(f"{tag}: {k} has non-finite values")
    if fail:
        return
    rot = np.asarray(r["rot"], np.float64)
    ortho = np.abs(rot @ np.swapaxes(rot, -1, -2) - np.eye(3)).max()
    det = np.linalg.det(rot)
    if ortho > 1e-3 or np.abs(det - 1).max() > 1e-3:
        fail.append(f"{tag}: rot not a rotation (max |R R^T - I| {ortho:.2e}, det range {det.min():.4f}..{det.max():.4f})")
    if r.get("frame", "ego_camera") != state.get("frame", "ego_camera"):
        fail.append(f"{tag}: frame {r.get('frame')!r} != observation frame {state.get('frame')!r}")
    jump = np.linalg.norm(np.asarray(r["pos"])[0] - np.asarray(state["pos"])[0], axis=-1).max()
    if jump > max_jump:  # absolute poses are expected: a large first-step jump usually means a delta was returned
        fail.append(f"{tag}: first action is {jump:.3f} m from the current wrist pose (> {max_jump} m); "
                    "pos must be ABSOLUTE in the observation frame")
    if "raw" not in r:
        fail.append(f"{tag}: missing 'raw' (model output before decode, logged for debugging)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--scene", default=sorted(TASKS)[0])
    p.add_argument("--max-jump", type=float, default=0.30, help="max first-step wrist distance from the state (m)")
    a = p.parse_args()

    fail: list[str] = []
    policy = PolicyClient(a.port)
    meta = policy.meta
    print(f"[meta] {meta}", flush=True)
    check_meta(meta, fail)
    if fail:
        sys.exit("FAIL\n  " + "\n  ".join(fail))

    env = make_env(meta, a.scene)
    offsets = sorted(int(o) for o in meta["obs_offsets"])
    obs0 = _pack(env.reset(seed_for(a.scene, 0)))
    req = {"obs": {o: obs0 for o in offsets}, "task": TASKS[a.scene]["task"], "seed": 0, "reset": True,
           "prev_cmd": None, "executed": None}
    lat = []
    for i in range(2):
        t0 = time.perf_counter()
        r = policy(req)
        lat.append(time.perf_counter() - t0)
        check_response(r, meta, req["obs"][0]["state"], a.max_jump, f"request {i}", fail)
        if fail:
            break
        n = int(meta["execute_steps"])  # follow-up as the runner sends it after executing n actions
        last = {k: np.asarray(r[k])[n - 1:n] for k in ("pos", "rot", "hand")}
        req = dict(req, reset=False, seed=1, prev_cmd=dict(last, frame=obs0["state"].get("frame", "ego_camera")),
                   executed=n)
    print(f"[latency] {', '.join(f'{x:.2f}s' for x in lat)} (first includes warm-up)", flush=True)
    if fail:
        sys.exit("FAIL\n  " + "\n  ".join(fail))
    print(f"PASS {meta['model']}: chunk {meta['chunk']}, execute {meta['execute_steps']}, offsets {offsets}")


if __name__ == "__main__":
    main()
