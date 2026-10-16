"""State-based task verifier for Sharpa sim rollouts, driven by per-scene stage chains in task_specs.json.

A spec is an ordered list of stages. Each stage is a sim check (grasp, place_in, joint_goal, ...) evaluated on the
replayed trajectory (trajectory.npz qpos through the scene model), or {"check": "judge"} for a final stage only a
vision judge can decide (pour outcome, stirring, ...). Output per rollout: verifier.json with every stage's
achieved flag + first time, success_any (every sim stage achieved, in order), needs_judge, and a short text summary
of the sim facts for the judge prompt.

usage: verify.py <rollout dirs...> [--force] [--max-steps N] [--specs task_specs.json]
Thresholds are provisional until calibrated on data/webxr_episodes (human success / recover / failure labels).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco  # noqa: E402
from env import SharpaWebXREnv  # noqa: E402
from limits import limit  # noqa: E402
from postprocess import DT, SCENES_ROOT, Analyzer  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SPECS_PATH = HERE / "task_specs.json"
DEFAULTS = {"hold_s": 0.5, "reach_m": 0.03, "lift_m": 0.03, "carry_m": 0.05, "rest_speed_m_s": 0.03,
            "xy_margin_m": 0.0, "z_tol_m": 0.03, "tilt_deg": 15.0}


def rot(q_wxyz):
    return Rotation.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]]).as_matrix()


class Replay:
    """Per-frame state of every scene object + hand/object contact pairs, for the whole trajectory."""

    def __init__(self, env: SharpaWebXREnv, qpos: np.ndarray):
        self.env, self.an = env, Analyzer(env)
        m, d = self.an.m, self.an.d
        self.n = len(qpos)
        self.t = np.arange(self.n) * DT
        self.obj = {o: {"pos": np.zeros((self.n, 3)), "quat": np.zeros((self.n, 4)), "contact": np.zeros(self.n, bool),
                        "dist": np.zeros(self.n), "joints": np.zeros((self.n, len(v["joints"])))} for o, v in self.an.objects.items()}
        self.pairs = []
        self.wrist = np.zeros((self.n, 2, 3))
        self.base = None
        # local AABB of each object's collision geometry (object frame), for footprints / regions
        self.local_aabb = {}
        for o, v in self.an.objects.items():
            pts = []
            for g in v["geoms"]:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    vs = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    pts.append((g, vs))
                else:
                    pts.append((g, np.array([[-1, -1, -1], [1, 1, 1]]) * m.geom_size[g][:3]))
            d.qpos[:] = qpos[0]
            mujoco.mj_forward(m, d)
            R, p = d.xmat[v["body"]].reshape(3, 3), d.xpos[v["body"]]
            P = np.concatenate([(d.geom_xmat[g].reshape(3, 3) @ vs.T).T + d.geom_xpos[g] for g, vs in pts])
            loc = (R.T @ (P - p).T).T
            self.local_aabb[o] = np.stack([loc.min(0), loc.max(0)])
        for i in range(self.n):
            out, pairs, wrist = self.an.frame(qpos[i])
            for o, s in out.items():
                r = self.obj[o]
                r["pos"][i], r["quat"][i], r["contact"][i], r["dist"][i] = s["pos"], s["quat"], s["contact"], s["dist"]
                if s["joints"].size:
                    r["joints"][i] = s["joints"]
            self.pairs.append(pairs)
            self.wrist[i] = wrist
            if i == 0:
                self.base = env.world_from_base()
        self.table_z = self.an.table_z
        self.joint_names = {o: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in v["joints"]]
                            for o, v in self.an.objects.items()}

    # ---- helpers -----------------------------------------------------------------------------
    def point(self, o: str, local=None) -> np.ndarray:
        """World trajectory of a point given in the object frame (default: body origin)."""
        if local is None:
            return self.obj[o]["pos"]
        P = self.obj[o]["pos"]
        return np.stack([P[i] + rot(self.obj[o]["quat"][i]) @ np.asarray(local, float) for i in range(self.n)])

    def target_box(self, target: str, i: int, box=None) -> tuple[np.ndarray, np.ndarray]:
        """World AABB of the target at frame i (its local AABB, or `box` override in the target frame, rotated by
        the current pose). Gravity-aligned, so it works for targets lying on any side."""
        lo, hi = np.asarray(box, float) if box is not None else self.local_aabb[target]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = (rot(self.obj[target]["quat"][i]) @ corners.T).T + self.obj[target]["pos"][i]
        return W.min(0), W.max(0)

    def in_target(self, o: str, target: str, point=None, xy_margin=0.0, z_lo=None, z_hi=None, box=None,
                  z_above=0.05) -> np.ndarray:
        """Per frame: object point inside the target's world AABB footprint (shrunk by xy_margin; negative = grown)
        with z in [bottom - 1 cm (or z_lo offset from bottom), top + z_above (or z_hi offset from top)]."""
        pw = self.point(o, point)
        ok = np.zeros(self.n, bool)
        for i in range(self.n):
            lo, hi = self.target_box(target, i, box)
            zl = lo[2] + (-0.01 if z_lo is None else z_lo)
            zh = hi[2] + (z_above if z_hi is None else z_hi)
            ok[i] = (lo[0] + xy_margin <= pw[i, 0] <= hi[0] - xy_margin and lo[1] + xy_margin <= pw[i, 1] <= hi[1] - xy_margin
                     and zl <= pw[i, 2] <= zh)
        return ok

    def in_region(self, o: str, region: dict, point=None) -> np.ndarray:
        """Static world/base region: {"frame": "base"|"world", "xy": [[x0,y0],[x1,y1]], "z": [z0,z1]}."""
        pw = self.point(o, point)
        if region.get("frame", "base") == "base":
            Tb = np.linalg.inv(self.base)
            pw = (Tb[:3, :3] @ pw.T).T + Tb[:3, 3]
        (x0, y0), (x1, y1) = region["xy"]
        ok = (pw[:, 0] >= x0) & (pw[:, 0] <= x1) & (pw[:, 1] >= y0) & (pw[:, 1] <= y1)
        if "z" in region:
            ok &= (pw[:, 2] >= region["z"][0]) & (pw[:, 2] <= region["z"][1])
        return ok

    def speed(self, o: str) -> np.ndarray:
        P = self.obj[o]["pos"]
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / DT
        return np.concatenate([[0.0], v])

    def ang_speed(self, o: str) -> np.ndarray:
        Q = self.obj[o]["quat"]
        w = [0.0] + [Rotation.from_matrix(rot(Q[i - 1]).T @ rot(Q[i])).magnitude() / DT for i in range(1, self.n)]
        return np.asarray(w)

    def touching(self, a: str, b: str) -> np.ndarray:
        return np.array([(a, b) in p or (b, a) in p for p in self.pairs])

    def hand_contact(self, o: str) -> np.ndarray:
        return self.obj[o]["contact"]

    def lift(self, o: str) -> np.ndarray:
        return self.obj[o]["pos"][:, 2] - self.obj[o]["pos"][0, 2]

    def disp(self, o: str) -> np.ndarray:
        return np.linalg.norm(self.obj[o]["pos"] - self.obj[o]["pos"][0], axis=1)

    def axis(self, o: str, local_axis) -> np.ndarray:
        a = np.asarray(local_axis, float)
        return np.stack([rot(q) @ a for q in self.obj[o]["quat"]])

    def joint(self, o: str, name: str | None = None) -> np.ndarray:
        names = self.joint_names[o]
        if not names:
            raise KeyError(f"{o} has no articulated joint")
        return self.obj[o]["joints"][:, names.index(name) if name else 0]

    def fallen(self, o: str) -> np.ndarray:
        return self.obj[o]["pos"][:, 2] < self.table_z - 0.10 if self.table_z is not None else np.zeros(self.n, bool)


def run_length(mask: np.ndarray) -> np.ndarray:
    """Length (frames) of the true-run ending at each frame."""
    out = np.zeros(len(mask), int)
    for i, m in enumerate(mask):
        out[i] = out[i - 1] + 1 if m and i else int(m)
    return out


def fill_gaps(mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Bridge false runs of <= max_gap frames between true frames (hand/object contact flickers at 20 Hz)."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a:b] = True
    return out


def held(mask: np.ndarray, hold_s: float) -> np.ndarray:
    return run_length(mask) >= max(1, int(round(hold_s / DT)))


def first_true(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    return (int(idx[0]), float(idx[0] * DT)) if len(idx) else (None, None)


# ---- stage checks: each returns (per-frame achieved mask, metrics) -----------------------------
def c_reach(r: Replay, s):
    return r.obj[s["obj"]]["dist"] < s.get("reach_m", DEFAULTS["reach_m"]), {"min_dist_m": float(r.obj[s["obj"]]["dist"].min())}


def c_grasp(r: Replay, s):
    """Hand contact sustained hold_s while the object is lifted >= lift_m or carried >= carry_m."""
    o = s["obj"]
    moved = (r.lift(o) >= s.get("lift_m", DEFAULTS["lift_m"])) | (r.disp(o) >= s.get("carry_m", DEFAULTS["carry_m"]))
    ok = held(fill_gaps(r.hand_contact(o)), s.get("hold_s", DEFAULTS["hold_s"])) & moved
    return ok, {"max_lift_m": float(r.lift(o).max()), "contact_s": float(r.hand_contact(o).sum() * DT)}


def c_hold_lifted(r: Replay, s):
    """Object lifted >= lift_m with hand contact for hold_s continuously (pick / grasp-only tasks)."""
    o = s["obj"]
    C, z = r.hand_contact(o), r.lift(o)
    up = fill_gaps(C) & (z >= s.get("lift_m", DEFAULTS["lift_m"]))
    ok = held(up, s.get("hold_s", DEFAULTS["hold_s"]))
    max_hold = float(run_length(up).max() * DT)
    # end-state diagnostics: was it put down gently (released while low/slow) or dropped?
    last_c = int(np.flatnonzero(C)[-1]) if C.any() else -1
    vz = np.concatenate([[0.0], np.diff(r.obj[o]["pos"][:, 2]) / DT])
    after = vz[last_c + 1:] if 0 <= last_c < r.n - 1 else np.zeros(1)
    fall = float(-after.min())
    if s.get("end_rule") == "not_dropped":
        # labelled on the end state: still held, or set down gently (no fast fall after the hand let go)
        keep = bool(C[-1]) or fall <= s.get("max_drop_speed_m_s", 0.3)
        ok = ok if keep else np.zeros(r.n, bool)
    return ok, {"max_lift_m": float(z.max()), "held_any": bool(ok.any()), "max_hold_s": max_hold, "final_lift_m": float(z[-1]), "final_contact": bool(C[-1]),
                "lift_at_release_m": float(z[last_c]) if last_c >= 0 else 0.0, "max_fall_speed_after_release": fall,
                "release_to_end_s": float((r.n - 1 - last_c) * DT) if last_c >= 0 else None}


def _rest(r: Replay, o, s):
    return r.speed(o) <= s.get("rest_speed_m_s", DEFAULTS["rest_speed_m_s"])


def _placed_common(r: Replay, s, inside):
    o = s["obj"]
    ok = inside
    if s.get("release", True):
        ok = ok & ~r.hand_contact(o)
    if s.get("contact"):
        ok = ok & r.touching(o, s["contact"])
    if s.get("level_deg"):
        ok = ok & (r.axis(o, s.get("up_axis", [0, 0, 1]))[:, 2] >= np.cos(np.deg2rad(s["level_deg"])))
    if s.get("rest", True):
        ok = ok & _rest(r, o, s)
    ok = held(ok, s.get("hold_s", DEFAULTS["hold_s"]))
    return ok


def c_place_in(r: Replay, s):
    """Object point inside target footprint + z range, released, at rest (optionally touching target)."""
    inside = r.in_target(s["obj"], s["target"], s.get("point"), s.get("xy_margin_m", DEFAULTS["xy_margin_m"]),
                         s.get("z_lo"), s.get("z_hi"), s.get("target_box"), s.get("z_above_m", 0.05))
    return _placed_common(r, s, inside), {"inside_s": float(inside.sum() * DT)}


def c_place_region(r: Replay, s):
    inside = r.in_region(s["obj"], s["region"], s.get("point"))
    return _placed_common(r, s, inside), {"inside_s": float(inside.sum() * DT)}


def c_move_out(r: Replay, s):
    """Object taken out of a container: point outside target footprint (or region) and displaced >= min_disp_m."""
    o = s["obj"]
    if "target" in s:
        inside = r.in_target(o, s["target"], s.get("point"), -s.get("xy_margin_m", 0.02), -1.0, 1.0, s.get("target_box"))
    elif "region" in s:
        inside = r.in_region(o, s["region"], s.get("point"))
    else:  # container not simulated: "out" = displaced from where it started
        inside = r.disp(o) < s.get("min_disp_m", 0.05)
    ok = ~inside & (r.disp(o) >= s.get("min_disp_m", 0.05))
    if s.get("release", True):
        ok = ok & ~r.hand_contact(o)
    if s.get("rest", True):
        ok = ok & _rest(r, o, s)
    ok = held(ok, s.get("hold_s", DEFAULTS["hold_s"]))
    return ok, {"max_disp_m": float(r.disp(o).max()), "final_inside": bool(inside[-1])}


def c_joint_goal(r: Replay, s):
    """Articulated joint reaches a goal: 'min'/'max' absolute, or 'delta' from q0 (sign = direction), held hold_s."""
    q = r.joint(s["obj"], s.get("joint"))
    ok = np.ones_like(q, bool)
    if "delta" in s:  # relative to the start; either_direction accepts |dq| >= |delta|
        dq = q - q[0]
        ok &= np.abs(dq) >= abs(s["delta"]) if s.get("either_direction") else (dq >= s["delta"] if s["delta"] > 0 else dq <= s["delta"])
    if "min" in s:
        ok &= q >= s["min"]
    if "max" in s:
        ok &= q <= s["max"]
    if s.get("release"):
        ok = ok & ~r.hand_contact(s["obj"])
    speed = np.concatenate([[0.0], np.abs(np.diff(q)) / DT])
    peak = float(np.maximum.accumulate(speed).max())
    if s.get("max_speed"):  # gentle: the joint never moved faster than this before the goal was reached
        ok = ok & (np.maximum.accumulate(speed) <= s["max_speed"])
    ok = held(ok, s.get("hold_s", DEFAULTS["hold_s"]))
    return ok, {"q0": float(q[0]), "q_min": float(q.min()), "q_max": float(q.max()), "q_final": float(q[-1]), "peak_speed": peak}


def c_relative_pose(r: Replay, s):
    """Lid/cap on body: obj point in ref frame within pos_tol of `offset` (ref-local), tilt <= tilt_deg, released,
    at rest, optionally touching ref. `offset` comes from calibration on success episodes (None = xy over ref center
    within xy_tol and z within z range of the ref's aabb top)."""
    o, ref = s["obj"], s["ref"]
    pw = r.point(o, s.get("point"))
    tp, tq = r.obj[ref]["pos"], r.obj[ref]["quat"]
    loc = np.stack([rot(tq[i]).T @ (pw[i] - tp[i]) for i in range(r.n)])
    if s.get("offset") is not None:
        err = np.linalg.norm(loc - np.asarray(s["offset"]), axis=1)
        inside = err <= s.get("pos_tol_m", 0.02)
    else:  # on top of the ref: over its world-AABB centre within xy_tol, z between (top - z_tol) and (top + z_above)
        inside = np.zeros(r.n, bool)
        err = np.zeros(r.n)
        for i in range(r.n):
            lo, hi = r.target_box(ref, i, s.get("target_box"))
            c = (lo + hi) / 2
            err[i] = np.linalg.norm(pw[i, :2] - c[:2])
            inside[i] = err[i] <= s.get("xy_tol_m", 0.03) and hi[2] - s.get("z_tol_m", DEFAULTS["z_tol_m"]) <= pw[i, 2] <= hi[2] + s.get("z_above_m", 0.08)
    Ro, Rr = [rot(q) for q in r.obj[o]["quat"]], [rot(q) for q in tq]
    tilt = np.array([np.arccos(np.clip(abs((Rr[i].T @ Ro[i])[2, 2]), 0, 1)) for i in range(r.n)])
    ok = inside & (tilt <= np.deg2rad(s.get("tilt_deg", DEFAULTS["tilt_deg"])))
    ok = _placed_common(r, {**s, "level_deg": None}, ok)
    return ok, {"min_err_m": float(err.min()), "final_err_m": float(err[-1]), "final_tilt_deg": float(np.degrees(tilt[-1])),
                "final_local": loc[-1].round(4).tolist(), "final_touching": bool(r.touching(o, ref)[-1])}


def c_attached(r: Replay, s):
    """Cap/lid seated on a body without a known offset: obj point within max_dist_m of the ref point, touching the
    ref, released, at rest (calibrate to relative_pose with an offset once success episodes give one)."""
    o, ref = s["obj"], s["ref"]
    dist = np.linalg.norm(r.point(o, s.get("point")) - r.point(ref, s.get("ref_point")), axis=1)
    ok = (dist <= s.get("max_dist_m", 0.05)) & r.touching(o, ref)
    ok = _placed_common(r, {**s, "contact": None}, ok)
    loc = rot(r.obj[ref]["quat"][-1]).T @ (r.point(o, s.get("point"))[-1] - r.obj[ref]["pos"][-1])
    return ok, {"min_dist_m": float(dist.min()), "final_dist_m": float(dist[-1]), "final_local": loc.round(4).tolist(),
                "final_touching": bool(r.touching(o, ref)[-1])}


def c_orientation(r: Replay, s):
    """mode flip: object's local `axis` z-component <= -cos(tol) (upside down). mode rotate: rotation angle from the
    initial orientation >= min_deg (about any axis, or `about` axis). mode face: local `axis` projected on the base xy
    plane points within tol_deg of base direction `toward` ([x,y] in base frame). Held hold_s, at rest, released."""
    o = s["obj"]
    mode = s["mode"]
    R0 = rot(r.obj[o]["quat"][0])
    total = np.array([Rotation.from_matrix(rot(q) @ R0.T).magnitude() for q in r.obj[o]["quat"]])
    Rb = r.base[:3, :3]
    xb = (Rb.T @ r.axis(o, s.get("heading_axis", [1, 0, 0])).T).T
    yaw_base = np.degrees(np.arctan2(xb[:, 1], xb[:, 0]))
    if mode == "flip":  # the axis that pointed up at the start now points down
        a = r.axis(o, s.get("axis", [0, 0, 1]))
        up0 = np.sign(a[0, 2]) or 1.0
        ok = a[:, 2] * up0 <= -np.cos(np.deg2rad(s.get("tol_deg", 35)))
        metric = {"min_axis_z_rel": float((a[:, 2] * up0).min()), "final_axis_z_rel": float(a[-1, 2] * up0)}
    elif mode == "rotate":
        R0 = rot(r.obj[o]["quat"][0])
        if "about" in s:
            ax = np.asarray(s["about"], float)
            ang = []
            for q in r.obj[o]["quat"]:
                rv = Rotation.from_matrix(rot(q) @ R0.T).as_rotvec()
                ang.append(abs(rv @ ax) if s.get("world_axis", True) else np.linalg.norm(rv))
            ang = np.asarray(ang)
        else:
            ang = np.array([Rotation.from_matrix(rot(q) @ R0.T).magnitude() for q in r.obj[o]["quat"]])
        ok = np.degrees(ang) >= s["min_deg"]
        if s.get("max_deg"):
            ok &= np.degrees(ang) <= s["max_deg"]
        metric = {"max_rot_deg": float(np.degrees(ang).max()), "final_rot_deg": float(np.degrees(ang[-1]))}
    elif mode == "face":
        a = r.axis(o, s["axis"])
        Rb = r.base[:3, :3]
        ab = (Rb.T @ a.T).T[:, :2]
        tw = np.asarray(s.get("toward", [-1, 0]), float)
        cosang = (ab @ tw) / (np.linalg.norm(ab, axis=1) * np.linalg.norm(tw) + 1e-9)
        ok = cosang >= np.cos(np.deg2rad(s.get("tol_deg", 30)))
        metric = {"final_face_deg": float(np.degrees(np.arccos(np.clip(cosang[-1], -1, 1))))}
    else:
        raise ValueError(mode)
    if s.get("keep_upright_deg"):
        ok &= r.axis(o, [0, 0, 1])[:, 2] >= np.cos(np.deg2rad(s["keep_upright_deg"]))
    if s.get("release", True):
        ok = ok & ~r.hand_contact(o)
    if s.get("rest", True):
        ok = ok & _rest(r, o, s)
    ok = held(ok, s.get("hold_s", DEFAULTS["hold_s"]))
    metric.update({"max_total_rot_deg": float(np.degrees(total).max()), "final_total_rot_deg": float(np.degrees(total[-1])),
                   "yaw_base0_deg": float(yaw_base[0]), "final_yaw_base_deg": float(yaw_base[-1])})
    return ok, metric


def c_oscillation(r: Replay, s):
    """Shake: while hand-held, the object reverses direction >= min_reversals times inside a window_s window, each
    leg travelling >= min_amp_m along the dominant motion axis."""
    o = s["obj"]
    P, C = r.obj[o]["pos"], r.hand_contact(o)
    V = np.diff(P, axis=0)
    held_v = V[C[1:]] if C[1:].any() else V
    if len(held_v) < 3:
        return np.zeros(r.n, bool), {"max_reversals_in_window": 0}
    ax = np.linalg.svd(held_v - held_v.mean(0), full_matrices=False)[2][0]
    x = P @ ax
    amp = s.get("min_amp_m", 0.03)
    # turning points with hysteresis: a new extremum counts once the signal has moved >= amp back from it
    ext, direction, cand = [0], 0, 0
    for i in range(1, r.n):
        if direction == 0:
            if abs(x[i] - x[ext[-1]]) >= amp:
                direction, cand = np.sign(x[i] - x[ext[-1]]), i
        elif (x[i] - x[cand]) * direction > 0:
            cand = i  # still travelling the same way: extend the leg
        elif abs(x[i] - x[cand]) >= amp:
            ext.append(cand)  # reversal confirmed
            direction, cand = -direction, i
    ext = np.array(ext[1:], int)
    win = int(round(s.get("window_s", 3.0) / DT))
    counts = np.array([int(((ext <= i) & (ext > i - win)).sum()) for i in range(r.n)])
    ok = counts >= s.get("min_reversals", 4)
    if s.get("while_held", True):
        ok &= C
    metrics = {"max_reversals_in_window": int(counts.max()), "n_reversals": int(len(ext))}
    for a2 in (0.02, 0.03, 0.05, 0.08):  # diagnostics at other amplitudes, for threshold calibration
        e2, d2, c2 = [0], 0, 0
        for i in range(1, r.n):
            if d2 == 0:
                if abs(x[i] - x[e2[-1]]) >= a2:
                    d2, c2 = np.sign(x[i] - x[e2[-1]]), i
            elif (x[i] - x[c2]) * d2 > 0:
                c2 = i
            elif abs(x[i] - x[c2]) >= a2:
                e2.append(c2)
                d2, c2 = -d2, i
        e2 = np.array(e2[1:], int)
        metrics[f"rev{int(a2 * 100)}cm"] = int(max((((e2 <= i) & (e2 > i - win)).sum() for i in range(r.n)), default=0))
    return ok, metrics


def c_free_flight(r: Replay, s):
    """Toss and catch, from kinematics (hand contact is unreliable for a coin leaving the palm): the object rises
    >= min_rise_m relative to the nearest wrist within rise_s, comes back down by >= 60% of that within fall_s, and
    is then within catch_dist_m of a wrist for catch_hold_s (caught, not dropped on the table)."""
    o = s["obj"]
    P = r.obj[o]["pos"]
    dw = np.linalg.norm(P[:, None, :] - r.wrist, axis=2)  # (n, 2) distance to each wrist
    near = dw.min(1)
    rel = P[:, 2] - r.wrist[:, :, 2].max(1)
    rise_n, fall_n = int(round(s.get("rise_s", 0.6) / DT)), int(round(s.get("fall_s", 1.0) / DT))
    hold_n = max(1, int(round(s.get("catch_hold_s", 0.3) / DT)))
    min_rise = s.get("min_rise_m", 0.05)
    ok = np.zeros(r.n, bool)
    best = 0.0
    z = P[:, 2]
    for i in range(1, r.n):  # i = apex candidate
        j0 = max(0, i - rise_n)
        rise = z[i] - z[j0:i + 1].min()  # the object itself went up ...
        rel_rise = rel[i] - rel[j0:i + 1].min()  # ... and away from the hand (not just lifted with it)
        best = max(best, rel_rise)
        if rise < min_rise or rel_rise < s.get("min_rel_rise_m", 0.02):
            continue
        k1 = min(r.n, i + fall_n + 1)
        down = np.flatnonzero(z[i:k1] <= z[i] - 0.6 * rise)
        if not len(down):
            continue
        k = i + int(down[0])
        caught = (near[k:k + hold_n] <= s.get("catch_dist_m", 0.22)).all() and (r.lift(o)[k:k + hold_n] >= s.get("min_catch_lift_m", 0.02)).all()
        if k + hold_n <= r.n and caught and not r.fallen(o)[k:k + hold_n].any():
            ok[k + hold_n - 1:] = True
            break
    return ok, {"best_rel_rise_m": float(best), "final_near_wrist_m": float(near[-1])}


def c_tilt_over(r: Replay, s):
    """Pour: object's local up axis tilted >= min_tilt_deg while `point` (spout, object-local) is inside the target
    footprint and above the target top, held hold_s, while hand-held."""
    o, tgt = s["obj"], s["target"]
    up = r.axis(o, s.get("up_axis", [0, 0, 1]))
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    over = r.in_target(o, tgt, s.get("point"), -s.get("xy_slack_m", 0.03), None, s.get("z_above_m", 0.30), s.get("target_box"))
    ok = (tilt >= s.get("min_tilt_deg", 60)) & over
    if s.get("while_held", True):
        ok &= r.hand_contact(o)
    ok = held(ok, s.get("hold_s", 0.3))
    return ok, {"max_tilt_deg": float(tilt.max()), "max_tilt_over_target_deg": float(tilt[over].max()) if over.any() else 0.0,
                "over_s": float(over.sum() * DT)}


def c_contact_events(r: Replay, s):
    """Tool strikes: count onsets of tool-target contact while the tool is hand-held (hit, tap, drum)."""
    tool, tgt = s["obj"], s["target"]
    T = r.touching(tool, tgt)
    onsets = np.zeros(r.n, bool)
    onsets[1:] = T[1:] & ~T[:-1]
    if s.get("while_held", True):
        onsets &= r.hand_contact(tool)
    if s.get("min_speed_m_s"):
        v = r.speed(tool)
        onsets &= np.concatenate([[False], v[:-1] >= s["min_speed_m_s"]]) | (v >= s["min_speed_m_s"])
    cum = np.cumsum(onsets)
    return cum >= s.get("min_events", 1), {"events": int(cum[-1]), "contact_s": float(T.sum() * DT)}


def c_sweep(r: Replay, s):
    """Tool point moves inside the target footprint (stir, brush, scramble, scoop): cumulative path length inside
    >= min_path_m, optionally requiring tool-target contact, while hand-held."""
    tool, tgt = s["obj"], s["target"]
    pw = r.point(tool, s.get("point"))
    inside = (r.in_target(tool, tgt, s.get("point"), -s.get("xy_slack_m", 0.0), s.get("z_lo"), s.get("z_hi"), s.get("target_box"), s.get("z_above_m", 0.15))
              if s.get("require_inside", True) else np.ones(r.n, bool))
    if s.get("contact"):
        inside &= r.touching(tool, tgt)
    if s.get("while_held", True):
        inside &= r.hand_contact(tool)
    step = np.concatenate([[0.0], np.linalg.norm(np.diff(pw, axis=0), axis=1)])
    path = np.cumsum(np.where(inside, step, 0.0))
    return path >= s.get("min_path_m", 0.2), {"path_in_target_m": float(path[-1]), "inside_s": float(inside.sum() * DT)}


def c_judge(r: Replay, s):
    return np.zeros(r.n, bool), {}


CHECKS = {"reach": c_reach, "grasp": c_grasp, "hold_lifted": c_hold_lifted, "place_in": c_place_in,
          "place_region": c_place_region, "move_out": c_move_out, "joint_goal": c_joint_goal,
          "relative_pose": c_relative_pose, "attached": c_attached, "orientation": c_orientation, "oscillation": c_oscillation,
          "free_flight": c_free_flight, "tilt_over": c_tilt_over, "contact_events": c_contact_events,
          "sweep": c_sweep, "judge": c_judge}


def verify(rollout_dir: Path, env: SharpaWebXREnv, spec: dict, max_steps: int | None = None) -> dict:
    meta = json.loads((rollout_dir / "rollout.json").read_text())
    qpos = np.load(rollout_dir / "trajectory.npz")["qpos"]
    if max_steps is not None:
        qpos = qpos[: max_steps + 1]
    r = Replay(env, qpos)
    stages = []
    for st in spec["stages"]:
        mask, metrics = CHECKS[st["check"]](r, st)
        i0, t0 = first_true(mask)
        any_time, end_state = i0 is not None, bool(mask[-1])
        # the spec decides which semantics this stage is graded on: end state for tasks the human labels judge
        # at the end (still holding, lid still closed), any time otherwise (RoboTwin "done on first success")
        achieved = end_state if st.get("final") else any_time
        stages.append({"name": st.get("name", st["check"]), "check": st["check"], "achieved": achieved, "any_time": any_time,
                       "end_state": end_state, "t_first_s": t0, "final": end_state if st["check"] != "judge" else None, "metrics": metrics})
    sim_stages = [s for s in stages if s["check"] != "judge"]
    needs_judge = any(s["check"] == "judge" for s in stages)
    fails = {o: bool(r.fallen(o).any()) for o in spec.get("no_fall", [])}
    clean = not any(fails.values())
    sim_ok = all(s["achieved"] for s in sim_stages) and clean
    ok_any = all(s["any_time"] for s in sim_stages) and clean
    ok_end = all(s["end_state"] for s in sim_stages) and clean
    times = [s["t_first_s"] for s in sim_stages]
    in_order = all(a is not None for a in times) and all(times[i] <= times[i + 1] for i in range(len(times) - 1))
    out = {"scene_id": meta["scene_id"], "seed": meta["seed"], "category": spec.get("category"), "spec_version": spec.get("version", 1),
           "stages": stages, "n_sim_stages": len(sim_stages), "sim_stages_achieved": sum(s["achieved"] for s in sim_stages),
           "graded_sim": round(sum(s["achieved"] for s in sim_stages) / max(1, len(stages)), 4),
           "success_any": bool(sim_ok) if not needs_judge else None,
           "sim_prefix_ok": bool(sim_ok), "sim_prefix_any_time": bool(ok_any), "sim_prefix_end_state": bool(ok_end),
           "in_order": bool(in_order), "fallen": fails,
           "needs_judge": needs_judge, "analyzed_steps": len(qpos) - 1, "recorded_steps": meta["steps"],
           "judge_context": judge_context(stages, spec), "label": meta.get("outcome")}
    (rollout_dir / "verifier.json").write_text(json.dumps(out, indent=1))
    return out


def judge_context(stages, spec) -> str:
    parts = []
    for s in stages:
        if s["check"] == "judge":
            continue
        m = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in s["metrics"].items())
        parts.append(f"{s['name']}: {'yes' if s['achieved'] else 'no'}" + (f" (first at {s['t_first_s']:.1f} s)" if s["achieved"] else "") + (f" [{m}]" if m else ""))
    return "Simulator state checks: " + "; ".join(parts) + "."


def load_specs(path: Path = SPECS_PATH) -> dict:
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--specs", type=Path, default=SPECS_PATH)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="default: whole trajectory")
    p.add_argument("--limit", action="store_true", help="cut each rollout at limits.limit(scene), like the judge frames")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    specs = load_specs(a.specs)
    env_cache = {}
    for d in a.dirs:
        if (d / "verifier.json").exists() and not a.force:
            continue
        meta = json.loads((d / "rollout.json").read_text())
        scene = meta["scene_id"]
        if meta.get("revision_match") is False:
            continue  # recorded on an older scene revision; qpos layout may not match the local scene
        if scene not in specs:
            print(d, "no spec", flush=True)
            continue
        if scene not in env_cache:
            env_cache.clear()
            env_cache[scene] = SharpaWebXREnv(SCENES_ROOT / scene)
        try:
            out = verify(d, env_cache[scene], specs[scene], limit(scene) if a.limit else a.max_steps)
        except Exception as e:  # noqa: BLE001  one bad rollout must not stop the batch
            (d / "verifier.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "scene_id": scene}))
            print(d, "ERROR", type(e).__name__, e, flush=True)
            continue
        if not a.quiet:
            print(d, out["label"], "ok" if out["sim_prefix_ok"] else "fail", [(s["name"], s["achieved"]) for s in out["stages"]], flush=True)


if __name__ == "__main__":
    main()
