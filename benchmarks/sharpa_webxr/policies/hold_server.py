"""Reference policy server: holds the current pose. No model, no GPU.

The smallest complete implementation of the policies/common.py contract, for two uses:
- run the whole eval end to end (rollout.py, postprocess.py, score.py) without any model;
- a template for a new model's server: subclass BasePolicy, set the meta fields, implement ``act``.

usage: python benchmarks/sharpa_webxr/policies/hold_server.py --port 11600
"""
from __future__ import annotations

import argparse

import numpy as np

from common import BasePolicy, C, serve


class HoldPolicy(BasePolicy):
    model = "hold"
    codec = "none"
    chunk = 16                # actions returned per request
    execute_steps = 16        # actions the runner executes before calling again
    obs_offsets = (0,)        # observation rows needed, relative to now (e.g. (-5, 0) for a t-5 history)
    control_hz = 20.0
    upstream = "reference implementation (no upstream)"
    sampler = {"name": "none"}
    ckpt = "none"

    def act(self, obs: dict, task: str, seed: int, prev_cmd: C.Canonical | None) -> tuple[C.Canonical, np.ndarray]:
        s = obs[0]["state"]  # canonical state of the current row: pos [1,2,3], rot [1,2,3,3], hand [1,2,22]
        h = self.chunk
        cmd = C.Canonical(np.repeat(s.pos, h, 0), np.repeat(s.rot, h, 0), np.repeat(s.hand, h, 0),
                          np.ones(h, bool), frame=s.frame)
        return cmd, np.zeros((h, 1), np.float32)  # raw: the model's undecoded output (logged only)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=11600)
    p.add_argument("--ckpt", default="none", help="ignored (no weights); accepted so watchers can start it like any server")
    a = p.parse_args()
    pol = HoldPolicy()
    pol.ckpt = a.ckpt
    serve(pol, a.port)
