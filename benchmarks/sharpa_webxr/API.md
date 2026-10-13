# Policy server API

The eval side (sim env, `rollout.py`, judge) knows nothing about any model. A model plugs in as a **policy
server**: a process that answers two endpoints over ZeroMQ. Everything model-specific lives in the server:
- image layout and resizing;
- state encoding;
- normalization stats;
- the action codec;
- the sampler;
- deploy settings (how many actions to execute, which past observations are needed).

The server belongs with the model's training code, not in the eval repo.

Reference implementation: `policies/hold_server.py` (holds the current pose; no GPU).
Base class and transport: `policies/common.py` (`BasePolicy`, `serve`).
Conformance check: `check_server.py --port N`. A server must pass it before it is evaluated.

## Endpoints

### `meta` (no input)

Returns a dict describing how the policy is deployed. The runner reads it once.

| key | type | meaning |
|---|---|---|
| `model` | str | name, logged with every rollout |
| `codec` | str | the model's action encoding (informational; decoding happens in the server) |
| `chunk` | int | H, actions returned per `inference` call |
| `execute_steps` | int | actions the runner executes before the next call, 1..H. Match the model's upstream deploy. `rollout.py --execute-steps` overrides it. |
| `obs_offsets` | list[int] | observation rows the policy needs, relative to now, all <= 0 and including 0. Example: `[-5, 0]` = now and 5 control steps ago. Rows before the episode start repeat the first observation. |
| `control_hz` | float | control rate the actions are meant for (the sim runs 20 Hz) |
| `upstream` | str | which upstream deploy path the settings copy (free text, logged) |
| `sampler` | dict | sampler settings, logged (e.g. `{"name": "unipc", "steps": 30, "cfg": 1.0}`) |
| `ckpt` | str | checkpoint path, logged |
| `cameras` | dict, optional | `{camera_name: [H, W]}` to render. Default: `ego_view` and `chest_view` at 480x640. |

### `inference`

Request:

```
{
  "obs":      {offset: {"images": {"ego_view": uint8[480,640,3], "chest_view": uint8[480,640,3]},
                        "state":  canonical (1 row)}            # one entry per meta obs_offsets
              },
  "task":     str,          # natural-language instruction
  "seed":     int,          # per-request seed; use it for sampling noise so rollouts are reproducible
  "reset":    bool,         # True on the first request of an episode: clear any per-episode state
  "prev_cmd": canonical (1 row) | None,   # last pose the runner commanded (None on reset)
  "executed": int | None    # actions executed since the previous request (None on reset)
}
```

Response:

```
{
  "pos":    float[H, 2, 3],      # wrist positions (m)
  "rot":    float[H, 2, 3, 3],   # wrist rotation matrices
  "hand":   float[H, 2, 22],     # hand joint targets (rad)
  "frame":  "ego_camera",        # must equal the observation's frame
  "raw":    float32[H, width],   # model output before decoding, logged for debugging
  "timing": {...},               # optional, e.g. {"server_ms": 812.0}
  "ckpt":   str
}
```

`BasePolicy.inference` builds this from `act()`. A subclass only implements
`act(obs, task, seed, prev_cmd) -> (Canonical [H], raw [H, width])`.

## Canonical pose

Both state and actions use one canonical layout (`scripts/sharpa_eef/canonical.py`). Arm axis 2 is
`(left, right)`.

- `pos [T, 2, 3]` wrist position and `rot [T, 2, 3, 3]` wrist orientation, both in the **ego-camera frame**;
- `hand [T, 2, 22]` absolute hand joint angles.

Actions are **absolute targets in that frame**, not deltas. A model trained on deltas or relative poses decodes
them in its server, using `obs[0]["state"]` (the measured pose now) or `prev_cmd` (the last commanded pose) as
the reference. `check_server.py` flags a first action more than 0.30 m from the current wrist pose; that
usually means a delta was returned.

## Control loop (what the runner does)

```
reset env (scene, seed)
loop until the episode limit (min(1.5 x p90 demo length, 20 s) at 20 Hz):
    request = observations at rows now + obs_offsets, task, seed, reset, prev_cmd, executed
    response = server.inference(request)                # H actions
    execute the first execute_steps actions: per step, IK to arm joints (seeded from the previous solve),
        hold for 50 ms (3 physics ticks)
    prev_cmd = last executed action; executed = execute_steps
```

There is no action ensembling or queue blending: each new chunk replaces the previous one. The runner saves the
full qpos trajectory. Judge frames, signals and videos are rendered offline from it (`postprocess.py`,
`render_video.py`), so they do not depend on the server.

## Writing a server for a new model

1. Copy `policies/hold_server.py`. Set the meta fields to the model's upstream deploy settings.
2. In `act`: build the model input from `obs` (images per offset, state), normalize with **the training run's
   own stats** (load them from the run directory, never from a default path), sample, denormalize, decode to
   absolute canonical poses.
3. Start it: `python <model>_server.py --ckpt <ckpt> --port 11600`.
4. Check it: `.venv_webxr/bin/python benchmarks/sharpa_webxr/check_server.py --port 11600`.
5. Evaluate: `RUN=<name> benchmarks/sharpa_webxr/run_suite.sh <tag> 11600 0 4 <workers>`, then judge with
   `postprocess.py --frames-only` + `score.py`. `watch_run.sh` automates this per checkpoint.
