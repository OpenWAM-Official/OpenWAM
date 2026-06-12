# OpenWAM Client Integration Guide

For users wiring their own robot or benchmark to an OpenWAM policy server.

**You don't need to know anything about the server** — its model, preprocessing,
multi-view composition, prompt wrapping, or checkpoint. Just speak the WebSocket
protocol below. Minimal client dependencies: `numpy`, `Pillow`, `websockets`
(plus `opencv-python` if you decode camera frames yourself). The wire contract
(message types) is mirrored on both sides: server constants in [`openwam/deploy/server.py`](../openwam/deploy/server.py), client constants in [`benchmarks/utils/transport.py`](utils/transport.py).

## 1. What the client sends

One call per control step: three raw camera JPEGs + a base task prompt. Proprioceptive checkpoints also require a raw `state` vector whose length matches the checkpoint's `model.architecture.state_dim`.

```json
// obs message (Client → Server)
{
  "type": "obs",
  "images": {
    "head_camera":        "<base64 JPEG>",    // required
    "left_wrist_camera":  "<base64 JPEG>|null", // optional
    "right_wrist_camera": "<base64 JPEG>|null"  // optional
  },
  "prompt": "pick up the red bottle",
  "state":  [float, ...]                       // required when use_proprioception=true
}
```

Response (action message, Server → Client):

```json
{"type": "action", "action": [float × 20 or 14], "step": int, "latency_ms": float}
```

## 2. Three things you don't need to handle

- **Image sizing / aspect ratio.** Server reads the checkpoint's `config.yaml` and resizes for you. Send the native camera output.
- **Prompt format.** Pass the base task prompt (`"pick up the red bottle"`). Server wraps it with the training/deploy FastWAM template internally. Do **not** pre-wrap the prompt yourself.
- **Action units.** For normalized checkpoints, the returned action is already denormalized to **physical units** (eef: xyz in meters, rot6d unitless, gripper 0-1; joint: radians). Feed it directly to your controller — do not multiply by any mean/std. If the checkpoint was trained with normalization disabled, deploy leaves actions and state in that raw training scale.

## 3. Camera field rules

- `head_camera`: **required**. Used as TI2V first-frame condition / single-view main view.
- `left_wrist_camera`, `right_wrist_camera`: optional. If missing or `null`:
  - Server is single-view → the field is ignored.
  - Server is multi-view → the slot is filled with a black frame. The model still runs, but accuracy degrades since you're out of the training distribution for wrist-conditioned checkpoints.
- `state`: required when the checkpoint has `model.architecture.use_proprioception: true`. The server validates the dimension before inference and returns a `ServerError` (status 400) for missing or mismatched state instead of failing later inside the model.

## 4. Episode lifecycle and reset

Within one episode, just keep calling `client.predict(payload)`. The server caches an action chunk internally: the first call runs full inference (~seconds), the next N-1 are buffer pops (<10 ms). It re-infers automatically when the buffer empties.

**You must call `client.reset()` between episodes.** The server keeps per-episode state that leaks across episode boundaries otherwise:

- `obs_history` — the rolling observation buffer used for temporal conditioning
- the action chunk buffer — pending actions from the last inference
- the ensemble buffer — overlapping predictions used in receding-horizon mode
- the step counter

Reset drops all of this and returns `{"type": "reset_ack"}`. It does **not** touch model weights or server-level config, so it's cheap (<1 ms) and safe to call defensively at the start of every episode.

When to call it:
- At the **start** of each new task / episode / rollout — including the very first one.
- After any hard failure (client timeout, controller fault) where you're not sure the action buffer is still valid.
- **Not** during normal step-to-step control. Calling `reset()` mid-episode forces the next `predict()` to pay full inference latency and throws away temporal ensembling.

Message shape:
```
reset message  → {"type": "reset"}

reset_ack      → {"type": "reset_ack"}
```

## 5. Using `benchmarks.utils`

Client helpers live under `benchmarks/utils/client.py` and can be imported directly:

```python
from benchmarks.utils import (
    build_payload,      # assemble the {"images": {...}, "prompt": ...} dict
    encode_path_b64,    # JPEG path -> base64 str
    WSPolicyClient,     # WebSocket transport — predict() / reset() / ping()
    ServerError,        # structured server error: .status / .code / .message
)

ws_url = "ws://127.0.0.1:8848"

# One persistent connection; obs/reset auto-reconnect once on a dropped socket,
# ping fails fast. open_timeout caps connection setup; timeout caps a round-trip.
with WSPolicyClient(ws_url, timeout=300.0, open_timeout=10.0) as client:
    client.ping()        # verify / wait for the server to be up (raises if unreachable)

    # --- start of episode ---
    client.reset()

    # --- per-step ---
    head_b64  = encode_path_b64("/path/to/head.jpg")
    left_b64  = encode_path_b64("/path/to/left.jpg")   # or None
    right_b64 = encode_path_b64("/path/to/right.jpg")  # or None
    current_state = [0.0] * 20                         # replace with your raw proprio vector

    payload = build_payload(
        head=head_b64,
        left_wrist=left_b64,
        right_wrist=right_b64,
        prompt="pick up the red bottle",
        state=current_state,  # optional raw proprio; required for proprio-conditioned checkpoints
    )
    try:
        action = client.predict(payload)["action"]   # already in physical units — feed to controller
    except ServerError as e:
        print(f"server rejected the request [{e.status} {e.code}]: {e.message}")
        raise
```

The bundled test script ([scripts/inference_single_test.py](../scripts/inference_single_test.py)) imports from here, so it doubles as a reference integration. For a full real-robot adapter, see [benchmarks/robotwin/openwam2robotwin_interface.py](robotwin/openwam2robotwin_interface.py).

## 6. Error cheatsheet

| ServerError (status 400) message contains | Cause |
|---|---|
| `head_camera is required` | Missing or `null` head_camera |
| `client must send 'images' dict` | Legacy single-field `image` payload (no longer supported) |
| `failed to decode base64 JPEG` | Corrupted base64 or bad JPEG bytes |
| `requires obs['state']` | Checkpoint uses proprioception but the payload omitted `state` |
| `state dimension mismatch` | Payload `state` length differs from checkpoint `state_dim` |
| `camera_layout` | Server config has fewer than 3 entries in `camera_layout` while multi-view is enabled — check the checkpoint |

## 7. See also

- Starting the server: [root README → Deployment](../README.md#deployment)
- Reference client: [scripts/inference_single_test.py](../scripts/inference_single_test.py)

## 8. Web control dashboard

Use `benchmarks/web_control.py` to inspect benchmark log directories from a
browser without installing frontend dependencies. The dashboard is a single
stdlib-only Python server with embedded HTML/CSS/JS.

```bash
python benchmarks/web_control.py <log_dir> \
    --benchmark auto \
    --host 0.0.0.0 \
    --port 8765
```

Open `http://<node-ip>:8765/` to view run progress, task/job state, parsed
metrics, failure snippets, log files, and a live tail pane. Use
`--host 127.0.0.1` when the dashboard should only be reachable locally.

Adapter choices:

| Adapter | Use case |
|---|---|
| `auto` | Default. Detects RoboTwin layouts from `run.env`, `summary.tsv`, `queue/`, or `node*/worker*`; otherwise uses generic log browsing. |
| `robotwin` | RoboTwin DLC/local evaluation logs with queue progress, success-rate parsing, node/worker/server logs, consistency checks, and CSV export. |
| `generic` | Any directory containing `*.log`, `*.out`, `*.err`, `stdout*`, or `stderr*` files. |

Useful options:

| Option | Default | Description |
|---|---:|---|
| `--tail-bytes` | `200000` | Bytes returned for the first tail request. |
| `--state-tail-bytes` | `256000` | Bytes scanned per task log for success-rate parsing. |
| `--max-logs` | `2000` | Maximum log-like files shown in the log browser. |
| `--max-task-log-bytes` | `4000000` | Bytes scanned per failed task for error snippets. |
| `--max-error-snippets` | `200` | Maximum failed-task snippets retained in `/api/state`. |
| `--refresh-sec` | `2.0` | Browser polling interval. |
| `--open` | off | Open a local browser after startup. |

HTTP endpoints:

| Endpoint | Description |
|---|---|
| `/` | Dashboard page. |
| `/api/state` | Unified JSON state for benchmark metadata, progress, metrics, jobs, nodes, logs, failures, and validation issues. |
| `/api/tail?file=<relative-path>&offset=<n>&max_bytes=<n>` | Safe incremental log tail. Paths are restricted to `<log_dir>`. |
| `/raw?file=<relative-path>` | Raw text view for one log file. |
| `/api/results.csv` | Adapter-provided CSV export. |

For RoboTwin, this compatibility command is equivalent to passing
`--benchmark robotwin`:

```bash
python benchmarks/robotwin/dlc_web_console.py <log_dir> \
    --host 0.0.0.0 \
    --port 8765
```
