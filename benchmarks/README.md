# OpenWAM Client Integration Guide

For users wiring their own robot or benchmark to an OpenWAM policy server.

**You don't need to know anything about the server** — its model, preprocessing,
multi-view composition, or checkpoint. Just speak the WebSocket
protocol below. Minimal client dependencies: `numpy`, `Pillow`, `websockets`
(plus `opencv-python` if you decode camera frames yourself). The wire contract
(message types) is mirrored on both sides: server constants in [`openwam/deploy/server.py`](../openwam/deploy/server.py), client constants in [`benchmarks/utils/transport.py`](utils/transport.py).

Bundled clients:

- [RoboTwin](robotwin/README.md)
- [LIBERO](libero/README.md)
- [LIBERO-plus](libero-plus/README.md)
- [RoboCasa365](robocasa365/README.md)
- [RoboCasa GR1](robocasa_gr1/README.md)
- [VLABench](vlabench/README.md)
- [EBench](ebench/README.md)

RoboDojo trains through OpenWAM but evaluates through the external
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) repository — see
[robodojo/README.md](robodojo/README.md).

## 1. What the client sends

One call per control step: up to three lossless PNG camera frames + the task `prompt` — the exact string the model should see (the server forwards it verbatim; wrap it in your checkpoint's template first). Proprioceptive checkpoints also require a raw `state` vector whose length matches the checkpoint's `model.architecture.state_dim`.

```json
// obs message (Client → Server)
{
  "type": "obs",
  "images": {
    "head_camera":        "<base64 PNG>",       // required
    "left_wrist_camera":  "<base64 PNG>|null",  // optional
    "right_wrist_camera": "<base64 PNG>|null"   // optional
  },
  "prompt": "pick up the red bottle",          // sent verbatim; wrap per your checkpoint's template
  "state":  [float, ...]                       // required when use_proprioception=true
}
```

> **Prompt formatting.** The server forwards `prompt` to the model **verbatim** — it does *not* wrap or reformat it. Send the exact string the model was trained on. Each benchmark owns its prompt template; for RoboTwin checkpoints, wrap the raw instruction with [`benchmarks/robotwin/prompt_template.py`](robotwin/prompt_template.py) (the RoboTwin eval adapter does this for you).

Response (action message, Server → Client):

```json
{"type": "action", "action": [float × 20 or 14], "step": int, "latency_ms": float}
```

## 2. Three things you don't need to handle

- **Image sizing / aspect ratio.** Use the bundled benchmark adapter. RoboCasa365,
  LIBERO, BEHAVIOR, EBench, VLABench, and RoboCasa-GR1 reproduce their unchanged
  training readers' LANCZOS tile resize before sending; RoboTwin sends native
  camera sizes because its reader performs BILINEAR composition directly.
- **Action units.** For normalized checkpoints, the returned action is already denormalized to **physical units** (eef: xyz in meters, rot6d unitless, gripper 0-1; joint: radians). Feed it directly to your controller — do not multiply by any mean/std. If the checkpoint was trained with normalization disabled, deploy leaves actions and state in that raw training scale.
- **Execution mode / chunking.** Whether the server runs the sync executor (buffer-and-replan) or the async one (background prefetch, `inference.inference_mode: async`) is invisible on the wire: the protocol is always one obs in, one action out.

## 3. Camera field rules

- `head_camera`: **required**. Used as TI2V first-frame condition / single-view main view.
- `left_wrist_camera`, `right_wrist_camera`: optional. If missing or `null`:
  - Server is single-view → the field is ignored.
  - Server is multi-view → the slot is filled with a black frame. The model still runs, but accuracy degrades since you're out of the training distribution for wrist-conditioned checkpoints.
- `state`: required when the checkpoint has `model.architecture.use_proprioception: true`. The server validates the dimension before inference and returns a `ServerError` (status 400) for missing or mismatched state instead of failing later inside the model.

## 4. Episode lifecycle and reset

Within one episode, just keep calling `client.predict(payload)`. The server caches an action chunk internally: the first call runs full inference (~seconds), the next N-1 are buffer pops (<10 ms). It re-infers automatically when the buffer empties.

**You must call `client.reset()` between episodes.** The server keeps per-episode executor state that leaks across episode boundaries otherwise:

- the action chunk buffer — pending actions from the last inference
- the step counter (and, in async mode, any in-flight background inference)

Reset drops all of this and returns `{"type": "reset_ack"}`. It does **not** touch model weights or server-level config, so it's cheap (<1 ms) and safe to call defensively at the start of every episode.

When to call it:
- At the **start** of each new task / episode / rollout — including the very first one.
- After any hard failure (client timeout, controller fault) where you're not sure the action buffer is still valid.
- **Not** during normal step-to-step control. Calling `reset()` mid-episode forces the next `predict()` to pay full inference latency and discards the remaining action chunk.

Message shape:
```
reset message  → {"type": "reset"}

reset_ack      → {"type": "reset_ack"}
```

## 5. Using `benchmarks.utils`

Client helpers live under `benchmarks/utils/client.py` and can be imported directly:

```python
from benchmarks.utils import (
    build_payload,  # assemble the {"images": {...}, "prompt": ...} dict
    encode_path_b64,  # PNG path -> base64 str
    WSPolicyClient,  # WebSocket transport — predict() / reset() / ping()
    ServerError,  # structured server error: .status / .code / .message
)

ws_url = "ws://127.0.0.1:8848"

# One persistent connection; obs/reset auto-reconnect once on a dropped socket,
# ping fails fast. open_timeout caps connection setup; timeout caps a round-trip.
with WSPolicyClient(ws_url, timeout=300.0, open_timeout=10.0) as client:
    client.ping()  # verify / wait for the server to be up (raises if unreachable)

    # --- start of episode ---
    client.reset()

    # --- per-step ---
    head_b64 = encode_path_b64("/path/to/head.png")
    left_b64 = encode_path_b64("/path/to/left.png")  # or None
    right_b64 = encode_path_b64("/path/to/right.png")  # or None
    current_state = [0.0] * 20  # replace with your raw proprio vector

    payload = build_payload(
        head=head_b64,
        left_wrist=left_b64,
        right_wrist=right_b64,
        prompt="pick up the red bottle",  # sent verbatim; wrap per your checkpoint's template first
        state=current_state,  # optional raw proprio; required for proprio-conditioned checkpoints
    )
    try:
        action = client.predict(payload)["action"]  # already in physical units — feed to controller
    except ServerError as e:
        print(f"server rejected the request [{e.status} {e.code}]: {e.message}")
        raise
```

The bundled test script ([scripts/inference_test/inference_single_test.py](../scripts/inference_test/inference_single_test.py)) imports from here, so it doubles as a reference integration. For a full real-robot adapter, see [benchmarks/robotwin/openwam2robotwin_interface.py](robotwin/openwam2robotwin_interface.py).

## 6. Error cheatsheet

| ServerError (status 400) message contains | Cause |
|---|---|
| `head_camera is required` | Missing or `null` head_camera |
| `client must send 'images' dict` | Legacy single-field `image` payload (no longer supported) |
| `failed to decode base64 image` | Corrupted base64 or invalid image bytes |
| `requires obs['state']` | Checkpoint uses proprioception but the payload omitted `state` |
| `state dimension mismatch` | Payload `state` length differs from checkpoint `state_dim` |
| `camera_layout` | Server config has fewer than 3 entries in `camera_layout` while multi-view is enabled — check the checkpoint |

## 7. See also

- Starting the server: [root README → Deployment](../README.md#deployment)
- Reference client: [scripts/inference_test/inference_single_test.py](../scripts/inference_test/inference_single_test.py)
