# OpenWAM Client Integration Guide

For users wiring their own robot or benchmark to an OpenWAM policy server.

## 1. What the client sends

One call per control step: three raw camera JPEGs + a base task prompt.

```json
POST /predict
{
  "images": {
    "head_camera":        "<base64 JPEG>",    // required
    "left_wrist_camera":  "<base64 JPEG>|null", // optional
    "right_wrist_camera": "<base64 JPEG>|null"  // optional
  },
  "prompt": "pick up the red bottle",
  "state":  [float, ...]                       // optional proprio
}
```

Response:

```json
{"action": [float × 20 or 14], "step": int, "latency_ms": float}
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

## 4. Episode lifecycle and reset

Within one episode, just keep calling `POST /predict`. The server caches an action chunk internally: the first call runs full inference (~seconds), the next N-1 are buffer pops (<10 ms). It re-infers automatically when the buffer empties.

**You must call `POST /reset` between episodes.** The server keeps per-episode state that leaks across episode boundaries otherwise:

- `obs_history` — the rolling observation buffer used for temporal conditioning
- the action chunk buffer — pending actions from the last inference
- the ensemble buffer — overlapping predictions used in receding-horizon mode
- the step counter

Reset drops all of this and returns `{"status": "ok"}`. It does **not** touch model weights or server-level config, so it's cheap (<1 ms) and safe to call defensively at the start of every episode.

When to call it:
- At the **start** of each new task / episode / rollout — including the very first one.
- After any hard failure (client timeout, controller fault) where you're not sure the action buffer is still valid.
- **Not** during normal step-to-step control. Calling `/reset` mid-episode forces the next `/predict` to pay full inference latency and throws away temporal ensembling.

Endpoint shape:
```
POST /reset
{}                        # empty body

→ {"status": "ok"}
```

## 5. Using `benchmarks.utils`

Client helpers live under `benchmarks/utils/client.py` and can be imported directly:

```python
from benchmarks.utils import (
    build_payload,      # assemble the {"images": {...}, "prompt": ...} dict
    encode_path_b64,    # JPEG path -> base64 str
    post, get,          # thin HTTP wrappers over urllib
    reset,              # POST /reset — call between episodes
)

server = "http://127.0.0.1:8848"

# --- start of episode ---
reset(server)

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
result = post(server, "/predict", payload)
action = result["action"]   # already in physical units — feed to controller
```

Both bundled test scripts ([scripts/inference_single_test.py](../scripts/inference_single_test.py), [scripts/inference_continuous_test.py](../scripts/inference_continuous_test.py)) import from here, so they double as reference integrations.

## 6. Error cheatsheet

| HTTP 400 message contains | Cause |
|---|---|
| `head_camera is required` | Missing or `null` head_camera |
| `client must send 'images' dict` | Legacy single-field `image` payload (no longer supported) |
| `failed to decode base64 JPEG` | Corrupted base64 or bad JPEG bytes |
| `camera_layout` | Server config has fewer than 3 entries in `camera_layout` while multi-view is enabled — check the checkpoint |

## 7. See also

- Starting the server: [root README → Deployment](../README.md#deployment)
- Reference clients: [scripts/inference_single_test.py](../scripts/inference_single_test.py), [scripts/inference_continuous_test.py](../scripts/inference_continuous_test.py)
