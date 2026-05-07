"""Continuous inference test — simulates a real robot control loop.

Demonstrates how the OpenWAM policy server handles action chunking:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Client sends 3 cams + prompt ──►  Server runs full inference   │
  │  (first request or buffer        ◄──  Returns action chunk      │
  │   exhausted)                           (e.g. 33 steps cached)   │
  │                                                                 │
  │  Client sends 3 cams + prompt ──►  Server pops from buffer      │
  │  (buffer still has actions)      ◄──  Returns cached action     │
  │                                         (NO inference, fast)    │
  │                                                                 │
  │  ... repeat until buffer empty, then re-infer with new images   │
  └─────────────────────────────────────────────────────────────────┘

Client contract (unified regardless of server multiview setting):
    payload["images"] = {
        "head_camera":        <base64 JPEG>,       # required
        "left_wrist_camera":  <base64 JPEG>|null,  # optional
        "right_wrist_camera": <base64 JPEG>|null   # optional
    }
    payload["state"] = [float, ...]                # raw proprio, sent by default

Usage:
    # Start the server first:
    bash scripts/deploy.sh /path/to/checkpoint_dir

    # Smoke / stream mode — every step sends 3 fresh random images:
    python scripts/inference_continuous_test.py --steps 100

    # Real robot mode — reuse the same static frames each step:
    python scripts/inference_continuous_test.py \
        --head-camera /path/to/head.jpg \
        --left-wrist-camera /path/to/left.jpg \
        --right-wrist-camera /path/to/right.jpg \
        --steps 50
"""

import argparse
import base64
import io
import json
import os
import sys

# Canonical client helpers live under benchmarks.utils — add project root so import works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.utils.client import (  # noqa: E402
    build_payload,
    encode_path_b64,
    get,
    post,
    reset,
)


def _make_random_image_b64(height: int, width: int) -> str:
    """Test-only: random RGB JPEG for smoke tests (not a real client helper)."""
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_state_file(path: str) -> list[float]:
    """Load a 1-D state vector from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("state")
    if not isinstance(data, list):
        raise ValueError("--state-file must contain a JSON list or an object with a 'state' list")
    return [float(x) for x in data]


def _resolve_state(args) -> list[float] | None:
    if args.no_state:
        return None
    if args.state_file:
        return _load_state_file(args.state_file)
    if args.state is not None:
        return [float(x) for x in args.state]
    return [0.0] * int(args.state_dim)


def main():
    parser = argparse.ArgumentParser(description="Continuous inference test for OpenWAM policy server.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8848")
    parser.add_argument("--steps", type=int, default=100, help="Total control steps to simulate.")
    parser.add_argument("--height", type=int, default=480, help="Random-image height (ignored if --head-camera given).")
    parser.add_argument("--width", type=int, default=640, help="Random-image width (ignored if --head-camera given).")
    parser.add_argument("--prompt", type=str, default="robot picks up the red bottle from the table")
    parser.add_argument("--state", type=float, nargs="*", default=None, help="Raw proprio state values to send.")
    parser.add_argument("--state-file", type=str, default=None, help="JSON list, or object with a 'state' list.")
    parser.add_argument("--state-dim", type=int, default=20, help="Dummy zero-state dimension when --state is omitted.")
    parser.add_argument("--no-state", action="store_true", help="Do not include state in the /predict payload.")
    parser.add_argument(
        "--head-camera", type=str, default=None, help="Static path for head camera (reused every step)."
    )
    parser.add_argument("--left-wrist-camera", type=str, default=None, help="Static path for left wrist camera.")
    parser.add_argument("--right-wrist-camera", type=str, default=None, help="Static path for right wrist camera.")
    args = parser.parse_args()

    static_head = encode_path_b64(args.head_camera) if args.head_camera else None
    static_left = encode_path_b64(args.left_wrist_camera) if args.left_wrist_camera else None
    static_right = encode_path_b64(args.right_wrist_camera) if args.right_wrist_camera else None
    stream_random = args.head_camera is None
    state = _resolve_state(args)

    print(f"Server:  {args.server}")
    print(f"Steps:   {args.steps}")
    if stream_random:
        print(f"Mode:    random {args.height}x{args.width} images every step (all 3 cameras)")
    else:
        print(
            f"Mode:    static frames  head={args.head_camera}  "
            f"left={args.left_wrist_camera or '(null)'}  "
            f"right={args.right_wrist_camera or '(null)'}"
        )
    print(f"Prompt:  {args.prompt}")
    print(f"State:   {'omitted' if state is None else f'{len(state)} dims'}")
    print("=" * 60)

    # Health check
    health = get(args.server, "/health")
    print(f"Health:  {health}")

    # Reset server state at the start of the episode
    reset(args.server)
    print("Reset:   OK")
    print("=" * 60)

    inference_count = 0
    total_latency = 0.0
    latencies = []

    print(f"\n{'Step':>5}  {'Latency':>10}  {'Type':>12}  {'Action (first 5 dims)':>30}")
    print("-" * 72)

    last_action = None

    for step in range(args.steps):
        if stream_random:
            head_b64 = _make_random_image_b64(args.height, args.width)
            left_b64 = _make_random_image_b64(args.height, args.width)
            right_b64 = _make_random_image_b64(args.height, args.width)
        else:
            head_b64, left_b64, right_b64 = static_head, static_left, static_right

        payload = build_payload(
            head=head_b64,
            left_wrist=left_b64,
            right_wrist=right_b64,
            prompt=args.prompt,
            state=state,
        )
        result = post(args.server, "/predict", payload)

        action = result["action"]
        last_action = action
        server_latency = result.get("latency_ms", 0)
        latencies.append(server_latency)
        total_latency += server_latency

        is_inference = server_latency > 500
        if is_inference:
            inference_count += 1
            step_type = "INFERENCE"
        else:
            step_type = "buffer pop"

        action_preview = [round(a, 4) for a in action[:5]]
        print(f"{step:5d}  {server_latency:8.1f}ms  {step_type:>12}  {action_preview}")

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total steps:        {args.steps}")
    print(f"  Action dim:         {len(last_action) if last_action else '?'}")
    print(f"  Inference calls:    {inference_count}")
    print(f"  Buffer pops:        {args.steps - inference_count}")
    if inference_count > 0:
        chunk_size_est = args.steps / inference_count
        print(f"  Est. chunk size:    ~{chunk_size_est:.0f} steps")
    print(f"  Avg latency:        {total_latency / args.steps:.1f}ms")
    inf_latencies = [latency_ms for latency_ms in latencies if latency_ms > 500]
    if inf_latencies:
        print(f"  Avg inference:      {sum(inf_latencies) / len(inf_latencies):.0f}ms")
    pop_latencies = [latency_ms for latency_ms in latencies if latency_ms <= 500]
    if pop_latencies:
        print(f"  Avg buffer pop:     {sum(pop_latencies) / len(pop_latencies):.1f}ms")


if __name__ == "__main__":
    main()
