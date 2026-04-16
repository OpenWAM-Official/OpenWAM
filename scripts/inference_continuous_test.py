"""Continuous inference test — simulates a real robot control loop.

Demonstrates how the OpenWAM policy server handles action chunking:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Client sends image+prompt  ──►  Server runs full inference     │
  │  (first request or buffer     ◄──  Returns action chunk         │
  │   exhausted)                       (e.g. 33 steps cached)       │
  │                                                                 │
  │  Client sends image+prompt  ──►  Server pops from buffer        │
  │  (buffer still has actions)   ◄──  Returns cached action        │
  │                                    (NO inference, fast)          │
  │                                                                 │
  │  ... repeat until buffer empty, then re-infer with new image    │
  └─────────────────────────────────────────────────────────────────┘

Key points:
  - The client ALWAYS sends the current image with every /predict call.
  - The server decides internally whether to run inference or pop from buffer.
  - In greedy mode (execute_horizon=None, default): the full chunk is consumed
    before re-inference. Chunk length = num_frames in training (e.g. 33).
  - In receding-horizon mode (execute_horizon=K): only K actions are executed
    before re-inference with a new image, enabling temporal ensembling.

Usage:
    # Start the server first:
    bash scripts/deploy.sh /path/to/checkpoint_dir

    # Then run this test:
    python scripts/inference_continuous_test.py

    # With custom settings:
    python scripts/inference_continuous_test.py \
        --server http://127.0.0.1:8766 \
        --steps 100 \
        --height 480 --width 640
"""

import argparse
import base64
import io
import json
from urllib import request


def _make_random_image_b64(height: int, width: int) -> str:
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _post(server: str, endpoint: str, payload: dict) -> dict:
    url = f"{server.rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(server: str, endpoint: str) -> dict:
    url = f"{server.rstrip('/')}{endpoint}"
    with request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Continuous inference test for OpenWAM policy server.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8766")
    parser.add_argument("--steps", type=int, default=100, help="Total control steps to simulate.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--prompt", type=str, default="robot picks up the red bottle from the table")
    args = parser.parse_args()

    print(f"Server:  {args.server}")
    print(f"Steps:   {args.steps}")
    print(f"Image:   {args.height}x{args.width} (random)")
    print(f"Prompt:  {args.prompt}")
    print("=" * 60)

    # Health check
    health = _get(args.server, "/health")
    print(f"Health:  {health}")

    # Reset server state
    _post(args.server, "/reset", {})
    print("Reset:   OK")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Control loop
    #
    # In a real robot scenario:
    #   - image = current camera frame from the robot
    #   - action = sent to the robot's motor controller
    #   - The loop runs at the robot's control frequency (e.g. 10-50 Hz)
    #
    # The server handles chunking internally:
    #   - Step 0: buffer empty → full inference (~seconds), caches chunk
    #   - Step 1..N-1: buffer has actions → instant pop (~ms)
    #   - Step N: buffer empty again → re-inference with fresh image
    # ----------------------------------------------------------------

    inference_count = 0
    total_latency = 0.0
    latencies = []

    print(f"\n{'Step':>5}  {'Latency':>10}  {'Type':>12}  {'Action (first 5 dims)':>30}")
    print("-" * 65)

    for step in range(args.steps):
        # In reality, this would be the current camera frame.
        # We generate a new random image each step to simulate changing observations.
        image_b64 = _make_random_image_b64(args.height, args.width)

        payload = {
            "image": image_b64,
            "prompt": args.prompt,
        }

        result = _post(args.server, "/predict", payload)

        action = result["action"]
        server_latency = result.get("latency_ms", 0)
        latencies.append(server_latency)
        total_latency += server_latency

        # Heuristic: if server latency > 500ms, it likely ran full inference.
        # Otherwise it was a buffer pop.
        is_inference = server_latency > 500
        if is_inference:
            inference_count += 1
            step_type = "INFERENCE"
        else:
            step_type = "buffer pop"

        action_preview = [round(a, 4) for a in action[:5]]
        print(f"{step:5d}  {server_latency:8.1f}ms  {step_type:>12}  {action_preview}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total steps:        {args.steps}")
    print(f"  Action dim:         {len(action)}")
    print(f"  Inference calls:    {inference_count}")
    print(f"  Buffer pops:        {args.steps - inference_count}")
    if inference_count > 0:
        chunk_size_est = args.steps / inference_count
        print(f"  Est. chunk size:    ~{chunk_size_est:.0f} steps")
    print(f"  Avg latency:        {total_latency / args.steps:.1f}ms")
    inf_latencies = [l for l in latencies if l > 500]
    pop_latencies = [l for l in latencies if l <= 500]
    if inf_latencies:
        print(f"  Avg inference:      {sum(inf_latencies) / len(inf_latencies):.1f}ms")
    if pop_latencies:
        print(f"  Avg buffer pop:     {sum(pop_latencies) / len(pop_latencies):.1f}ms")

    # Reset
    _post(args.server, "/reset", {})
    print("\nServer reset. Done.")


if __name__ == "__main__":
    main()
