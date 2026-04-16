"""Single inference test for the OpenWAM policy server.

Sends a single image and prompt, verifies the server returns a valid action.

Usage:
    # Quick smoke test with a random image (no file needed)
    python scripts/inference_single_test.py --test

    # With a real image
    python scripts/inference_single_test.py --image /path/to/frame.jpg --prompt "robot picks up the bottle"
"""

import argparse
import base64
import io
import json
from pathlib import Path
from urllib import request


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the OpenWAM HTTP policy server.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8766")
    parser.add_argument("--image", type=str, default=None, help="Path to a JPEG/PNG image file.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--state", type=float, nargs="*", default=None)
    parser.add_argument("--test", action="store_true", help="Smoke test with a random 480x640 image and dummy prompt.")
    return parser


def _make_random_image_b64(height: int = 480, width: int = 640) -> str:
    """Generate a random RGB image and return its base64-encoded JPEG bytes."""
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _post(server: str, endpoint: str, payload: dict = None) -> dict:
    """POST JSON to server and return parsed response."""
    url = f"{server.rstrip('/')}{endpoint}"
    data = json.dumps(payload or {}).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(server: str, endpoint: str) -> dict:
    """GET from server and return parsed response."""
    url = f"{server.rstrip('/')}{endpoint}"
    with request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_smoke_test(server: str):
    """Run a full smoke test: health check, predict, reset."""
    print(f"Server: {server}")
    print("-" * 40)

    # 1. Health check
    print("[1/3] Health check ...", end=" ")
    health = _get(server, "/health")
    print(f"OK — {health}")

    # 2. Predict with random image
    print("[2/3] Predict (random 480x640 image, dummy prompt) ...", end=" ", flush=True)
    payload = {
        "image": _make_random_image_b64(480, 640),
        "prompt": "robot picks up the red bottle from the table",
    }
    result = _post(server, "/predict", payload)
    action = result.get("action", [])
    latency = result.get("latency_ms", "?")
    print(f"OK — action dim={len(action)}, latency={latency}ms")
    print(f"       action[:5] = {[round(a, 4) for a in action[:5]]}")

    # 3. Reset
    print("[3/3] Reset ...", end=" ")
    reset = _post(server, "/reset")
    print(f"OK — {reset}")

    print("-" * 40)
    print("Smoke test passed.")


def main() -> None:
    args = _build_argparser().parse_args()

    if args.test:
        run_smoke_test(args.server)
        return

    if args.image is None:
        print("Error: --image is required (or use --test for smoke test)")
        return

    image_bytes = Path(args.image).read_bytes()
    payload = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "prompt": args.prompt,
    }
    if args.state is not None:
        payload["state"] = args.state

    result = _post(args.server, "/predict", payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
