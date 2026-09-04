"""Single inference test for the OpenWAM policy server (WebSocket).

Client contract (unified, regardless of server multiview setting):
    payload["images"] = {
        "head_camera":        <base64 JPEG>,       # required
        "left_wrist_camera":  <base64 JPEG>|null,  # optional
        "right_wrist_camera": <base64 JPEG>|null   # optional
    }
    payload["state"] = [float, ...]                # raw proprio, sent by default

Server picks the right preprocessing branch based on its checkpoint's
``cfg.dataloader.multiview``. See ``benchmarks/README.md`` for details.

Usage:
    # Quick smoke test with 3 random images (no files needed)
    python scripts/inference_test/inference_single_test.py --test

    # Real head camera, wrist cameras sent as null (server black-fills if multiview):
    python scripts/inference_test/inference_single_test.py \
        --head-camera /path/to/head.jpg \
        --prompt "pick up the red bottle"

    # All three real cameras
    python scripts/inference_test/inference_single_test.py \
        --head-camera /path/to/head.jpg \
        --left-wrist-camera /path/to/left.jpg \
        --right-wrist-camera /path/to/right.jpg \
        --prompt "pick up the red bottle"
"""

import argparse
import base64
import io
import json
import os
import sys

# Canonical client helpers live under benchmarks.utils — add project root so import works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.utils.client import build_payload, encode_path_b64  # noqa: E402
from benchmarks.utils.transport import WSPolicyClient  # noqa: E402


def _make_random_image_b64(height: int = 480, width: int = 640) -> str:
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


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the OpenWAM WebSocket policy server.")
    parser.add_argument("--server", type=str, default="ws://127.0.0.1:8848")
    parser.add_argument(
        "--head-camera", type=str, default=None, help="Path to head camera JPEG/PNG (required in run mode)."
    )
    parser.add_argument("--left-wrist-camera", type=str, default=None, help="Optional path to left wrist camera image.")
    parser.add_argument(
        "--right-wrist-camera", type=str, default=None, help="Optional path to right wrist camera image."
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--state", type=float, nargs="*", default=None, help="Raw proprio state values to send.")
    parser.add_argument("--state-file", type=str, default=None, help="JSON list, or object with a 'state' list.")
    parser.add_argument("--state-dim", type=int, default=20, help="Dummy zero-state dimension when --state is omitted.")
    parser.add_argument("--no-state", action="store_true", help="Do not include state in the obs payload.")
    parser.add_argument("--test", action="store_true", help="Smoke test with 3 random images and dummy prompt.")
    return parser


def run_smoke_test(server: str, state: list[float] | None):
    """Run a full smoke test: ping, predict, reset."""
    print(f"Server: {server}")
    print("-" * 60)

    with WSPolicyClient(server) as client:
        # 1. Liveness
        print("[1/3] Ping ...", end=" ")
        print(f"OK — {client.ping()}")

        # 2. Predict with 3 random images
        print("[2/3] Predict (3 random images, dummy prompt) ...", end=" ", flush=True)
        payload = build_payload(
            head=_make_random_image_b64(480, 640),
            left_wrist=_make_random_image_b64(480, 640),
            right_wrist=_make_random_image_b64(480, 640),
            prompt="robot picks up the red bottle from the table",
            state=state,
        )
        result = client.predict(payload)
        action = result.get("action", [])
        latency = result.get("latency_ms", "?")
        print(f"OK — action dim={len(action)}, latency={latency}ms")
        print(f"       action[:5] = {[round(a, 4) for a in action[:5]]}")

        # 3. Reset
        print("[3/3] Reset ...", end=" ")
        ack = client.reset()
        print(f"OK — {ack}")

    print("-" * 60)
    print("Smoke test passed.")


def main() -> None:
    args = _build_argparser().parse_args()
    state = _resolve_state(args)

    if args.test:
        run_smoke_test(args.server, state)
        return

    if args.head_camera is None:
        print("Error: --head-camera is required (or use --test for smoke test)", file=sys.stderr)
        sys.exit(2)

    payload = build_payload(
        head=encode_path_b64(args.head_camera),
        left_wrist=encode_path_b64(args.left_wrist_camera) if args.left_wrist_camera else None,
        right_wrist=encode_path_b64(args.right_wrist_camera) if args.right_wrist_camera else None,
        prompt=args.prompt,
        state=state,
    )
    with WSPolicyClient(args.server) as client:
        result = client.predict(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
