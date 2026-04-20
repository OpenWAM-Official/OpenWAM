"""Single inference test for the OpenWAM policy server.

Client contract (unified, regardless of server multiview setting):
    payload["images"] = {
        "head_camera":        <base64 JPEG>,       # required
        "left_wrist_camera":  <base64 JPEG>|null,  # optional
        "right_wrist_camera": <base64 JPEG>|null   # optional
    }

Server picks the right preprocessing branch based on its checkpoint's
``cfg.dataloader.multiview``. See ``benchmarks/README.md`` for details.

Usage:
    # Quick smoke test with 3 random images (no files needed)
    python scripts/inference_single_test.py --test

    # Real head camera, wrist cameras sent as null (server black-fills if multiview):
    python scripts/inference_single_test.py \
        --head-camera /path/to/head.jpg \
        --prompt "pick up the red bottle"

    # All three real cameras
    python scripts/inference_single_test.py \
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.utils.client import (  # noqa: E402
    build_payload,
    encode_path_b64,
    get,
    post,
    reset,
)


def _make_random_image_b64(height: int = 480, width: int = 640) -> str:
    """Test-only: random RGB JPEG for smoke tests (not a real client helper)."""
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the OpenWAM HTTP policy server.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8848")
    parser.add_argument(
        "--head-camera", type=str, default=None, help="Path to head camera JPEG/PNG (required in run mode)."
    )
    parser.add_argument("--left-wrist-camera", type=str, default=None, help="Optional path to left wrist camera image.")
    parser.add_argument(
        "--right-wrist-camera", type=str, default=None, help="Optional path to right wrist camera image."
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--state", type=float, nargs="*", default=None)
    parser.add_argument("--test", action="store_true", help="Smoke test with 3 random images and dummy prompt.")
    return parser


def run_smoke_test(server: str):
    """Run a full smoke test: health check, predict, reset."""
    print(f"Server: {server}")
    print("-" * 60)

    # 1. Health check
    print("[1/3] Health check ...", end=" ")
    health = get(server, "/health")
    print(f"OK — {health}")

    # 2. Predict with 3 random images
    print("[2/3] Predict (3 random images, dummy prompt) ...", end=" ", flush=True)
    payload = build_payload(
        head=_make_random_image_b64(480, 640),
        left_wrist=_make_random_image_b64(480, 640),
        right_wrist=_make_random_image_b64(480, 640),
        prompt="robot picks up the red bottle from the table",
    )
    result = post(server, "/predict", payload)
    action = result.get("action", [])
    latency = result.get("latency_ms", "?")
    print(f"OK — action dim={len(action)}, latency={latency}ms")
    print(f"       action[:5] = {[round(a, 4) for a in action[:5]]}")

    # 3. Reset
    print("[3/3] Reset ...", end=" ")
    ack = reset(server)
    print(f"OK — {ack}")

    print("-" * 60)
    print("Smoke test passed.")


def main() -> None:
    args = _build_argparser().parse_args()

    if args.test:
        run_smoke_test(args.server)
        return

    if args.head_camera is None:
        print("Error: --head-camera is required (or use --test for smoke test)", file=sys.stderr)
        sys.exit(2)

    payload = build_payload(
        head=encode_path_b64(args.head_camera),
        left_wrist=encode_path_b64(args.left_wrist_camera) if args.left_wrist_camera else None,
        right_wrist=encode_path_b64(args.right_wrist_camera) if args.right_wrist_camera else None,
        prompt=args.prompt,
        state=args.state,
    )
    result = post(args.server, "/predict", payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
