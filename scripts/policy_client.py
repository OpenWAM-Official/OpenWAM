"""Minimal HTTP client for the OpenWAM policy server.

Example:
    python scripts/policy_client.py \
        --server http://127.0.0.1:8766 \
        --image /path/to/frame.jpg
"""

import argparse
import base64
import json
from pathlib import Path
from urllib import request


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the OpenWAM HTTP policy server.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8766")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--state", type=float, nargs="*", default=None)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()

    image_bytes = Path(args.image).read_bytes()
    payload = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "prompt": args.prompt,
    }
    if args.state is not None:
        payload["state"] = args.state

    req = request.Request(
        f"{args.server.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as resp:
        print(resp.read().decode("utf-8"))


if __name__ == "__main__":
    main()
