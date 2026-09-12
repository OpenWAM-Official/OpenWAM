"""Check the policy protocol without loading torch or mutating episode state."""

import argparse
import json
import os
import sys

from websockets.sync.client import connect


def check(url: str, timeout: float) -> None:
    with connect(url, open_timeout=timeout, close_timeout=2, ping_interval=None, proxy=None) as websocket:
        websocket.send(json.dumps({"type": "ping"}))
        response = json.loads(websocket.recv(timeout=timeout))
        if not isinstance(response, dict) or response.get("type") != "pong":
            raise ValueError(f"expected policy pong, received {response!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("OPENWAM_SERVER_URL", "ws://127.0.0.1:8848"))
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    try:
        check(args.url, args.timeout)
    except Exception as exc:
        print(f"OpenWAM health check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
