"""Continuous inference test for the OpenWAM policy server (WebSocket).

Fires N back-to-back predict requests over a single connection to exercise
the server under sustained load: latency stability, no state leak / crash /
OOM across requests, and -- with ``--reset-every`` -- periodic reset behaviour.
Reuses the client contract and helpers from ``scripts/inference_test/inference_single_test.py``.

Usage:
    # Back-to-back requests with fresh random images (no files needed; defaults: 128 requests, log every step)
    python scripts/inference_test/inference_continuous_test.py --test

    # Real head camera, 100 requests, reset every 20, 100 ms apart
    python scripts/inference_test/inference_continuous_test.py \
        --head-camera /path/to/head.jpg --prompt "pick up the red bottle" \
        -n 100 --reset-every 20 --interval 0.1
"""

import argparse
import os
import statistics
import sys
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPTS))
sys.path.insert(0, _ROOT)  # benchmarks.utils
sys.path.insert(0, _SCRIPTS)  # inference_single_test (same dir)

from inference_single_test import _make_random_image_b64, _resolve_state  # noqa: E402

from benchmarks.utils.client import build_payload, encode_path_b64  # noqa: E402
from benchmarks.utils.transport import WSPolicyClient  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fire N back-to-back requests at the OpenWAM WebSocket policy server.")
    parser.add_argument("--server", type=str, default="ws://127.0.0.1:8848")
    parser.add_argument("-n", "--num-requests", type=int, default=128, help="Number of back-to-back predict requests.")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds to sleep between requests (default 0).")
    parser.add_argument("--reset-every", type=int, default=0, help="Reset the server every N requests (0 = never).")
    parser.add_argument(
        "--log-every", type=int, default=1, help="Print every k-th request (default 1 = every; 0 = ~10 evenly-spaced)."
    )
    parser.add_argument("--head-camera", type=str, default=None, help="Path to head camera image (or use --test).")
    parser.add_argument("--left-wrist-camera", type=str, default=None, help="Optional path to left wrist camera image.")
    parser.add_argument(
        "--right-wrist-camera", type=str, default=None, help="Optional path to right wrist camera image."
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--state", type=float, nargs="*", default=None, help="Raw proprio state values to send.")
    parser.add_argument("--state-file", type=str, default=None, help="JSON list, or object with a 'state' list.")
    parser.add_argument("--state-dim", type=int, default=20, help="Dummy zero-state dimension when --state is omitted.")
    parser.add_argument("--no-state", action="store_true", help="Do not include state in the obs payload.")
    parser.add_argument("--test", action="store_true", help="Use fresh random images each request (no files needed).")
    return parser


def _build_one_payload(args, state):
    """Random images in --test mode, else the given camera files."""
    if args.test:
        return build_payload(
            head=_make_random_image_b64(480, 640),
            left_wrist=_make_random_image_b64(480, 640),
            right_wrist=_make_random_image_b64(480, 640),
            prompt=args.prompt or "robot picks up the red bottle from the table",
            state=state,
        )
    return build_payload(
        head=encode_path_b64(args.head_camera),
        left_wrist=encode_path_b64(args.left_wrist_camera) if args.left_wrist_camera else None,
        right_wrist=encode_path_b64(args.right_wrist_camera) if args.right_wrist_camera else None,
        prompt=args.prompt,
        state=state,
    )


def _report(label: str, xs: list[float]) -> None:
    xs_sorted = sorted(xs)
    p50 = statistics.median(xs_sorted)
    p95 = xs_sorted[min(len(xs_sorted) - 1, int(0.95 * len(xs_sorted)))]
    print(
        f"{label:>6} latency ms:  min={min(xs):.1f}  mean={statistics.fmean(xs):.1f}  "
        f"p50={p50:.1f}  p95={p95:.1f}  max={max(xs):.1f}"
    )


def run_continuous(args, state) -> None:
    n = int(args.num_requests)
    log_every = args.log_every if args.log_every > 0 else max(1, n // 10)
    print(f"Server: {args.server}  |  requests: {n}  reset-every: {args.reset_every}  interval: {args.interval}s")
    print("-" * 70)

    client_latencies: list[float] = []
    server_latencies: list[float] = []
    failures = 0

    with WSPolicyClient(args.server) as client:
        print("ping ...", end=" ", flush=True)
        print(f"OK — {client.ping()}")

        # In --test mode rebuild each request (fresh random frames); otherwise
        # reuse one payload so the measured variance is server-side, not encoding.
        fixed_payload = None if args.test else _build_one_payload(args, state)

        for i in range(1, n + 1):
            payload = fixed_payload if fixed_payload is not None else _build_one_payload(args, state)
            t0 = time.perf_counter()
            try:
                result = client.predict(payload)
            except Exception as e:  # noqa: BLE001 -- report and keep the loop going
                failures += 1
                print(f"[{i:>4}/{n}] FAILED: {type(e).__name__}: {e}")
                continue
            client_ms = (time.perf_counter() - t0) * 1000.0
            client_latencies.append(client_ms)
            action = result.get("action", [])
            server_ms = result.get("latency_ms")
            if isinstance(server_ms, (int, float)):
                server_latencies.append(float(server_ms))

            # Log the first, last, and every log_every-th request (default: ~10 evenly-spaced).
            if i == 1 or i == n or i % log_every == 0:
                stail = f"  server={server_ms}ms" if server_ms is not None else ""
                print(f"[{i:>4}/{n}] OK  client={client_ms:7.1f}ms{stail}  action_dim={len(action)}")

            if args.reset_every and i % args.reset_every == 0 and i < n:
                client.reset()
            if args.interval > 0:
                time.sleep(args.interval)

        client.reset()

    print("-" * 70)
    print(f"done: {n - failures}/{n} ok, {failures} failed")
    if client_latencies:
        _report("client", client_latencies)
    if server_latencies:
        _report("server", server_latencies)
    if failures:
        sys.exit(1)


def main() -> None:
    args = _build_argparser().parse_args()
    state = _resolve_state(args)
    if not args.test and args.head_camera is None:
        print("Error: --head-camera is required (or use --test for random images)", file=sys.stderr)
        sys.exit(2)
    run_continuous(args, state)


if __name__ == "__main__":
    main()
