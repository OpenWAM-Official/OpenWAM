"""Continuous inference test — simulates a real robot control loop, and
benchmarks HTTP vs WebSocket transport speed.

The control loop demonstrates server-side action chunking:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Client sends 3 cams + prompt ──►  Server runs full inference   │
  │  (first request or buffer        ◄──  Returns action chunk      │
  │   exhausted)                           (e.g. 33 steps cached)   │
  │                                                                 │
  │  Client sends 3 cams + prompt ──►  Server pops from buffer      │
  │  (buffer still has actions)      ◄──  Returns cached action     │
  │                                         (NO inference, fast)    │
  └─────────────────────────────────────────────────────────────────┘

Transport comparison:
    The server reports ``latency_ms`` = inference time only. This script also
    times each round-trip on the *client* side (``rtt_ms``) and reports the
    transport overhead ``rtt_ms - latency_ms`` per transport. The same
    pre-built payload sequence is replayed over every transport so inputs are
    byte-identical. For the cleanest signal, point it at a low-latency mock
    server (``python scripts/deploy.py --mock --mock-latency-ms 50``).

Transports (``--transport``):
    http  http.client over one reused keep-alive connection (keep-alive is the default)
    ws    websockets.sync, one persistent connection
    all   run both transports and print a comparison table (default)

Client contract (unified regardless of server multiview setting):
    payload["images"] = {
        "head_camera":        <base64 JPEG>,       # required
        "left_wrist_camera":  <base64 JPEG>|null,  # optional
        "right_wrist_camera": <base64 JPEG>|null   # optional
    }
    payload["state"] = [float, ...]                # raw proprio, sent by default

Usage:
    # Start the server first (mock is enough for a transport benchmark):
    python scripts/deploy.py --mock --mock-latency-ms 50

    # Compare all transports with random images every step:
    python scripts/inference_continuous_test.py --transport all --steps 100

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
import time
from urllib.parse import urlparse

# Canonical client helpers live under benchmarks.utils — add project root so import works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.utils.client import build_payload, encode_path_b64, get  # noqa: E402
from benchmarks.utils.transport import TRANSPORTS, make_policy_client  # noqa: E402


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


def _build_payloads(args, state) -> list[dict]:
    """Pre-build the payload sequence once so every transport sees identical input."""
    if args.head_camera is None:
        return [
            build_payload(
                head=_make_random_image_b64(args.height, args.width),
                left_wrist=_make_random_image_b64(args.height, args.width),
                right_wrist=_make_random_image_b64(args.height, args.width),
                prompt=args.prompt,
                state=state,
            )
            for _ in range(args.steps)
        ]
    static = build_payload(
        head=encode_path_b64(args.head_camera),
        left_wrist=encode_path_b64(args.left_wrist_camera) if args.left_wrist_camera else None,
        right_wrist=encode_path_b64(args.right_wrist_camera) if args.right_wrist_camera else None,
        prompt=args.prompt,
        state=state,
    )
    return [static] * args.steps


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); empty list -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_transport(client, payloads: list[dict], verbose: bool = False) -> dict:
    """Drive one transport through the payload sequence, timing each round-trip.

    Resets server state first, then wraps every ``predict`` in a client-side
    timer. ``overhead = rtt - server_latency`` isolates pure transport cost
    (the engine's inference time cancels out across transports).
    """
    client.reset()
    rtts: list[float] = []
    server_lats: list[float] = []
    overheads: list[float] = []
    last_action = None
    for step, payload in enumerate(payloads):
        t0 = time.perf_counter()
        result = client.predict(payload)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        server_lat = float(result.get("latency_ms", 0.0))
        last_action = result.get("action", last_action)
        rtts.append(rtt_ms)
        server_lats.append(server_lat)
        overheads.append(max(rtt_ms - server_lat, 0.0))
        if verbose:
            print(f"{step:5d}  rtt={rtt_ms:8.2f}ms  server={server_lat:8.2f}ms  overhead={overheads[-1]:8.2f}ms")
    return {"rtts": rtts, "server_lats": server_lats, "overheads": overheads, "last_action": last_action}


def _summarize(name: str, data: dict) -> dict:
    rtts, server_lats, overheads = data["rtts"], data["server_lats"], data["overheads"]
    return {
        "name": name,
        "n": len(rtts),
        "rtt_avg": _avg(rtts),
        "rtt_p50": _percentile(rtts, 50),
        "rtt_p95": _percentile(rtts, 95),
        "rtt_min": min(rtts) if rtts else 0.0,
        "srv_avg": _avg(server_lats),
        "ovh_avg": _avg(overheads),
        "ovh_p50": _percentile(overheads, 50),
        "ovh_p95": _percentile(overheads, 95),
        "ovh_min": min(overheads) if overheads else 0.0,
    }


def _print_comparison(summaries: list[dict]) -> None:
    print("\n" + "=" * 84)
    print("Transport comparison (ms; overhead = client RTT - server inference, lower is better)")
    print("=" * 84)
    print(
        f"{'transport':>16} {'rtt_avg':>9} {'rtt_p50':>9} {'rtt_p95':>9} "
        f"{'rtt_min':>9} {'srv_avg':>9} {'ovh_avg':>9} {'ovh_p50':>9}"
    )
    print("-" * 84)
    for s in summaries:
        print(
            f"{s['name']:>16} {s['rtt_avg']:9.2f} {s['rtt_p50']:9.2f} {s['rtt_p95']:9.2f} "
            f"{s['rtt_min']:9.2f} {s['srv_avg']:9.2f} {s['ovh_avg']:9.2f} {s['ovh_p50']:9.2f}"
        )

    if len(summaries) < 2:
        return

    by_name = {s["name"]: s for s in summaries}
    baseline = by_name.get("http", summaries[0])
    print(f"\nVerdict (vs {baseline['name']} baseline):")
    for s in summaries:
        if s is baseline:
            continue
        d_ovh = baseline["ovh_avg"] - s["ovh_avg"]
        p_ovh = d_ovh / baseline["ovh_avg"] * 100.0 if baseline["ovh_avg"] else 0.0
        d_rtt = baseline["rtt_p50"] - s["rtt_p50"]
        p_rtt = d_rtt / baseline["rtt_p50"] * 100.0 if baseline["rtt_p50"] else 0.0
        print(
            f"  {s['name']:>16}: overhead_avg {-d_ovh:+7.2f}ms ({-p_ovh:+5.1f}%)   rtt_p50 {-d_rtt:+7.2f}ms ({-p_rtt:+5.1f}%)"
        )
    best = min(summaries, key=lambda s: s["ovh_avg"])
    print(f"\nLowest transport overhead: {best['name']} ({best['ovh_avg']:.2f}ms avg)")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous inference + transport benchmark for OpenWAM policy server."
    )
    parser.add_argument(
        "--transport",
        choices=[*TRANSPORTS, "all"],
        default="all",
        help="Transport(s) to benchmark (default: all).",
    )
    parser.add_argument(
        "--http-url",
        "--server",
        dest="http_url",
        type=str,
        default="http://127.0.0.1:8848",
        help="HTTP base URL (legacy --server alias accepted).",
    )
    parser.add_argument("--ws-url", type=str, default=None, help="WebSocket URL (default: ws://<http-host>:8850).")
    parser.add_argument("--steps", type=int, default=100, help="Total control steps to simulate.")
    parser.add_argument("--height", type=int, default=480, help="Random-image height (ignored if --head-camera given).")
    parser.add_argument("--width", type=int, default=640, help="Random-image width (ignored if --head-camera given).")
    parser.add_argument("--prompt", type=str, default="robot picks up the red bottle from the table")
    parser.add_argument("--state", type=float, nargs="*", default=None, help="Raw proprio state values to send.")
    parser.add_argument("--state-file", type=str, default=None, help="JSON list, or object with a 'state' list.")
    parser.add_argument("--state-dim", type=int, default=20, help="Dummy zero-state dimension when --state is omitted.")
    parser.add_argument("--no-state", action="store_true", help="Do not include state in the payload.")
    parser.add_argument(
        "--head-camera", type=str, default=None, help="Static path for head camera (reused every step)."
    )
    parser.add_argument("--left-wrist-camera", type=str, default=None, help="Static path for left wrist camera.")
    parser.add_argument("--right-wrist-camera", type=str, default=None, help="Static path for right wrist camera.")
    parser.add_argument("--verbose", action="store_true", help="Print per-step timing for each transport.")
    args = parser.parse_args()

    # ws-url defaults to the http host so `--server http://remote:8848` also points WS there.
    ws_url = args.ws_url or f"ws://{urlparse(args.http_url).hostname or '127.0.0.1'}:8850"
    transports = list(TRANSPORTS) if args.transport == "all" else [args.transport]
    state = _resolve_state(args)
    stream_random = args.head_camera is None

    print(f"HTTP:       {args.http_url}")
    print(f"WS:         {ws_url}")
    print(f"Transports: {', '.join(transports)}")
    print(f"Steps:      {args.steps}")
    if stream_random:
        print(f"Mode:       random {args.height}x{args.width} images every step (all 3 cameras)")
    else:
        print(
            f"Mode:       static frames  head={args.head_camera}  "
            f"left={args.left_wrist_camera or '(null)'}  right={args.right_wrist_camera or '(null)'}"
        )
    print(f"Prompt:     {args.prompt}")
    print(f"State:      {'omitted' if state is None else f'{len(state)} dims'}")
    print("=" * 60)

    # Health check over HTTP (best-effort: the server may be ws-only).
    try:
        print(f"Health:  {get(args.http_url, '/health')}")
    except Exception as exc:
        print(f"Health:  (skipped: {exc})")

    # Build the payload sequence once; every transport replays the same bytes.
    payloads = _build_payloads(args, state)

    summaries = []
    for name in transports:
        print(f"\n>>> transport = {name}  ({args.steps} steps)")
        try:
            with make_policy_client(name, http_url=args.http_url, ws_url=ws_url) as client:
                data = run_transport(client, payloads, verbose=args.verbose)
        except Exception as exc:
            # One transport being unreachable (e.g. a ws-only server) must not
            # abort the others or discard results already printed.
            print(f"    [skip] transport {name} unavailable: {exc}")
            continue
        s = _summarize(name, data)
        action = data["last_action"]
        print(
            f"    rtt avg {s['rtt_avg']:.2f}ms  p50 {s['rtt_p50']:.2f}ms  p95 {s['rtt_p95']:.2f}ms  | "
            f"overhead avg {s['ovh_avg']:.2f}ms  | server avg {s['srv_avg']:.2f}ms  | "
            f"action_dim {len(action) if action else '?'}"
        )
        summaries.append(s)

    if not summaries:
        print("\nNo transport completed successfully.")
        return
    _print_comparison(summaries)


if __name__ == "__main__":
    main()
