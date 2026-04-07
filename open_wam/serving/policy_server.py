"""WebSocket + HTTP policy server for real-time robot deployment.

Provides a network-accessible policy server that wraps WAMPolicy with
receding-horizon execution. Robot controllers connect via WebSocket for
low-latency streaming or HTTP for request-response patterns.

Protocol (WebSocket):
    Client sends JSON: {"type": "obs", "image": <base64_jpeg>, "state": [floats], "prompt": "..."}
    Server responds:   {"type": "action", "action": [floats], "step": int, "latency_ms": float}

    Client sends: {"type": "reset"}
    Server responds: {"type": "reset_ack"}

Protocol (HTTP):
    POST /predict  — same JSON as WebSocket obs message
    POST /reset    — reset policy state
    GET  /health   — server health check
    GET  /info     — model info and config

Usage:
    server = PolicyServer(engine, cfg)
    server.run(host="0.0.0.0", port=8765)
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PolicyServer:
    """WebSocket + HTTP policy server for WAM deployment.

    Args:
        engine: Inference engine (BaseInferenceEngine).
        cfg: Config with policy and server settings.
        embodiment: Optional embodiment name for action space conversion.
    """

    def __init__(self, engine, cfg, embodiment: Optional[str] = None):
        self.engine = engine
        self.cfg = cfg
        self.embodiment = embodiment

        # Lazy imports at init time to validate availability
        self._policy = None
        self._adapter = None
        self._request_count = 0
        self._total_latency = 0.0

    def _init_policy(self):
        """Initialize policy and optional embodiment adapter."""
        if self._policy is not None:
            return

        from open_wam.evaluation.policy import WAMPolicy

        policy_cfg = getattr(self.cfg, "policy", self.cfg)
        deploy = getattr(self.cfg, "deploy", None)
        async_config = getattr(deploy, "async_execution", None) if deploy else None
        self._policy = WAMPolicy(
            engine=self.engine,
            cfg=policy_cfg,
            async_config=async_config,
        )

        if self.embodiment:
            from open_wam.data.embodiment import ActionSpaceAdapter

            self._adapter = ActionSpaceAdapter(self.embodiment)

    def predict(self, obs: dict) -> dict:
        """Synchronous prediction for a single observation.

        Args:
            obs: Observation dict with "image" (PIL Image or base64 string)
                and optional "state" (list of floats), "prompt" (str).

        Returns:
            dict with "action" (list of floats), "step", "latency_ms".
        """
        self._init_policy()
        t0 = time.monotonic()

        # Decode base64 image if needed
        obs = self._decode_obs(obs)

        action = self._policy.predict_action(obs)

        # Convert from canonical to native action space
        if self._adapter is not None:
            action = self._adapter.canonical_to_native(action.reshape(1, -1))[0]

        latency_ms = (time.monotonic() - t0) * 1000
        self._request_count += 1
        self._total_latency += latency_ms

        return {
            "action": action.tolist() if isinstance(action, np.ndarray) else list(action),
            "step": self._request_count,
            "latency_ms": round(latency_ms, 2),
        }

    def reset(self):
        """Reset policy state."""
        if self._policy is not None:
            self._policy.reset()
        self._request_count = 0
        self._total_latency = 0.0

    def shutdown(self):
        """Clean up async resources."""
        if self._policy is not None:
            self._policy.shutdown()

    def get_info(self) -> dict:
        """Return server info and statistics."""
        avg_latency = self._total_latency / self._request_count if self._request_count > 0 else 0.0
        return {
            "model": "OpenWAM",
            "embodiment": self.embodiment,
            "total_requests": self._request_count,
            "avg_latency_ms": round(avg_latency, 2),
            "policy_config": {
                "execute_horizon": getattr(getattr(self.cfg, "policy", self.cfg), "execute_horizon", None),
                "temporal_ensemble": getattr(getattr(self.cfg, "policy", self.cfg), "temporal_ensemble", True),
            },
        }

    def _decode_obs(self, obs: dict) -> dict:
        """Decode base64 image in observation if present."""
        from PIL import Image

        if "image" in obs and isinstance(obs["image"], str):
            try:
                img_bytes = base64.b64decode(obs["image"])
                obs["image"] = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                logger.warning("Failed to decode base64 image: %s", e)

        if "state" in obs and isinstance(obs["state"], list):
            obs["state"] = np.array(obs["state"], dtype=np.float32)

        return obs

    def run(self, host: str = "0.0.0.0", port: int = 8765, http_port: Optional[int] = None):
        """Start the WebSocket + HTTP server.

        Requires ``websockets`` and ``aiohttp`` packages.
        """
        try:
            import aiohttp  # noqa: F401
            import websockets
            from aiohttp import web
        except ImportError:
            raise ImportError("Server dependencies required. Install with:\n  pip install websockets aiohttp")

        self._init_policy()

        async def ws_handler(websocket):
            """Handle WebSocket connections."""
            logger.info("Client connected: %s", websocket.remote_address)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "obs")

                        if msg_type == "reset":
                            self.reset()
                            await websocket.send(json.dumps({"type": "reset_ack"}))
                        elif msg_type == "obs":
                            result = self.predict(data)
                            result["type"] = "action"
                            await websocket.send(json.dumps(result))
                        else:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": f"Unknown message type: {msg_type}",
                                    }
                                )
                            )
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": str(e),
                                }
                            )
                        )
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected")

        async def http_predict(request):
            """HTTP POST /predict endpoint."""
            data = await request.json()
            result = self.predict(data)
            return web.json_response(result)

        async def http_reset(request):
            """HTTP POST /reset endpoint."""
            self.reset()
            return web.json_response({"status": "ok"})

        async def http_health(request):
            """HTTP GET /health endpoint."""
            return web.json_response({"status": "healthy"})

        async def http_info(request):
            """HTTP GET /info endpoint."""
            return web.json_response(self.get_info())

        async def start_servers():
            # HTTP server
            app = web.Application()
            app.router.add_post("/predict", http_predict)
            app.router.add_post("/reset", http_reset)
            app.router.add_get("/health", http_health)
            app.router.add_get("/info", http_info)

            runner = web.AppRunner(app)
            await runner.setup()
            resolved_http_port = port + 1 if http_port is None else http_port
            site = web.TCPSite(runner, host, resolved_http_port)
            await site.start()
            logger.info("HTTP server started on %s:%d", host, resolved_http_port)

            # WebSocket server
            async with websockets.serve(ws_handler, host, port):
                logger.info("WebSocket server started on ws://%s:%d", host, port)
                await asyncio.Future()  # Run forever

        resolved_http_port = port + 1 if http_port is None else http_port
        logger.info(
            "Starting PolicyServer on %s:%d (WS) and %s:%d (HTTP)",
            host,
            port,
            host,
            resolved_http_port,
        )
        asyncio.run(start_servers())


def build_server_from_config(cfg, ckpt_path: str, device: str = "cuda", embodiment: Optional[str] = None):
    """Build a PolicyServer from Hydra-style config and checkpoint path."""
    from open_wam.inference import JointInferenceEngine, load_wam_models

    cfg.eval.ckpt_path = ckpt_path
    pipe, action_dit = load_wam_models(cfg, device=device)
    engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, action_dit=action_dit)
    return PolicyServer(engine=engine, cfg=cfg, embodiment=embodiment)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the OpenWAM policy server.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config. Defaults to configs/config.yaml from the repo root.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Checkpoint path passed to the model loader.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--host", type=str, default=None, help="WebSocket/HTTP bind host override.")
    parser.add_argument("--ws-port", type=int, default=None, help="WebSocket port override.")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP port override.")
    parser.add_argument("--embodiment", type=str, default=None, help="Optional embodiment name.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional OmegaConf dotlist overrides, e.g. model/backbone=ti2v_5b",
    )
    return parser


def main(argv: Optional[list[str]] = None):
    """CLI entrypoint for running the OpenWAM policy server."""
    from omegaconf import OmegaConf

    parser = _build_argparser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = Path(args.config) if args.config else project_root / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    server_cfg = getattr(cfg, "server", None)
    if server_cfg is None:
        deploy_cfg = getattr(cfg, "deploy", None)
        server_cfg = getattr(deploy_cfg, "server", None) if deploy_cfg is not None else None

    host = args.host or getattr(server_cfg, "host", "0.0.0.0")
    ws_port = args.ws_port or getattr(server_cfg, "ws_port", 8765)
    http_port = args.http_port or getattr(server_cfg, "http_port", ws_port + 1)
    embodiment = args.embodiment if args.embodiment is not None else getattr(cfg, "embodiment", None)

    server = build_server_from_config(
        cfg=cfg,
        ckpt_path=args.ckpt_path,
        device=args.device,
        embodiment=embodiment,
    )
    server.run(host=host, port=ws_port, http_port=http_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
