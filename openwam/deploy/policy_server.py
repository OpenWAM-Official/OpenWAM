"""WebSocket + HTTP policy server for real-time robot deployment.

Provides a network-accessible policy server that wraps WAMPolicy with
receding-horizon execution. Robot controllers connect via WebSocket for
low-latency streaming or HTTP for request-response patterns.

The client is thin on purpose: it always sends raw per-camera JPEGs plus a
base task prompt. All image composition, resize, and prompt wrapping happen
server-side, driven by the saved training config (``cfg.dataloader.multiview``
/ ``camera_layout`` / ``height`` / ``width``).

Protocol (unified — same shape for single-view and multi-view checkpoints):
    Client → {
        "type": "obs",
        "images": {
            "head_camera":        <base64_jpeg>,       # required
            "left_wrist_camera":  <base64_jpeg>|null,  # optional
            "right_wrist_camera": <base64_jpeg>|null   # optional
        },
        "prompt": "<base task prompt>",
        "state":  [floats]                             # optional proprio
    }

Server-side behavior:
- ``multiview=False``: ignores wrist fields, crop+resize ``head_camera``.
- ``multiview=True``:  black-fills missing/None wrists, then composes the
  L-shape layout defined by ``camera_layout``.
- ``prompt`` is always re-wrapped via ``format_prompt_for_inference`` so that
  multiview checkpoints see the same prompt template as at training time.

Responses:
    {"type": "action", "action": [floats], "step": int, "latency_ms": float}
    {"type": "error",  "code": str, "message": "<what went wrong>"}

    Client sends: {"type": "reset"}
    Server responds: {"type": "reset_ack"}

HTTP endpoints:
    POST /predict  — same JSON as WebSocket obs message (400 on bad payload)
    POST /reset    — reset policy state
    GET  /health   — server health check
    GET  /info     — model info and config

Usage:
    server = PolicyServer(engine, cfg)
    server.run(host="0.0.0.0", port=8850)
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ObsValidationError(ValueError):
    """Raised when a client observation payload does not match the server's
    view configuration (single-view vs multi-view, missing cameras, bad image
    encoding, etc.).  Server turns it into a structured error response to the
    client (HTTP 400 / WebSocket ``{"type":"error"}``).
    """


class PolicyServer:
    """WebSocket + HTTP policy server for WAM deployment.

    Args:
        engine: Inference engine (BaseInferenceEngine).
        cfg: Config with policy and server settings.
        embodiment: Optional embodiment name for action space conversion.
    """

    def __init__(
        self,
        engine,
        cfg,
        embodiment: Optional[str] = None,
        debug: bool = False,
        debug_dir: str = "./server_debug",
    ):
        self.engine = engine
        self.cfg = cfg
        self.embodiment = embodiment

        # Lazy imports at init time to validate availability
        self._policy = None
        self._adapter = None
        self._request_count = 0
        self._total_latency = 0.0

        # Debug mode: save received images + actions + metadata per step.
        # ``_debug_episode`` is lazily initialized the first time reset() or
        # predict() is called (see ``_ensure_debug_episode``). This prevents
        # the stray ``ep-001/`` directory that used to appear when /predict was
        # called before any /reset.
        self._debug = debug
        self._debug_dir = debug_dir
        self._debug_episode: Optional[int] = None
        self._debug_step = 0
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
            logger.info("Debug mode enabled — saving to %s", debug_dir)

    def _init_policy(self):
        """Initialize policy and optional embodiment adapter."""
        if self._policy is not None:
            return

        from openwam.deploy.policy import WAMPolicy

        policy_cfg = getattr(self.cfg, "policy", self.cfg)
        deploy = getattr(self.cfg, "deploy", None)
        async_config = getattr(deploy, "async_execution", None) if deploy else None
        self._policy = WAMPolicy(
            engine=self.engine,
            cfg=policy_cfg,
            async_config=async_config,
        )

        # Resolve view mode from saved config so every predict() can validate
        # + preprocess the client payload without re-reading it per-request.
        from openwam.dataloader.transforms.multiview import DEFAULT_MULTIVIEW_CAMERA_LAYOUT

        dl = getattr(self.cfg, "dataloader", None)
        self._multiview = bool(getattr(dl, "multiview", False)) if dl is not None else False
        _layout = getattr(dl, "camera_layout", None) if dl is not None else None
        self._camera_layout = list(_layout) if _layout is not None else list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT)
        self._target_camera = getattr(dl, "target_camera", "head_camera") if dl is not None else "head_camera"
        # Output canvas size: prefer inference.{height,width}, fall back to dataloader
        _inf = getattr(self.cfg, "inference", None)
        _h = getattr(_inf, "height", None) if _inf is not None else None
        _w = getattr(_inf, "width", None) if _inf is not None else None
        if _h is None and dl is not None:
            _h = getattr(dl, "height", 384)
        if _w is None and dl is not None:
            _w = getattr(dl, "width", 320)
        self._img_height = int(_h if _h is not None else 384)
        self._img_width = int(_w if _w is not None else 320)
        logger.info(
            "[obs] View config: multiview=%s, camera_layout=%s, target_camera=%s, canvas=%dx%d",
            self._multiview,
            self._camera_layout if self._multiview else "[unused]",
            self._target_camera if not self._multiview else "[unused]",
            self._img_height,
            self._img_width,
        )

        if self.embodiment:
            # embodiment module moved to previous_codebase/; restore it to use this feature
            from openwam.dataloader.embodiment import ActionSpaceAdapter  # noqa: F401

            self._adapter = ActionSpaceAdapter(self.embodiment)

    def predict(self, obs: dict) -> dict:
        """Synchronous prediction for a single observation.

        Args:
            obs: Observation dict with ``images`` (dict of camera name →
                base64 JPEG / bytes / PIL.Image, with ``head_camera`` required
                and ``left_wrist_camera`` / ``right_wrist_camera`` optional),
                a base ``prompt`` (str), and optional ``state`` (list of floats).
                Server does all preprocessing and prompt wrapping internally.

        Returns:
            dict with "action" (list of floats in physical units),
            "step", "latency_ms".
        """
        self._init_policy()
        t0 = time.monotonic()

        state_raw = obs.get("state")

        # _decode_obs populates obs["image"] with a composite PIL image (post
        # crop/resize for single-view, post L-shape composition for multi-view)
        # and obs["prompt"] with the wrapped string. Capture them after decode
        # so debug artifacts match what the model actually saw.
        obs = self._decode_obs(obs)
        image_pil = obs.get("image")
        wrapped_prompt: str = obs.get("prompt", "")

        action = self._policy.predict_action(obs)

        # Convert from canonical to native action space
        if self._adapter is not None:
            action = self._adapter.canonical_to_native(action.reshape(1, -1))[0]

        latency_ms = (time.monotonic() - t0) * 1000
        self._request_count += 1
        self._total_latency += latency_ms

        result = {
            "action": action.tolist() if isinstance(action, np.ndarray) else list(action),
            "step": self._request_count,
            "latency_ms": round(latency_ms, 2),
        }

        if self._debug:
            self._ensure_debug_episode()  # lazy-open episode 0 on first predict without reset
            self._debug_step += 1
            self._save_debug_step(
                image_pil=image_pil,
                prompt=wrapped_prompt,
                state_raw=state_raw,
                result=result,
            )

        return result

    def _ensure_debug_episode(self) -> None:
        """Lazily open the current debug episode directory.

        On first call (``_debug_episode is None``) this sets episode index to
        0 and creates ``<debug_dir>/ep0000/``. Subsequent calls are no-ops —
        incrementing is the responsibility of ``reset()``.
        """
        if not self._debug or self._debug_episode is not None:
            return
        self._debug_episode = 0
        self._debug_step = 0
        ep_dir = os.path.join(self._debug_dir, f"ep{self._debug_episode:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        logger.info("Debug episode %d (lazy-start) → %s", self._debug_episode, ep_dir)

    def _save_debug_step(
        self,
        image_pil,
        prompt: str,
        state_raw,
        result: dict,
    ) -> None:
        """Save per-step debug data: processed image + metadata JSON.

        Directory structure:
          debug_dir/ep{N:04d}/step_{N:04d}/
            image_processed.jpg  — the PIL image the pipeline actually saw
                                   (post crop/resize or multi-view composition)
            meta.json            — wrapped prompt, state, action, latency, step, episode
        """
        ep_dir = os.path.join(self._debug_dir, f"ep{self._debug_episode:04d}")
        step_dir = os.path.join(ep_dir, f"step_{self._debug_step:04d}")
        os.makedirs(step_dir, exist_ok=True)

        # Save the post-preprocessing image that went into the pipeline.
        if image_pil is not None:
            try:
                image_pil.save(os.path.join(step_dir, "image_processed.jpg"), format="JPEG", quality=95)
            except Exception as exc:
                logger.warning("Debug: failed to save processed image: %s", exc)

        # Normalise state to a plain list for JSON serialisation
        if isinstance(state_raw, np.ndarray):
            state_list = state_raw.tolist()
        elif state_raw is not None:
            state_list = list(state_raw)
        else:
            state_list = None

        meta = {
            "episode": self._debug_episode,
            "step": self._debug_step,
            "server_request_count": result["step"],
            "prompt": prompt,
            "state": state_list,
            "action": result["action"],
            "latency_ms": result["latency_ms"],
        }
        with open(os.path.join(step_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def reset(self):
        """Reset policy state."""
        if self._policy is not None:
            self._policy.reset()
        self._request_count = 0
        self._total_latency = 0.0
        if self._debug:
            # Lazy-init on first reset (-> episode 0); otherwise advance by 1.
            self._debug_episode = 0 if self._debug_episode is None else self._debug_episode + 1
            self._debug_step = 0
            ep_dir = os.path.join(self._debug_dir, f"ep{self._debug_episode:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            logger.info("Debug episode %d → %s", self._debug_episode, ep_dir)

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
        """Validate and preprocess a client observation payload.

        Unified payload contract across single- and multi-view checkpoints:
        client sends ``obs["images"]`` as a dict with fixed keys (``head_camera``
        required, ``left_wrist_camera`` / ``right_wrist_camera`` optional and
        may be ``None``). Server dispatches by ``cfg.dataloader.multiview``:

        - ``multiview=False``: use ``head_camera`` only, ``crop_and_resize`` to
          (W, H). Wrist fields are ignored.
        - ``multiview=True``: black-fill missing/None wrists, then compose the
          L-shape layout keyed by ``cfg.dataloader.camera_layout``.

        On success the returned obs always has:
            obs["image"]  -> PIL.Image sized (self._img_width, self._img_height)
            obs["prompt"] -> str wrapped via ``format_prompt_for_inference``

        Raises :class:`ObsValidationError` on malformed payload.
        """
        from PIL import Image

        from openwam.dataloader.transforms.multiview import (
            assemble_multiview_layout,
            crop_and_resize,
            format_prompt_for_inference,
        )

        def _as_pil(x, *, ctx: str) -> Image.Image:
            if isinstance(x, Image.Image):
                return x if x.mode == "RGB" else x.convert("RGB")
            if isinstance(x, (bytes, bytearray)):
                try:
                    return Image.open(io.BytesIO(bytes(x))).convert("RGB")
                except Exception as e:
                    raise ObsValidationError(f"{ctx}: failed to decode raw image bytes ({e})")
            if isinstance(x, str):
                try:
                    raw = base64.b64decode(x)
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as e:
                    raise ObsValidationError(f"{ctx}: failed to decode base64 JPEG ({e})")
            raise ObsValidationError(
                f"{ctx}: expected base64 JPEG string, raw bytes, or PIL.Image, got {type(x).__name__}"
            )

        # --- Payload shape validation ---
        if "images" not in obs or not isinstance(obs.get("images"), dict):
            raise ObsValidationError(
                "client must send 'images' dict with head_camera key "
                "(left_wrist_camera / right_wrist_camera optional, may be null). "
                "The legacy single-field 'image' payload is no longer supported."
            )

        imgs = obs["images"]
        head_raw = imgs.get("head_camera")
        if head_raw is None:
            raise ObsValidationError(
                "head_camera is required in obs['images'] (got None or missing). "
                "The head camera feed is never optional on either single-view or multi-view servers."
            )
        head_pil = _as_pil(head_raw, ctx="images['head_camera']")

        # --- Dispatch by server's configured view mode ---
        if not self._multiview:
            if imgs.get("left_wrist_camera") is not None or imgs.get("right_wrist_camera") is not None:
                logger.info(
                    "[obs] single-view mode (target_camera=%s); ignoring wrist camera inputs.",
                    self._target_camera,
                )
            obs["image"] = crop_and_resize(head_pil, self._img_height, self._img_width)
        else:
            # Multi-view: assemble via camera_layout with black-fill for missing wrists.
            if len(self._camera_layout) < 3:
                raise ObsValidationError(
                    f"multi-view server requires camera_layout with >= 3 entries; "
                    f"got {self._camera_layout}. Check the checkpoint's config.yaml."
                )

            def _decode_or_black(raw, ctx: str) -> Image.Image:
                if raw is None:
                    return Image.new("RGB", (self._img_width, self._img_height), (0, 0, 0))
                return _as_pil(raw, ctx=ctx)

            left_pil = _decode_or_black(imgs.get("left_wrist_camera"), ctx="images['left_wrist_camera']")
            right_pil = _decode_or_black(imgs.get("right_wrist_camera"), ctx="images['right_wrist_camera']")

            # Map the fixed client-side keys to camera_layout positions:
            #   head_camera        -> layout[0]  (top)
            #   left_wrist_camera  -> layout[1]  (bottom-left)
            #   right_wrist_camera -> layout[2]  (bottom-right)
            frames = {
                self._camera_layout[0]: head_pil,
                self._camera_layout[1]: left_pil,
                self._camera_layout[2]: right_pil,
            }
            obs["image"] = assemble_multiview_layout(
                frames,
                camera_layout=self._camera_layout,
                out_h=self._img_height,
                out_w=self._img_width,
            )

        # --- Prompt wrapping (must match training-time _get_prompt byte-for-byte) ---
        obs["prompt"] = format_prompt_for_inference(
            obs.get("prompt", "") or "",
            self._multiview,
            self._camera_layout,
        )

        if "state" in obs and isinstance(obs["state"], list):
            obs["state"] = np.array(obs["state"], dtype=np.float32)

        return obs

    def run(self, host: str = "0.0.0.0", port: int = 8850, http_port: Optional[int] = None):
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
                                        "code": "unknown_message_type",
                                        "message": f"Unknown message type: {msg_type}",
                                    }
                                )
                            )
                    except ObsValidationError as e:
                        # Client-side mistake: bad payload shape / missing cameras / bad base64.
                        # Logged at INFO so it doesn't look like a server crash.
                        logger.info("[obs] validation failed: %s", e)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "code": "obs_validation_error",
                                    "message": str(e),
                                }
                            )
                        )
                    except Exception as e:
                        logger.exception("Error processing message")
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "code": "internal_error",
                                    "message": str(e),
                                }
                            )
                        )
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected")

        async def http_predict(request):
            """HTTP POST /predict endpoint."""
            try:
                data = await request.json()
                result = self.predict(data)
                return web.json_response(result)
            except ObsValidationError as e:
                logger.info("[obs] validation failed: %s", e)
                return web.json_response(
                    {"type": "error", "code": "obs_validation_error", "message": str(e)},
                    status=400,
                )
            except Exception as e:
                logger.exception("/predict failed")
                return web.json_response(
                    {"type": "error", "code": "internal_error", "message": str(e)},
                    status=500,
                )

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
            resolved_http_port = 8848 if http_port is None else http_port
            site = web.TCPSite(runner, host, resolved_http_port)
            await site.start()
            logger.info("HTTP server started on %s:%d", host, resolved_http_port)

            # WebSocket server
            async with websockets.serve(ws_handler, host, port):
                logger.info("WebSocket server started on ws://%s:%d", host, port)
                await asyncio.Future()  # Run forever

        resolved_http_port = 8848 if http_port is None else http_port
        logger.info(
            "Starting PolicyServer on %s:%d (WS) and %s:%d (HTTP)",
            host,
            port,
            host,
            resolved_http_port,
        )
        asyncio.run(start_servers())


def build_server_from_config(cfg, ckpt_dir: str, device: str = "cuda", embodiment: Optional[str] = None):
    """Build a PolicyServer from a self-contained checkpoint directory.

    Aligns with ``scripts/deploy.py`` — uses ``load_from_checkpoint_dir`` so
    the CLI's ``--ckpt-dir`` means the same thing in both entrypoints: the
    directory that contains ``config.yaml`` + ``checkpoint_step_*.safetensors``.
    """
    from openwam.deploy import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    cfg, pipe, architecture = load_from_checkpoint_dir(ckpt_dir, device=device)
    engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, architecture=architecture)
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
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory (config.yaml + checkpoint_step_*.safetensors). "
        "Same meaning as scripts/deploy.py --ckpt-dir.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--host", type=str, default=None, help="WebSocket/HTTP bind host override.")
    parser.add_argument("--ws-port", type=int, default=None, help="WebSocket port override.")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP port override.")
    parser.add_argument("--embodiment", type=str, default=None, help="Optional embodiment name.")
    # Mock mode
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (random actions, no weights required).")
    parser.add_argument(
        "--mock-action-dim", type=int, default=20, help="Action dimension for mock engine (default: 20)."
    )
    parser.add_argument(
        "--mock-latency-ms", type=float, default=2000.0, help="Simulated inference latency in ms (default: 2000)."
    )
    # Debug mode
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode: save received images + actions + metadata per step."
    )
    parser.add_argument(
        "--debug-dir", type=str, default="./server_debug", help="Directory for debug output (default: ./server_debug)."
    )
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

    if not args.mock and args.ckpt_dir is None:
        parser.error("--ckpt-dir is required unless --mock is set")

    project_root = Path(__file__).resolve().parent.parent.parent

    if args.mock:
        from openwam.deploy.mock_engine import MockInferenceEngine

        cfg = OmegaConf.create({})
        if args.config:
            cfg = OmegaConf.merge(OmegaConf.load(args.config), cfg)
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
        engine = MockInferenceEngine(
            cfg=cfg,
            action_dim=args.mock_action_dim,
            latency_ms=args.mock_latency_ms,
        )
    else:
        config_path = Path(args.config) if args.config else project_root / "configs" / "config.yaml"
        cfg = OmegaConf.load(config_path)
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
        engine = None  # built inside build_server_from_config

    server_cfg = getattr(cfg, "server", None)
    if server_cfg is None:
        deploy_cfg = getattr(cfg, "deploy", None)
        server_cfg = getattr(deploy_cfg, "server", None) if deploy_cfg is not None else None

    host = args.host or getattr(server_cfg, "host", "0.0.0.0")
    ws_port = args.ws_port or getattr(server_cfg, "ws_port", 8850)
    http_port = args.http_port or getattr(server_cfg, "http_port", 8848)
    embodiment = args.embodiment if args.embodiment is not None else getattr(cfg, "embodiment", None)

    if args.mock:
        server = PolicyServer(
            engine=engine,
            cfg=cfg,
            embodiment=embodiment,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
    else:
        server = build_server_from_config(
            cfg=cfg,
            ckpt_dir=args.ckpt_dir,
            device=args.device,
            embodiment=embodiment,
        )
        server._debug = args.debug
        server._debug_dir = args.debug_dir
        if args.debug:
            os.makedirs(args.debug_dir, exist_ok=True)
            logger.info("Debug mode enabled — saving to %s", args.debug_dir)
    server.run(host=host, port=ws_port, http_port=http_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
