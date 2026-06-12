"""WebSocket policy server for real-time robot deployment.

Provides a network-accessible policy server that wraps WAMPolicy with
receding-horizon execution. Robot controllers connect over a single
persistent WebSocket.

The client is thin on purpose: it always sends raw per-camera JPEGs plus a
base task prompt. Image composition and resize happen server-side, driven by
the saved training config (``cfg.dataloader.multiview`` / ``camera_layout`` /
``height`` / ``width``); the prompt is wrapped with the FastWAM deploy
template.

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
- ``prompt`` is always re-wrapped via ``format_prompt_for_inference``.

Messages:
    obs   → {"type": "action", "action": [floats], "step": int, "latency_ms": float}
    reset → {"type": "reset_ack"}
    ping  → {"type": "pong"}
    error → {"type": "error", "code": str, "message": "<what went wrong>"}

Usage:
    server = PolicyServer(engine, cfg)
    server.run(host="0.0.0.0", port=8848)
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from openwam import ws_protocol as wsp
from openwam.deploy.obs_decoder import ObsDecoder, ObsValidationError

logger = logging.getLogger(__name__)
_COMPILE_MODES = ("auto", "none")

# Cap for a single obs message. A multi-camera base64 frame can exceed the
# 1 MB websockets default, so lift it for the obs stream.
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def _infer_video_num_frames(dl) -> int:
    """Return the video frame count seen by Wan after dataloader sub-sampling."""
    from omegaconf import OmegaConf

    raw_frames = int(OmegaConf.select(dl, "num_frames", default=33))
    video_stride = int(OmegaConf.select(dl, "video_stride", default=1) or 1)
    if video_stride <= 0:
        video_stride = 1
    return (raw_frames - 1) // video_stride + 1


def _normalize_compile_mode_in_cfg(cfg) -> None:
    """Keep package and script entrypoints aligned on compile-mode validation."""
    from omegaconf import OmegaConf

    from openwam.model.compile_options import normalize_compile_mode

    mode = OmegaConf.select(cfg, "optimization.compile.mode", default=None)
    if mode is not None:
        OmegaConf.update(cfg, "optimization.compile.mode", normalize_compile_mode(mode), merge=False)


def _apply_compile_mode_override(cfg, compile_mode: Optional[str]) -> None:
    """Apply a named CLI compile-mode override, then normalize the config."""
    from omegaconf import OmegaConf

    if compile_mode is not None:
        OmegaConf.update(cfg, "optimization.compile.mode", compile_mode, merge=False)
    _normalize_compile_mode_in_cfg(cfg)


def _compile_mode_choices() -> tuple[str, ...]:
    return _COMPILE_MODES


def _normalize_compile_mode_arg(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _COMPILE_MODES:
        raise argparse.ArgumentTypeError(f"Unknown compile mode '{value}'. Choose from: {', '.join(_COMPILE_MODES)}")
    return normalized


class PolicyServer:
    """WebSocket policy server for WAM deployment.

    Args:
        engine: Inference engine (BaseInferenceEngine).
        cfg: Config with policy and server settings.
    """

    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg

        # Lazy imports at init time to validate availability
        self._policy = None
        self._request_count = 0
        self._total_latency = 0.0

    def _init_policy(self):
        """Initialize the receding-horizon policy."""
        if self._policy is not None:
            return

        from openwam.deploy.optimizations import resolve_async_inference_config
        from openwam.deploy.policy import WAMPolicy

        policy_cfg = getattr(self.cfg, "policy", self.cfg)
        async_config = resolve_async_inference_config(self.cfg, policy_cfg=policy_cfg)
        self._policy = WAMPolicy(
            engine=self.engine,
            cfg=policy_cfg,
            async_config=async_config,
        )

        # Resolve obs preprocessing config from the saved checkpoint cfg so every
        # predict() validates + preprocesses without re-reading it per request.
        self._obs_decoder = ObsDecoder.from_cfg(self.cfg, self.engine)
        d = self._obs_decoder
        logger.info(
            "[obs] View config: multiview=%s, camera_layout=%s, target_camera=%s, canvas=%dx%d",
            d.multiview,
            d.camera_layout if d.multiview else "[unused]",
            d.target_camera if not d.multiview else "[unused]",
            d.img_height,
            d.img_width,
        )

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

        obs = self._obs_decoder.decode(obs)
        action = self._policy.predict_action(obs)

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
        policy_cfg = getattr(self.cfg, "policy", self.cfg)
        if self._policy is not None:
            async_info = self._policy.async_info
        else:
            from openwam.deploy.optimizations import resolve_async_inference_config
            from openwam.deploy.policy import build_async_info

            async_config = resolve_async_inference_config(self.cfg, policy_cfg=policy_cfg)
            async_info = build_async_info(async_config, policy_cfg)
        return {
            "model": "OpenWAM",
            "total_requests": self._request_count,
            "avg_latency_ms": round(avg_latency, 2),
            "policy_config": {
                "execute_horizon": getattr(policy_cfg, "execute_horizon", None),
                "temporal_ensemble": getattr(policy_cfg, "temporal_ensemble", True),
            },
            "async_inference": async_info,
        }

    def run(self, host: str = "0.0.0.0", port: int = 8848):
        """Start the WebSocket policy server.

        One persistent listener accepts obs / reset / ping messages up to
        ``MAX_MESSAGE_BYTES`` so multi-camera payloads above the 1 MB default
        aren't rejected.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("Server dependency required. Install with:\n  pip install websockets")

        self._init_policy()

        async def ws_handler(websocket):
            """Handle WebSocket connections."""
            logger.info("Client connected: %s", websocket.remote_address)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", wsp.OBS)

                        if msg_type == wsp.RESET:
                            self.reset()
                            await websocket.send(json.dumps({"type": wsp.RESET_ACK}))
                        elif msg_type == wsp.OBS:
                            result = self.predict(data)
                            result["type"] = wsp.ACTION
                            await websocket.send(json.dumps(result))
                        elif msg_type == wsp.PING:
                            await websocket.send(json.dumps({"type": wsp.PONG}))
                        else:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": wsp.ERROR,
                                        "code": wsp.ERR_UNKNOWN_TYPE,
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
                                    "type": wsp.ERROR,
                                    "code": wsp.ERR_OBS_VALIDATION,
                                    "message": str(e),
                                }
                            )
                        )
                    except Exception as e:
                        logger.exception("Error processing message")
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": wsp.ERROR,
                                    "code": wsp.ERR_INTERNAL,
                                    "message": str(e),
                                }
                            )
                        )
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected")

        async def serve():
            async with websockets.serve(ws_handler, host, port, max_size=MAX_MESSAGE_BYTES):
                logger.info("WebSocket server started on ws://%s:%d", host, port)
                await asyncio.Future()  # run forever

        logger.info("Starting PolicyServer: ws://%s:%d", host, port)
        asyncio.run(serve())


def merge_deploy_cfg(training_cfg, deploy_cfg):
    """Fill inference frame/resolution fallbacks from the training dataloader,
    then merge deploy overrides on top of the training config (deploy wins).

    ``inference.num_frames`` stays the raw action/state window (actions returned
    = num_frames - 1); ``inference.video_num_frames`` is the Wan video length
    after ``dataloader.video_stride`` sub-sampling.
    """
    from omegaconf import OmegaConf

    deploy_cfg = deploy_cfg if deploy_cfg is not None else OmegaConf.create({})
    dl = OmegaConf.select(training_cfg, "dataloader", default=None)
    if dl is not None:
        inf = OmegaConf.select(deploy_cfg, "inference", default=OmegaConf.create({}))
        if OmegaConf.select(inf, "num_frames", default=None) is None:
            OmegaConf.update(inf, "num_frames", OmegaConf.select(dl, "num_frames", default=33), merge=False)
        if OmegaConf.select(inf, "video_num_frames", default=None) is None:
            OmegaConf.update(inf, "video_num_frames", _infer_video_num_frames(dl), merge=False)
        if OmegaConf.select(inf, "height", default=None) is None:
            OmegaConf.update(inf, "height", OmegaConf.select(dl, "height", default=384), merge=False)
        if OmegaConf.select(inf, "width", default=None) is None:
            OmegaConf.update(inf, "width", OmegaConf.select(dl, "width", default=320), merge=False)
        OmegaConf.update(deploy_cfg, "inference", inf, merge=True)
    return OmegaConf.merge(training_cfg, deploy_cfg)


def build_server_from_config(
    cfg,
    ckpt_dir: str,
    device: str = "cuda",
    ckpt_name: Optional[str] = None,
):
    """Build a PolicyServer from a self-contained checkpoint directory.

    Single construction path shared by both entrypoints (``scripts/deploy.py``
    and ``openwam-serve``): load the checkpoint (``config.yaml`` +
    ``checkpoint_step_*.safetensors``), merge deploy-side overrides on top via
    :func:`merge_deploy_cfg`, build the engine, and wrap it in a PolicyServer.
    """
    from omegaconf import OmegaConf

    from openwam.deploy import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    training_cfg, architecture = load_from_checkpoint_dir(ckpt_dir, device=device, ckpt_name=ckpt_name)
    deploy_cfg = cfg if cfg is not None else OmegaConf.create({})
    _normalize_compile_mode_in_cfg(deploy_cfg)
    merged = merge_deploy_cfg(training_cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=merged, architecture=architecture)
    return PolicyServer(engine=engine, cfg=merged)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the OpenWAM policy server.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config. Defaults to configs/deploy.yaml from the repo root.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory (config.yaml + checkpoint_step_*.safetensors). "
        "Same meaning as scripts/deploy.py --ckpt-dir.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--host", type=str, default=None, help="WebSocket bind host override.")
    parser.add_argument("--port", type=int, default=None, help="WebSocket port override.")
    parser.add_argument(
        "--compile-mode",
        type=_normalize_compile_mode_arg,
        choices=_compile_mode_choices(),
        default=None,
        help="Override compile strategy: auto or none.",
    )
    parser.add_argument(
        "--async-mode",
        choices=("none", "vanilla"),
        default=None,
        help="Override optimization.async_inference.mode.",
    )
    parser.add_argument(
        "--async-execution-horizon",
        type=int,
        default=None,
        dest="async_execution_horizon",
        help="Override optimization.async_inference.vanilla.execution_horizon.",
    )
    parser.add_argument(
        "--async-inference-delay-steps",
        type=int,
        default=None,
        dest="async_inference_delay_steps",
        help="Override optimization.async_inference.vanilla.inference_delay_steps.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional OmegaConf dotlist overrides, e.g. model/backbone=ti2v_5b",
    )
    return parser


def _apply_async_cli_overrides(cfg, args):
    """Apply async inference CLI flags to the nested deploy config."""
    from openwam.deploy.optimizations import apply_async_cli_overrides

    return apply_async_cli_overrides(cfg, args)


def main(argv: Optional[list[str]] = None):
    """CLI entrypoint for running the OpenWAM policy server."""
    from omegaconf import OmegaConf

    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.ckpt_dir is None:
        parser.error("--ckpt-dir is required")

    project_root = Path(__file__).resolve().parent.parent.parent

    config_path = Path(args.config) if args.config else project_root / "configs" / "deploy.yaml"
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        OmegaConf.update(cfg, "defaults", OmegaConf.create([]), merge=False)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    _apply_compile_mode_override(cfg, args.compile_mode)
    try:
        cfg = _apply_async_cli_overrides(cfg, args)
    except ValueError as exc:
        parser.error(str(exc))

    server_cfg = getattr(cfg, "server", None)
    if server_cfg is None:
        deploy_cfg = getattr(cfg, "deploy", None)
        server_cfg = getattr(deploy_cfg, "server", None) if deploy_cfg is not None else None

    host = args.host or getattr(server_cfg, "host", "0.0.0.0")
    port = args.port or getattr(server_cfg, "port", 8848)
    server = build_server_from_config(
        cfg=cfg,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
