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
            "head_camera":        <base64_jpeg>,       # required; carries the view
                                                       # training's target_camera referred to
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

from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError

logger = logging.getLogger(__name__)

# --- WebSocket message protocol (single source of truth for the server) ---
# Benchmark clients keep their own mirror in benchmarks/utils/transport.py;
# these string values are a frozen wire contract and must never change.
# Client -> server
OBS = "obs"
RESET = "reset"
PING = "ping"
# Server -> client
ACTION = "action"
RESET_ACK = "reset_ack"
PONG = "pong"
ERROR = "error"
# Error codes (the "code" field of an ERROR message)
ERR_UNKNOWN_TYPE = "unknown_message_type"
ERR_OBS_VALIDATION = "obs_validation_error"
ERR_INTERNAL = "internal_error"

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


def _normalize_compile_enabled_in_cfg(cfg) -> None:
    """Keep package and script entrypoints aligned on compile-enabled validation."""
    from omegaconf import OmegaConf

    from openwam.model.compile_options import compile_enabled, normalize_compile_enabled

    enabled = OmegaConf.select(cfg, "optimization.compile.enabled", default=None)
    if enabled is not None:
        OmegaConf.update(cfg, "optimization.compile.enabled", normalize_compile_enabled(enabled), merge=False)
        return
    compile_cfg = OmegaConf.select(cfg, "optimization.compile", default=None)
    if compile_cfg is not None:
        OmegaConf.update(cfg, "optimization.compile.enabled", compile_enabled(compile_cfg, strict=True), merge=False)


def _apply_compile_enabled_override(cfg, compile_enabled: Optional[bool]) -> None:
    """Apply a CLI compile-enabled override, then normalize the config."""
    from omegaconf import OmegaConf

    if compile_enabled is not None:
        OmegaConf.update(cfg, "optimization.compile.enabled", compile_enabled, merge=False)
    _normalize_compile_enabled_in_cfg(cfg)


def _normalize_compile_enabled_arg(value: str) -> bool:
    from openwam.model.compile_options import normalize_compile_enabled

    try:
        return normalize_compile_enabled(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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

        from openwam.deploy.executors import resolve_execution_config
        from openwam.deploy.policy import WAMPolicy

        policy_cfg = getattr(self.cfg, "policy", self.cfg)
        execution_config = resolve_execution_config(self.cfg, policy_cfg=policy_cfg)
        self._policy = WAMPolicy(
            engine=self.engine,
            cfg=policy_cfg,
            execution_config=execution_config,
        )

        # Resolve obs preprocessing config from the saved checkpoint cfg so every
        # predict() validates + preprocesses without re-reading it per request.
        self._obs_preprocessor = ObsPreprocessor.from_cfg(self.cfg, self.engine)
        d = self._obs_preprocessor
        logger.info(
            "[obs] View config: multiview=%s, camera_layout=%s, canvas=%dx%d",
            d.multiview,
            d.camera_layout if d.multiview else "[unused]",
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

        obs = self._obs_preprocessor.preprocess(obs)
        action = self._policy.predict_action(obs)

        latency_ms = (time.monotonic() - t0) * 1000
        self._request_count += 1
        self._total_latency += latency_ms

        return {
            "action": action.tolist(),
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
                        msg_type = data.get("type", OBS)

                        if msg_type == RESET:
                            self.reset()
                            await websocket.send(json.dumps({"type": RESET_ACK}))
                        elif msg_type == OBS:
                            result = self.predict(data)
                            result["type"] = ACTION
                            await websocket.send(json.dumps(result))
                        elif msg_type == PING:
                            await websocket.send(json.dumps({"type": PONG}))
                        else:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": ERROR,
                                        "code": ERR_UNKNOWN_TYPE,
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
                                    "type": ERROR,
                                    "code": ERR_OBS_VALIDATION,
                                    "message": str(e),
                                }
                            )
                        )
                    except Exception as e:
                        logger.exception("Error processing message")
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": ERROR,
                                    "code": ERR_INTERNAL,
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
    _normalize_compile_enabled_in_cfg(deploy_cfg)
    merged = merge_deploy_cfg(training_cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=merged, architecture=architecture)
    return PolicyServer(engine=engine, cfg=merged)


def _log_attention_backends(logger):
    """Log which attention backend is active for each subsystem.

    Not a pure logger call: imports below trigger flash/sage availability
    probes; invoke only after the heavy imports have already been paid for.
    """
    lines = ["Attention backend diagnostics:"]

    # --- ActionDiT backend (components.py, lazy, env: WAM_ATTENTION_IMPL) ---
    try:
        from openwam.model.action_backbone.components import get_attention_fn

        fn = get_attention_fn()
        name = fn.__name__ if hasattr(fn, "__name__") else repr(fn)
        lines.append(f"  ActionDiT          : {name}")
    except Exception as e:
        lines.append(f"  ActionDiT          : ERROR ({e})")

    # --- Video DiT backend (Wan first-class path, checked at import time, no env var) ---
    try:
        import openwam.model.video_backbone.wan.models.dit as _vdit

        if getattr(_vdit, "FLASH_ATTN_3_AVAILABLE", False):
            vdit_backend = "flash_attention_3"
        elif getattr(_vdit, "FLASH_ATTN_2_AVAILABLE", False):
            vdit_backend = "flash_attention_2"
        elif getattr(_vdit, "SAGE_ATTN_AVAILABLE", False):
            vdit_backend = "sage_attention"
        else:
            vdit_backend = "torch_sdpa"
        lines.append(f"  Video DiT          : {vdit_backend}")
    except Exception as e:
        lines.append(f"  Video DiT          : ERROR ({e})")

    # --- Wan shared core backend (attention.py, env: DIFFSYNTH_ATTENTION_IMPLEMENTATION) ---
    try:
        from openwam.model.video_backbone.wan.shared.core.attention.attention import ATTENTION_IMPLEMENTATION

        lines.append(f"  Wan shared core    : {ATTENTION_IMPLEMENTATION}")
    except Exception as e:
        lines.append(f"  Wan shared core    : ERROR ({e})")

    logger.info("\n".join(lines))


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the OpenWAM policy server. Base config from configs/deploy.yaml; CLI overrides it."
    )
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
        "Falls back to checkpoint_path in the deploy yaml.",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default=None,
        help="Specific checkpoint filename (default: latest checkpoint_step_*.safetensors)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (default: device from the deploy yaml, else cuda).",
    )
    parser.add_argument("--host", type=str, default=None, help="WebSocket bind host override.")
    parser.add_argument("--port", type=int, default=None, help="WebSocket port override.")
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=None,
        dest="denoise_steps",
        help="Override inference.denoise_steps",
    )
    parser.add_argument(
        "--schedule-type",
        type=str,
        choices=["sync", "independent"],
        default=None,
        dest="schedule_type",
        help="Override inference.schedule_type: 'sync' (lockstep) or 'independent' (per-stream random timesteps)",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=None,
        dest="schedule_seed",
        help="Override inference.schedule_seed: RNG seed for the 'independent' schedule (reproducible random timesteps)",
    )
    parser.add_argument(
        "--compile-enabled",
        type=_normalize_compile_enabled_arg,
        default=None,
        help="Enable architecture-specific compile fast paths: true or false.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("sync", "async"),
        default=None,
        dest="execution_mode",
        help="Override inference.execution_mode.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        dest="execution_horizon",
        help="Override inference.execution_horizon (async only).",
    )
    parser.add_argument(
        "--inference-delay-steps",
        type=int,
        default=None,
        dest="inference_delay_steps",
        help="Override inference.inference_delay_steps (async only).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional OmegaConf dotlist overrides, e.g. model/video_backbone=cosmos25",
    )
    return parser


def _apply_execution_cli_overrides(cfg, args):
    """Apply execution-mode CLI flags to the deploy config."""
    from openwam.deploy.executors import apply_execution_cli_overrides

    return apply_execution_cli_overrides(cfg, args)


def _load_deploy_yaml(config_path: Optional[str] = None):
    """Load the deploy yaml (Hydra defaults list stripped) as the base config."""
    from omegaconf import OmegaConf

    project_root = Path(__file__).resolve().parent.parent.parent
    path = Path(config_path) if config_path else project_root / "configs" / "deploy.yaml"
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        OmegaConf.update(cfg, "defaults", OmegaConf.create([]), merge=False)
    return cfg


def _apply_inference_overrides(cfg, args):
    """Write the inference CLI flags into cfg.inference (CLI wins over yaml)."""
    from omegaconf import OmegaConf

    if args.denoise_steps is not None:
        OmegaConf.update(cfg, "inference.denoise_steps", args.denoise_steps, merge=False)
    if args.schedule_type is not None:
        OmegaConf.update(cfg, "inference.schedule_type", args.schedule_type, merge=False)
    if args.schedule_seed is not None:
        OmegaConf.update(cfg, "inference.schedule_seed", args.schedule_seed, merge=False)
    return cfg


def main(argv: Optional[list[str]] = None):
    """CLI entrypoint for running the OpenWAM policy server."""
    from omegaconf import OmegaConf

    parser = _build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _log_attention_backends(logging.getLogger("deploy"))

    cfg = _load_deploy_yaml(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    cfg = _apply_inference_overrides(cfg, args)
    _apply_compile_enabled_override(cfg, args.compile_enabled)
    try:
        cfg = _apply_execution_cli_overrides(cfg, args)
    except ValueError as exc:
        parser.error(str(exc))

    # Checkpoint dir: CLI --ckpt-dir > checkpoint_path in the deploy yaml.
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        yaml_ckpt = OmegaConf.select(cfg, "checkpoint_path", default=None)
        if yaml_ckpt:
            ckpt_dir = str(yaml_ckpt)
            logging.getLogger("deploy").info("Using checkpoint from deploy yaml: %s", ckpt_dir)
        else:
            parser.error("--ckpt-dir is required (or set checkpoint_path in the deploy yaml)")

    # Device: CLI --device > yaml device > cuda.
    device = args.device or str(OmegaConf.select(cfg, "device", default="cuda"))

    server_cfg = getattr(cfg, "server", None)
    host = args.host or getattr(server_cfg, "host", "0.0.0.0")
    port = args.port or getattr(server_cfg, "port", 8848)
    server = build_server_from_config(
        cfg=cfg,
        ckpt_dir=ckpt_dir,
        device=device,
        ckpt_name=args.ckpt_name,
    )
    logging.getLogger("deploy").info(
        "Inference engine ready — steps=%d schedule=%s",
        OmegaConf.select(server.cfg, "inference.denoise_steps", default=20),
        OmegaConf.select(server.cfg, "inference.schedule_type", default="sync"),
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
