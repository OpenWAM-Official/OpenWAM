"""WebSocket policy server for real-time robot deployment.

Provides a network-accessible policy server that wraps WAMPolicy with
receding-horizon execution. Robot controllers connect over a single
persistent WebSocket.

The client is thin on purpose: it sends per-camera lossless PNGs plus the
task prompt. Image composition and resize happen server-side, driven by the
saved training config (``cfg.dataloader.multiview`` / ``camera_layout`` /
``height`` / ``width``). The server is prompt-agnostic — it forwards the prompt
to the model verbatim; each benchmark client owns whatever prompt template its
checkpoints were trained with.

Protocol (unified — same shape for single-view and multi-view checkpoints):
    Client → {
        "type": "obs",
        "images": {
            "head_camera":        <base64_png>,       # required; carries the view
                                                      # training's target_camera referred to
            "left_wrist_camera":  <base64_png>|null,  # optional
            "right_wrist_camera": <base64_png>|null   # optional
        },
        "prompt": "<prompt fed to the model verbatim>",
        "state":  [floats]                             # optional proprio
    }

Server-side behavior:
- ``multiview=False``: ignores wrist fields, crop+resize ``head_camera``.
- ``multiview=True``:  black-fills missing/None wrists, then composes the
  L-shape layout defined by ``camera_layout``.
- ``prompt`` is forwarded to the model verbatim (no server-side wrapping).

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

        execution_config = resolve_execution_config(self.cfg)
        self._policy = WAMPolicy(
            engine=self.engine,
            cfg=self.cfg,
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

    def _ckpt_contract(self) -> dict:
        """Representation-contract fields advertised in the PONG for client-side validation.

        Source is ``architecture.repr_contract`` — bound by ``load_model`` to the TRAINING cfg
        BEFORE ``merge_deploy_cfg`` (where deploy overrides win). Reading the merged ``self.cfg``
        here would let a deploy yaml carrying stray ``dataloader.*`` keys make the PONG advertise
        values the architecture isn't using (its normalizer/binary dims were built from the
        training cfg) — re-opening exactly the silent mismatch the handshake exists to kill.
        Engines without an architecture (or pre-contract load paths) advertise nothing — clients
        then apply their old-server compatibility rules.
        """
        contract = getattr(getattr(self.engine, "architecture", None), "repr_contract", None)
        return dict(contract) if contract else {}

    def predict(self, obs: dict) -> dict:
        """Synchronous prediction for a single observation.

        Args:
            obs: Observation dict with ``images`` (dict of camera name →
                base64 image / bytes / PIL.Image, with ``head_camera`` required
                and ``left_wrist_camera`` / ``right_wrist_camera`` optional),
                the ``prompt`` (str, forwarded to the model verbatim), and
                optional ``state`` (list of floats). Server does all image
                preprocessing internally; prompt wrapping is the client's job.

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
                            await websocket.send(json.dumps({"type": PONG, **self._ckpt_contract()}))
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
            # ping_interval=None: slow inference (notably torch.compile warmup on
            # the first request) blocks this event loop past the 20s default ping
            # deadline; keepalive pings would drop the connection mid-inference.
            async with websockets.serve(ws_handler, host, port, max_size=MAX_MESSAGE_BYTES, ping_interval=None):
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

    deploy_cfg = cfg if cfg is not None else OmegaConf.create({})
    _validate_inference_config(deploy_cfg)
    _normalize_compile_enabled_in_cfg(deploy_cfg)
    training_cfg, architecture = load_from_checkpoint_dir(ckpt_dir, device=device, ckpt_name=ckpt_name)
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
        "--denoise-mode",
        type=str,
        choices=["sync", "async"],
        default=None,
        dest="denoise_mode",
        help="Override inference.denoise_mode trajectory (separate from inference_mode).",
    )
    parser.add_argument(
        "--lead-modality",
        type=str,
        choices=["action", "video"],
        default=None,
        dest="lead_modality",
        help="Override inference.lead_modality (async denoising only).",
    )
    parser.add_argument(
        "--variance-shift-alpha",
        type=float,
        default=None,
        dest="variance_shift_alpha",
        help="Override inference.variance_shift_alpha (async denoising only, >= 1).",
    )
    parser.add_argument(
        "--linear-offset",
        type=float,
        default=None,
        dest="linear_offset",
        help="Override inference.linear_offset (async denoising only, 0 <= value < 1).",
    )
    parser.add_argument(
        "--compile-enabled",
        type=_normalize_compile_enabled_arg,
        default=None,
        help="Enable architecture-specific compile fast paths: true or false.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("sync", "async"),
        default=None,
        dest="inference_mode",
        help="Override inference.inference_mode executor (separate from denoise_mode).",
    )
    parser.add_argument(
        "--inference-horizon",
        type=int,
        default=None,
        dest="inference_horizon",
        help="Override inference.inference_horizon (actions executed per generated chunk in both modes).",
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
        help="Additional OmegaConf dotlist overrides, e.g. model/video_backbone=cosmos_predict25_2b",
    )
    return parser


def _apply_execution_cli_overrides(cfg, args):
    """Apply inference executor CLI flags to the deploy config."""
    from openwam.deploy.executors import apply_execution_cli_overrides

    return apply_execution_cli_overrides(cfg, args)


def _validate_denoise_config(cfg):
    """Validate denoising settings and warn when ``async`` is a no-op."""
    from omegaconf import OmegaConf

    from openwam.deploy.denoise_schedule import denoise_async_is_noop, normalize_denoise_config

    resolved = normalize_denoise_config(OmegaConf.select(cfg, "inference", default=None))
    if denoise_async_is_noop(resolved):
        logger.warning(
            "inference.denoise_mode='async' with variance_shift_alpha=%s and linear_offset=%s "
            "reproduces the sync trajectory bit-for-bit; set variance_shift_alpha > 1 and/or "
            "linear_offset > 0 for the schedule to actually shift.",
            resolved.variance_shift_alpha,
            resolved.linear_offset,
        )
    return resolved


def _validate_inference_config(cfg):
    """Validate denoising and executor settings before model loading."""
    from openwam.deploy.executors import resolve_execution_config

    _validate_denoise_config(cfg)
    resolve_execution_config(cfg)


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
    """Apply denoising CLI flags and validate the resulting config."""
    from omegaconf import OmegaConf

    from openwam.deploy.denoise_schedule import DenoiseConfig

    if args.denoise_steps is not None:
        OmegaConf.update(cfg, "inference.denoise_steps", args.denoise_steps, merge=False)
    if args.denoise_mode is not None:
        OmegaConf.update(cfg, "inference.denoise_mode", args.denoise_mode, merge=False)

    defaults = DenoiseConfig()
    async_controls = (
        ("lead_modality", "--lead-modality", defaults.lead_modality),
        ("variance_shift_alpha", "--variance-shift-alpha", defaults.variance_shift_alpha),
        ("linear_offset", "--linear-offset", defaults.linear_offset),
    )
    mode = str(OmegaConf.select(cfg, "inference.denoise_mode", default=defaults.denoise_mode)).strip().lower()
    # Compare against the default instead of testing flag presence, so the CLI
    # is no stricter than the identical value written in the yaml — the flags
    # stay safe to emit unconditionally from a wrapper script.
    conflicting = [flag for name, flag, default in async_controls if getattr(args, name, None) not in (None, default)]
    if conflicting and mode != "async":
        verb = "requires" if len(conflicting) == 1 else "require"
        raise ValueError(f"{', '.join(conflicting)} {verb} --denoise-mode async or inference.denoise_mode=async")

    for name, _flag, _default in async_controls:
        value = getattr(args, name, None)
        if value is not None:
            OmegaConf.update(cfg, f"inference.{name}", value, merge=False)

    if args.denoise_mode == "sync":
        # An explicit switch down to sync resets its dependents rather than
        # erroring on values the yaml already holds: running the sync baseline
        # against an async-tuned deploy is the A/B people ask for most, and
        # `--denoise-mode` is documented as a same-name override of the yaml.
        for name, _flag, default in async_controls:
            OmegaConf.update(cfg, f"inference.{name}", default, merge=False)

    _validate_denoise_config(cfg)
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

    try:
        cfg = _apply_inference_overrides(cfg, args)
        cfg = _apply_execution_cli_overrides(cfg, args)
        _validate_inference_config(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    _apply_compile_enabled_override(cfg, args.compile_enabled)

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
        "Inference engine ready — steps=%d denoise_mode=%s",
        OmegaConf.select(server.cfg, "inference.denoise_steps", default=20),
        OmegaConf.select(server.cfg, "inference.denoise_mode", default="sync"),
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
