"""Deploy an OpenWAM policy server from a training checkpoint directory.

Usage:
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir --device cuda:1 --port 9000
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir --denoise-steps 10 --schedule-type sync

Base configuration is read from configs/deploy.yaml.  CLI flags take precedence
over values in the yaml for the fields they cover.

Inference overrides (all optional; yaml values used when absent):
  --denoise-steps N       Denoising step count
  --schedule-type TYPE    Schedule type (only "sync" is supported)
  --shift SHIFT           Flow-matching shift parameter
  --compile-mode MODE     Compile strategy: auto | none
  --async-mode MODE       Async inference mode: none | vanilla
  --async-execution-horizon N
                          Number of actions executed before switching chunks
  --async-inference-delay-steps N
                          Expected inference latency in controller steps
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

_DEPLOY_CFG_PATH = PROJECT_ROOT / "configs" / "deploy.yaml"
_COMPILE_MODES = ("auto", "none")


def _normalize_compile_mode(value: str) -> str:
    """Normalize the public compile-mode spelling."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _COMPILE_MODES:
        raise ValueError(f"Unknown compile mode '{value}'. Choose from: {', '.join(_COMPILE_MODES)}")
    return normalized


def _normalize_compile_mode_arg(value: str) -> str:
    try:
        return _normalize_compile_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _load_deploy_config():
    """Load configs/deploy.yaml as a base deploy config."""
    from omegaconf import OmegaConf

    if _DEPLOY_CFG_PATH.exists():
        cfg = OmegaConf.load(str(_DEPLOY_CFG_PATH))
        # Strip the Hydra defaults list — it is not resolved here.
        if "defaults" in cfg:
            OmegaConf.update(cfg, "defaults", OmegaConf.create([]), merge=False)
        compile_mode = OmegaConf.select(cfg, "optimization.compile.mode", default=None)
        if compile_mode is not None:
            OmegaConf.update(cfg, "optimization.compile.mode", _normalize_compile_mode(compile_mode), merge=False)
        return cfg
    return OmegaConf.create({})


def _apply_cli_overrides(deploy_cfg, args):
    """Propagate argparse values into the deploy config (CLI wins over yaml)."""
    from omegaconf import OmegaConf

    from openwam.deploy.optimizations import apply_async_cli_overrides

    # Server / device
    if args.device is not None:
        OmegaConf.update(deploy_cfg, "device", args.device, merge=False)
    if args.host is not None:
        OmegaConf.update(deploy_cfg, "server.host", args.host, merge=False)
    if args.port is not None:
        OmegaConf.update(deploy_cfg, "server.port", args.port, merge=False)

    # Inference
    if args.denoise_steps is not None:
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", args.denoise_steps, merge=False)
    if args.schedule_type is not None:
        OmegaConf.update(deploy_cfg, "inference.schedule_type", args.schedule_type, merge=False)
    if args.shift is not None:
        OmegaConf.update(deploy_cfg, "inference.shift", args.shift, merge=False)
    apply_async_cli_overrides(deploy_cfg, args)

    compile_mode = getattr(args, "compile_mode", None)
    if compile_mode is not None:
        OmegaConf.update(deploy_cfg, "optimization.compile.mode", _normalize_compile_mode(compile_mode), merge=False)

    return deploy_cfg


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
        import openwam.model.video_backbone.wan.dit as _vdit

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


def main():
    parser = argparse.ArgumentParser(
        description="Deploy OpenWAM policy server. Base config from configs/deploy.yaml; CLI overrides it."
    )
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint directory (config.yaml + .safetensors)")
    parser.add_argument("--ckpt-name", type=str, default=None, help="Specific checkpoint filename (default: latest)")
    parser.add_argument("--device", type=str, default=None, help="Inference device (default: from deployment.yaml)")
    parser.add_argument("--host", type=str, default=None, help="Bind host (default: from deployment.yaml)")
    parser.add_argument("--port", type=int, default=None, dest="port", help="WebSocket port")
    # Inference overrides
    parser.add_argument(
        "--denoise-steps", type=int, default=None, dest="denoise_steps", help="Override denoising steps"
    )
    parser.add_argument(
        "--schedule-type",
        type=str,
        choices=["sync"],
        default=None,
        dest="schedule_type",
        help="Override schedule type (only 'sync' is supported)",
    )
    parser.add_argument("--shift", type=float, default=None, help="Override flow-matching shift")
    parser.add_argument(
        "--compile-mode",
        type=_normalize_compile_mode_arg,
        choices=_COMPILE_MODES,
        default=None,
        help="Override compile strategy: auto or none",
    )
    parser.add_argument(
        "--async-mode",
        choices=("none", "vanilla"),
        default=None,
        help="Override optimization.async_inference.mode",
    )
    parser.add_argument(
        "--async-execution-horizon",
        type=int,
        default=None,
        dest="async_execution_horizon",
        help="Override optimization.async_inference.vanilla.execution_horizon",
    )
    parser.add_argument(
        "--async-inference-delay-steps",
        type=int,
        default=None,
        dest="async_inference_delay_steps",
        help="Override optimization.async_inference.vanilla.inference_delay_steps",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("deploy")
    _log_attention_backends(logger)

    from omegaconf import OmegaConf

    # Load base deploy config from yaml, then apply CLI overrides.
    deploy_cfg = _load_deploy_config()
    try:
        deploy_cfg = _apply_cli_overrides(deploy_cfg, args)
    except ValueError as exc:
        parser.error(str(exc))

    # Resolve checkpoint dir: CLI --ckpt-dir > checkpoint_path in yaml.
    if args.ckpt_dir is None:
        yaml_ckpt = OmegaConf.select(deploy_cfg, "checkpoint_path", default=None)
        if yaml_ckpt:
            args.ckpt_dir = str(yaml_ckpt)
            logger.info("Using checkpoint from deploy.yaml: %s", args.ckpt_dir)
        else:
            parser.error("--ckpt-dir is required (or set checkpoint_path in configs/deploy.yaml)")

    # Resolve server params (with yaml fallbacks).
    server_cfg = OmegaConf.select(deploy_cfg, "server", default=OmegaConf.create({}))
    device = str(OmegaConf.select(deploy_cfg, "device", default="cuda"))
    host = str(OmegaConf.select(server_cfg, "host", default="0.0.0.0"))
    port = int(OmegaConf.select(server_cfg, "port", default=8848))

    # Build the server: load checkpoint → merge deploy cfg → engine → PolicyServer.
    from openwam.deploy.server import build_server_from_config

    server = build_server_from_config(
        cfg=deploy_cfg,
        ckpt_dir=args.ckpt_dir,
        device=device,
        ckpt_name=args.ckpt_name,
    )
    logger.info(
        "Inference engine ready — steps=%d schedule=%s",
        OmegaConf.select(server.cfg, "inference.denoise_steps", default=20),
        OmegaConf.select(server.cfg, "inference.schedule_type", default="sync"),
    )
    logger.info("Starting server: ws://%s:%d", host, port)
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
