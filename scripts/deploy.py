"""Deploy an OpenWAM policy server from a training checkpoint directory.

Usage:
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir --device cuda:1 --ws-port 9000
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir --denoise-steps 10 --schedule-type sync
    python scripts/deploy.py --mock --mock-action-dim 20   # no checkpoint or GPU needed

Base configuration is read from configs/deploy.yaml.  CLI flags take precedence
over values in the yaml for the fields they cover.

Inference overrides (all optional; yaml values used when absent):
  --denoise-steps N       Denoising step count
  --schedule-type TYPE    Schedule type: sync | video_leading | cascade | decoupled_flash | decoupled_asymmetric
  --shift SHIFT           Flow-matching shift parameter
  --compile-mode MODE     Compile strategy: none | default | mot_loop
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

_DEPLOY_CFG_PATH = PROJECT_ROOT / "configs" / "deploy.yaml"
_COMPILE_MODES = ("none", "default", "mot_loop")


def _normalize_compile_mode(value: str) -> str:
    """Normalize the public compile-mode spelling."""
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in _COMPILE_MODES:
        raise ValueError(f"Unknown compile mode '{value}'. Choose from: {', '.join(_COMPILE_MODES)}")
    return mode


def _infer_video_num_frames(dl) -> int:
    """Return the video frame count seen by Wan after dataloader sub-sampling.

    ``dataloader.num_frames`` is the raw state/action window length. RoboTwin
    keeps actions at that raw rate but sub-samples video by ``video_stride``
    before VAE encoding, so deploy must pass the sampled video length to Wan.
    """
    from omegaconf import OmegaConf

    raw_frames = int(OmegaConf.select(dl, "num_frames", default=33))
    video_stride = int(OmegaConf.select(dl, "video_stride", default=1) or 1)
    if video_stride <= 0:
        video_stride = 1
    return (raw_frames - 1) // video_stride + 1


def _infer_video_num_frames(dl) -> int:
    """Return the video frame count seen by Wan after dataloader sub-sampling.

    ``dataloader.num_frames`` is the raw state/action window length. RoboTwin
    keeps actions at that raw rate but sub-samples video by ``video_stride``
    before VAE encoding, so deploy must pass the sampled video length to Wan.
    """
    from omegaconf import OmegaConf

    raw_frames = int(OmegaConf.select(dl, "num_frames", default=33))
    video_stride = int(OmegaConf.select(dl, "video_stride", default=1) or 1)
    if video_stride <= 0:
        video_stride = 1
    return (raw_frames - 1) // video_stride + 1


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

    # Server / device
    if args.device is not None:
        OmegaConf.update(deploy_cfg, "device", args.device, merge=False)
    if args.host is not None:
        OmegaConf.update(deploy_cfg, "server.host", args.host, merge=False)
    if args.ws_port is not None:
        OmegaConf.update(deploy_cfg, "server.ws_port", args.ws_port, merge=False)
    if args.http_port is not None:
        OmegaConf.update(deploy_cfg, "server.http_port", args.http_port, merge=False)

    # Inference
    if args.denoise_steps is not None:
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", args.denoise_steps, merge=False)
    if args.schedule_type is not None:
        OmegaConf.update(deploy_cfg, "inference.schedule_type", args.schedule_type, merge=False)
    if args.shift is not None:
        OmegaConf.update(deploy_cfg, "inference.shift", args.shift, merge=False)

    compile_mode = getattr(args, "compile_mode", None)
    if compile_mode is not None:
        OmegaConf.update(deploy_cfg, "optimization.compile.mode", _normalize_compile_mode(compile_mode), merge=False)

    return deploy_cfg


def _merge_with_training_cfg(training_cfg, deploy_cfg):
    """Merge deploy config on top of training config (deploy wins on overlap).

    Not pure: ``OmegaConf.select(deploy_cfg, "inference")`` returns a live
    view, so the ``OmegaConf.update`` calls below also mutate the caller's
    ``deploy_cfg``.
    """
    from omegaconf import OmegaConf

    # Let dataloader params provide fallback for inference frame/resolution dims.
    # ``inference.num_frames`` remains the raw action/state horizon (actions
    # returned = num_frames - 1). ``inference.video_num_frames`` is the Wan
    # video length after dataloader.video_stride sub-sampling.
    dl = OmegaConf.select(training_cfg, "dataloader", default=None)
    if dl is not None:
        inf = OmegaConf.select(deploy_cfg, "inference", default=OmegaConf.create({}))
        if OmegaConf.select(inf, "num_frames", default=None) is None:
            OmegaConf.update(inf, "num_frames", OmegaConf.select(dl, "num_frames", default=33), merge=False)
        if OmegaConf.select(inf, "video_num_frames", default=None) is None:
            OmegaConf.update(inf, "video_num_frames", _infer_video_num_frames(dl), merge=False)
        if OmegaConf.select(inf, "height", default=None) is None:
            OmegaConf.update(inf, "height", OmegaConf.select(dl, "height", default=480), merge=False)
        if OmegaConf.select(inf, "width", default=None) is None:
            OmegaConf.update(inf, "width", OmegaConf.select(dl, "width", default=832), merge=False)

    return OmegaConf.merge(training_cfg, deploy_cfg)


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
    parser.add_argument("--ws-port", type=int, default=None, dest="ws_port", help="WebSocket port")
    parser.add_argument("--http-port", type=int, default=None, dest="http_port", help="HTTP port")
    # Inference overrides
    parser.add_argument(
        "--denoise-steps", type=int, default=None, dest="denoise_steps", help="Override denoising steps"
    )
    parser.add_argument("--schedule-type", type=str, default=None, dest="schedule_type", help="Override schedule type")
    parser.add_argument("--shift", type=float, default=None, help="Override flow-matching shift")
    parser.add_argument(
        "--compile-mode",
        choices=_COMPILE_MODES,
        default=None,
        help="Override compile strategy: none, default, or mot_loop",
    )
    # Mock mode: no weights or GPU needed
    parser.add_argument(
        "--mock", action="store_true", help="Run in mock mode (random actions, no model weights required)"
    )
    parser.add_argument(
        "--mock-action-dim", type=int, default=20, help="Action dimension for mock engine (default: 20)"
    )
    parser.add_argument(
        "--mock-latency-ms",
        type=float,
        default=2000.0,
        help="Simulated inference latency in ms for mock engine (default: 2000)",
    )
    # Debug mode
    parser.add_argument("--debug", action="store_true", help="Save received images + actions + metadata per step")
    parser.add_argument(
        "--debug-dir", type=str, default="./server_debug", help="Directory for debug output (default: ./server_debug)"
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
    deploy_cfg = _apply_cli_overrides(deploy_cfg, args)

    # Resolve checkpoint dir: CLI --ckpt-dir > checkpoint_path in yaml.
    if not args.mock and args.ckpt_dir is None:
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
    ws_port = int(OmegaConf.select(server_cfg, "ws_port", default=8850))
    http_port = int(OmegaConf.select(server_cfg, "http_port", default=8848))

    if args.mock:
        # Mock mode: skip model loading entirely
        from openwam.deploy.mock_engine import MockInferenceEngine

        cfg = OmegaConf.merge(OmegaConf.create({}), deploy_cfg)
        engine = MockInferenceEngine(
            cfg=cfg,
            action_dim=args.mock_action_dim,
            latency_ms=args.mock_latency_ms,
        )
        logger.info(
            "Mock engine ready (action_dim=%d, latency=%.0fms)",
            args.mock_action_dim,
            args.mock_latency_ms,
        )
    else:
        # Real mode: load weights from checkpoint directory
        from openwam.deploy.model_loader import load_from_checkpoint_dir

        training_cfg, architecture = load_from_checkpoint_dir(
            ckpt_dir=args.ckpt_dir,
            device=device,
            ckpt_name=args.ckpt_name,
        )

        # Merge: training config + deploy config (deploy wins on overlap)
        cfg = _merge_with_training_cfg(training_cfg, deploy_cfg)

        from openwam.deploy.joint_engine import JointInferenceEngine

        engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
        logger.info(
            "Inference engine ready — steps=%d schedule=%s",
            OmegaConf.select(cfg, "inference.denoise_steps", default=20),
            OmegaConf.select(cfg, "inference.schedule_type", default="sync"),
        )

    # Start server
    from openwam.deploy.policy_server import PolicyServer

    server = PolicyServer(engine=engine, cfg=cfg, debug=args.debug, debug_dir=args.debug_dir)
    logger.info("Starting server: ws://%s:%d  http://%s:%d", host, ws_port, host, http_port)
    server.run(host=host, port=ws_port, http_port=http_port)


if __name__ == "__main__":
    main()
