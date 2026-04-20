"""Deploy an OpenWAM policy server from a training checkpoint directory.

Usage:
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir
    python scripts/deploy.py --ckpt-dir /path/to/checkpoint_dir --device cuda:1 --ws-port 9000

The checkpoint directory must contain config.yaml and checkpoint_step_*.safetensors
files produced by training.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


def _add_inference_defaults(cfg):
    """Inject sensible inference defaults when config lacks them."""
    from omegaconf import OmegaConf

    defaults = OmegaConf.create(
        {
            "inference": {
                "num_steps": 50,
                "schedule_type": "sync",
                "cfg_scale": 1.0,
                "shift": 5.0,
                "num_frames": cfg.get("dataloader", {}).get("num_frames", 33),
                "height": cfg.get("dataloader", {}).get("height", 480),
                "width": cfg.get("dataloader", {}).get("width", 640),
            },
        }
    )
    return OmegaConf.merge(defaults, cfg)


def main():
    parser = argparse.ArgumentParser(description="Deploy OpenWAM policy server from a checkpoint directory.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint directory (config.yaml + .safetensors)")
    parser.add_argument("--ckpt-name", type=str, default=None, help="Specific checkpoint filename (default: latest)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--ws-port", type=int, default=8850, help="WebSocket port (default: 8850)")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP port (default: 8848)")
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

    if not args.mock and args.ckpt_dir is None:
        parser.error("--ckpt-dir is required unless --mock is set")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("deploy")

    from omegaconf import OmegaConf

    if args.mock:
        # Mock mode: skip model loading entirely
        from openwam.deployment.mock_engine import MockInferenceEngine

        cfg = OmegaConf.create({})
        cfg = _add_inference_defaults(cfg)
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
        from openwam.deployment.model_loader import load_from_checkpoint_dir

        cfg, pipe, architecture = load_from_checkpoint_dir(
            ckpt_dir=args.ckpt_dir,
            device=args.device,
            ckpt_name=args.ckpt_name,
        )

        # Ensure inference config exists
        cfg = _add_inference_defaults(cfg)

        from openwam.deployment.joint_engine import JointInferenceEngine

        engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, architecture=architecture)
        logger.info("Inference engine ready")

    # Start server
    from openwam.deployment.policy_server import PolicyServer

    server = PolicyServer(engine=engine, cfg=cfg, debug=args.debug, debug_dir=args.debug_dir)
    http_port = args.http_port if args.http_port is not None else 8848
    logger.info("Starting server: ws://%s:%d  http://%s:%d", args.host, args.ws_port, args.host, http_port)
    server.run(host=args.host, port=args.ws_port, http_port=http_port)


if __name__ == "__main__":
    main()
