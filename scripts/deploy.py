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
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory (config.yaml + .safetensors)")
    parser.add_argument("--ckpt-name", type=str, default=None, help="Specific checkpoint filename (default: latest)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP port (default: ws-port + 1)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("deploy")

    # Load model
    from openwam.deployment.model_loader import load_from_checkpoint_dir

    cfg, pipe, architecture = load_from_checkpoint_dir(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        ckpt_name=args.ckpt_name,
    )

    # Ensure inference config exists
    cfg = _add_inference_defaults(cfg)

    # Build engine
    from openwam.deployment.joint_engine import JointInferenceEngine

    engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, architecture=architecture)
    logger.info("Inference engine ready")

    # Start server
    from openwam.deployment.policy_server import PolicyServer

    server = PolicyServer(engine=engine, cfg=cfg)
    http_port = args.http_port if args.http_port is not None else args.ws_port + 1
    logger.info("Starting server: ws://%s:%d  http://%s:%d", args.host, args.ws_port, args.host, http_port)
    server.run(host=args.host, port=args.ws_port, http_port=http_port)


if __name__ == "__main__":
    main()
