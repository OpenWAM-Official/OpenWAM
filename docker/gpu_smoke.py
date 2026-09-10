"""Exercise CUDA, torch.compile, and optionally DeepSpeed/NCCL without model weights."""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepspeed", action="store_true", help="compile and exercise the FusedAdam training optimizer"
    )
    parser.add_argument("--all-visible", action="store_true", help="run one worker per visible GPU, including NCCL")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the container")
    if args.all_visible:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            # Static rendezvous avoids advertising an unresolvable container
            # hostname when the smoke service has network_mode: none.
            "--master-addr=127.0.0.1",
            "--master-port=29500",
            "--nnodes=1",
            "--nproc-per-node=" + str(torch.cuda.device_count()),
            __file__,
        ]
        if args.deepspeed:
            command.append("--deepspeed")
        os.execv(sys.executable, command)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    print(f"torch={torch.__version__} CUDA={torch.version.cuda} device={torch.cuda.get_device_name(rank)}", flush=True)
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def operation(value):
        return (value @ value.T).float().square().mean()

    expected = operation(x)
    actual = torch.compile(operation)(x)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    actual.backward()
    assert torch.isfinite(x.grad).all()
    if args.deepspeed:
        from deepspeed.ops.adam import FusedAdam

        parameter = torch.nn.Parameter(torch.ones(32, device="cuda"))
        optimizer = FusedAdam([parameter], lr=0.01)
        parameter.square().mean().backward()
        optimizer.step()
        assert torch.isfinite(parameter).all() and bool((parameter < 1).all())
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl")
        try:
            value = torch.ones(1, device="cuda")
            dist.all_reduce(value)
            assert value.item() == dist.get_world_size()
        finally:
            dist.destroy_process_group()
    torch.cuda.synchronize()
    print("GPU smoke passed.", flush=True)


if __name__ == "__main__":
    main()
