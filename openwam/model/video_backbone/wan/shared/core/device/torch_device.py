from typing import Any

import torch

IS_CUDA_AVAILABLE = torch.cuda.is_available()


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA."""
    return "cuda" if IS_CUDA_AVAILABLE else "cpu"


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        print(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def get_nccl_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if IS_CUDA_AVAILABLE:
        return "nccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def enable_high_precision_for_bf16():
    """
    Set high accumulation dtype for matmul and reduction.
    """
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def parse_device_type(device):
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        else:
            return "cpu"
    elif isinstance(device, torch.device):
        return device.type


def parse_nccl_backend(device_type):
    if device_type == "cuda":
        return "nccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {device_type}.")


def get_available_device_type():
    return get_device_type()
