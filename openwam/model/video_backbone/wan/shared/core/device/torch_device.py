from typing import Any

import torch

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()


def get_device_type() -> str:
    """Get device type based on current machine: CUDA, then Intel XPU, then CPU."""
    if IS_CUDA_AVAILABLE:
        return "cuda"
    if IS_XPU_AVAILABLE:
        return "xpu"
    return "cpu"


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        print(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def parse_device_type(device):
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        elif device.startswith("xpu"):
            return "xpu"
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
