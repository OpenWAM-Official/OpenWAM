"""Opt-in runtime settings that change scheduling only, never what is computed.

- ``OPENWAM_DS_TIMER_NOSYNC=1``: DeepSpeed's throughput timer (on by default,
  ``synchronized: true``) calls ``torch.cuda.synchronize()`` at the start of every
  engine forward and at the end of every step, only to time itself; the host then
  waits for the whole VAE encode before it may queue the first MoT layer.
  ``synchronized: false`` keeps the timer (host clock) without the device syncs.
- ``OPENWAM_DS_OVERLAP_COMM=1``: ZeRO-2 ``overlap_comm`` (off by default in
  DeepSpeed): the gradient bucket all-reduces run on DeepSpeed's reduction stream
  instead of the compute stream; same buckets, same collectives, same sums.
- ``OPENWAM_GC_FREEZE=1``: after steps 1, 2 and 4, collect once and move the
  surviving objects into the cyclic GC's permanent generation, on every rank at the
  same step. Otherwise the long-lived objects built in the first steps
  (torch.compile, DeepSpeed, the data pipeline) trigger full collections of about
  0.5 s a few steps later, one rank at a time, which stall every rank at the next
  collective. The GC stays enabled.
- ``OPENWAM_NUMA_BIND=1``: run each rank, and the DataLoader workers it forks, on
  the CPU cores of its GPU's NUMA node, so its host work and pinned buffers stay
  local to the GPU's socket. Uses ``nvidia-smi`` and the kernel's sysfs topology;
  does nothing if either is unavailable.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

_GC_FREEZE_STEPS = (1, 2, 4)


def _on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def apply_deepspeed_runtime_settings(ds_config: dict) -> None:
    """Add the enabled DeepSpeed runtime settings to ``ds_config`` (keys set explicitly are kept)."""
    timer_nosync, overlap_comm = _on("OPENWAM_DS_TIMER_NOSYNC"), _on("OPENWAM_DS_OVERLAP_COMM")
    if not (timer_nosync or overlap_comm):
        return
    if timer_nosync:
        ds_config.setdefault("timers", {}).setdefault("throughput", {}).setdefault("synchronized", False)
    if overlap_comm:
        zero = ds_config.setdefault("zero_optimization", {})
        if int(zero.get("stage", 0)) == 2:
            zero.setdefault("overlap_comm", True)
    logger.info(
        "DeepSpeed runtime settings: timers=%s overlap_comm=%s",
        ds_config.get("timers"),
        ds_config.get("zero_optimization", {}).get("overlap_comm"),
    )


def maybe_freeze_gc(global_step: int) -> None:
    if global_step in _GC_FREEZE_STEPS and _on("OPENWAM_GC_FREEZE"):
        gc.collect()
        gc.freeze()


def _parse_cpulist(spec: str) -> set[int]:
    cpus = set()
    for part in spec.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def _gpu_numa_cpus():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = visible.split(",")[local].strip() if visible else str(local)
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    bus = None
    for line in out.splitlines():
        idx, uuid, pci = [x.strip() for x in line.split(",")]
        if phys in (idx, uuid):
            bus = pci
    if bus is None:
        return None
    dom, rest = bus.split(":", 1)
    sysname = f"{dom[-4:]}:{rest}".lower()  # nvidia-smi: 00000000:1B:00.0, sysfs: 0000:1b:00.0
    with open(f"/sys/bus/pci/devices/{sysname}/numa_node") as f:
        node = int(f.read().strip())
    if node < 0:
        return None
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        return node, _parse_cpulist(f.read())


def bind_to_gpu_numa_node() -> None:
    """Restrict this process (and the processes it forks later) to its GPU's NUMA node."""
    if not _on("OPENWAM_NUMA_BIND") or not hasattr(os, "sched_setaffinity"):
        return
    try:
        res = _gpu_numa_cpus()
        if not res:
            return
        node, cpus = res
        allowed = os.sched_getaffinity(0) & cpus
        if allowed:
            os.sched_setaffinity(0, allowed)
            logger.info("rank %s bound to NUMA node %d (%d CPUs)", os.environ.get("LOCAL_RANK"), node, len(allowed))
    except Exception as exc:  # best effort
        logger.warning("NUMA binding skipped: %s", exc)
