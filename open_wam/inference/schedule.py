"""Package-native denoising schedule generators."""

from __future__ import annotations

from typing import List, Tuple

from open_wam.inference.flow_match_scheduler import FlowMatchScheduler
from open_wam.inference.optimizations.decoupled_schedule import (
    schedule_decoupled_asymmetric,
    schedule_decoupled_flash,
)

Schedule = List[Tuple[float, float]]


def _base_timesteps(num_steps: int, shift: float) -> List[float]:
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(num_steps, shift=shift)
    return scheduler.timesteps.tolist()


def schedule_sync(num_steps: int = 50, shift: float = 5.0) -> Schedule:
    ts = _base_timesteps(num_steps, shift)
    return [(t, t) for t in ts] + [(0.0, 0.0)]


def schedule_video_leading(
    num_steps: int = 50,
    shift: float = 5.0,
    lead_steps: int = 10,
) -> Schedule:
    v_ts = _base_timesteps(num_steps + lead_steps, shift)
    a_ts = _base_timesteps(num_steps, shift)
    t_a_max = a_ts[0]

    schedule = []
    for i, v_t in enumerate(v_ts):
        if i < lead_steps:
            schedule.append((v_t, t_a_max))
        else:
            schedule.append((v_t, a_ts[i - lead_steps]))
    schedule.append((0.0, 0.0))
    return schedule


def schedule_cascade(
    video_steps: int = 50,
    action_steps: int = 50,
    shift: float = 5.0,
) -> Schedule:
    v_ts = _base_timesteps(video_steps, shift)
    a_ts = _base_timesteps(action_steps, shift)
    t_a_max = a_ts[0]

    schedule = [(v_t, t_a_max) for v_t in v_ts]
    schedule.extend((0.0, a_t) for a_t in a_ts)
    schedule.append((0.0, 0.0))
    return schedule


def schedule_action_only(num_steps: int = 50, shift: float = 5.0) -> Schedule:
    a_ts = _base_timesteps(num_steps, shift)
    return [(0.0, a_t) for a_t in a_ts] + [(0.0, 0.0)]


_SCHEDULE_REGISTRY = {
    "sync": schedule_sync,
    "video_leading": schedule_video_leading,
    "cascade": schedule_cascade,
    "action_only": schedule_action_only,
    "decoupled_flash": schedule_decoupled_flash,
    "decoupled_asymmetric": schedule_decoupled_asymmetric,
}


def make_schedule(strategy: str, num_steps: int = 50, shift: float = 5.0, **kwargs) -> Schedule:
    if strategy not in _SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule strategy '{strategy}'. Choose from: {list(_SCHEDULE_REGISTRY.keys())}")

    fn = _SCHEDULE_REGISTRY[strategy]
    call_kwargs = {"shift": shift}
    if strategy == "cascade":
        call_kwargs["video_steps"] = kwargs.get("video_steps", num_steps)
        call_kwargs["action_steps"] = kwargs.get("action_steps", num_steps)
    elif strategy == "decoupled_flash":
        call_kwargs["action_steps"] = kwargs.get("action_steps", num_steps)
    elif strategy == "decoupled_asymmetric":
        call_kwargs["video_steps"] = kwargs.get("video_steps", num_steps)
        call_kwargs["action_steps"] = kwargs.get("action_steps", max(1, num_steps // 5))
    else:
        call_kwargs["num_steps"] = num_steps
    if strategy == "video_leading":
        call_kwargs["lead_steps"] = kwargs.get("lead_steps", 10)
    return fn(**call_kwargs)


__all__ = [
    "Schedule",
    "schedule_sync",
    "schedule_video_leading",
    "schedule_cascade",
    "schedule_action_only",
    "schedule_decoupled_flash",
    "schedule_decoupled_asymmetric",
    "make_schedule",
]
