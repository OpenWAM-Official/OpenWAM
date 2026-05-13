"""Configuration helpers for deploy-time async inference."""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

VALID_ASYNC_MODES = ("none", "vanilla")
ASYNC_CLI_NUMERIC_OVERRIDES = ("async_execution_horizon", "async_inference_delay_steps")


@dataclass(frozen=True)
class AsyncInferenceConfig:
    """Normalized async inference config used by deployment policy code."""

    mode: str = "none"
    execution_horizon: Optional[int] = None
    inference_delay_steps: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "execution_horizon": self.execution_horizon,
            "inference_delay_steps": self.inference_delay_steps,
        }


def _select(cfg, path: str, default=None):
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.select(cfg, path, default=default)
    except ImportError:
        pass

    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _coerce_optional_int(value, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def normalize_async_inference_config(async_cfg=None, policy_cfg=None) -> AsyncInferenceConfig:
    """Normalize async inference config into a stable dataclass."""
    if async_cfg is None:
        return AsyncInferenceConfig()

    mode = _select(async_cfg, "mode", default=None)
    if mode is None:
        mode = "vanilla" if bool(_select(async_cfg, "enabled", default=False)) else "none"
    mode = str(mode).strip().lower()
    if mode not in VALID_ASYNC_MODES:
        raise ValueError(f"Unsupported async inference mode {mode!r}; expected one of {VALID_ASYNC_MODES}")

    vanilla_cfg = _select(async_cfg, "vanilla", default=None)
    execution_horizon = _select(vanilla_cfg, "execution_horizon", default=None)
    if execution_horizon is None:
        execution_horizon = _select(async_cfg, "execution_horizon", default=None)
    if execution_horizon is None:
        execution_horizon = _select(policy_cfg, "execute_horizon", default=None)

    inference_delay_steps = _select(vanilla_cfg, "inference_delay_steps", default=None)
    if inference_delay_steps is None:
        inference_delay_steps = _select(async_cfg, "inference_delay_steps", default=None)

    execution_horizon = _coerce_optional_int(execution_horizon, "execution_horizon")
    inference_delay_steps = _coerce_optional_int(inference_delay_steps, "inference_delay_steps")

    if mode == "vanilla":
        if execution_horizon is not None and execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if inference_delay_steps is not None and inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be non-negative")
        if (
            execution_horizon is not None
            and inference_delay_steps is not None
            and inference_delay_steps >= execution_horizon
        ):
            raise ValueError("inference_delay_steps must be < execution_horizon")

    return AsyncInferenceConfig(
        mode=mode,
        execution_horizon=execution_horizon,
        inference_delay_steps=inference_delay_steps,
    )


def _arg_value(args, name: str, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def apply_async_cli_overrides(root_cfg, args):
    """Apply async CLI overrides and validate the resulting nested config."""
    from omegaconf import OmegaConf

    async_mode = _arg_value(args, "async_mode")
    if async_mode is not None:
        mode = str(async_mode).strip().lower()
        if mode not in VALID_ASYNC_MODES:
            raise ValueError(f"Unsupported async inference mode {mode!r}; expected one of {VALID_ASYNC_MODES}")
        OmegaConf.update(root_cfg, "optimization.async_inference.mode", mode, merge=False)

    has_timing_override = any(_arg_value(args, name) is not None for name in ASYNC_CLI_NUMERIC_OVERRIDES)
    if has_timing_override:
        resolved = resolve_async_inference_config(root_cfg)
        if resolved.mode != "vanilla":
            raise ValueError(
                "--async-execution-horizon and --async-inference-delay-steps require "
                "--async-mode vanilla or optimization.async_inference.mode=vanilla"
            )

    execution_horizon = _arg_value(args, "async_execution_horizon")
    if execution_horizon is not None:
        OmegaConf.update(
            root_cfg,
            "optimization.async_inference.vanilla.execution_horizon",
            execution_horizon,
            merge=False,
        )

    inference_delay_steps = _arg_value(args, "async_inference_delay_steps")
    if inference_delay_steps is not None:
        OmegaConf.update(
            root_cfg,
            "optimization.async_inference.vanilla.inference_delay_steps",
            inference_delay_steps,
            merge=False,
        )

    if async_mode is not None or has_timing_override:
        resolve_async_inference_config(root_cfg)

    return root_cfg


def resolve_async_inference_config(root_cfg, policy_cfg=None) -> AsyncInferenceConfig:
    """Resolve async inference config from the deploy config tree.

    Async inference is configured through ``optimization.async_inference``.
    """
    async_cfg = _select(root_cfg, "optimization.async_inference", default=None)
    return normalize_async_inference_config(async_cfg, policy_cfg=policy_cfg)
