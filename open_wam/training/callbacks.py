"""
Extensible callback system for OpenWAM training (DynamiCrafter-inspired).

Instead of a single monolithic val_callback function, training hooks are
distributed across independent Callback objects.  Each callback implements
only the hooks it cares about; the CallbackRunner dispatches events to all
registered callbacks.

Hook lifecycle inside ``launch_training_task``::

    on_train_start(state)
    for epoch:
        on_epoch_start(state)
        for batch:
            on_step_end(state)       # after optimizer.step()
        on_epoch_end(state)
    on_train_end(state)

Adding a new callback is as simple as subclassing ``TrainingCallback`` and
overriding the relevant hooks — no changes to the training loop required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Training state container — passed to every hook
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Mutable state bag threaded through all callback hooks."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    loss_components: Dict[str, float] = field(default_factory=dict)
    model: Any = None
    accelerator: Any = None
    model_logger: Any = None
    wandb_run: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base callback
# ---------------------------------------------------------------------------

class TrainingCallback:
    """Base class for training callbacks.

    Override any hook method.  All hooks receive a ``TrainingState`` and are
    no-ops by default so subclasses only implement what they need.

    Attributes:
        every_n_steps: If set, ``on_step_end`` is only called every N steps.
            The CallbackRunner handles the modulo check so subclasses don't
            need to.  Set to ``None`` to be called on every step.
    """

    every_n_steps: Optional[int] = None

    def on_train_start(self, state: TrainingState) -> None:
        pass

    def on_epoch_start(self, state: TrainingState) -> None:
        pass

    def on_step_end(self, state: TrainingState) -> None:
        pass

    def on_epoch_end(self, state: TrainingState) -> None:
        pass

    def on_train_end(self, state: TrainingState) -> None:
        pass


# ---------------------------------------------------------------------------
# Callback runner — dispatches events to a list of callbacks
# ---------------------------------------------------------------------------

class CallbackRunner:
    """Manages a list of callbacks and dispatches training events.

    Also provides ``as_legacy_val_callback()`` to bridge with the existing
    ``launch_training_task(val_callback=..., val_steps=...)`` interface.
    """

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks: List[TrainingCallback] = list(callbacks or [])
        self._state = TrainingState()

    def add(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)

    @property
    def state(self) -> TrainingState:
        return self._state

    # -- dispatch helpers --

    def on_train_start(self) -> None:
        for cb in self.callbacks:
            cb.on_train_start(self._state)

    def on_epoch_start(self) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(self._state)

    def on_step_end(self) -> None:
        for cb in self.callbacks:
            if cb.every_n_steps is None or self._state.step % cb.every_n_steps == 0:
                cb.on_step_end(self._state)

    def on_epoch_end(self) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(self._state)

    def on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end(self._state)

    # -- legacy bridge --

    def compute_callback_interval(self) -> Optional[int]:
        """Compute the GCD of all ``every_n_steps`` values.

        Used as ``val_steps`` for the legacy ``launch_training_task`` API.
        Returns ``None`` if no callbacks have ``every_n_steps`` set.
        """
        intervals = [cb.every_n_steps for cb in self.callbacks if cb.every_n_steps is not None]
        if not intervals:
            return None
        result = intervals[0]
        for i in intervals[1:]:
            result = math.gcd(result, i)
        return result

    def as_legacy_val_callback(self):
        """Return a ``(val_callback, val_steps)`` tuple compatible with
        ``launch_training_task``.

        The returned function delegates to ``on_step_end`` for all
        registered callbacks, with each callback's ``every_n_steps``
        filtering handled internally.
        """
        interval = self.compute_callback_interval()
        if interval is None:
            return None, None

        def _dispatch(step: int):
            self._state.step = step
            self.on_step_end()

        return _dispatch, interval


# ---------------------------------------------------------------------------
# Concrete callbacks
# ---------------------------------------------------------------------------

class ValidationLossCallback(TrainingCallback):
    """Compute validation loss on one or more datasets at regular intervals."""

    def __init__(self, model, datasets: Dict[str, Any], wandb_run=None,
                 every_n_steps: int = 500, max_samples: int = 500):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.model = model
        self.datasets = datasets
        self.wandb_run = wandb_run
        self.max_samples = max_samples

    def on_step_end(self, state: TrainingState) -> None:
        wandb_run = self.wandb_run or state.wandb_run
        for prefix, dataset in self.datasets.items():
            self.model.compute_val_losses(
                dataset, state.step, wandb_run,
                prefix=prefix, max_samples=self.max_samples,
            )


class VideoLogCallback(TrainingCallback):
    """Generate and log sample videos/actions to W&B at regular intervals."""

    def __init__(self, model, datasets: Dict[str, Any], wandb_run=None,
                 every_n_steps: int = 1000):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.model = model
        self.datasets = datasets
        self.wandb_run = wandb_run

    def on_step_end(self, state: TrainingState) -> None:
        wandb_run = self.wandb_run or state.wandb_run
        for prefix, dataset in self.datasets.items():
            self.model.validate_during_training(
                dataset[0], state.step, wandb_run, prefix=prefix,
            )


class SetupCallback(TrainingCallback):
    """Save resolved config and create output directories at training start."""

    def __init__(self, output_dir: str, config_dict: Optional[dict] = None):
        super().__init__()
        self.output_dir = output_dir
        self.config_dict = config_dict

    def on_train_start(self, state: TrainingState) -> None:
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        if self.config_dict is not None:
            import json
            config_path = os.path.join(self.output_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config_dict, f, indent=2, default=str)


class LearningRateLogCallback(TrainingCallback):
    """Log learning rate to W&B (useful for LR schedulers)."""

    def __init__(self, every_n_steps: int = 1):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_step_end(self, state: TrainingState) -> None:
        if state.wandb_run is not None and "learning_rate" in state.extra:
            state.wandb_run.log(
                {"train/lr_callback": state.extra["learning_rate"]},
                step=state.step,
            )
