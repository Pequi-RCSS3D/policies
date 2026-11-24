"""Logs scalars to TensorBoard."""

import math
from pathlib import Path
from typing import Any

from policies.core.config import Config
from policies.training.monitoring.events import Event
from policies.training.monitoring.hooks.base import hooks
from policies.training.monitoring.hooks.scalars import scalars


class TensorboardHookConfig(Config):
    every: int = 1


class TensorboardHook:
    """Logs scalars under `run_directory/tensorboard`."""

    def __init__(self, config: TensorboardHookConfig, run_directory: Path, **_: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "the tensorboard hook needs the `tensorboard` extra plus `cpu` or `cu126`"
            ) from error
        self.every = config.every
        self._writer = SummaryWriter(log_dir=str(Path(run_directory) / "tensorboard"))

    def __call__(self, event: Event) -> None:
        if not event.traits.scalar:
            return
        for key, value in scalars(event.data).items():
            if math.isfinite(value):
                self._writer.add_scalar(f"{event.kind}/{key}", value, event.step)

    def close(self) -> None:
        self._writer.close()


hooks.register("tensorboard", TensorboardHookConfig)(TensorboardHook)
