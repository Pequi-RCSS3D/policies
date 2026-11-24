"""Logs scalars to Weights & Biases."""

from importlib import import_module
from pathlib import Path
from typing import Any

from policies.core.config import Config
from policies.training.monitoring.events import Event
from policies.training.monitoring.hooks.base import hooks
from policies.training.monitoring.hooks.scalars import scalars


class WandbHookConfig(Config):
    project: str
    entity: str | None = None
    name: str | None = None
    every: int = 1


class WandbHook:
    """Logs scalars to Weights and Biases."""

    def __init__(self, config: WandbHookConfig, run_directory: Path, **_: Any) -> None:
        try:
            wandb = import_module("wandb")
        except ImportError as error:
            raise ImportError("the wandb hook needs the `wandb` package installed") from error
        self.every = config.every
        self._run = wandb.init(
            project=config.project, entity=config.entity, name=config.name, dir=str(run_directory)
        )

    def __call__(self, event: Event) -> None:
        if not event.traits.scalar:
            return
        values = {f"{event.kind}/{key}": value for key, value in scalars(event.data).items()}
        if values:
            self._run.log(values, step=event.step)

    def close(self) -> None:
        self._run.finish()


hooks.register("wandb", WandbHookConfig)(WandbHook)
