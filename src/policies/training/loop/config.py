"""Trainer and run configs."""

from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from policies.core.config import Config
from policies.learning import LearnerConfig
from policies.spec import PolicySpec
from policies.training.loop.device import AUTO, resolve_device


class TrainerConfig(Config):
    iterations: int
    checkpoint_every: int = 100
    keep_checkpoints: int = 3
    seed: int = 0
    device: str = AUTO
    hooks: tuple[dict[str, Any], ...] = ({"type": "console"}, {"type": "jsonl"})
    wrappers: tuple[dict[str, Any], ...] = ()
    """Environment wrappers as `{type: ...}`, applied in order around the environment."""
    tunable: tuple[str, ...] = ("optimizer.lr",)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        try:
            resolve_device(value)
        except RuntimeError as error:
            raise ValueError(
                f"unknown device {value!r}, expected {AUTO!r} or a torch device"
            ) from error
        return value

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        unknown = sorted(set(self.bounds) - set(self.tunable))
        if unknown:
            raise ValueError(f"bounds for paths that are not tunable: {', '.join(unknown)}")
        return self


class RunConfig(Config):
    """Everything a run needs, as one YAML file."""

    policy: PolicySpec
    learner: LearnerConfig = LearnerConfig()
    trainer: TrainerConfig
