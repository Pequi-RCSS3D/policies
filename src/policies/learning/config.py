"""Learner settings."""

from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator

from policies.core.config import Config
from policies.io.inputs import Inputs
from policies.learning.objectives import ObjectiveConfig
from policies.learning.optimizers import DEFAULT_OPTIMIZER, OptimizerConfig
from policies.learning.schedules import DEFAULT_SCHEDULE, ScheduleConfig
from policies.learning.teachers import ArtifactReference
from policies.spec.model import DEFAULT_MODEL, ModelConfig, default_model_type


class RolloutConfig(Config):
    length: int = 24
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True


def _default_objectives() -> dict[str, dict[str, Any]]:
    return {"ppo": {}}


def _typed(value: Any, default: str) -> Any:
    """`value` with `type` defaulting to `default`."""
    return {"type": default, **value} if isinstance(value, Mapping) else value


class LearnerConfig(Config):
    critic_inputs: Inputs | None = None
    """Defaults to the policy inputs. May include privileged parts."""
    critic_model: ModelConfig = Field(default_factory=lambda: {"type": DEFAULT_MODEL})
    """Any architecture in the `models` registry. Without `type`, a `parts` model."""
    initial_std: float = 1.0
    objectives: dict[str, ObjectiveConfig] = Field(default_factory=_default_objectives)
    """Instance name -> objective config. `type` defaults to the instance name.

    `{}` suits learners that only `fit`, since the default `ppo` builds a critic.
    """
    optimizer: OptimizerConfig = Field(default_factory=lambda: {"type": DEFAULT_OPTIMIZER})
    """Any optimizer in the `optimizers` registry. Without `type`, `adamw`."""
    schedule: ScheduleConfig = Field(default_factory=lambda: {"type": DEFAULT_SCHEDULE})
    """Any schedule in the `schedules` registry. Without `type`, `constant`."""
    rollout: RolloutConfig = RolloutConfig()
    max_grad_norm: float = 1.0
    normalizer_clip: float = 5.0
    auxiliary: dict[str, float] = Field(default_factory=lambda: {"balance": 0.01})
    """Weights of the losses models record, by the last segment of their name."""
    teachers: dict[str, str | ArtifactReference] = Field(default_factory=dict)
    """Teacher name -> artifact path or registry reference, loaded as `ArtifactTeacher`."""

    @field_validator("critic_model", mode="before")
    @classmethod
    def _model_type(cls, value: Any) -> Any:
        return default_model_type(value)

    @field_validator("optimizer", mode="before")
    @classmethod
    def _optimizer_type(cls, value: Any) -> Any:
        return _typed(value, DEFAULT_OPTIMIZER)

    @field_validator("schedule", mode="before")
    @classmethod
    def _schedule_type(cls, value: Any) -> Any:
        return _typed(value, DEFAULT_SCHEDULE)

    @field_validator("objectives", mode="before")
    @classmethod
    def _objective_types(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        dotted = [name for name in value if "." in name]
        if dotted:
            raise ValueError(f"objective names must not contain dots, got {dotted}")
        return {name: _typed(settings or {}, name) for name, settings in value.items()}
