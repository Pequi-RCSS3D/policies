"""One config that defines a policy. Its fingerprint identifies the behavior contract."""

from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from policies.core.config import Config
from policies.core.fingerprint import fingerprint
from policies.io.bookkeeping import TrackerConfig
from policies.io.inputs import Inputs
from policies.io.outputs import Outputs
from policies.spec.model import DEFAULT_MODEL, ModelConfig, default_model_type


class PolicySpec(Config):
    """Component configs are stored canonical, so the fingerprint sees every default."""

    name: str
    inputs: Inputs
    outputs: Outputs
    bookkeeping: tuple[TrackerConfig, ...] = ()
    model: ModelConfig = Field(default_factory=lambda: {"type": DEFAULT_MODEL})
    """Any architecture in the `models` registry. Without `type`, a `parts` model."""

    @field_validator("model", mode="before")
    @classmethod
    def _model_type(cls, value: Any) -> Any:
        return default_model_type(value)

    @model_validator(mode="after")
    def _observable(self) -> Self:
        privileged = [part.name for part in self.inputs.parts if part.privileged]
        if privileged:
            raise ValueError(f"policy inputs must be observable, got privileged {privileged}")
        return self

    def fingerprint(self) -> str:
        return fingerprint(self)
