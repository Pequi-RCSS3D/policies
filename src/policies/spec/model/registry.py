"""The model registry, importable without torch."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any

from policies.core.registry import Registry
from policies.spec.model.parts import PartsModelConfig

if TYPE_CHECKING:
    from policies.models import Model

models: "Registry[Model]" = Registry("model")

type ModelConfig = Annotated[dict[str, Any], models]
"""Any model's config, validated to its canonical form."""

DEFAULT_MODEL = "parts"
"""The architecture of a model config without `type`."""


def default_model_type(value: Any) -> Any:
    """`value` with `type` defaulting to `DEFAULT_MODEL`, for use as a before-validator."""
    return {"type": DEFAULT_MODEL, **value} if isinstance(value, Mapping) else value


models.register_lazy(DEFAULT_MODEL, PartsModelConfig, "policies.models.parts:PartsModel")
