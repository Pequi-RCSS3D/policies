"""Keeps the policy close to a frozen copy of itself."""

from typing import Any

from torch import Tensor

from policies.core.config import Config
from policies.learning.objectives.base import BaseObjective, UpdateContext, objectives


class KLToReferenceConfig(Config):
    weight: float = 0.1


@objectives.register("kl_to_reference", KLToReferenceConfig)
class KLToReference(BaseObjective):
    """Keeps the policy close to a frozen copy of itself, taken at the first update."""

    needs_reference = True

    def __init__(self, config: KLToReferenceConfig, **_: Any) -> None:
        self.hyperparameters = config.model_dump()

    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]:
        assert context.reference is not None
        kl = context.reference.kl(context.distribution).mean()
        return kl, {"kl": kl.detach()}
