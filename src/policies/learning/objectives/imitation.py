"""Behavior cloning from targets the environment provides."""

from typing import Any

from torch import Tensor

from policies.core.config import Config
from policies.learning.objectives.base import (
    ActContext,
    BaseObjective,
    UpdateContext,
    imitation_loss,
    objectives,
)


class ImitationConfig(Config):
    weight: float = 1.0
    targets: dict[str, str]
    """Output part -> observation key holding its target (raw action space)."""


@objectives.register("imitation", ImitationConfig)
class Imitation(BaseObjective):
    """Behavior cloning from targets the environment provides, e.g. a reference motion."""

    def __init__(self, config: ImitationConfig, **_: Any) -> None:
        self.targets = config.targets
        self.hyperparameters = {"weight": config.weight}

    def record(self, context: ActContext) -> dict[str, Tensor]:
        return {part: context.observation[key] for part, key in self.targets.items()}

    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]:
        loss = imitation_loss(context.distribution, context.recorded)
        return loss, {"loss": loss.detach()}
