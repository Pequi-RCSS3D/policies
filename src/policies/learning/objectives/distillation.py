"""DAgger-style distillation from a teacher."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from policies.core.config import Config
from policies.learning.objectives.base import (
    ActContext,
    BaseObjective,
    UpdateContext,
    imitation_loss,
    objectives,
)
from policies.learning.teachers import Teacher


class DistillationConfig(Config):
    weight: float = 1.0
    teacher: str
    """Name of a learner teacher: a callable or an artifact from `LearnerConfig.teachers`."""


@objectives.register("distillation", DistillationConfig)
class Distillation(BaseObjective):
    """DAgger: the student acts, the teacher labels what the student saw."""

    def __init__(
        self, config: DistillationConfig, teachers: Mapping[str, Teacher] | None = None, **_: Any
    ) -> None:
        if not teachers or config.teacher not in teachers:
            raise KeyError(
                f"distillation needs teacher {config.teacher!r}, got {sorted(teachers or {})}"
            )
        self.teacher = teachers[config.teacher]
        self.hyperparameters = {"weight": config.weight}

    def record(self, context: ActContext) -> dict[str, Tensor]:
        with torch.no_grad():
            return dict(self.teacher(context.observation, context.reset))

    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]:
        loss = imitation_loss(context.distribution, context.recorded)
        return loss, {"loss": loss.detach()}
