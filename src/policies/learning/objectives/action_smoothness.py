"""Penalizes jerky continuous actions."""

from typing import Any

import torch
from torch import Tensor

from policies.core.config import Config
from policies.learning.objectives.base import ActContext, BaseObjective, UpdateContext, objectives


class ActionSmoothnessConfig(Config):
    weight: float = 0.01


@objectives.register("action_smoothness", ActionSmoothnessConfig)
class ActionSmoothness(BaseObjective):
    """Penalizes changing actions between consecutive steps, as each distribution defines it.

    Only floating-point actions are recorded. Distributions without a penalty are skipped.
    """

    def __init__(self, config: ActionSmoothnessConfig, **_: Any) -> None:
        self.hyperparameters = config.model_dump()
        self.previous: dict[str, Tensor] = {}

    def record(self, context: ActContext) -> dict[str, Tensor]:
        out = {}
        for name, action in context.actions.items():
            if not torch.is_floating_point(action):
                continue
            previous = self.previous.get(name, torch.zeros_like(action))
            keep = (~context.reset).float().reshape(-1, *([1] * (action.dim() - 1)))
            out[f"previous/{name}"] = previous * keep
            out[f"previous_valid/{name}"] = keep.reshape(-1) * (name in self.previous)
            self.previous[name] = action
        return out

    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]:
        total = torch.zeros((), device=context.log_prob.device)
        recorded = context.recorded
        for name, part in context.distribution.parts.items():
            if f"previous/{name}" not in recorded:
                continue
            penalty = part.smoothness(recorded[f"previous/{name}"])
            if penalty is not None:
                valid = recorded[f"previous_valid/{name}"]
                total = total + (penalty * valid).sum() / valid.sum().clamp(min=1)
        return total, {"loss": total.detach()}
