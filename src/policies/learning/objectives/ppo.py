"""Proximal policy optimization: clipped surrogate, clipped value loss, entropy bonus."""

from typing import Any

import torch
from torch import Tensor

from policies.core.config import Config
from policies.learning.objectives.base import BaseObjective, UpdateContext, objectives


class PPOConfig(Config):
    weight: float = 1.0
    clip: float = 0.2
    value_weight: float = 1.0
    value_clip: float = 0.2
    entropy_weight: float = 0.0


@objectives.register("ppo", PPOConfig)
class PPO(BaseObjective):
    """Policy loss from the rollout's advantages, value loss for the critic."""

    needs_values = True

    def __init__(self, config: PPOConfig, **_: Any) -> None:
        self.hyperparameters = config.model_dump()

    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]:
        settings, batch = self.hyperparameters, context.batch
        assert context.values is not None  # `needs_values` asks the learner for a critic
        ratio = torch.exp(context.log_prob - batch["log_prob"])
        advantages = batch["advantages"]
        surrogate = -torch.min(
            ratio * advantages, ratio.clamp(1 - settings["clip"], 1 + settings["clip"]) * advantages
        ).mean()
        clipped = batch["values"] + (context.values - batch["values"]).clamp(
            -settings["value_clip"], settings["value_clip"]
        )
        value = torch.max(
            (context.values - batch["returns"]) ** 2, (clipped - batch["returns"]) ** 2
        ).mean()
        entropy = context.distribution.entropy().mean()
        loss = surrogate + settings["value_weight"] * value - settings["entropy_weight"] * entropy
        kl = (batch["log_prob"] - context.log_prob).mean()
        return loss, {
            "surrogate": surrogate.detach(),
            "value": value.detach(),
            "entropy": entropy.detach(),
            "kl": kl.detach(),
        }
