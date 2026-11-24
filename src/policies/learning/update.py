"""The update loop: epochs of minibatch steps through the objectives' weighted losses."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from policies.learning.actor import Actor
from policies.learning.auxiliary import Auxiliary
from policies.learning.config import RolloutConfig
from policies.learning.critic import Critic
from policies.learning.metrics import Metrics
from policies.learning.objectives import Objective, UpdateContext
from policies.learning.optimization import Optimization
from policies.learning.rollout import Steps, minibatches
from policies.learning.sequences import group

RECORDED = "objectives"
"""Recorded keys are `objectives/<name>/<key>`, handed back to objective `name` only."""


@dataclass
class Updater:
    actor: Actor
    critic: Critic | None
    objectives: Mapping[str, Objective]
    auxiliary: Auxiliary
    optimization: Optimization
    config: RolloutConfig
    device: torch.device

    @property
    def recurrent(self) -> bool:
        """Recurrent models train on whole environment sequences, others on shuffled steps."""
        critic = self.critic is not None and bool(self.critic.model.state_shapes)
        return bool(self.actor.model.state_shapes) or critic

    def run(self, data: Steps, reference: Actor | None) -> Metrics:
        """Optimize over `[time, env]` steps. `reference` is a frozen actor for objectives."""
        metrics = Metrics()
        for _ in range(self.config.epochs):
            for minibatch in minibatches(
                data, self.config.minibatches, self.recurrent, self.device
            ):
                loss = self.loss(minibatch, reference, metrics)
                metrics.add({"grad_norm": self.optimization.step(loss)})
                metrics.step()
        return metrics

    def loss(self, minibatch: Steps, reference: Actor | None, metrics: Metrics) -> Tensor:
        with self.auxiliary.collection:
            distribution = self.actor.evaluate(minibatch)
            values = None if self.critic is None else self.critic.evaluate(minibatch)
        frozen = None
        if reference is not None:
            with torch.no_grad():
                frozen = reference.evaluate(minibatch)
        batch = minibatch.reshape(-1)
        log_prob = distribution.log_prob(group(batch, "actions"))
        loss = torch.zeros((), device=self.device)
        for name, objective in self.objectives.items():
            recorded = group(batch, f"{RECORDED}/{name}")
            context = UpdateContext(batch, distribution, log_prob, values, frozen, recorded)
            term, terms = objective.loss(context)
            loss = loss + objective.hyperparameters["weight"] * term
            metrics.add(terms, f"{name}/")
        auxiliary = self.auxiliary.loss(metrics)
        return loss if auxiliary is None else loss + auxiliary
