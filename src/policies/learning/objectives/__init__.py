"""Losses and how they combine. Register more with `objectives.register`."""

from policies.learning.objectives.action_smoothness import ActionSmoothness
from policies.learning.objectives.base import (
    ActContext,
    BaseObjective,
    Objective,
    ObjectiveConfig,
    UpdateContext,
    imitation_loss,
    objectives,
)
from policies.learning.objectives.distillation import Distillation
from policies.learning.objectives.imitation import Imitation
from policies.learning.objectives.kl_to_reference import KLToReference
from policies.learning.objectives.ppo import PPO

__all__ = [
    "PPO",
    "ActContext",
    "ActionSmoothness",
    "BaseObjective",
    "Distillation",
    "Imitation",
    "KLToReference",
    "Objective",
    "ObjectiveConfig",
    "UpdateContext",
    "imitation_loss",
    "objectives",
]
