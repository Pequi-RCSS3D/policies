"""Learning from rollouts and data. Requires the torch extra."""

from policies.learning.actor import Actor
from policies.learning.config import LearnerConfig, RolloutConfig
from policies.learning.critic import Critic
from policies.learning.distributions import DistributionHead, PartDistribution, PartsDistribution
from policies.learning.learner import Learner
from policies.learning.objectives import (
    ActContext,
    BaseObjective,
    Objective,
    ObjectiveConfig,
    UpdateContext,
    objectives,
)
from policies.learning.optimizers import optimizers
from policies.learning.schedules import schedules
from policies.learning.teachers import ArtifactReference, ArtifactTeacher, Teacher, TeacherSource

__all__ = [
    "ActContext",
    "Actor",
    "ArtifactReference",
    "ArtifactTeacher",
    "BaseObjective",
    "Critic",
    "DistributionHead",
    "Learner",
    "LearnerConfig",
    "Objective",
    "ObjectiveConfig",
    "PartDistribution",
    "PartsDistribution",
    "RolloutConfig",
    "Teacher",
    "TeacherSource",
    "UpdateContext",
    "objectives",
    "optimizers",
    "schedules",
]
