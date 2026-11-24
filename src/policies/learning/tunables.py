"""The learner's values that may change while running, by dotted path."""

from collections.abc import Mapping

from policies.learning.objectives import Objective
from policies.learning.optimization import Optimization
from policies.training.control import Tunable

LR = "optimizer.lr"
OBJECTIVES = "objectives"
"""Objective values are `objectives.<name>.<key>`."""


class LearnerTunables:
    """The learning rate and every objective's `hyperparameters`."""

    def __init__(self, optimization: Optimization, objectives: Mapping[str, Objective]) -> None:
        self.optimization, self.objectives = optimization, objectives

    def values(self) -> dict[str, Tunable]:
        values: dict[str, Tunable] = {LR: self.optimization.lr}
        for name, objective in self.objectives.items():
            values.update(
                {f"{OBJECTIVES}.{name}.{k}": v for k, v in objective.hyperparameters.items()}
            )
        return values

    def set(self, path: str, value: Tunable) -> None:
        """Set a value as given. The caller checks its type, as `training.control` does."""
        if path == LR:
            self.optimization.set_lr(float(value))
            return
        root, _, rest = path.partition(".")
        name, _, key = rest.partition(".")
        objective = self.objectives.get(name) if root == OBJECTIVES else None
        if objective is None or key not in objective.hyperparameters:
            raise KeyError(f"{path} is not tunable. Tunable: {', '.join(self.values())}")
        objective.hyperparameters[key] = value
