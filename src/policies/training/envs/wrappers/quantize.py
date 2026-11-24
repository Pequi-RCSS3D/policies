"""Rounds observation parts to a fixed step."""

from collections.abc import Mapping
from typing import Any

from policies.core.config import Config
from policies.training.envs.base import Environment, Observation
from policies.training.envs.wrappers.base import wrappers
from policies.training.envs.wrappers.observation import ObservationWrapper
from policies.training.envs.wrappers.rounding import ROUNDERS, Rounding


class Quantize(ObservationWrapper):
    """Rounds listed parts to a multiple of their step, e.g. `{"position": 0.01}`."""

    def __init__(
        self, env: Environment, steps: Mapping[str, float], rounding: Rounding = Rounding.NEAREST
    ) -> None:
        super().__init__(env)
        self.steps = dict(steps)
        # Scaling by the exact reciprocal of a decimal step keeps values on its grid, as
        # `trunc(x * 100) / 100` does. Dividing by 0.001 in float32 would take 1.0 to 0.999.
        self.scales = {name: 1 / step for name, step in self.steps.items()}
        self.round = ROUNDERS[Rounding(rounding)]

    def observation(self, observation: Observation) -> Observation:
        return {
            name: self.round(value * self.scales[name]) / self.scales[name]
            if name in self.scales
            else value
            for name, value in observation.items()
        }


class QuantizeConfig(Config):
    steps: dict[str, float]
    """Part name to step."""
    rounding: Rounding = Rounding.NEAREST


@wrappers.register("quantize", QuantizeConfig)
def quantize(config: QuantizeConfig, env: Environment, **_: Any) -> Quantize:
    return Quantize(env, config.steps, config.rounding)
