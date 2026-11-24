"""Adds gaussian noise to observation parts, like an imprecise sensor."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from policies.core.config import Config
from policies.training.envs.base import FINAL_OBSERVATION, Environment, Observation, Step
from policies.training.envs.wrappers.base import Wrapper, wrappers


class Noise(Wrapper):
    """Adds zero-mean gaussian noise with a standard deviation per part, e.g. `{"velocity": 0.05}`.

    The final observation in `info` gets the same draw as the returned one.
    """

    def __init__(self, env: Environment, scales: Mapping[str, float], seed: int = 0) -> None:
        super().__init__(env)
        self.scales = dict(scales)
        self.seed = seed
        self.generator: torch.Generator | None = None

    def _draw(self, observation: Observation) -> dict[str, Tensor]:
        noise: dict[str, Tensor] = {}
        for name, scale in self.scales.items():
            if name not in observation:
                continue
            value = observation[name]
            # Drawn on the observations' device to avoid a host transfer per step.
            if self.generator is None:
                self.generator = torch.Generator(value.device).manual_seed(self.seed)
            sample = torch.randn(
                value.shape, generator=self.generator, device=value.device, dtype=value.dtype
            )
            noise[name] = sample * scale
        return noise

    @staticmethod
    def _add(observation: Observation, noise: Mapping[str, Tensor]) -> Observation:
        return {**observation, **{name: observation[name] + value for name, value in noise.items()}}

    def reset(self) -> Observation:
        observation = self.env.reset()
        return self._add(observation, self._draw(observation))

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        observation, rewards, dones, info = self.env.step(outputs)
        noise = self._draw(observation)
        if FINAL_OBSERVATION in info:
            info = {**info, FINAL_OBSERVATION: self._add(info[FINAL_OBSERVATION], noise)}
        return self._add(observation, noise), rewards, dones, info


class NoiseConfig(Config):
    scales: dict[str, float]
    """Part name to standard deviation."""
    seed: int | None = None
    """Defaults to the run seed."""


@wrappers.register("noise", NoiseConfig)
def noise(config: NoiseConfig, env: Environment, seed: int = 0, **_: Any) -> Noise:
    return Noise(env, config.scales, seed if config.seed is None else config.seed)
