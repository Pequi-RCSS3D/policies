"""Randomly repeats the previous output, like a late message."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from policies.core.config import Config
from policies.training.envs.base import Environment, Observation, Step
from policies.training.envs.wrappers.base import Wrapper, wrappers


class DroppedCycles(Wrapper):
    """With probability `rate`, an environment keeps its previous output, like a late message."""

    def __init__(self, env: Environment, rate: float, seed: int = 0) -> None:
        super().__init__(env)
        self.rate = rate
        self.seed = seed
        self.generator: torch.Generator | None = None
        self.previous: dict[str, Tensor] | None = None

    def reset(self) -> Observation:
        self.previous = None
        return self.env.reset()

    def _drops(self, device: torch.device) -> Tensor:
        # Sampled on the outputs' device to avoid a host transfer per step.
        if self.generator is None:
            self.generator = torch.Generator(device).manual_seed(self.seed)
        return torch.rand(self.num_envs, generator=self.generator, device=device) < self.rate

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        applied = dict(outputs)
        if self.previous is not None:
            drop = self._drops(next(iter(applied.values())).device)
            for name, value in applied.items():
                mask = drop.to(value.device).reshape(-1, *[1] * (value.dim() - 1))
                applied[name] = torch.where(mask, self.previous[name], value)
        self.previous = applied
        return self.env.step(applied)


class DroppedCyclesConfig(Config):
    rate: float
    seed: int | None = None
    """Defaults to the run seed."""


@wrappers.register("dropped_cycles", DroppedCyclesConfig)
def dropped_cycles(
    config: DroppedCyclesConfig, env: Environment, seed: int = 0, **_: Any
) -> DroppedCycles:
    return DroppedCycles(env, config.rate, seed if config.seed is None else config.seed)
