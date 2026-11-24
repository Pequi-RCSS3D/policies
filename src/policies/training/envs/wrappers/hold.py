"""Refreshes observation parts at a lower rate, like a slower sensor."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from pydantic import Field
from torch import Tensor

from policies.core.config import Config
from policies.training.envs.base import FINAL_OBSERVATION, Environment, Observation, Step
from policies.training.envs.wrappers.base import Wrapper, wrappers


class Hold(Wrapper):
    """Refreshes `parts` every `every` steps and repeats their last value in between.

    An environment refreshes on the first observation of an episode and whenever its step in
    the episode plus its phase is a multiple of `every`. With `random_phase`, each episode
    draws its phase instead. Parts missing from the observation are left alone.
    """

    def __init__(
        self,
        env: Environment,
        parts: Sequence[str],
        every: int,
        phase: int = 0,
        random_phase: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self.parts = tuple(parts)
        self.every = every
        self.phase = phase % every
        self.random_phase = random_phase
        self.seed = seed
        self.generator: torch.Generator | None = None
        self.count: Tensor | None = None
        self.held: dict[str, Tensor] = {}

    def _phases(self, device: torch.device) -> Tensor:
        if not self.random_phase:
            return torch.full((self.num_envs,), self.phase, device=device)
        # Drawn on the observations' device to avoid a host transfer per step.
        if self.generator is None:
            self.generator = torch.Generator(device).manual_seed(self.seed)
        return torch.randint(self.every, (self.num_envs,), generator=self.generator, device=device)

    def reset(self) -> Observation:
        observation = self.env.reset()
        self.held = {name: observation[name].clone() for name in self.parts if name in observation}
        if self.held:
            self.count = self._phases(next(iter(self.held.values())).device)
        return observation

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        observation, rewards, dones, info = self.env.step(outputs)
        if self.count is None:
            return observation, rewards, dones, info
        self.count += 1
        fresh = self.count % self.every == 0
        if FINAL_OBSERVATION in info:
            final = info[FINAL_OBSERVATION]
            info = {**info, FINAL_OBSERVATION: {**final, **self._refresh(fresh, final)}}
        done = dones.to(fresh.device)
        if self.random_phase:
            self.count = torch.where(done, self._phases(fresh.device), self.count)
        else:
            self.count.masked_fill_(done, self.phase)
        self.held = self._refresh(fresh | done, observation)
        return {**observation, **self.held}, rewards, dones, info

    def _refresh(self, mask: Tensor, observation: Observation) -> dict[str, Tensor]:
        return {
            name: torch.where(mask.view(-1, *[1] * (value.dim() - 1)), observation[name], value)
            for name, value in self.held.items()
        }


class HoldConfig(Config):
    parts: tuple[str, ...]
    every: int = Field(ge=1)
    """Steps between refreshes."""
    phase: int = Field(default=0, ge=0)
    """Offset of the refresh steps, modulo `every`."""
    random_phase: bool = False
    """Draw each episode's phase instead."""
    seed: int | None = None
    """Defaults to the run seed."""


@wrappers.register("hold", HoldConfig)
def hold(config: HoldConfig, env: Environment, seed: int = 0, **_: Any) -> Hold:
    return Hold(
        env,
        config.parts,
        config.every,
        config.phase,
        config.random_phase,
        seed if config.seed is None else config.seed,
    )
