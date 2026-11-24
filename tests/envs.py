"""Small batched environments for tests and examples."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


class Reach:
    """A point mass on a line must reach a random target. Continuous output `force`."""

    def __init__(self, num_envs: int, horizon: int = 40, seed: int = 0) -> None:
        self.num_envs, self.horizon = num_envs, horizon
        self.generator = torch.Generator().manual_seed(seed)
        self.position = torch.zeros(num_envs)
        self.velocity = torch.zeros(num_envs)
        self.target = torch.zeros(num_envs)
        self.t = torch.zeros(num_envs, dtype=torch.long)
        self._reset(torch.ones(num_envs, dtype=torch.bool))

    def _reset(self, mask: Tensor) -> None:
        n = int(mask.sum())
        self.position[mask] = torch.rand(n, generator=self.generator) * 2 - 1
        self.velocity[mask] = 0
        self.target[mask] = torch.rand(n, generator=self.generator) * 2 - 1
        self.t[mask] = 0

    def _observation(self) -> dict[str, Tensor]:
        return {
            "state": torch.stack([self.position, self.velocity], -1),
            "target": self.target[:, None].clone(),
        }

    def reset(self) -> dict[str, Tensor]:
        return self._observation()

    def step(
        self, outputs: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Any]]:
        force = outputs["force"][:, 0].clamp(-1, 1)
        self.velocity = 0.8 * self.velocity + 0.1 * force
        self.position = (self.position + self.velocity).clamp(-2, 2)
        self.t += 1
        reward = -(self.position - self.target).abs()
        done = self.t >= self.horizon
        truncated = done.clone()
        self._reset(done)
        return self._observation(), reward, done, {"truncated": truncated}


class Choose:
    """A context selects which of three choices pays. Categorical output `choice`."""

    def __init__(self, num_envs: int, seed: int = 0) -> None:
        self.num_envs = num_envs
        self.generator = torch.Generator().manual_seed(seed)
        self.context = torch.randint(0, 3, (num_envs,), generator=self.generator)

    def _observation(self) -> dict[str, Tensor]:
        return {"context": torch.nn.functional.one_hot(self.context, 3).float()}

    def reset(self) -> dict[str, Tensor]:
        return self._observation()

    def step(
        self, outputs: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Any]]:
        reward = (outputs["choice"] == self.context).float()
        self.context = torch.randint(0, 3, (self.num_envs,), generator=self.generator)
        done = torch.ones(self.num_envs, dtype=torch.bool)
        return self._observation(), reward, done, {}
