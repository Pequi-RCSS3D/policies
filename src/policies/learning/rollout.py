"""Steps recorded while acting, stacked to `[time, env]` for an update."""

from collections.abc import Iterator, Mapping
from typing import Self

import torch
from torch import Tensor


class Steps(dict[str, Tensor]):
    """Tensors by key sharing `batch_dims` leading dimensions, e.g. `[time, env]`.

    The few TensorDict operations updates need, on a plain dict, without TensorDict's import of
    torch._dynamo and sympy, which took most of the package's import time.
    """

    def __init__(self, tensors: Mapping[str, Tensor], batch_dims: int) -> None:
        super().__init__(tensors)
        self.batch_dims = batch_dims

    def reshape(self, *shape: int) -> Self:
        """Batch dimensions reshaped to `shape`, e.g. `-1` to flatten them."""
        dims = self.batch_dims
        return type(self)(
            {key: value.reshape(*shape, *value.shape[dims:]) for key, value in self.items()},
            len(shape),
        )

    def select(self, dim: int, indices: Tensor) -> Self:
        """Entries `indices` along batch dimension `dim`."""
        return type(self)(
            {key: value.index_select(dim, indices) for key, value in self.items()},
            self.batch_dims,
        )


class Rollout:
    """A step is what `act` records, completed by `rewards` and `dones` in `observe`."""

    def __init__(self, length: int, num_envs: int) -> None:
        self.length, self.num_envs = length, num_envs
        self.steps: list[dict[str, Tensor]] = []
        self.pending: dict[str, Tensor] = {}

    def record(self, values: Mapping[str, Tensor]) -> None:
        self.pending.update(values)

    def complete(self, rewards: Tensor, dones: Tensor) -> None:
        self.steps.append({**self.pending, "rewards": rewards, "dones": dones.float()})
        self.pending = {}

    @property
    def ready(self) -> bool:
        return len(self.steps) >= self.length

    def take(self) -> Steps:
        """The recorded steps as `[time, env]`. Starts a new rollout."""
        if not self.steps:
            raise RuntimeError("no recorded steps: call act and observe first")
        steps, self.steps = self.steps, []
        data = {key: torch.stack([step[key] for step in steps]) for key in steps[0]}
        wrong = [key for key, value in data.items() if value.shape[1:2] != (self.num_envs,)]
        if wrong:
            raise ValueError(f"recorded {wrong} do not lead with {self.num_envs} environments")
        return Steps(data, batch_dims=2)


def minibatches(data: Steps, chunks: int, sequences: bool, device: torch.device) -> Iterator[Steps]:
    """Shuffled minibatches of steps `[batch]`, or of environment sequences `[time, batch]`."""
    if sequences:
        for indices in torch.randperm(data["dones"].shape[1], device=device).chunk(chunks):
            yield data.select(1, indices)
    else:
        flat = data.reshape(-1)
        for indices in torch.randperm(flat["dones"].shape[0], device=device).chunk(chunks):
            yield flat.select(0, indices)
