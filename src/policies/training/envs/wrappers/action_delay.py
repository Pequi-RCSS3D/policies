"""Applies outputs late, like a link with latency."""

from collections import deque
from collections.abc import Mapping
from typing import Any

from torch import Tensor

from policies.core.config import Config
from policies.training.envs.base import Environment, Observation, Step
from policies.training.envs.wrappers.base import Wrapper, wrappers


class ActionDelay(Wrapper):
    """Applies each output `cycles` steps late. Until then, the first output repeats."""

    def __init__(self, env: Environment, cycles: int = 1) -> None:
        super().__init__(env)
        self.cycles = cycles
        self.queue: deque[dict[str, Tensor]] = deque()

    def reset(self) -> Observation:
        self.queue.clear()
        return self.env.reset()

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        self.queue.append(dict(outputs))
        applied = self.queue.popleft() if len(self.queue) > self.cycles else self.queue[0]
        return self.env.step(applied)


class ActionDelayConfig(Config):
    cycles: int = 1


@wrappers.register("action_delay", ActionDelayConfig)
def action_delay(config: ActionDelayConfig, env: Environment, **_: Any) -> ActionDelay:
    return ActionDelay(env, config.cycles)
