"""The wrapper base, the wrapper registry and applying configured wrappers."""

from collections.abc import Iterable, Mapping
from typing import Any

from torch import Tensor

from policies.core.registry import Registry
from policies.training.envs.base import Environment, Observation, Step


class Wrapper:
    """Forwards everything to `env`. Override what you change."""

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.num_envs = env.num_envs

    def reset(self) -> Observation:
        return self.env.reset()

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        return self.env.step(outputs)


wrappers: Registry[Environment] = Registry("wrapper")
"""Factories are called as `factory(config, env=env, seed=seed)`."""


def wrap(env: Environment, configs: Iterable[Mapping[str, Any]], **context: Any) -> Environment:
    """Apply each `{type: ...}` config in order, so the first wraps `env` directly."""
    for config in configs:
        env = wrappers.build(config, env=env, **context)
    return env
