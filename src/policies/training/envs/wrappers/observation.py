"""A wrapper that changes every observation, including the final one in `info`."""

from collections.abc import Mapping

from torch import Tensor

from policies.training.envs.base import FINAL_OBSERVATION, Observation, Step
from policies.training.envs.wrappers.base import Wrapper


class ObservationWrapper(Wrapper):
    """Override `observation`. Applies to resets, steps and `info[FINAL_OBSERVATION]`."""

    def observation(self, observation: Observation) -> Observation:
        return observation

    def reset(self) -> Observation:
        return self.observation(self.env.reset())

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        observation, rewards, dones, info = self.env.step(outputs)
        if FINAL_OBSERVATION in info:
            info = {**info, FINAL_OBSERVATION: self.observation(info[FINAL_OBSERVATION])}
        return self.observation(observation), rewards, dones, info
