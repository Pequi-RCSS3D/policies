"""Any Gymnasium vector environment as an `Environment`."""

from collections.abc import Mapping
from typing import Any

import torch
from gymnasium.vector import AutoresetMode, VectorEnv
from torch import Tensor

from policies.training.envs.base import FINAL_OBSERVATION, TRUNCATED, Observation, Step
from policies.training.envs.gymnasium.actions import Actions
from policies.training.envs.gymnasium.observations import Observations

SUPPORTED = (AutoresetMode.SAME_STEP, AutoresetMode.DISABLED)


class GymnasiumEnvironment:
    """Adapts a Gymnasium vector env. Dict spaces map entry by entry to parts.

    The contract returns the next episode's first observation on `done`. SAME_STEP does
    that natively, DISABLED is reset here through `reset_mask`. NEXT_STEP is refused: its
    reset step ignores the action, which the contract cannot express. `info` carries
    `TRUNCATED` and `FINAL_OBSERVATION`.
    """

    def __init__(
        self,
        env: VectorEnv[Any, Any, Any],
        *,
        observation_name: str = "observation",
        action_name: str = "action",
        seed: int | None = None,
    ) -> None:
        """Single spaces become the parts `observation_name` and `action_name`.

        `seed` seeds the first reset.
        """
        if "autoreset_mode" not in env.metadata:
            raise ValueError(f"{env} has no `autoreset_mode` in its metadata")
        self.mode = AutoresetMode(env.metadata["autoreset_mode"])
        if self.mode not in SUPPORTED:
            raise ValueError(
                f"{env} uses {self.mode}, expected SAME_STEP or DISABLED autoreset. Build it "
                'with `vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP}`'
            )
        self.env = env
        self.num_envs = env.num_envs
        self.observations = Observations(env.single_observation_space, observation_name)
        self.actions = Actions(env.single_action_space, env.num_envs, action_name)
        self.seed = seed

    def reset(self) -> Observation:
        observation, _ = self.env.reset(seed=self.seed)
        self.seed = None
        return self.observations(observation)

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        observation, rewards, terminated, truncated, info = self.env.step(self.actions(outputs))
        done = terminated | truncated
        parts = final = self.observations(observation)
        if self.mode == AutoresetMode.DISABLED and done.any():
            observation, _ = self.env.reset(options={"reset_mask": done})
            parts = self.observations(observation)
        elif "final_obs" in info:
            final = self.observations.replace(parts, info["final_obs"], info["_final_obs"])
        return (
            parts,
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(done),
            {
                **info,
                TRUNCATED: torch.as_tensor(truncated & ~terminated),
                FINAL_OBSERVATION: final,
            },
        )
