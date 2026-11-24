"""Truncated episodes bootstrap with the value of the observation that ended them."""

import torch
from torch import Tensor

from policies.learning import Learner, LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from policies.training.envs import FINAL_OBSERVATION, TRUNCATED, Step

GAMMA = 0.9
A, B = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
TRUE = {"A": GAMMA / (1 - GAMMA**2), "B": 1 / (1 - GAMMA**2)}
"""Values without the time limit: rewards 0, 1, 0, 1, ... from A, and 1, 0, 1, ... from B."""


class Alternating:
    """States A and B alternate, B pays 1. A time limit ends half the steps, back in A.

    The time limit is not part of the task, so values must not depend on it.
    """

    def __init__(self, num_envs: int, seed: int = 0) -> None:
        self.num_envs = num_envs
        self.generator = torch.Generator().manual_seed(seed)
        self.phase = torch.zeros(num_envs, dtype=torch.long)

    @staticmethod
    def _observation(phase: Tensor) -> dict[str, Tensor]:
        return {"phase": torch.nn.functional.one_hot(phase, 2).float()}

    def reset(self) -> dict[str, Tensor]:
        return self._observation(self.phase)

    def step(self, outputs: object) -> Step:
        rewards = self.phase.float()
        self.phase = 1 - self.phase
        final = self._observation(self.phase)
        truncated = torch.rand(self.num_envs, generator=self.generator) < 0.5
        self.phase = torch.where(truncated, 0, self.phase)
        info = {TRUNCATED: truncated, FINAL_OBSERVATION: final}
        return self._observation(self.phase), rewards, truncated, info


SPEC = PolicySpec.model_validate(
    {
        "name": "alternating",
        "inputs": {"parts": [{"name": "phase", "source": "phase", "shape": [2]}]},
        "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 1}}]},
        "model": {"block": {"type": "mlp", "hidden": [8]}},
    }
)


def learned_values(final_observation: bool) -> dict[str, float]:
    torch.manual_seed(0)
    config = LearnerConfig(
        critic_model={"block": {"type": "mlp", "hidden": [16]}},
        objectives={"ppo": {"value_clip": 100.0}},
        optimizer={"type": "adam", "lr": 1e-2},
        rollout=RolloutConfig(length=16, gamma=GAMMA),
    )
    learner = Learner(SPEC, config, num_envs=32)
    env = Alternating(32)
    observation = env.reset()
    for _ in range(25):
        while not learner.ready:
            observation, rewards, dones, info = env.step(learner.act(observation))
            final = info[FINAL_OBSERVATION] if final_observation else None
            learner.observe(observation, rewards, dones, info[TRUNCATED], final)
        learner.update()
    assert learner.critic is not None
    with torch.no_grad():
        return {
            name: float(learner.critic({"phase": x}, {})[0]) for name, x in (("A", A), ("B", B))
        }


def gap(values: dict[str, float]) -> float:
    return values["B"] - values["A"]


def test_the_final_observation_bootstraps_truncated_episodes() -> None:
    final, naive = learned_values(True), learned_values(False)
    assert all(abs(final[name] - TRUE[name]) < 0.15 for name in TRUE)
    # Bootstrapping with the last observation's value would make the gap 1 instead of 0.53.
    assert abs(gap(final) - gap(TRUE)) < 0.05
    assert abs(gap(naive) - gap(TRUE)) > 0.3
