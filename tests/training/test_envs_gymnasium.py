from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv

from policies.learning import LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from policies.training.envs import FINAL_OBSERVATION
from policies.training.envs.gymnasium import GymnasiumEnvironment
from policies.training.envs.gymnasium.actions import Actions
from policies.training.loop import RunConfig, Trainer, TrainerConfig


def make(name: str, num_envs: int, mode: AutoresetMode, **kwargs: Any) -> VectorEnv[Any, Any, Any]:
    return gym.make_vec(
        name,
        num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": mode},
        **kwargs,
    )


class Echo(gym.Env[dict[str, Any], dict[str, Any]]):
    """Observes its last action. Truncates after `horizon` steps."""

    observation_space = spaces.Dict(
        {"position": spaces.Box(-1, 1, (2,), np.float64), "mode": spaces.Discrete(3)}
    )
    action_space = spaces.Dict(
        {"force": spaces.Box(-1, 1, (2,), np.float32), "mode": spaces.Discrete(3, start=1)}
    )

    def __init__(self, horizon: int = 2) -> None:
        self.horizon = horizon

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        return {"position": np.zeros(2), "mode": 0}, {}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        self.steps += 1
        observation = {"position": action["force"].astype(np.float64), "mode": action["mode"] - 1}
        return observation, 1.0, False, self.steps >= self.horizon, {}


@pytest.mark.parametrize("mode", [AutoresetMode.SAME_STEP, AutoresetMode.DISABLED])
def test_dict_spaces_round_trip_and_truncate(mode: AutoresetMode) -> None:
    env = GymnasiumEnvironment(SyncVectorEnv([Echo] * 3, autoreset_mode=mode))
    first = env.reset()
    assert first["position"].dtype == torch.float32 and first["mode"].dtype == torch.int64
    force = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    choice = torch.tensor([0, 1, 2])
    observation, rewards, dones, info = env.step({"force": force, "mode": choice})
    torch.testing.assert_close(observation["position"], force)
    assert observation["mode"].tolist() == [0, 1, 2]
    assert rewards.dtype == torch.float32 and not dones.any()
    observation, _, dones, info = env.step({"force": force, "mode": choice})
    assert dones.all() and info["truncated"].all()
    assert not observation["position"].any(), "the done step returns the next first observation"


@pytest.mark.parametrize("mode", [AutoresetMode.SAME_STEP, AutoresetMode.DISABLED])
def test_final_observation_holds_the_observation_before_the_reset(mode: AutoresetMode) -> None:
    envs = [lambda: Echo(horizon=1), lambda: Echo(horizon=2)]
    env = GymnasiumEnvironment(SyncVectorEnv(envs, autoreset_mode=mode))
    env.reset()
    force = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    choice = torch.tensor([1, 2])
    observation, _, dones, info = env.step({"force": force, "mode": choice})
    assert dones.tolist() == [True, False]
    final = info[FINAL_OBSERVATION]
    torch.testing.assert_close(final["position"], force)
    assert final["mode"].tolist() == [1, 2]
    assert final["position"].dtype == torch.float32 and final["mode"].dtype == torch.int64
    torch.testing.assert_close(observation["position"], torch.tensor([[0.0, 0.0], [0.3, 0.4]]))
    assert observation["mode"].tolist() == [0, 2]


def test_final_observation_without_done_is_the_observation() -> None:
    vector = SyncVectorEnv([lambda: Echo(horizon=5)] * 2, autoreset_mode=AutoresetMode.SAME_STEP)
    env = GymnasiumEnvironment(vector)
    env.reset()
    force = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    observation, _, _, info = env.step({"force": force, "mode": torch.tensor([0, 1])})
    assert info[FINAL_OBSERVATION] is observation


def test_same_step_and_disabled_agree() -> None:
    runs = []
    for mode in (AutoresetMode.SAME_STEP, AutoresetMode.DISABLED):
        env = GymnasiumEnvironment(make("CartPole-v1", 4, mode, max_episode_steps=5), seed=0)
        observations = [env.reset()["observation"]]
        for _ in range(12):
            observation, _, _, _ = env.step({"action": torch.ones(4, dtype=torch.long)})
            observations.append(observation["observation"])
        runs.append(torch.stack(observations))
    torch.testing.assert_close(runs[0], runs[1])


def test_box_action_and_truncation() -> None:
    env = GymnasiumEnvironment(
        make("Pendulum-v1", 2, AutoresetMode.SAME_STEP, max_episode_steps=3), seed=0
    )
    assert env.reset()["observation"].shape == (2, 3)
    for _ in range(3):
        _, rewards, dones, info = env.step({"action": torch.zeros(2, 1)})
        assert rewards.shape == (2,)
    assert dones.all() and info["truncated"].all()


def test_next_step_autoreset_is_refused() -> None:
    with pytest.raises(ValueError, match="SAME_STEP"):
        GymnasiumEnvironment(make("CartPole-v1", 2, AutoresetMode.NEXT_STEP))


def test_unsupported_action_space() -> None:
    with pytest.raises(TypeError, match="unsupported space"):
        Actions(spaces.MultiBinary(3), 2, "action")


CARTPOLE = {
    "name": "cartpole",
    "inputs": {
        "parts": [{"name": "observation", "source": "observation", "shape": [4], "normalize": True}]
    },
    "outputs": {
        "parts": [{"name": "action", "kind": {"type": "categorical", "choices": ["left", "right"]}}]
    },
    "model": {"block": {"type": "mlp", "hidden": [32]}},
}
PENDULUM = {
    "name": "pendulum",
    "inputs": {"parts": [{"name": "observation", "source": "observation", "shape": [3]}]},
    "outputs": {
        "parts": [
            {"name": "action", "kind": {"type": "continuous", "size": 1, "low": -2, "high": 2}}
        ]
    },
    "model": {"block": {"type": "mlp", "hidden": [32]}},
}


@pytest.mark.parametrize(
    ("name", "spec"),
    [("CartPole-v1", CARTPOLE), ("Pendulum-v1", PENDULUM)],
    ids=["cartpole", "pendulum"],
)
def test_trains_end_to_end(tmp_path: Path, name: str, spec: dict[str, Any]) -> None:
    config = RunConfig(
        policy=PolicySpec.model_validate(spec),
        learner=LearnerConfig(rollout=RolloutConfig(length=16)),
        trainer=TrainerConfig(iterations=3, hooks=({"type": "jsonl"},)),
    )
    vector = make(name, 8, AutoresetMode.SAME_STEP, max_episode_steps=10)
    trainer = Trainer(config, GymnasiumEnvironment(vector, seed=0), tmp_path)
    assert trainer.run().iteration == 3
    assert "episode_return" in trainer.metrics
