from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from policies.learning import Learner, LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from tests.envs import Choose, Reach

REACH = PolicySpec.model_validate(
    {
        "name": "reach",
        "inputs": {
            "parts": [
                {"name": "state", "source": "state", "shape": [2], "normalize": True},
                {"name": "target", "source": "target", "shape": [1]},
                {"name": "last_force", "source": "last.force", "shape": [1]},
            ]
        },
        "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 1}}]},
        "bookkeeping": [{"type": "last", "source": "force"}],
        "model": {"block": {"type": "mlp", "hidden": [64, 64]}},
    }
)

CHOOSE = PolicySpec.model_validate(
    {
        "name": "choose",
        "inputs": {"parts": [{"name": "context", "source": "context", "shape": [3]}]},
        "outputs": {
            "parts": [
                {"name": "choice", "kind": {"type": "categorical", "choices": ["a", "b", "c"]}}
            ]
        },
        "model": {"block": {"type": "mlp", "hidden": [32]}},
    }
)


def train(learner: Learner, env: Any, iterations: int) -> list[float]:
    torch.manual_seed(0)
    observation = env.reset()
    rewards = []
    for _ in range(iterations):
        while not learner.ready:
            out = learner.act(observation)
            observation, reward, done, info = env.step(out)
            learner.observe(observation, reward, done, info.get("truncated"))
        rewards.append(learner.update()["reward_mean"])
    return rewards


def test_ppo_learns_continuous_reach() -> None:
    config = LearnerConfig(optimizer={"type": "adam", "lr": 3e-3}, rollout=RolloutConfig(length=40))
    learner = Learner(REACH, config, num_envs=64)
    rewards = train(learner, Reach(64), 60)
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.3


def test_ppo_learns_categorical_choice() -> None:
    config = LearnerConfig(
        optimizer={"type": "adam", "lr": 1e-2}, rollout=RolloutConfig(length=8, gamma=0.0)
    )
    learner = Learner(CHOOSE, config, num_envs=64)
    rewards = train(learner, Choose(64), 30)
    assert rewards[-1] > 0.9


def test_gru_block_and_extra_objectives() -> None:
    block = {"type": "gru", "hidden_size": 32}
    spec = PolicySpec.model_validate({**REACH.model_dump(), "model": {"block": block}})
    config = LearnerConfig(
        objectives={"ppo": {}, "kl_to_reference": {"weight": 0.1}, "action_smoothness": {}},
        rollout=RolloutConfig(length=8),
    )
    learner = Learner(spec, config, num_envs=8)
    metrics = train(learner, Reach(8), 2)
    assert len(metrics) == 2
    assert "block.h" in learner.actor_state


def test_distillation_from_teacher() -> None:
    def teacher(observation: Mapping[str, Any], reset: Any) -> Mapping[str, Any]:
        return {"force": (observation["target"] - observation["state"][:, :1]).clamp(-1, 1)}

    config = LearnerConfig(
        objectives={"distillation": {"teacher": "expert"}},
        optimizer={"type": "adam", "lr": 3e-3},
        rollout=RolloutConfig(length=16),
    )
    learner = Learner(REACH, config, num_envs=32, teachers={"expert": teacher})
    env = Reach(32)
    first = distillation_loss(learner, env)
    for _ in range(30):
        last = distillation_loss(learner, env)
    assert last < first * 0.5


def distillation_loss(learner: Learner, env: Reach) -> float:
    observation = env.reset()
    while not learner.ready:
        outputs = learner.act(observation)
        observation, reward, done, info = env.step(outputs)
        learner.observe(observation, reward, done, info.get("truncated"))
    return learner.update()["distillation/loss"]


def test_tunables_and_state_round_trip() -> None:
    learner = Learner(REACH, num_envs=4)
    learner.set("optimizer.lr", 1e-4)
    learner.set("objectives.ppo.entropy_weight", 0.01)
    assert learner.tunables()["optimizer.lr"] == 1e-4
    other = Learner(REACH, num_envs=4)
    other.load_state_dict(learner.state_dict())
    assert other.tunables() == learner.tunables()


def test_optimizer_and_schedule_types_default() -> None:
    config = LearnerConfig(optimizer={"lr": 1e-3}, schedule={})
    assert config.optimizer == {
        "type": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
    }
    assert config.schedule == {"type": "constant"}
    assert Learner(REACH, config, num_envs=1).lr == 1e-3


def test_optimizer_and_schedule_are_validated_with_the_config() -> None:
    with pytest.raises(ValidationError, match="unknown optimizer 'sgd'"):
        LearnerConfig(optimizer={"type": "sgd"})
    with pytest.raises(ValidationError, match="optimizer"):
        LearnerConfig(optimizer={"momentum": 0.9})
    with pytest.raises(ValidationError, match="iterations"):
        LearnerConfig(schedule={"type": "cosine"})


@pytest.mark.parametrize("trainer", [True, False])
def test_from_checkpoint_acts_deterministically(trainer: bool, tmp_path: Path) -> None:
    config = LearnerConfig(objectives={"ppo": {"entropy_weight": 0.01}}, schedule={})
    learner = Learner(REACH, config, num_envs=4)
    train(learner, Reach(4), 1)
    state = learner.state_dict()
    torch.save({"agent": state} if trainer else state, tmp_path / "checkpoint.pt")
    loaded = Learner.from_checkpoint(tmp_path / "checkpoint.pt", num_envs=2)
    assert not loaded.training
    assert loaded.spec == REACH
    assert loaded.config == config
    assert loaded.iteration == 1
    observation = Reach(2).reset()
    first, second = loaded.act(observation), loaded.act(observation)
    actor = learner.actor.eval()
    parts = loaded.actor_inputs(observation, {"last.force": first["force"]})
    with torch.no_grad():
        expected = actor(parts, actor.model.initial_state((2,)))[0].mode()["force"]
    torch.testing.assert_close(second["force"], expected)
    assert not loaded.rollout.ready


def test_from_checkpoint_needs_a_spec(tmp_path: Path) -> None:
    state = Learner(REACH, num_envs=1).state_dict()
    del state["spec"]
    torch.save(state, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="no spec"):
        Learner.from_checkpoint(tmp_path / "old.pt")
