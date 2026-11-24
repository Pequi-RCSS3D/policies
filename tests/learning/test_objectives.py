from collections.abc import Iterable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from policies.core.config import Config
from policies.learning import (
    BaseObjective,
    Learner,
    LearnerConfig,
    RolloutConfig,
    UpdateContext,
    objectives,
)
from policies.learning.schedules import AdaptiveKL
from tests.envs import Reach
from tests.learning.test_learner import REACH, train


class Targets(Reach):
    """Reach with two constant imitation targets for `force`."""

    def _observation(self) -> dict[str, Tensor]:
        n = self.num_envs
        return super()._observation() | {
            "walk": torch.full((n, 1), 0.5),
            "kick": torch.full((n, 1), -0.5),
        }


def test_two_objectives_of_one_type_without_a_critic() -> None:
    config = LearnerConfig(
        objectives={
            "bc_walk": {"type": "imitation", "weight": 3.0, "targets": {"force": "walk"}},
            "bc_kick": {"type": "imitation", "weight": 1.0, "targets": {"force": "kick"}},
        },
        optimizer={"type": "adam", "lr": 1e-2},
        rollout=RolloutConfig(length=8),
    )
    learner = Learner(REACH, config, num_envs=32)
    assert learner.critic is None and learner.critic_state == {}
    env = Targets(32)
    train(learner, env, 25)
    learner.training = False
    force = learner.act(env.reset())["force"]
    # Squared errors weighted 3 to 1 meet at the weighted mean of the targets.
    torch.testing.assert_close(force, torch.full_like(force, 0.25), atol=0.05, rtol=0)
    other = Learner(REACH, config, num_envs=32)
    other.load_state_dict(learner.state_dict())


def test_instances_report_under_their_names() -> None:
    config = LearnerConfig(
        objectives={
            "ppo": {},
            "bc_walk": {"type": "imitation", "targets": {"force": "walk"}},
            "bc_kick": {"type": "imitation", "targets": {"force": "kick"}},
        },
        rollout=RolloutConfig(length=4),
    )
    learner = Learner(REACH, config, num_envs=4)
    env = Targets(4)
    observation = env.reset()
    while not learner.ready:
        observation, reward, done, info = env.step(learner.act(observation))
        learner.observe(observation, reward, done, info["truncated"])
    metrics = learner.update()
    assert {"ppo/kl", "bc_walk/loss", "bc_kick/loss"} <= set(metrics)
    assert metrics["bc_walk/loss"] != metrics["bc_kick/loss"]
    assert learner.tunables()["objectives.bc_kick.weight"] == 1.0


def test_types_default_to_the_name_and_names_have_no_dots() -> None:
    config = LearnerConfig.model_validate(
        {"objectives": {"ppo": None, "kl_to_reference": {"weight": 0.5}}}
    )
    assert config.objectives["ppo"]["type"] == "ppo"
    assert config.objectives["kl_to_reference"] == {"type": "kl_to_reference", "weight": 0.5}
    with pytest.raises(ValueError, match="dots"):
        LearnerConfig(objectives={"a.b": {"type": "ppo"}})
    with pytest.raises(ValueError, match="unknown objective"):
        LearnerConfig(objectives={"walk": {}})


class AnchorConfig(Config):
    weight: float = 1.0
    goal: float = 3.0
    steps: int = 1
    enabled: bool = True


@objectives.register("tests.anchor", AnchorConfig)
class Anchor(BaseObjective):
    """Pulls its own parameter toward `goal`. Its hyperparameters include non-float tunables."""

    def __init__(self, config: AnchorConfig, device: torch.device, **_: Any) -> None:
        self.hyperparameters = config.model_dump()
        self.value = nn.Parameter(torch.zeros((), device=device))

    def parameters(self) -> Iterable[nn.Parameter]:
        return [self.value]

    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]:
        loss = (self.value - self.hyperparameters["goal"]) ** 2
        return loss, {"value": self.value.detach()}


def test_an_objective_trains_its_own_parameters() -> None:
    config = LearnerConfig(
        objectives={"ppo": {}, "anchor": {"type": "tests.anchor"}},
        optimizer={"type": "adam", "lr": 0.1},
        rollout=RolloutConfig(length=4),
    )
    learner = Learner(REACH, config, num_envs=4)
    train(learner, Reach(4), 3)
    anchor = learner.objectives["anchor"]
    assert isinstance(anchor, Anchor)
    assert anchor.value.detach().item() > 1.5
    other = Learner(REACH, config, num_envs=4)
    other.load_state_dict(learner.state_dict())
    other_anchor = other.objectives["anchor"]
    assert isinstance(other_anchor, Anchor)
    assert torch.equal(other_anchor.value, anchor.value)


def test_non_float_tunables_keep_their_type() -> None:
    config = LearnerConfig(objectives={"ppo": {}, "anchor": {"type": "tests.anchor"}})
    learner = Learner(REACH, config, num_envs=2)
    learner.set("objectives.anchor.steps", 4)
    learner.set("objectives.anchor.enabled", False)
    tunables = learner.tunables()
    assert tunables["objectives.anchor.steps"] == 4 and isinstance(
        tunables["objectives.anchor.steps"], int
    )
    assert tunables["objectives.anchor.enabled"] is False
    with pytest.raises(KeyError, match="not tunable"):
        learner.set("objectives.anchor.missing", 1)


def adaptive_kl(objectives: dict[str, Any], **schedule: Any) -> AdaptiveKL:
    config = LearnerConfig(objectives=objectives, schedule={"type": "adaptive_kl", **schedule})
    built = Learner(REACH, config, num_envs=2).optimization.schedule
    assert isinstance(built, AdaptiveKL)
    return built


def test_adaptive_kl_follows_the_ppo_instance() -> None:
    assert adaptive_kl({"walk": {"type": "ppo"}}).metric == "walk/kl"
    two = {"walk": {"type": "ppo"}, "kick": {"type": "ppo"}}
    with pytest.raises(ValueError, match="Set `metric`"):
        adaptive_kl(two)
    assert adaptive_kl(two, metric="kick/kl").metric == "kick/kl"
