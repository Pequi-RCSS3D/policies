"""Whole architectures from the models registry, and the auxiliary losses models record."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from policies.core.config import Config
from policies.data import InMemoryDataset
from policies.deploy.inference import Policy, validate_artifact
from policies.learning import Learner, LearnerConfig, RolloutConfig
from policies.models import Model, State, models
from policies.spec import PolicySpec
from policies.training.loop import RunConfig, Trainer, TrainerConfig
from tests.envs import Choose, Reach
from tests.learning.test_learner import CHOOSE, REACH


class FlatConfig(Config):
    hidden: int = 32


@models.register("tests.flat", FlatConfig)
class Flat(Model):
    """Every part flattened into one hidden layer, then a linear map per output."""

    def __init__(
        self,
        config: FlatConfig,
        in_shapes: Mapping[str, tuple[int, ...]],
        out_sizes: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.shapes = dict(in_shapes)
        size = sum(int(torch.Size(shape).numel()) for shape in in_shapes.values())
        self.body = nn.Sequential(nn.Linear(size, config.hidden), nn.Tanh())
        self.outputs = nn.ModuleDict(
            {name: nn.Linear(config.hidden, out) for name, out in out_sizes.items()}
        )
        self.state_shapes = {}

    def forward(self, parts: Mapping[str, Tensor], state: State) -> tuple[dict[str, Tensor], State]:
        flat = [parts[name].flatten(parts[name].dim() - len(s)) for name, s in self.shapes.items()]
        x = self.body(torch.cat(flat, dim=-1))
        return {name: head(x) for name, head in self.outputs.items()}, state


def test_a_custom_architecture_trains_and_exports(tmp_path: Path) -> None:
    torch.manual_seed(0)
    spec = PolicySpec.model_validate({**CHOOSE.model_dump(), "model": {"type": "tests.flat"}})
    config = RunConfig(
        policy=spec,
        learner=LearnerConfig(
            critic_model={"type": "tests.flat", "hidden": 8},
            optimizer={"type": "adam", "lr": 1e-2},
            rollout=RolloutConfig(length=8, gamma=0.0),
        ),
        trainer=TrainerConfig(iterations=30, hooks=()),
    )
    trainer = Trainer(config, Choose(64), tmp_path / "run")
    learner = trainer.run()
    assert isinstance(learner.actor.model, Flat)
    assert learner.critic is not None and isinstance(learner.critic.model, Flat)
    assert trainer.metrics["reward_mean"] > 0.9
    trainer.export(tmp_path / "artifact")
    validate_artifact(tmp_path / "artifact")
    policy = Policy.load(tmp_path / "artifact")
    assert policy.spec.model == {"type": "tests.flat", "hidden": 32}


MIXTURE = {"type": "mixture", "experts": 2, "expert": {"type": "gru", "hidden_size": 8}}


def update_with_balance(weight: float) -> tuple[dict[str, float], dict[str, Tensor]]:
    """One update of a recurrent mixture. Returns the metrics and the gate's parameters."""
    torch.manual_seed(0)
    spec = PolicySpec.model_validate({**REACH.model_dump(), "model": {"block": MIXTURE}})
    config = LearnerConfig(
        objectives={"ppo": {}, "kl_to_reference": {}},
        auxiliary={"balance": weight},
        rollout=RolloutConfig(length=8),
    )
    learner = Learner(spec, config, num_envs=8)
    env = Reach(8)
    observation = env.reset()
    while not learner.ready:
        observation, reward, done, info = env.step(learner.act(observation))
        learner.observe(observation, reward, done, info["truncated"])
    metrics = learner.update()
    gate = {
        name: value.detach().clone()
        for name, value in learner.actor.model.named_parameters()
        if name.startswith("block.gate")
    }
    return metrics, gate


def test_the_balance_loss_is_reported_and_trains_the_gate() -> None:
    unweighted, before = update_with_balance(0.0)
    weighted, after = update_with_balance(10.0)
    # Only the actor records: not the reference copy, and the critic has no mixture.
    assert {name for name in weighted if name.startswith("auxiliary/")} == {
        "auxiliary/actor.block.balance"
    }
    assert 1.0 <= weighted["auxiliary/actor.block.balance"] <= 2.0
    # Weighted, the gate moves toward balance, which is 1.
    assert weighted["auxiliary/actor.block.balance"] < unweighted["auxiliary/actor.block.balance"]
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_fit_reports_the_balance_loss() -> None:
    block = {"type": "mixture", "experts": 3, "top_k": 1}
    spec = PolicySpec.model_validate(
        {
            "name": "echo",
            "inputs": {"parts": [{"name": "x", "source": "x", "shape": [3]}]},
            "outputs": {"parts": [{"name": "y", "kind": {"type": "continuous", "size": 2}}]},
            "model": {"block": block},
        }
    )
    dataset = InMemoryDataset({"x": torch.randn(64, 3)}, {"y": torch.randn(64, 2)}, batch_size=16)
    metrics = Learner(spec, num_envs=1).fit(dataset)
    assert metrics["auxiliary/actor.block.balance"] >= 1.0
