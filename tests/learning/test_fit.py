from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from policies.data import InMemoryDataset
from policies.deploy.export import ExportSource, export_artifact
from policies.deploy.inference import Policy
from policies.learning import Learner, LearnerConfig
from policies.spec import PolicySpec


def spec(block: dict[str, Any]) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "name": "echo",
            "inputs": {"parts": [{"name": "x", "source": "x", "shape": [3], "normalize": True}]},
            "outputs": {"parts": [{"name": "y", "kind": {"type": "continuous", "size": 2}}]},
            "model": {"block": block},
        }
    )


def learner(block: dict[str, Any]) -> Learner:
    torch.manual_seed(0)
    config = LearnerConfig(optimizer={"type": "adam", "lr": 3e-3})
    return Learner(spec(block), config, num_envs=1)


def test_fit_clones_a_function() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 3, generator=generator) * 2 + 1
    y = torch.stack([torch.sin(x[:, 0]) * x[:, 1], x[:, 2] - x[:, 0]], -1)
    dataset = InMemoryDataset({"x": x}, {"y": y}, batch_size=256)
    model = learner({"type": "mlp", "hidden": [64, 64]})
    first = model.fit(dataset)["imitation/loss"]
    for _ in range(15):
        last = model.fit(dataset)["imitation/loss"]
    assert last < first * 0.2


def previous_input() -> InMemoryDataset:
    """Targets are the first two dims of the previous input, zero at an episode's start."""
    generator = torch.Generator().manual_seed(0)
    episodes, length = 256, 16
    x = torch.randn(episodes, length, 3, generator=generator)
    y = torch.cat([torch.zeros(episodes, 1, 2), x[:, :-1, :2]], 1)
    reset = torch.zeros(episodes, length, dtype=torch.bool)
    reset[:, 0] = True
    return InMemoryDataset(
        {"x": x.flatten(0, 1)},
        {"y": y.flatten(0, 1)},
        reset=reset.flatten(),
        batch_size=64,
        sequence_length=24,
    )


def test_fit_recurrent_sequences_needs_memory() -> None:
    dataset = previous_input()
    recurrent = learner({"type": "gru", "hidden_size": 32})
    feedforward = learner({"type": "mlp", "hidden": [32]})
    first = recurrent.fit(dataset)["imitation/loss"]
    for _ in range(40):
        last = recurrent.fit(dataset)["imitation/loss"]
        memoryless = feedforward.fit(dataset)["imitation/loss"]
    assert last < first * 0.2
    assert last < memoryless * 0.5


def test_recurrent_fit_needs_sequences() -> None:
    dataset = InMemoryDataset({"x": torch.zeros(8, 3)}, {"y": torch.zeros(8, 2)})
    with pytest.raises(ValueError, match="sequences"):
        learner({"type": "gru", "hidden_size": 8}).fit(dataset)


HISTORY = PolicySpec.model_validate(
    {
        "name": "history",
        "bookkeeping": [{"type": "history", "source": "x", "length": 3}],
        "inputs": {
            "parts": [
                {"name": "x", "source": "x", "shape": [3]},
                {"name": "past", "source": "history.x", "shape": [3, 3]},
            ]
        },
        "outputs": {"parts": [{"name": "y", "kind": {"type": "continuous", "size": 1}}]},
    }
)


def offline(spec: PolicySpec) -> Learner:
    return Learner(spec, LearnerConfig(objectives={}), num_envs=1)


def test_offline_learners_build_no_critic() -> None:
    model = offline(spec({"type": "mlp", "hidden": [8]}))
    assert model.critic is None
    trained = {id(parameter) for parameter in model.optimization.parameters}
    assert trained == {id(parameter) for parameter in model.actor.parameters()}


def test_fit_records_the_observations_export_needs(tmp_path: Path) -> None:
    x = torch.randn(64, 3)
    model = offline(HISTORY)
    model.fit(InMemoryDataset({"x": x, "past": x[:, None].expand(64, 3, 3)}, {"y": x[:, :1]}))
    assert model.observation_shapes["x"] == (3,)
    assert model.observation_dtypes["x"] == "float32"
    export_artifact(model, tmp_path)
    policy = Policy.load(tmp_path)
    assert policy.observation_shapes == {"x": (3,)}
    assert policy.step({"x": np.zeros(3, np.float32)})["y"].shape == (1,)


def test_export_refuses_what_fitting_cannot_know(tmp_path: Path) -> None:
    only_history = PolicySpec.model_validate(
        {**HISTORY.model_dump(), "inputs": {"parts": [HISTORY.inputs["past"].model_dump()]}}
    )
    model = offline(only_history)
    model.fit(InMemoryDataset({"past": torch.randn(16, 3, 3)}, {"y": torch.randn(16, 1)}))
    with pytest.raises(ValueError, match=r"unknown shape: \['x'\]"):
        export_artifact(model, tmp_path)
    export_artifact(ExportSource(model.actor, {"x": (3,)}), tmp_path)
    assert Policy.load(tmp_path).observation_keys == ("x",)
