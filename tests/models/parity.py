"""Train a small reach policy, export it and check ONNX against PyTorch."""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from policies.deploy.export import export_artifact
from policies.deploy.inference import Policy, validate_artifact
from policies.learning import Learner, LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from tests.envs import Reach


def reach_spec(model: dict[str, Any]) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "name": "reach",
            "inputs": {
                "parts": [
                    {"name": "state", "source": "state", "shape": [2], "normalize": True},
                    {"name": "target", "source": "target", "shape": [1]},
                ]
            },
            "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 1}}]},
            "model": model,
        }
    )


def train_export_and_match(model: dict[str, Any], directory: Path) -> Learner:
    """Two updates, an export to `directory`, then five steps of PyTorch and ONNX in lockstep."""
    learner = Learner(reach_spec(model), LearnerConfig(rollout=RolloutConfig(length=8)), num_envs=8)
    env = Reach(8)
    observation = env.reset()
    for _ in range(2):
        while not learner.ready:
            outputs = learner.act(observation)
            observation, reward, done, info = env.step(outputs)
            learner.observe(observation, reward, done, info["truncated"])
        learner.update()
    export_artifact(learner, directory)
    validate_artifact(directory)
    policy = Policy.load(directory)
    actor = learner.actor.cpu().eval()
    state = actor.model.initial_state((1,))
    for _ in range(5):
        parts = {"state": torch.randn(1, 2), "target": torch.randn(1, 1)}
        raw, policy.state = policy._run({key: value[0].numpy() for key, value in parts.items()})
        with torch.no_grad():
            expected, state = actor.model(actor.normalizer(parts), state)
        np.testing.assert_allclose(raw["force"], expected["force"][0].numpy(), atol=1e-4)
    return learner


def exported_state(directory: Path) -> list[str]:
    """State names of the exported graph's inputs, without the `state.` prefix."""
    graph = onnx.load(str(directory / "policy.onnx")).graph
    prefix = "state."
    return [tensor.name[len(prefix) :] for tensor in graph.input if tensor.name.startswith(prefix)]
