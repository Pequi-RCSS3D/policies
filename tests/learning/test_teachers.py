from pathlib import Path
from typing import Any

import pytest
import torch

from policies.deploy.export import ExportSource, export_artifact
from policies.deploy.registry import ArtifactRegistry
from policies.learning import (
    Actor,
    ArtifactReference,
    ArtifactTeacher,
    Learner,
    LearnerConfig,
    RolloutConfig,
)
from policies.spec import PolicySpec
from tests.envs import Reach
from tests.learning.test_learner import REACH

BLOCKS: dict[str, dict[str, Any]] = {
    "mlp": {"type": "mlp", "hidden": [32, 32]},
    "gru": {"type": "gru", "hidden_size": 16},
}


def spec(block: str) -> PolicySpec:
    return PolicySpec.model_validate({**REACH.model_dump(), "model": {"block": BLOCKS[block]}})


def exported(block: str, directory: Path, gain: float = 1.0) -> Learner:
    """A learner with fitted normalization and weights scaled by `gain`, exported."""
    torch.manual_seed(1)
    learner = Learner(spec(block), num_envs=8)
    env = Reach(8, seed=1)
    observation = env.reset()
    for _ in range(5):
        outputs = learner.act(observation)
        observation, reward, done, info = env.step(outputs)
        learner.observe(observation, reward, done, info["truncated"])
    with torch.no_grad():
        for parameter in learner.actor.model.parameters():
            parameter.mul_(gain)
    export_artifact(learner, directory)
    return learner


@pytest.mark.parametrize("block", BLOCKS)
def test_artifact_teacher_matches_the_learner(block: str, tmp_path: Path) -> None:
    trained = exported(block, tmp_path)
    learner = Learner(spec(block), num_envs=8)
    learner.load_state_dict(trained.state_dict())
    learner.training = False
    teacher = ArtifactTeacher(tmp_path)
    env = Reach(8, horizon=10, seed=2)
    observation = env.reset()
    reset = torch.zeros(8, dtype=torch.bool)
    for _ in range(25):
        student = learner.act(observation)
        labels = learner.decode(teacher(observation, reset))
        torch.testing.assert_close(labels["force"], student["force"], atol=1e-5, rtol=1e-4)
        observation, reward, reset, info = env.step(student)
        learner.observe(observation, reward, reset, info["truncated"])


def test_distillation_from_an_artifact_reduces_loss(tmp_path: Path) -> None:
    exported("gru", tmp_path, gain=3.0)
    config = LearnerConfig(
        objectives={"distillation": {"teacher": "expert"}},
        optimizer={"type": "adam", "lr": 3e-3},
        rollout=RolloutConfig(length=16),
        teachers={"expert": str(tmp_path)},
    )
    learner = Learner(spec("mlp"), config, num_envs=32)
    env = Reach(32)
    observation = env.reset()
    losses = []
    for _ in range(30):
        while not learner.ready:
            outputs = learner.act(observation)
            observation, reward, done, info = env.step(outputs)
            learner.observe(observation, reward, done, info["truncated"])
        losses.append(learner.update()["distillation/loss"])
    assert losses[-1] < losses[0] * 0.5


def test_teacher_from_a_registry(tmp_path: Path) -> None:
    exported("mlp", tmp_path / "artifact")
    registry = ArtifactRegistry(tmp_path / "registry")
    manifest = registry.register(tmp_path / "artifact")
    registry.tag(manifest.id, "competition")
    reference = ArtifactReference(
        root=str(tmp_path / "registry"),
        name="reach",
        tag="competition",
        cache_dir=str(tmp_path / "cache"),
    )
    teacher = ArtifactTeacher.load(reference)
    observation = Reach(4).reset()
    outputs = teacher(observation, torch.ones(4, dtype=torch.bool))
    assert outputs["force"].shape == (4, 1)
    assert (tmp_path / "cache" / manifest.id / "policy.onnx").is_file()


def test_artifact_references_need_one_address() -> None:
    with pytest.raises(ValueError, match="either"):
        ArtifactReference(root="registry", id="a", name="b", tag="c")
    with pytest.raises(ValueError, match="either"):
        ArtifactReference(root="registry", name="b")


def test_teacher_names_must_not_clash(tmp_path: Path) -> None:
    config = LearnerConfig(teachers={"expert": str(tmp_path)})
    with pytest.raises(ValueError, match="expert"):
        Learner(REACH, config, num_envs=2, teachers={"expert": lambda o, r: o})


def test_distilled_students_record_their_teachers_as_parents(tmp_path: Path) -> None:
    coach_spec = PolicySpec.model_validate({**spec("mlp").model_dump(), "name": "coach"})
    expert = export_artifact(ExportSource(Actor(spec("mlp"))), tmp_path / "expert")
    coach = export_artifact(ExportSource(Actor(coach_spec)), tmp_path / "coach")
    config = LearnerConfig(
        objectives={"distillation": {"teacher": "expert"}},
        teachers={"expert": str(tmp_path / "expert"), "coach": str(tmp_path / "coach")},
    )
    student = Learner(spec("gru"), config, num_envs=2)
    assert student.parents == (expert.id, coach.id)
    student.act(Reach(2).reset())
    metrics = {"evaluation/return": 1.5}
    manifest = export_artifact(student, tmp_path / "student", metrics=metrics)
    assert manifest.parents == (expert.id, coach.id)
    assert manifest.metrics == metrics
    given = export_artifact(student, tmp_path / "given", parents=["given"])
    assert given.parents == ("given",)
    assert export_artifact(ExportSource(Actor(spec("mlp"))), tmp_path / "none").parents == ()
