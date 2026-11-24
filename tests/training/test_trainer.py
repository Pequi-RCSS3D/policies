import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from policies.core.config import Config
from policies.deploy.inference import Policy, validate_artifact
from policies.deploy.registry import ArtifactRegistry
from policies.learning import Actor, Learner, LearnerConfig, RolloutConfig
from policies.training.control import ControlMessage, Tunable, send_changes, send_command
from policies.training.envs import (
    FINAL_OBSERVATION,
    ActionDelay,
    DroppedCycles,
    Quantize,
    Step,
    Wrapper,
)
from policies.training.loop import AUTO, RunConfig, Trainer, TrainerConfig, resolve_device
from policies.training.monitoring import Event, EventKind, FunctionHook, hooks
from tests.envs import Reach
from tests.learning.test_learner import REACH
from tests.learning.test_objectives import Anchor
from tests.training.test_envs import Offset


def config(iterations: int) -> RunConfig:
    return RunConfig(
        policy=REACH,
        learner=LearnerConfig(
            optimizer={"type": "adam", "lr": 3e-3}, rollout=RolloutConfig(length=16)
        ),
        trainer=TrainerConfig(
            iterations=iterations,
            checkpoint_every=2,
            hooks=({"type": "jsonl"},),
            tunable=("optimizer.lr", "objectives.ppo.entropy_weight"),
        ),
    )


def events(run: Path) -> list[dict[str, object]]:
    lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_live_control_resume_export_and_inference(tmp_path: Path) -> None:
    run = tmp_path / "run"
    trainer = Trainer(config(4), Reach(16), run)
    send_changes(run, {"optimizer.lr": 1e-3})
    send_command(run, "explode")
    learner = trainer.run()
    assert learner.iteration == 4
    assert learner.lr == pytest.approx(1e-3)
    kinds = [event["kind"] for event in events(run)]
    assert "control" in kinds and "error" in kinds and kinds[-1] == "end"

    resumed = Trainer(config(6), Reach(16), run)
    assert resumed.run().iteration == 6
    assert resumed.agent.lr == pytest.approx(1e-3)
    assert len(resumed.checkpoints) == 3
    kinds = [event["kind"] for event in events(run)]
    assert kinds.count("control") == 1 and kinds.count("error") == 1

    manifest = resumed.export()
    validate_artifact(run / "export")
    policy = Policy.load(run / "export")
    assert policy.fingerprint == manifest.fingerprint
    observation = {"state": np.zeros(2, np.float32), "target": np.ones(1, np.float32)}
    first = policy.step(observation)["force"]
    assert first.shape == (1,)
    policy.reset()
    np.testing.assert_allclose(policy.step(observation)["force"], first)

    registry = ArtifactRegistry(tmp_path / "registry")
    registry.register(run / "export")
    registry.tag(manifest.id, "candidate")
    assert registry.tagged("reach", "candidate").id == manifest.id


def test_stop_command(tmp_path: Path) -> None:
    run = tmp_path / "run"
    trainer = Trainer(config(10), Reach(4), run)
    send_command(run, "stop")
    assert trainer.run().iteration == 0
    assert Trainer(config(2), Reach(4), run).run().iteration == 2


def test_device_defaults_to_auto() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert TrainerConfig(iterations=1).device == AUTO
    assert resolve_device(AUTO).type == expected
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValidationError, match="unknown device"):
        TrainerConfig(iterations=1, device="abacus")


def test_pause_applies_changes_until_resume(tmp_path: Path) -> None:
    trainer = Trainer(config(1), Reach(4), tmp_path)
    send_command(tmp_path, "pause")
    send_changes(tmp_path, {"optimizer.lr": 1e-3})
    send_command(tmp_path, "resume")
    learner = trainer.run()
    assert learner.iteration == 1
    assert learner.lr == pytest.approx(1e-3)


class Counter:
    """A fake agent: zero outputs, an update every `length` steps, one tunable."""

    def __init__(self, num_envs: int, length: int = 3) -> None:
        self.num_envs, self.length = num_envs, length
        self.iteration = 0
        self.device = torch.device("cpu")
        self.steps = 0
        self.values: dict[str, Tunable] = {"optimizer.lr": 0.1}
        self.truncated: list[bool] = []
        self.finals: list[Mapping[str, Tensor] | None] = []

    @property
    def ready(self) -> bool:
        return self.steps >= self.length

    def act(self, observation: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {"force": torch.zeros(self.num_envs, 1)}

    def observe(
        self,
        next_observation: Mapping[str, Tensor],
        rewards: Tensor,
        dones: Tensor,
        truncated: Tensor | None = None,
        final_observation: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.steps += 1
        self.truncated.append(truncated is not None)
        self.finals.append(final_observation)

    def update(self) -> dict[str, float]:
        self.steps = 0
        self.iteration += 1
        return {"loss": float(self.iteration)}

    def tunables(self) -> dict[str, Tunable]:
        return dict(self.values)

    def set(self, path: str, value: Tunable) -> None:
        self.values[path] = value

    def state_dict(self) -> dict[str, Any]:
        return {"iteration": self.iteration, "values": dict(self.values)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.iteration = state["iteration"]
        self.values = dict(state["values"])


class Scripted:
    """A control source that hands out one batch of messages per poll."""

    def __init__(self, *batches: list[ControlMessage]) -> None:
        self.batches = list(batches)

    def poll(self) -> list[ControlMessage]:
        return self.batches.pop(0) if self.batches else []


def quiet(iterations: int, **trainer: Any) -> RunConfig:
    return RunConfig(
        policy=REACH, trainer=TrainerConfig(iterations=iterations, **{"hooks": (), **trainer})
    )


def test_an_injected_agent_drives_the_trainer(tmp_path: Path) -> None:
    trainer = Trainer(quiet(4), Reach(2), tmp_path, agent=Counter(2))
    seen: list[Event] = []
    trainer.hooks.add("record", seen.append)
    agent = trainer.run()
    assert agent is trainer.agent and agent.iteration == 4
    assert all(agent.truncated)
    updates = [event for event in seen if event.kind == EventKind.UPDATE]
    assert [(event.step, event.data) for event in updates] == [
        (step, {"loss": float(step)}) for step in range(1, 5)
    ]
    kinds = [event.kind for event in seen]
    assert kinds.index(EventKind.UPDATE) + 1 == kinds.index(EventKind.ITERATION)
    assert trainer.metrics["loss"] == 4.0 and "seconds" in trainer.metrics
    assert "loss" not in next(e for e in seen if e.kind == EventKind.ITERATION).data

    again = Trainer(quiet(6), Reach(2), tmp_path, agent=Counter(2))
    assert again.run().iteration == 6
    with pytest.raises(TypeError, match="needs an agent with an actor"):
        again.export()


def test_an_injected_control_source(tmp_path: Path) -> None:
    bumped: list[int] = []
    control = Scripted(
        [ControlMessage(1, {"optimizer.lr": 0.5}), ControlMessage(2, command="bump")],
        [ControlMessage(3, command="explode")],
        [ControlMessage(4, command="stop")],
    )
    trainer = Trainer(quiet(10), Reach(2), tmp_path, agent=Counter(2), control=control)
    trainer.commands["bump"] = lambda: bumped.append(trainer.agent.iteration)
    seen: list[Event] = []
    trainer.hooks.add("record", seen.append)
    agent = trainer.run()
    assert agent.iteration == 2
    assert agent.values["optimizer.lr"] == 0.5
    assert bumped == [0]
    [error] = [event for event in seen if event.kind == EventKind.ERROR]
    assert "explode" in error.data["control"] and "bump" in error.data["control"]
    assert not (tmp_path / "control.jsonl").exists()


def test_wrappers_come_from_the_config_in_order(tmp_path: Path) -> None:
    wrappers = (
        {"type": "quantize", "steps": {"state": 0.1}, "rounding": "ceil"},
        {"type": "tests.offset", "amount": 1.0},
        {"type": "dropped_cycles", "rate": 0.1},
        {"type": "action_delay", "cycles": 2},
    )
    trainer = Trainer(quiet(1, wrappers=wrappers), Reach(2), tmp_path, agent=Counter(2))
    chain, env = [], trainer.env
    while isinstance(env, Wrapper):
        chain.append(type(env))
        env = env.env
    assert chain == [ActionDelay, DroppedCycles, Offset, Quantize]
    assert isinstance(env, Reach)


class ContextConfig(Config):
    pass


contexts: list[dict[str, Any]] = []


@hooks.register("tests.context", ContextConfig)
def context_hook(config: ContextConfig, **context: Any) -> FunctionHook:
    contexts.append(context)
    return FunctionHook(lambda event: None)


def test_hooks_get_the_trainer_and_spec(tmp_path: Path) -> None:
    config = quiet(1, hooks=({"type": "tests.context"},))
    trainer = Trainer(config, Reach(2), tmp_path, agent=Counter(2))
    context = contexts[-1]
    assert context["trainer"] is trainer
    assert context["spec"] == config.policy
    assert context["run_directory"] == tmp_path


def test_evaluation_events(tmp_path: Path) -> None:
    trainer = Trainer(quiet(5), Reach(2), tmp_path, agent=Counter(2))
    trainer.add_evaluation(lambda trainer: {"score": trainer.agent.iteration * 10}, every=2)
    seen: list[Event] = []
    trainer.hooks.add("record", seen.append)
    trainer.run()
    evaluations = [(e.step, e.data) for e in seen if e.kind == EventKind.EVALUATION]
    assert evaluations == [(2, {"score": 20}), (4, {"score": 40})]


def test_bounds_must_name_tunable_paths() -> None:
    with pytest.raises(ValidationError, match=r"not tunable: optimizer\.momentum"):
        TrainerConfig(iterations=1, bounds={"optimizer.momentum": (0.0, 1.0)})


class Exporting(Counter):
    """A fake agent with what export needs."""

    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.actor = Actor(REACH)
        self.observation_shapes = {"state": (2,), "target": (1,)}
        self.observation_dtypes: dict[str, str] = {}


def test_any_exportable_agent_exports(tmp_path: Path) -> None:
    trainer = Trainer(quiet(1), Reach(2), tmp_path, agent=Exporting(2))
    trainer.run()
    manifest = trainer.export()
    validate_artifact(tmp_path / "export")
    assert manifest.run == {"seed": 0, "iteration": 1}


def test_exports_record_the_latest_evaluation(tmp_path: Path) -> None:
    trainer = Trainer(quiet(2), Reach(2), tmp_path, agent=Exporting(2))
    trainer.add_evaluation(
        lambda trainer: {"score": np.float32(trainer.agent.iteration), "best": "a", "won": True},
        every=1,
    )
    trainer.add_evaluation(lambda trainer: {"steps": 7}, every=2, name="steps")
    trainer.run()
    assert trainer.evaluation == {"score": 2.0, "best": "a", "won": True, "steps": 7}
    manifest = trainer.export(parents=["teacher"])
    assert manifest.parents == ("teacher",)
    assert manifest.metrics == {
        **trainer.metrics,
        "evaluation/score": 2.0,
        "evaluation/steps": 7.0,
    }


def test_learners_load_trainer_checkpoints(tmp_path: Path) -> None:
    trainer = Trainer(config(1), Reach(2), tmp_path)
    trainer.run()
    assert Learner.from_checkpoint(trainer.checkpoints[-1]).iteration == 1


class Finishing(Reach):
    """Reach that reports the observation that ended each episode."""

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        before = self._observation()
        observation, rewards, dones, info = super().step(outputs)
        final = {
            name: torch.where(dones[:, None], before[name], v) for name, v in observation.items()
        }
        return observation, rewards, dones, {**info, FINAL_OBSERVATION: final}


def test_the_final_observation_reaches_the_agent(tmp_path: Path) -> None:
    agent = Counter(2)
    Trainer(quiet(1), Finishing(2), tmp_path, agent=agent).run()
    assert all(final is not None and set(final) == {"state", "target"} for final in agent.finals)


def test_custom_commands_and_non_float_tunables_through_the_run_directory(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        policy=REACH,
        learner=LearnerConfig(
            objectives={"ppo": {}, "anchor": {"type": "tests.anchor"}},
            rollout=RolloutConfig(length=4),
        ),
        trainer=TrainerConfig(
            iterations=2,
            hooks=(),
            tunable=("objectives.anchor.steps", "objectives.anchor.enabled"),
        ),
    )
    trainer = Trainer(config, Reach(2), tmp_path)
    send_changes(tmp_path, {"objectives.anchor.steps": 5, "objectives.anchor.enabled": False})
    send_command(tmp_path, "bump")
    bumped: list[int] = []
    trainer.commands["bump"] = lambda: bumped.append(trainer.agent.iteration)
    learner = trainer.run()
    anchor = learner.objectives["anchor"]
    assert isinstance(anchor, Anchor)
    hyperparameters = anchor.hyperparameters
    assert hyperparameters["steps"] == 5 and type(hyperparameters["steps"]) is int
    assert hyperparameters["enabled"] is False
    assert bumped == [0]


def recorded(trainer: Trainer[Counter]) -> list[Event]:
    seen: list[Event] = []
    trainer.hooks.add("record", seen.append)
    return seen


def test_resuming_applies_only_messages_newer_than_the_checkpoint(tmp_path: Path) -> None:
    trainer = Trainer(quiet(2), Reach(2), tmp_path, agent=Counter(2))
    send_changes(tmp_path, {"optimizer.lr": 0.5})
    send_command(tmp_path, "checkpoint")
    send_command(tmp_path, "pause")
    send_command(tmp_path, "resume")
    trainer.run()
    assert trainer.last_message == 4
    # Sent while the run is down.
    send_changes(tmp_path, {"optimizer.lr": 0.7})

    resumed = Trainer(quiet(4), Reach(2), tmp_path, agent=Counter(2))
    seen = recorded(resumed)
    agent = resumed.run()
    assert agent.iteration == 4 and agent.values["optimizer.lr"] == 0.7
    controls = [event.data for event in seen if event.kind == EventKind.CONTROL]
    assert controls == [{"changes": {"optimizer.lr": [0.5, 0.7]}}]
    assert resumed.last_message == 5


def test_a_stop_after_a_checkpoint_is_saved_and_not_replayed(tmp_path: Path) -> None:
    trainer = Trainer(quiet(10, checkpoint_every=1), Reach(2), tmp_path, agent=Counter(2))

    def stop_at_two(event: Event) -> None:
        if event.kind == EventKind.ITERATION and event.step == 2:
            send_command(tmp_path, "stop")

    trainer.hooks.add("stop", stop_at_two)
    assert trainer.run().iteration == 2
    assert Trainer(quiet(3), Reach(2), tmp_path, agent=Counter(2)).run().iteration == 3


def test_load_and_run_from_a_given_checkpoint(tmp_path: Path) -> None:
    first = tmp_path / "first"
    trainer = Trainer(quiet(3, checkpoint_every=1), Reach(2), first, agent=Counter(2))
    assert not trainer.load()
    send_changes(first, {"optimizer.lr": 0.5})
    trainer.run()
    one, two, three = trainer.checkpoints

    again = Trainer(quiet(3), Reach(2), first, agent=Counter(2))
    assert again.load(one) and again.agent.iteration == 1 and again.last_message == 1
    seen = recorded(again)
    again.run(checkpoint=two)
    assert [event.step for event in seen if event.kind == EventKind.UPDATE] == [3]
    assert not any(event.kind == EventKind.CONTROL for event in seen)

    # Another run's checkpoint: its message ids say nothing about this run's control file.
    second = tmp_path / "second"
    other = Trainer(quiet(4), Reach(2), second, agent=Counter(2))
    send_changes(second, {"optimizer.lr": 0.25})
    agent = other.run(checkpoint=three)
    assert agent.iteration == 4 and agent.values["optimizer.lr"] == 0.25
    assert [path.name for path in other.checkpoints] == ["iteration_0000004.pt"]
    with pytest.raises(FileNotFoundError):
        other.load(second / "missing.pt")
