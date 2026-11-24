"""The training loop: rollouts, updates, checkpoints, hooks and live control."""

import random
import time
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Self, cast, overload

import numpy as np
import torch

from policies.core.config import dump_yaml, load_yaml
from policies.deploy.export import export_artifact
from policies.deploy.registry import Manifest
from policies.learning import Learner, Teacher
from policies.learning.learner import CHECKPOINT_AGENT
from policies.training.control import (
    RUN_CONFIG,
    Command,
    ControlError,
    ControlMessage,
    ControlSource,
    FileControl,
    Tunables,
)
from policies.training.envs import FINAL_OBSERVATION, TRUNCATED, Environment, wrap
from policies.training.loop.agent import Agent
from policies.training.loop.config import RunConfig
from policies.training.loop.device import resolve_device
from policies.training.loop.exportable import ExportableAgent
from policies.training.monitoring import Event, EventKind, build_hooks

PAUSE_POLL_SECONDS = 0.5


class Trainer[A: Agent]:
    """Runs an agent in an environment. Builds a `Learner` from the config unless given one."""

    @overload
    def __init__(
        self: "Trainer[Learner]",
        config: RunConfig,
        env: Environment,
        run_directory: str | Path,
        *,
        teachers: Mapping[str, Teacher] | None = None,
        control: ControlSource | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        config: RunConfig,
        env: Environment,
        run_directory: str | Path,
        *,
        agent: A,
        control: ControlSource | None = None,
    ) -> None: ...

    def __init__(
        self,
        config: RunConfig,
        env: Environment,
        run_directory: str | Path,
        *,
        agent: A | None = None,
        teachers: Mapping[str, Teacher] | None = None,
        control: ControlSource | None = None,
    ) -> None:
        """`config.learner` and `teachers` only apply to the default `Learner`.

        `control` defaults to `FileControl` on the run directory.
        """
        self.config = config
        self.directory = Path(run_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        dump_yaml(config, self.directory / RUN_CONFIG)
        self.env = wrap(env, config.trainer.wrappers, seed=config.trainer.seed)
        if agent is None:
            learner = Learner(
                config.policy,
                config.learner,
                num_envs=self.env.num_envs,
                device=resolve_device(config.trainer.device),
                teachers=teachers,
            )
            # The overloads only allow this branch for `Trainer[Learner]`.
            agent = cast("A", learner)
        self.agent: A = agent
        self.control = control if control is not None else FileControl(self.directory)
        self.commands: dict[str, Callable[[], object]] = {
            Command.CHECKPOINT: self.save,
            Command.STOP: self.stop,
            Command.PAUSE: self.pause,
            Command.RESUME: self.resume,
        }
        """Control commands by name. Add your own."""
        self.last_message = 0
        """Id of the last control message handled, saved with each checkpoint."""
        self._saved: tuple[int, int] | None = None
        self.metrics: dict[str, float] = {}
        self.evaluation: dict[str, Any] = {}
        """The latest value of each key evaluations returned."""
        self.stopped = False
        self.paused = False
        self.hooks = build_hooks(
            config.trainer.hooks, self.directory, trainer=self, spec=config.policy
        )

    @staticmethod
    def from_yaml(
        path: str | Path,
        env: Environment,
        run_directory: str | Path,
        *,
        teachers: Mapping[str, Teacher] | None = None,
        control: ControlSource | None = None,
    ) -> "Trainer[Learner]":
        return Trainer(
            load_yaml(path, RunConfig), env, run_directory, teachers=teachers, control=control
        )

    # Events

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        """Send an event to every hook, at the current iteration."""
        self.hooks.emit(Event(kind, self.agent.iteration, data))

    def add_evaluation(
        self, evaluate: Callable[[Self], Mapping[str, Any]], every: int, name: str = "evaluation"
    ) -> None:
        """Every `every` iterations, emit `evaluate(self)` as an evaluation event.

        The results are kept in `evaluation`, and `export` records their numbers.
        """

        def hook(event: Event) -> None:
            if event.kind == EventKind.ITERATION:
                result = dict(evaluate(self))
                self.evaluation.update(result)
                self.emit(EventKind.EVALUATION, result)

        self.hooks.add(name, hook, every)

    # Checkpoints

    @property
    def checkpoints(self) -> list[Path]:
        return sorted((self.directory / "checkpoints").glob("iteration_*.pt"))

    def _checkpoint_path(self) -> Path:
        return self.directory / "checkpoints" / f"iteration_{self.agent.iteration:07d}.pt"

    def save(self) -> Path:
        """Checkpoint the agent, the random generators and `last_message`."""
        path = self._checkpoint_path()
        path.parent.mkdir(exist_ok=True)
        state = {
            CHECKPOINT_AGENT: self.agent.state_dict(),
            "control": self.last_message,
            "random": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(state, path)
        self._saved = (self.agent.iteration, self.last_message)
        for old in self.checkpoints[: -self.config.trainer.keep_checkpoints]:
            old.unlink()
        self.emit(EventKind.CHECKPOINT, {"path": str(path)})
        return path

    def load(self, path: str | Path | None = None) -> bool:
        """Load `path`, else the latest checkpoint of this run. False when there is none.

        Values in the checkpoint, such as an `optimizer.lr` tuned live, override the config. To
        change one on resume, send it with `send_changes` or `pm set`. A checkpoint of this run
        also restores `last_message`, so messages it already handled are not replayed. One from
        elsewhere starts this run's `control.jsonl` from its first message.
        """
        if path is None:
            if not self.checkpoints:
                return False
            path = self.checkpoints[-1]
        path = Path(path)
        state = torch.load(path, map_location=self.agent.device, weights_only=False)
        self.agent.load_state_dict(state[CHECKPOINT_AGENT])
        if path.resolve().parent == (self.directory / "checkpoints").resolve():
            self.last_message = state["control"]
            self._saved = (self.agent.iteration, self.last_message)
        # Generator states must be CPU byte tensors, whatever `map_location` did.
        torch.set_rng_state(state["random"]["torch"].cpu())
        cuda = [rng.cpu() for rng in state["random"].get("cuda", [])]
        if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(state["random"]["numpy"])
        random.setstate(state["random"]["python"])
        return True

    # Control

    def stop(self) -> None:
        self.stopped = True

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def _apply_control(self) -> bool:
        """Apply pending control messages, waiting while paused. Returns True to stop."""
        self._handle(self.control.poll())
        while self.paused and not self.stopped:
            time.sleep(PAUSE_POLL_SECONDS)
            self._handle(self.control.poll())
        return self.stopped

    def _handle(self, messages: list[ControlMessage]) -> None:
        for message in messages:
            if message.id <= self.last_message:
                continue
            # Before acting, so a checkpoint the message asks for records it as handled.
            self.last_message = message.id
            if message.changes:
                self._change(message.changes)
            if message.command:
                self._command(message.command)

    def _change(self, changes: Mapping[str, Any]) -> None:
        tunable = self.config.trainer.tunable
        allowed = {k: v for k, v in self.agent.tunables().items() if k in tunable}
        bounds = {k: v for k, v in self.config.trainer.bounds.items() if k in allowed}
        try:
            applied = Tunables(allowed, bounds).apply(dict(changes))
        except ControlError as error:
            self.emit(EventKind.ERROR, {"control": str(error)})
            return
        for path, (_, new) in applied.items():
            self.agent.set(path, new)
        self.emit(EventKind.CONTROL, {"changes": {p: list(pair) for p, pair in applied.items()}})

    def _command(self, command: str) -> None:
        action = self.commands.get(command)
        if action is None:
            known = ", ".join(self.commands)
            self.emit(EventKind.ERROR, {"control": f"unknown command {command!r}. Known: {known}"})
            return
        self.emit(EventKind.CONTROL, {"command": command})
        action()

    # Running

    def run(self, checkpoint: str | Path | None = None) -> A:
        """Train until `trainer.iterations` or a stop command.

        Resumes from `checkpoint`, else from this run's latest. See `load` for what it restores.
        Control messages newer than the checkpoint apply at the first iteration, including those
        sent while the run was down: a `stop` sent then stops it again at once, and a `pause` not
        followed by a `resume` pauses it.
        """
        trainer = self.config.trainer
        resumed = self.load(checkpoint)
        if not resumed:
            torch.manual_seed(trainer.seed)
            np.random.seed(trainer.seed)
            random.seed(trainer.seed)
        self.emit(EventKind.START, {"resumed": resumed})
        observation = self.env.reset()
        # Outputs go back to the device the environment observes on.
        env_device = next(iter(observation.values())).device
        device = self.agent.device
        returns = torch.zeros(self.env.num_envs, device=device)
        try:
            while self.agent.iteration < trainer.iterations and not self.stopped:
                if self._apply_control():
                    break
                # Episode accounting stays on the device: one sync per iteration.
                finished_sum = torch.zeros((), device=device)
                finished_count = torch.zeros((), device=device)
                started = time.perf_counter()
                while not self.agent.ready:
                    outputs = self.agent.act(observation)
                    outputs = {name: value.to(env_device) for name, value in outputs.items()}
                    observation, rewards, dones, info = self.env.step(outputs)
                    # Moved once here, so the agent's own moves are no-ops.
                    rewards, dones = rewards.to(device).float(), dones.to(device).bool()
                    truncated = info.get(TRUNCATED)
                    if truncated is not None:
                        truncated = truncated.to(device)
                    final = info.get(FINAL_OBSERVATION)
                    if final is not None:
                        final = {name: value.to(device) for name, value in final.items()}
                    self.agent.observe(observation, rewards, dones, truncated, final)
                    returns += rewards
                    finished_sum += (returns * dones).sum()
                    finished_count += dones.sum()
                    returns *= ~dones
                update = dict(self.agent.update())
                self.emit(EventKind.UPDATE, update)
                iteration = {"seconds": time.perf_counter() - started}
                count = float(finished_count)
                if count:
                    iteration["episode_return"] = float(finished_sum) / count
                self.metrics = {**update, **iteration}
                self.emit(EventKind.ITERATION, iteration)
                if self.agent.iteration % trainer.checkpoint_every == 0:
                    self.save()
            if self._saved != (self.agent.iteration, self.last_message):
                self.save()
        except Exception as error:
            self.emit(EventKind.ERROR, {"error": repr(error)})
            raise
        finally:
            self.emit(EventKind.END, {"stopped": self.stopped})
            self.hooks.close()
        return self.agent

    def export(
        self, directory: str | Path | None = None, parents: Sequence[str] | None = None
    ) -> Manifest:
        """Export the current policy as an artifact directory, ready to register.

        The agent must be an `ExportableAgent`. The manifest records the latest `metrics` and
        each number in `evaluation` as `evaluation/<key>`. `parents` default to the agent's.
        """
        agent: object = self.agent
        if not isinstance(agent, ExportableAgent):
            raise TypeError(
                f"export needs an agent with an actor and its observations, "
                f"got {type(agent).__name__}"
            )
        run = {"seed": self.config.trainer.seed, "iteration": self.agent.iteration}
        evaluation = {
            f"{EventKind.EVALUATION}/{key}": float(value)
            for key, value in self.evaluation.items()
            if not isinstance(value, bool) and isinstance(value, Real)
        }
        return export_artifact(
            agent,
            directory or self.directory / "export",
            metrics={**self.metrics, **evaluation},
            parents=parents,
            run=run,
        )
