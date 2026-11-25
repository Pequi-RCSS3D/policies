"""The learner: acts in batched environments, records rollouts and updates the policy.

It also fits recorded data offline.
"""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import array_api_compat.torch as torch_xp
import torch
from torch import Tensor

from policies.core.config import canonical
from policies.data import Batch, Dataset
from policies.io.bookkeeping import Compose, build_bookkeeping
from policies.learning.actor import Actor
from policies.learning.advantages import estimate_advantages
from policies.learning.auxiliary import Auxiliary
from policies.learning.config import LearnerConfig
from policies.learning.critic import Critic
from policies.learning.fit import fit
from policies.learning.objectives import ActContext, Objective, objectives
from policies.learning.observed import copied_observations, observed
from policies.learning.optimization import Optimization
from policies.learning.rollout import Rollout
from policies.learning.sequences import mask
from policies.learning.teachers import ArtifactTeacher, Teacher
from policies.learning.tunables import LearnerTunables
from policies.learning.update import RECORDED, Updater
from policies.models import State
from policies.spec import PolicySpec
from policies.training.control import Tunable

CHECKPOINT_AGENT = "agent"
"""Key of the learner's state in a trainer checkpoint."""


def _prefixed(prefix: str, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {f"{prefix}/{name}": value for name, value in values.items()}


class Learner:
    """Owns sampling, the rollout, objectives and optimization. Environments only see outputs.

    The critic exists only if an objective `needs_values`. With `training` False, `act` is
    deterministic: it takes the mode of each part's distribution, records nothing and leaves
    normalization fixed.
    """

    def __init__(
        self,
        spec: PolicySpec,
        config: LearnerConfig | None = None,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        teachers: Mapping[str, Teacher] | None = None,
    ) -> None:
        self.spec, self.config = spec, config or LearnerConfig()
        self.num_envs, self.device = num_envs, torch.device(device)
        config = self.config
        clash = set(config.teachers) & set(teachers or {})
        if clash:
            raise ValueError(f"teachers given both in the config and as callables: {sorted(clash)}")
        self.teachers: dict[str, Teacher] = {
            **{name: ArtifactTeacher.load(source) for name, source in config.teachers.items()},
            **(teachers or {}),
        }
        self.objectives: dict[str, Objective] = {
            name: objectives.build(settings, teachers=self.teachers, device=self.device)
            for name, settings in config.objectives.items()
        }
        self.actor_inputs = spec.inputs.compile()
        self.actor = Actor(spec, config.initial_std, config.normalizer_clip).to(self.device)
        self.critic: Critic | None = None
        if any(objective.needs_values for objective in self.objectives.values()):
            critic = Critic(
                config.critic_inputs or spec.inputs, config.critic_model, config.normalizer_clip
            )
            self.critic = critic.to(self.device)
        models = {"actor": self.actor.model}
        trained = [*self.actor.parameters()]
        if self.critic is not None:
            models["critic"] = self.critic.model
            trained += self.critic.parameters()
        for objective in self.objectives.values():
            trained += objective.parameters()
        types = {name: settings["type"] for name, settings in config.objectives.items()}
        self.optimization = Optimization(
            trained, config.optimizer, config.schedule, config.max_grad_norm, objectives=types
        )
        self.auxiliary = Auxiliary(models, config.auxiliary)
        self.updater = Updater(
            self.actor,
            self.critic,
            self.objectives,
            self.auxiliary,
            self.optimization,
            config.rollout,
            self.device,
        )
        self.tuning = LearnerTunables(self.optimization, self.objectives)
        self.rollout = Rollout(config.rollout.length, num_envs)
        self.reference: Actor | None = None
        """A frozen copy of the actor, taken at the first update if an objective needs it."""
        self.iteration = 0
        self.bookkeeping: Compose | None = None
        self.observation_shapes: dict[str, tuple[int, ...]] = {}
        self.observation_dtypes: dict[str, str] = {}
        self.actor_state = self.actor.model.initial_state((num_envs,), self.device)
        self.critic_state: State = (
            {} if self.critic is None else self.critic.model.initial_state((num_envs,), self.device)
        )
        self.noise = self.actor.head.noise(num_envs, self.device)
        """Exploration noise state per output part that has one."""
        self.reset = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        """Environments whose episode starts with the next observation."""
        self.next_observation: Mapping[str, Tensor] | None = None
        self.training = True
        """False to act deterministically, as deployed."""

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        teachers: Mapping[str, Teacher] | None = None,
    ) -> "Learner":
        """A learner rebuilt from a trainer checkpoint or a saved `state_dict()`, not training.

        The spec and config come from the checkpoint. `teachers` are callables its objectives
        need. Checkpoints are pickles: load only trusted files.
        """
        saved = torch.load(path, map_location=device, weights_only=False)
        state = saved.get(CHECKPOINT_AGENT, saved)
        if "spec" not in state:
            raise ValueError(f"{path} holds no spec. It was saved before learners recorded one.")
        spec = PolicySpec.model_validate(state["spec"])
        config = LearnerConfig.model_validate(state["config"])
        learner = cls(spec, config, num_envs=num_envs, device=device, teachers=teachers)
        learner.load_state_dict(state)
        learner.training = False
        return learner

    @property
    def parents(self) -> tuple[str, ...]:
        """Ids of the artifacts its teachers run, recorded as an export's `parents`."""
        ids = (getattr(teacher, "artifact_id", None) for teacher in self.teachers.values())
        return tuple(id for id in ids if id is not None)

    @property
    def lr(self) -> float:
        return self.optimization.lr

    @property
    def recurrent(self) -> bool:
        """Recurrent models train on whole environment sequences, others on shuffled steps."""
        return self.updater.recurrent

    @property
    def ready(self) -> bool:
        """True once a full rollout is recorded."""
        return self.rollout.ready

    # Acting

    def _slots(self, observation: Mapping[str, Tensor], update: bool) -> dict[str, Tensor]:
        if self.bookkeeping is None:
            self.observation_shapes, self.observation_dtypes = observed(observation)
            decoder = self.actor.decoder
            self.bookkeeping = build_bookkeeping(
                self.spec.bookkeeping,
                shapes={**self.observation_shapes, **decoder.shapes},
                dtypes={**self.observation_dtypes, **decoder.dtypes},
                batch=(self.num_envs,),
                xp=torch_xp,
                device=self.device,
            )
        return self.bookkeeping.observe(observation, update)

    def _peek_values(self, observation: Mapping[str, Tensor]) -> Tensor:
        """Values in the current critic and bookkeeping state, which stay unchanged."""
        assert self.critic is not None
        parts = self.critic.inputs(observation, self._slots(observation, update=False))
        values: Tensor
        values, _ = self.critic(parts, self.critic_state)
        return values

    def decode(self, actions: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Actions to output values."""
        return self.actor.decoder.decode(actions)

    @torch.no_grad()
    def act(self, observation: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Outputs for every environment: sampled and recorded while `training`, else the mode."""
        observation = {k: v.to(self.device) for k, v in observation.items()}
        slots = self._slots(observation, update=True)
        parts = self.actor_inputs(observation, slots)
        if self.training:
            self.actor.normalizer.update(parts)
        distribution, actor_state = self.actor(parts, self.actor_state)
        actions = distribution.sample(self.noise) if self.training else distribution.mode()
        if self.training:
            context = ActContext(observation, actions, self.reset)
            self.rollout.record(
                _prefixed("actor", parts)
                | _prefixed("actions", actions)
                | _prefixed("actor_state", self.actor_state)
                | {"log_prob": distribution.log_prob(actions)}
            )
            for name, objective in self.objectives.items():
                self.rollout.record(_prefixed(f"{RECORDED}/{name}", objective.record(context)))
        if self.critic is not None:
            critic_parts = self.critic.inputs(observation, slots)
            if self.training:
                self.critic.normalizer.update(critic_parts)
            values, critic_state = self.critic(critic_parts, self.critic_state)
            if self.training:
                self.rollout.record(
                    _prefixed("critic", critic_parts)
                    | _prefixed("critic_state", self.critic_state)
                    | {"values": values}
                )
            self.critic_state = critic_state
        self.actor_state = actor_state
        assert self.bookkeeping is not None
        self.bookkeeping.commit(actions)
        return self.decode(actions)

    @torch.no_grad()
    def observe(
        self,
        next_observation: Mapping[str, Tensor],
        rewards: Tensor,
        dones: Tensor,
        truncated: Tensor | None = None,
        final_observation: Mapping[str, Tensor] | None = None,
    ) -> None:
        """Record the result of the last action. Done environments restart their state.

        Truncated episodes bootstrap with the value of `final_observation`, the one that ended the
        episode, or else with the value of the last observation.
        """
        dones = dones.to(self.device).bool()
        rewards = rewards.to(self.device).float()
        if self.training:
            if truncated is not None and self.critic is not None:
                if final_observation is None:
                    bootstrap = self.rollout.pending["values"]
                else:
                    final = {k: v.to(self.device) for k, v in final_observation.items()}
                    # Before the reset, so the final observation continues the ended episode.
                    bootstrap = self._peek_values(final)
                cut = truncated.to(self.device).float()
                rewards = rewards + self.config.rollout.gamma * bootstrap * cut
            self.rollout.complete(rewards, dones)
        assert self.bookkeeping is not None
        self.bookkeeping.reset(dones)
        for noise in self.noise.values():
            noise.reset(dones)
        keep = (~dones).float()
        self.actor_state = {k: mask(v, keep) for k, v in self.actor_state.items()}
        self.critic_state = {k: mask(v, keep) for k, v in self.critic_state.items()}
        self.reset = dones
        self.next_observation = {k: v.to(self.device) for k, v in next_observation.items()}

    # Updating

    def update(self) -> dict[str, float]:
        """One learning update over the recorded rollout. Returns averaged metrics."""
        data = self.rollout.take()
        if self.critic is not None:
            assert self.next_observation is not None
            with torch.no_grad():
                last = self._peek_values(self.next_observation)
            rollout = self.config.rollout
            estimate_advantages(
                data, last, rollout.gamma, rollout.gae_lambda, rollout.normalize_advantages
            )
        if self.reference is None and any(o.needs_reference for o in self.objectives.values()):
            self.reference = copy.deepcopy(self.actor).requires_grad_(False)
        metrics = self.updater.run(data, self.reference).averages()
        self.iteration += 1
        self.optimization.schedule_step(self.iteration, metrics)
        metrics["lr"] = self.lr
        metrics["reward_mean"] = float(data["rewards"].mean())
        return metrics

    def fit(self, dataset: Dataset, epochs: int = 1) -> dict[str, float]:
        """Behavior cloning on recorded data through the imitation loss. Returns averages.

        Objectives are not used, so `LearnerConfig(objectives={})` avoids building a critic.
        Records the observations that untransformed input parts copy, for export.
        """
        return fit(
            self.actor,
            dataset,
            epochs,
            self.auxiliary,
            self.optimization,
            self.device,
            self.training,
            first=self._record_inputs,
        )

    def _record_inputs(self, batch: Batch) -> None:
        shapes, dtypes = copied_observations(self.spec.inputs, batch)
        # What acting observed is complete, so it wins.
        self.observation_shapes = {**shapes, **self.observation_shapes}
        self.observation_dtypes = {**dtypes, **self.observation_dtypes}

    # Tuning and persistence

    def tunables(self) -> dict[str, Tunable]:
        """Values that may change while running, by dotted path."""
        return self.tuning.values()

    def set(self, path: str, value: Tunable) -> None:
        self.tuning.set(path, value)

    def state_dict(self) -> dict[str, Any]:
        """Everything to resume, with the spec and config to rebuild from."""
        return {
            "spec": canonical(self.spec),
            "config": canonical(self.config),
            "actor": self.actor.state_dict(),
            "critic": None if self.critic is None else self.critic.state_dict(),
            "optimization": self.optimization.state_dict(),
            "objectives": {
                name: {
                    "hyperparameters": dict(objective.hyperparameters),
                    "parameters": [p.detach().clone() for p in objective.parameters()],
                }
                for name, objective in self.objectives.items()
            },
            "iteration": self.iteration,
            "observation_shapes": self.observation_shapes,
            "observation_dtypes": self.observation_dtypes,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        if self.critic is not None:
            self.critic.load_state_dict(state["critic"])
        self.optimization.load_state_dict(state["optimization"])
        for name, saved in state["objectives"].items():
            objective = self.objectives[name]
            objective.hyperparameters.update(saved["hyperparameters"])
            with torch.no_grad():
                for parameter, value in zip(
                    objective.parameters(), saved["parameters"], strict=True
                ):
                    parameter.copy_(value)
        self.iteration = state["iteration"]
        self.observation_shapes = {k: tuple(v) for k, v in state["observation_shapes"].items()}
        self.observation_dtypes = dict(state["observation_dtypes"])
