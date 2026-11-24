"""End-to-end use as a training repo would: a joint-level policy and a macro-level policy."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from policies.deploy.inference import Policy, validate_artifact
from policies.learning import LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from policies.training.loop import RunConfig, Trainer, TrainerConfig

JOINTS = 12
BEND = torch.linspace(-1.0, 1.0, JOINTS)
"""How each joint moves as the commanded crouch goes from 0 to 1."""


class PoseTracking:
    """Joints follow position targets with lag. Reward: joints reach a commanded crouch."""

    def __init__(self, num_envs: int, seed: int = 0) -> None:
        self.num_envs = num_envs
        self.generator = torch.Generator().manual_seed(seed)
        self.position = torch.zeros(num_envs, JOINTS)
        self.velocity = torch.zeros(num_envs, JOINTS)
        self.crouch = torch.zeros(num_envs, 1)
        self.time = torch.randint(0, 50, (num_envs,), generator=self.generator).float()

    def _goal(self) -> Tensor:
        return self.crouch * 0.6 * BEND

    def _observation(self) -> dict[str, Tensor]:
        return {
            "joint_position": self.position.clone(),
            "joint_velocity": self.velocity.clone(),
            "command": self.crouch.clone(),
        }

    def reset(self) -> dict[str, Tensor]:
        self.crouch = torch.rand(self.num_envs, 1, generator=self.generator)
        return self._observation()

    def step(
        self, outputs: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Any]]:
        target = outputs["joints"]
        previous = self.position
        self.position = self.position + 0.3 * (target - self.position)
        self.velocity = (self.position - previous) / 0.02
        error = (self.position - self._goal()).abs().mean(-1)
        self.time += 1
        done = self.time >= 50
        self.time[done] = 0
        self.crouch[done] = torch.rand(int(done.sum()), 1, generator=self.generator)
        self.position[done] = 0
        return self._observation(), -error, done, {"truncated": done.clone()}


JOINT_POLICY = PolicySpec.model_validate(
    {
        "name": "crouch",
        "inputs": {
            "parts": [
                {
                    "name": "positions",
                    "source": "history.joint_position",
                    "shape": [3, JOINTS],
                    "normalize": True,
                },
                {
                    "name": "velocity",
                    "source": "joint_velocity",
                    "shape": [JOINTS],
                    "transforms": [{"type": "scale", "factor": 0.05}],
                },
                {"name": "command", "source": "command", "shape": [1]},
                {"name": "last_joints", "source": "last.joints", "shape": [JOINTS]},
            ]
        },
        "outputs": {
            "parts": [
                {
                    "name": "joints",
                    "kind": {"type": "continuous", "size": JOINTS, "low": -1.0, "high": 1.0},
                }
            ]
        },
        "bookkeeping": [
            {"type": "history", "source": "joint_position", "length": 3},
            {"type": "last", "source": "joints"},
        ],
        "model": {
            "encoders": {"positions": {"type": "mlp", "hidden": [64]}},
            "block": {"type": "mlp", "hidden": [128, 64]},
        },
    }
)


def test_joint_level_training_to_deployment(tmp_path: Path) -> None:
    config = RunConfig(
        policy=JOINT_POLICY,
        learner=LearnerConfig(
            optimizer={"type": "adam", "lr": 1e-3},
            initial_std=0.5,
            objectives={"ppo": {}, "action_smoothness": {"weight": 0.001}},
            rollout=RolloutConfig(length=24),
        ),
        trainer=TrainerConfig(iterations=60, checkpoint_every=30, hooks=({"type": "jsonl"},)),
    )
    trainer = Trainer(config, PoseTracking(64), tmp_path / "run")
    returns: list[float] = []
    trainer.hooks.add(
        "record",
        lambda event: (
            returns.append(event.data["episode_return"]) if "episode_return" in event.data else None
        ),
    )
    trainer.run()
    # Skip the first iterations: their episodes started part-way and are short.
    assert np.mean(returns[-5:]) > np.mean(returns[5:10]) + 1.0

    trainer.export(tmp_path / "artifact")
    validate_artifact(tmp_path / "artifact")
    policy = Policy.load(tmp_path / "artifact")
    observation = {
        "joint_position": np.zeros(JOINTS, np.float32),
        "joint_velocity": np.zeros(JOINTS, np.float32),
        "command": np.array([0.5], np.float32),
    }
    targets = policy.step(observation)["joints"]
    assert targets.shape == (JOINTS,)
    assert np.all(np.abs(targets) <= 1.0)


MACRO_POLICY = PolicySpec.model_validate(
    {
        "name": "macro",
        "inputs": {
            "parts": [
                {"name": "ball", "source": "ball", "shape": [2]},
                {"name": "since", "source": "timer.skill", "shape": [1]},
            ]
        },
        "outputs": {
            "parts": [
                {
                    "name": "skill",
                    "kind": {"type": "categorical", "choices": ["walk", "kick", "stand"]},
                },
                {"name": "goal", "kind": {"type": "continuous", "size": 3, "low": -1, "high": 1}},
            ]
        },
        "bookkeeping": [{"type": "timer", "name": "skill", "dt": 0.02}],
        "model": {"block": {"type": "gru", "hidden_size": 32}},
    }
)


class Pitch:
    """Kick when the ball is close, otherwise walk toward it."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.ball = self._place()

    def _place(self) -> Tensor:
        """Half the balls land within kicking distance."""
        angle = torch.rand(self.num_envs) * 2 * torch.pi
        radius = torch.rand(self.num_envs)
        return torch.stack([angle.cos(), angle.sin()], -1) * radius[:, None]

    def reset(self) -> dict[str, Tensor]:
        return {"ball": self.ball.clone()}

    def step(
        self, outputs: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Any]]:
        close = self.ball.norm(dim=-1) < 0.5
        right = torch.where(close, outputs["skill"] == 1, outputs["skill"] == 0).float()
        aim = -(outputs["goal"][:, :2] - self.ball).abs().sum(-1)
        self.ball = self._place()
        done = torch.ones(self.num_envs, dtype=torch.bool)
        return {"ball": self.ball.clone()}, right + 0.2 * aim, done, {}


def test_macro_level_training_to_deployment(tmp_path: Path) -> None:
    config = RunConfig(
        policy=MACRO_POLICY,
        learner=LearnerConfig(
            optimizer={"type": "adam", "lr": 3e-3}, rollout=RolloutConfig(length=8, gamma=0.0)
        ),
        trainer=TrainerConfig(iterations=80, hooks=()),
    )
    trainer = Trainer(config, Pitch(64), tmp_path / "run")
    trainer.run()
    assert trainer.metrics["reward_mean"] > 0.6
    trainer.export(tmp_path / "artifact")
    validate_artifact(tmp_path / "artifact")
    policy = Policy.load(tmp_path / "artifact")
    balls = Pitch(200)._place().numpy()
    close = np.linalg.norm(balls, axis=-1) < 0.5
    skills = []
    for ball in balls:
        policy.reset()
        out = policy.step({"ball": ball})
        skills.append(int(out["skill"]))
    assert out["goal"].shape == (3,)
    assert set(policy.state) == {"block.h"}
    correct = np.where(close, np.array(skills) == 1, np.array(skills) == 0)
    assert correct.mean() > 0.8
