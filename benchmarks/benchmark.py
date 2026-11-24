"""Training throughput and inference latency for a policy with 23 joint-sized outputs.

uv run --locked python benchmarks/benchmark.py training --envs 4096 --device cuda
uv run --locked python benchmarks/benchmark.py training --block gru
uv run --locked python benchmarks/benchmark.py inference
"""

import argparse
import statistics
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from policies.spec import PolicySpec

BLOCKS: dict[str, dict[str, Any]] = {
    "mlp": {"type": "mlp", "hidden": [512, 256, 128]},
    "gru": {"type": "gru", "hidden_size": 256, "after": [128]},
    "lstm": {"type": "lstm", "hidden_size": 256, "after": [128]},
}
"""Model blocks to compare. Recurrent ones train on whole environment sequences."""


def spec(block: str) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "name": "benchmark",
            "inputs": {
                "parts": [
                    {
                        "name": "joints",
                        "source": "history.joint_position",
                        "shape": [3, 23],
                        "normalize": True,
                    },
                    {
                        "name": "velocity",
                        "source": "joint_velocity",
                        "shape": [23],
                        "normalize": True,
                    },
                    {"name": "orientation", "source": "torso_quaternion", "shape": [4]},
                    {"name": "command", "source": "command", "shape": [3]},
                    {"name": "last_joints", "source": "last.joints", "shape": [23]},
                ]
            },
            "outputs": {"parts": [{"name": "joints", "kind": {"type": "continuous", "size": 23}}]},
            "bookkeeping": [
                {"type": "history", "source": "joint_position", "length": 3},
                {"type": "last", "source": "joints"},
            ],
            "model": {"block": BLOCKS[block]},
        }
    )


class Synthetic:
    """Random observations on the device. Measures everything except a real simulator."""

    def __init__(self, num_envs: int, device: str) -> None:
        import torch

        self.torch, self.num_envs, self.device = torch, num_envs, device

    def _observation(self) -> dict[str, Any]:
        torch, n, device = self.torch, self.num_envs, self.device
        quaternion = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
        return {
            "joint_position": torch.randn(n, 23, device=device),
            "joint_velocity": torch.randn(n, 23, device=device),
            "torso_quaternion": quaternion,
            "command": torch.randn(n, 3, device=device),
        }

    def reset(self) -> dict[str, Any]:
        return self._observation()

    def step(self, outputs: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        torch = self.torch
        rewards = torch.randn(self.num_envs, device=self.device)
        dones = torch.rand(self.num_envs, device=self.device) < 0.005
        return self._observation(), rewards, dones, {"truncated": dones}


def training(envs: int, device: str, iterations: int, block: str) -> None:
    import torch

    from policies.learning import Learner, LearnerConfig, RolloutConfig

    learner = Learner(
        spec(block), LearnerConfig(rollout=RolloutConfig(length=24)), num_envs=envs, device=device
    )
    env = Synthetic(envs, device)
    observation = env.reset()
    collect, update = [], []
    for iteration in range(iterations + 1):
        started = time.perf_counter()
        while not learner.ready:
            outputs = learner.act(observation)
            observation, rewards, dones, info = env.step(outputs)
            learner.observe(observation, rewards, dones, info["truncated"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        collected = time.perf_counter()
        learner.update()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if iteration:  # the first iteration warms up
            collect.append(collected - started)
            update.append(time.perf_counter() - collected)
    steps = envs * learner.config.rollout.length
    report(
        f"Training a {block} policy on {device}, {envs} environments",
        {
            "collection (policy side)": f"{steps / statistics.median(collect):,.0f} env steps/s",
            "update": f"{statistics.median(update) * 1000:,.0f} ms per iteration",
            "iteration": f"{statistics.median(collect) + statistics.median(update):.2f} s",
        },
    )


def inference(samples: int, block: str) -> None:
    from policies.deploy.export import export_onnx
    from policies.deploy.inference import Policy
    from policies.learning import Learner

    learner = Learner(spec(block), num_envs=1)
    env = Synthetic(1, "cpu")
    learner.act(env.reset())
    with tempfile.TemporaryDirectory() as directory:
        export_onnx(learner, f"{directory}/policy.onnx")
        policy = Policy.load(f"{directory}/policy.onnx")
        observation = {key: value[0].numpy() for key, value in env.reset().items()}
        for _ in range(100):
            policy.step(observation)
        timings = []
        for _ in range(samples):
            started = time.perf_counter()
            policy.step(observation)
            timings.append(time.perf_counter() - started)
    timings_ms = np.array(timings) * 1000
    report(
        f"Inference of a {block} policy on 1 CPU thread, {samples} steps",
        {
            "median": f"{np.median(timings_ms):.3f} ms",
            "p99": f"{np.percentile(timings_ms, 99):.3f} ms",
        },
    )


def report(title: str, rows: dict[str, str]) -> None:
    table = Table(title=title, show_header=False, box=box.SIMPLE)
    for key, value in rows.items():
        table.add_row(key, value)
    Console().print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("training")
    train.add_argument("--envs", type=int, default=4096)
    train.add_argument("--device", default="cuda")
    train.add_argument("--iterations", type=int, default=10)
    infer = commands.add_parser("inference")
    infer.add_argument("--samples", type=int, default=2000)
    for command in (train, infer):
        command.add_argument("--block", choices=list(BLOCKS), default="mlp")
    arguments = parser.parse_args()
    if arguments.command == "training":
        training(arguments.envs, arguments.device, arguments.iterations, arguments.block)
    else:
        inference(arguments.samples, arguments.block)


if __name__ == "__main__":
    main()
