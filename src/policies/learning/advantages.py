"""Generalized advantage estimation over a rollout."""

from collections.abc import MutableMapping

import torch
from torch import Tensor


def estimate_advantages(
    data: MutableMapping[str, Tensor],
    last_values: Tensor,
    gamma: float,
    gae_lambda: float,
    normalize: bool,
) -> None:
    """Add `returns` and `advantages` to `[time, env]` steps with `rewards`, `dones`, `values`.

    `last_values` are the values of the observation after the last step.
    """
    rewards, dones, values = data["rewards"], data["dones"], data["values"]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    next_values = last_values
    for t in reversed(range(rewards.shape[0])):
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * alive - values[t]
        gae = delta + gamma * gae_lambda * alive * gae
        advantages[t] = gae
        next_values = values[t]
    data["returns"] = advantages + values
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    data["advantages"] = advantages
