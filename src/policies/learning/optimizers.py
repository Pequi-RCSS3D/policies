"""Optimizers built by name."""

from typing import Annotated, Any

import torch

from policies.core.config import Config
from policies.core.registry import Registry


class AdamConfig(Config):
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


optimizers: Registry[torch.optim.Optimizer] = Registry("optimizer")

type OptimizerConfig = Annotated[dict[str, Any], optimizers]
"""Any optimizer's config, validated to its canonical form."""

DEFAULT_OPTIMIZER = "adamw"
"""The optimizer of a config without `type`."""


@optimizers.register("adam", AdamConfig)
def adam(config: AdamConfig, params: Any) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay
    )


@optimizers.register("adamw", AdamConfig)
def adamw(config: AdamConfig, params: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay
    )
