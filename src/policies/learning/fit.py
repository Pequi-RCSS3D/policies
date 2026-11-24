"""Behavior cloning on recorded data through the imitation loss."""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from policies.core.logs import get_logger
from policies.data import Batch, Dataset
from policies.learning.actor import Actor
from policies.learning.auxiliary import Auxiliary
from policies.learning.metrics import Metrics
from policies.learning.objectives import imitation_loss
from policies.learning.optimization import Optimization
from policies.learning.sequences import unroll

log = get_logger(__name__)


def fit_loss(actor: Actor, batch: Batch, device: torch.device, training: bool) -> Tensor:
    """The actor's imitation loss on one batch. `training` updates its normalization."""

    def tensors(values: Mapping[str, Any]) -> dict[str, Tensor]:
        return {name: torch.as_tensor(value, device=device) for name, value in values.items()}

    inputs, targets = tensors(batch.inputs), tensors(batch.targets)
    recurrent = bool(actor.model.state_shapes)
    if batch.reset is None and recurrent:
        raise ValueError("a recurrent model fits `[time, batch]` sequences with `reset` flags")
    if batch.reset is not None and not recurrent:
        inputs = {name: value.flatten(0, 1) for name, value in inputs.items()}
    if training:
        actor.normalizer.update(inputs)
    if batch.reset is None or not recurrent:
        state = actor.model.initial_state(next(iter(inputs.values())).shape[:1], device)
        distribution, _ = actor(inputs, state)
    else:
        reset = torch.as_tensor(batch.reset, device=device).float()
        # `unroll` clears state after a step that ends an episode, so shift by one.
        dones = torch.cat([reset[1:], torch.zeros_like(reset[:1])])
        initial = actor.model.initial_state((1, reset.shape[1]), device)
        distribution = actor.head(unroll(actor.model, actor.normalizer, inputs, initial, dones))
    if batch.reset is not None:
        targets = {name: value.flatten(0, 1) for name, value in targets.items()}
    return imitation_loss(distribution, targets)


def fit(
    actor: Actor,
    dataset: Dataset,
    epochs: int,
    auxiliary: Auxiliary,
    optimization: Optimization,
    device: torch.device,
    training: bool,
    first: Callable[[Batch], None] | None = None,
) -> dict[str, float]:
    """Epochs over `dataset`. Returns averaged metrics. `first` sees the first batch."""
    metrics = Metrics()
    for epoch in range(epochs):
        for batch in dataset:
            if first is not None and not metrics.count:
                first(batch)
            with auxiliary.collection:
                loss = fit_loss(actor, batch, device, training)
            metrics.add({"imitation/loss": loss.detach()})
            extra = auxiliary.loss(metrics)
            total = loss if extra is None else loss + extra
            metrics.add({"grad_norm": optimization.step(total)})
            metrics.step()
        log.debug("fit epoch", epoch=epoch)
    if not metrics.count:
        raise ValueError("the dataset yielded no batches")
    return metrics.averages()
