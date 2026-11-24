"""A mixture of experts block."""

from typing import cast

import torch
from torch import Tensor, nn

from policies.models.auxiliary import record, recording
from policies.models.blocks.base import Block
from policies.models.layers import mlp
from policies.models.state import State, namespaced, run_scoped, run_scoped_sequence
from policies.spec.model.blocks import MixtureBlockConfig, blocks


class MixtureBlock(Block):
    """Experts weighted by a softmax gate over the input.

    Every expert runs on every sample and top-k only masks the gate, which keeps the block
    exportable and fast at policy sizes. So stateful experts all advance every step, each under
    `experts.<i>.`. Records auxiliary loss `balance`, Switch Transformer's load balancing.
    """

    def __init__(self, config: MixtureBlockConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        experts = [blocks.build(config.expert, in_sizes=in_sizes) for _ in range(config.experts)]
        self.experts = nn.ModuleList(experts)
        self.prefixes = [f"experts.{i}" for i in range(config.experts)]
        self.gate = mlp(
            [sum(in_sizes), *config.gate, config.experts], config.activation, last_activation=False
        )
        self.top_k = config.top_k
        self.out_size = experts[0].out_size
        self.state_shapes = {
            name: shape
            for prefix, expert in zip(self.prefixes, experts, strict=True)
            for name, shape in namespaced(prefix, expert.state_shapes).items()
        }

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        weights = self._weights(x, steps=1)
        outputs: list[Tensor] = []
        next_state: State = {}
        for prefix, expert in zip(self.prefixes, self.experts, strict=True):
            y, inner = run_scoped(cast(Block, expert), prefix, x, state)
            outputs.append(y)
            next_state |= inner
        return self._mix(outputs, weights), next_state

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        weights = self._weights(x, steps=x.shape[0])
        outputs = [
            run_scoped_sequence(cast(Block, expert), prefix, x, state, reset)
            for prefix, expert in zip(self.prefixes, self.experts, strict=True)
        ]
        return self._mix(outputs, weights)

    def _weights(self, x: Tensor, steps: int) -> Tensor:
        """Gate weights over the experts. Records the balance of `steps` leading steps."""
        dense: Tensor = self.gate(x)
        logits = dense
        if self.top_k is not None:
            threshold = logits.topk(self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if recording(self):
            record(self, "balance", self._imbalance(dense, logits, steps))
        return logits.softmax(dim=-1)

    @staticmethod
    def _mix(outputs: list[Tensor], weights: Tensor) -> Tensor:
        return (torch.stack(outputs, dim=-1) @ weights.unsqueeze(-1)).squeeze(-1)

    def _imbalance(self, dense: Tensor, logits: Tensor, steps: int) -> Tensor:
        """Experts times the dot of dispatch fractions and mean gate probabilities. 1 is balanced.

        Averaged over `steps`, the leading dimension if above 1, as stepping would record it.
        Without top-k every expert is dispatched, so the probabilities stand in for the fractions.
        """
        count = dense.shape[-1]
        probabilities = dense.softmax(dim=-1).reshape(steps, -1, count).mean(1)
        dispatch = (
            probabilities
            if self.top_k is None
            else logits.isfinite().reshape(steps, -1, count).float().mean(1) / self.top_k
        )
        return count * (dispatch * probabilities).sum(-1).mean()
