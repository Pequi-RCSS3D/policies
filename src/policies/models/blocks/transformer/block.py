"""A transformer over the current step's tokens and, optionally, past steps' tokens."""

import torch
from torch import Tensor, nn

from policies.models.blocks.base import Block
from policies.models.layers import Transformer, mlp
from policies.models.state import State
from policies.spec.model.blocks import Pooling, Tokens, TransformerBlockConfig


class TransformerBlock(Block):
    """Tokens attend to each other across the current step and up to `memory` past steps.

    Past token embeddings are recurrent state `memory` (memory, tokens, width), and
    `filled` (1,) counts the valid ones, so a reset sees no stale memory. With no memory
    the block is stateless.
    """

    steps: Tensor
    """Step index of each token, the current step being `memory`."""
    eye: Tensor
    allowed: Tensor | None

    def __init__(self, config: TransformerBlockConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.width, self.memory, self.pooling = config.width, config.memory, config.pooling
        self.splits = list(in_sizes) if config.tokens == Tokens.PARTS else [sum(in_sizes)]
        self.embed = nn.ModuleList(nn.Linear(size, config.width) for size in self.splits)
        self.tokens = len(self.splits)
        self.positions = (
            nn.Parameter(0.02 * torch.randn(config.memory + 1, self.tokens, config.width))
            if config.positions
            else None
        )
        summary = config.pooling == Pooling.SUMMARY
        self.summary = nn.Parameter(0.02 * torch.randn(1, 1, config.width)) if summary else None
        self.transformer = Transformer(
            config.width, config.heads, config.layers, config.feedforward, config.activation
        )
        self.after = mlp([config.width, *config.after], config.activation)
        self.out_size = config.after[-1] if config.after else config.width
        self.state_shapes = {}
        if config.memory:
            self.state_shapes = {
                "memory": (config.memory, self.tokens, config.width),
                "filled": (1,),
            }
            steps = torch.arange(config.memory + 1).repeat_interleave(self.tokens)
            if summary:
                steps = torch.cat([steps, steps[-1:]])
            self.register_buffer("steps", steps.float(), persistent=False)
            self.register_buffer("eye", torch.eye(len(steps), dtype=torch.bool), persistent=False)
            causal = steps[None, :] <= steps[:, None] if config.causal else None
            self.register_buffer("allowed", causal, persistent=False)

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        batch = x.shape[:-1]
        current = self._embed(x.reshape(-1, x.shape[-1]))
        if not self.memory:
            return self._read(current[:, None], None).reshape(*batch, self.out_size), {}
        past = state["memory"].reshape(-1, self.memory, self.tokens, self.width)
        window = torch.cat([past, current[:, None]], dim=1)
        filled = state["filled"].reshape(-1, 1)
        y = self._read(window, filled)
        next_state = {
            "memory": window[:, 1:].reshape(*batch, self.memory, self.tokens, self.width),
            "filled": (filled + 1).clamp(max=self.memory).reshape(*batch, 1),
        }
        return y.reshape(*batch, self.out_size), next_state

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        """Every step's window at once, sliding over the memory and the sequence's own tokens.

        Steps before a reset stay in the window but are masked out, as stepping zeroes `filled`.
        """
        if not self.memory:
            return super().forward_sequence(x, state, reset)
        time, batch = x.shape[0], x.shape[1:-1]
        current = self._embed(x.reshape(-1, x.shape[-1])).reshape(time, -1, self.tokens, self.width)
        past = state["memory"].reshape(-1, self.memory, self.tokens, self.width).transpose(0, 1)
        history = torch.cat([past, current])
        window = history.unfold(0, self.memory + 1, 1).permute(0, 1, 4, 2, 3).flatten(0, 1)
        filled = self._filled(state["filled"].reshape(1, -1), reset.reshape(time, -1))
        y = self._read(window, filled.reshape(-1, 1))
        return y.reshape(time, *batch, self.out_size)

    def _embed(self, x: Tensor) -> Tensor:
        """(n, sum(in_sizes)) to the current step's tokens (n, tokens, width)."""
        chunks = torch.split(x, self.splits, dim=-1)
        return torch.stack(
            [embed(chunk) for embed, chunk in zip(self.embed, chunks, strict=True)], dim=1
        )

    def _read(self, window: Tensor, filled: Tensor | None) -> Tensor:
        """Output (n, out_size) of windows (n, steps, tokens, width), `filled` (n, 1) of them."""
        tokens = (window if self.positions is None else window + self.positions).flatten(1, 2)
        if self.summary is not None:
            tokens = torch.cat([tokens, self.summary.expand(tokens.shape[0], -1, -1)], dim=1)
        attend = None if filled is None else self._attend(filled)
        output: Tensor = self.after(self._pool(self.transformer(tokens, attend)))
        return output

    def _filled(self, initial: Tensor, reset: Tensor) -> Tensor:
        """Valid past steps (time, n) of each step, from `initial` (1, n) and `reset` (time, n)."""
        steps = torch.arange(reset.shape[0], device=reset.device, dtype=initial.dtype)[:, None]
        last = torch.where(reset, steps, -1).cummax(0).values
        return torch.where(last >= 0, steps - last, initial + steps).clamp(max=self.memory)

    def _attend(self, filled: Tensor) -> Tensor:
        """Keys from filled steps only, within the causal limit, and always the token itself."""
        attend = (self.steps >= self.memory - filled)[:, None, :]
        if self.allowed is not None:
            attend = attend & self.allowed
        return attend | self.eye

    def _pool(self, y: Tensor) -> Tensor:
        start = self.memory * self.tokens
        current = y[:, start : start + self.tokens]
        match self.pooling:
            case Pooling.MEAN:
                return current.mean(1)
            case Pooling.MAX:
                return current.amax(1)
            case Pooling.FIRST:
                return current[:, 0]
            case Pooling.LAST:
                return current[:, -1]
            case Pooling.SUMMARY:
                return y[:, -1]
