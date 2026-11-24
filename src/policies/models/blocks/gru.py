"""A gated recurrent unit block."""

from torch import Tensor, nn

from policies.models.blocks.base import Block
from policies.models.episodes import Episodes
from policies.models.layers import mlp
from policies.models.state import State
from policies.spec.model.blocks import GRUBlockConfig


class GRUBlock(Block):
    """State `h` has shape (layers, hidden_size) per sample."""

    def __init__(self, config: GRUBlockConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.gru = nn.GRU(sum(in_sizes), config.hidden_size, config.layers, batch_first=True)
        self.after = mlp([config.hidden_size, *config.after], config.activation)
        self.out_size = config.after[-1] if config.after else config.hidden_size
        self.state_shapes = {"h": (config.layers, config.hidden_size)}

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        batch = x.shape[:-1]
        flat = x.reshape(-1, 1, x.shape[-1])
        shape = self.state_shapes["h"]
        hidden = state["h"].reshape(-1, *shape).transpose(0, 1).contiguous()
        y, hidden = self.gru(flat, hidden)
        y = self.after(y[:, 0])
        hidden = hidden.transpose(0, 1).reshape(*batch, *shape)
        return y.reshape(*batch, self.out_size), {"h": hidden}

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        """One fused call over every episode in the sequence."""
        episodes = Episodes(reset)
        hidden = episodes.initial(state["h"]).transpose(0, 1).contiguous()
        y, _ = self.gru(episodes.pad(x), hidden)
        output: Tensor = self.after(episodes.unpad(y))
        return output
