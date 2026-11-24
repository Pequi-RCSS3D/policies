"""A long short-term memory block."""

from torch import Tensor, nn

from policies.models.blocks.base import Block
from policies.models.episodes import Episodes
from policies.models.layers import mlp
from policies.models.state import State
from policies.spec.model.blocks import LSTMBlockConfig


class LSTMBlock(Block):
    """State `h` and `c`, each of shape (layers, hidden_size) per sample."""

    def __init__(self, config: LSTMBlockConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.lstm = nn.LSTM(sum(in_sizes), config.hidden_size, config.layers, batch_first=True)
        self.after = mlp([config.hidden_size, *config.after], config.activation)
        self.out_size = config.after[-1] if config.after else config.hidden_size
        shape = (config.layers, config.hidden_size)
        self.state_shapes = {"h": shape, "c": shape}

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        batch = x.shape[:-1]
        shape = self.state_shapes["h"]
        hidden, cell = (
            state[name].reshape(-1, *shape).transpose(0, 1).contiguous() for name in ("h", "c")
        )
        y, (hidden, cell) = self.lstm(x.reshape(-1, 1, x.shape[-1]), (hidden, cell))
        y = self.after(y[:, 0])
        next_state = {
            name: value.transpose(0, 1).reshape(*batch, *shape)
            for name, value in (("h", hidden), ("c", cell))
        }
        return y.reshape(*batch, self.out_size), next_state

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        """One fused call over every episode in the sequence."""
        episodes = Episodes(reset)
        hidden, cell = (
            episodes.initial(state[name]).transpose(0, 1).contiguous() for name in ("h", "c")
        )
        y, _ = self.lstm(episodes.pad(x), (hidden, cell))
        output: Tensor = self.after(episodes.unpad(y))
        return output
