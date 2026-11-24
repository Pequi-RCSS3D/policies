"""`[time, batch]` steps split at resets into episodes, padded for fused recurrences."""

import torch
from torch import Tensor


class Episodes:
    """Each sample's steps, cut before every reset, as rows of an `(episodes, length)` grid.

    A fused recurrence, e.g. cuDNN's GRU, then runs every episode from its own initial state in
    one call, as rsl_rl and TorchRL do. Building it costs one host sync, for the grid's size.
    """

    def __init__(self, reset: Tensor) -> None:
        self.time, self.batch = reset.shape[0], reset.shape[1:]
        flat = reset.reshape(self.time, -1)
        starts = flat.clone()
        starts[0] = True
        steps = torch.arange(self.time, device=reset.device)[:, None]
        self.position = steps - torch.where(starts, steps, 0).cummax(0).values
        """Step within its episode, (time, samples)."""
        self.episode = starts.T.reshape(-1).cumsum(0).reshape(-1, self.time).T - 1
        """Episode of each step, (time, samples), numbered sample-major."""
        self.carried = ~flat[0]
        """Samples whose given state carries into their first step."""
        count, length = torch.stack([self.episode[-1, -1] + 1, self.position.max() + 1]).tolist()
        self.count, self.length = int(count), int(length)

    def pad(self, x: Tensor) -> Tensor:
        """(time, *batch, size) to (episodes, length, size), zero past each episode's end."""
        flat = x.reshape(self.time, -1, x.shape[-1])
        padded = flat.new_zeros(self.count, self.length, x.shape[-1])
        return padded.index_put((self.episode, self.position), flat)

    def unpad(self, y: Tensor) -> Tensor:
        """(episodes, length, size) back to (time, *batch, size)."""
        return y[self.episode, self.position].reshape(self.time, *self.batch, y.shape[-1])

    def initial(self, state: Tensor) -> Tensor:
        """Per-sample state (*batch, *shape) to per-episode, zero where an episode starts afresh."""
        flat = state.reshape(-1, *state.shape[len(self.batch) :])
        carried = self.carried.reshape(-1, *[1] * (flat.dim() - 1))
        initial = flat.new_zeros(self.count, *flat.shape[1:])
        return initial.index_put((self.episode[0],), flat.masked_fill(~carried, 0))
