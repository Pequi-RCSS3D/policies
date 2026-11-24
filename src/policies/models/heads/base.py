"""The head contract. The registry and configs live in `policies.spec.model.heads`."""

from torch import Tensor, nn


class Head(nn.Module):
    """Built from `(config, in_size, out_size)`. Stateless.

    `forward(x)` maps the block's output, of shape `(*batch, in_size)`, to one output part's raw
    values, of shape `(*batch, out_size)`. A factory may also return a plain module that does so.
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
