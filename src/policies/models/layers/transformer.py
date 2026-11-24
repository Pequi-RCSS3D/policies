"""Pre-norm transformer layers on `scaled_dot_product_attention`, which export cleanly to ONNX."""

from torch import Tensor, nn
from torch.nn import functional

from policies.models.layers.mlp import mlp


class TransformerLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, activation: str) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by {heads} heads")
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.query_key_value = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = mlp([width, feedforward, width], activation, last_activation=False)

    def forward(self, x: Tensor, attend: Tensor | None = None) -> Tensor:
        """`x` is (batch, tokens, width). See `Transformer.forward` for `attend`."""
        batch, tokens, width = x.shape
        query, key, value = (
            part.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)
            for part in self.query_key_value(self.attention_norm(x)).chunk(3, dim=-1)
        )
        mask = None if attend is None else attend.unsqueeze(-3)
        mixed = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        x = x + self.projection(mixed.transpose(1, 2).reshape(batch, tokens, width))
        output: Tensor = x + self.feedforward(self.feedforward_norm(x))
        return output


class Transformer(nn.Module):
    """A stack of layers with a final norm."""

    def __init__(
        self, width: int, heads: int, layers: int, feedforward: int, activation: str
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerLayer(width, heads, feedforward, activation) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, attend: Tensor | None = None) -> Tensor:
        """`x` is (batch, tokens, width).

        `attend[b, i, j]`, broadcastable to (batch, tokens, tokens), says whether token i sees
        token j. Every row needs at least one true entry. None means full attention.
        """
        for layer in self.layers:
            x = layer(x, attend)
        output: Tensor = self.norm(x)
        return output
