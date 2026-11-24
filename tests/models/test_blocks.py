from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from policies.models import Block, State, blocks
from policies.models.blocks import MixtureBlock
from tests.models.parity import exported_state, train_export_and_match

IN_SIZES = (2, 4)
TRANSFORMER: dict[str, Any] = {
    "type": "transformer",
    "width": 16,
    "heads": 2,
    "layers": 2,
    "feedforward": 32,
}
LSTM: dict[str, Any] = {"type": "lstm", "hidden_size": 8}
MIXTURE: dict[str, Any] = {
    "type": "mixture",
    "experts": 3,
    "expert": {"type": "mlp", "hidden": [16]},
}
BLOCKS: dict[str, dict[str, Any]] = {
    "mlp": {"type": "mlp", "hidden": [32, 16]},
    "gru": {"type": "gru", "hidden_size": 16},
    "lstm": {"type": "lstm", "hidden_size": 16, "layers": 2},
    "transformer": TRANSFORMER,
    "transformer-input": {**TRANSFORMER, "tokens": "input", "positions": False},
    "transformer-memory": {**TRANSFORMER, "memory": 3},
    "transformer-causal": {**TRANSFORMER, "memory": 3, "causal": True},
    **{
        f"transformer-{pooling}": {**TRANSFORMER, "memory": 2, "pooling": pooling, "after": [8]}
        for pooling in ("max", "first", "last", "summary")
    },
    "mixture": MIXTURE,
    "mixture-top-k": {**MIXTURE, "top_k": 2, "gate": [8]},
    "mixture-gru": {**MIXTURE, "top_k": 1, "expert": {"type": "gru", "hidden_size": 8}},
    "sequential": {
        "type": "sequential",
        "blocks": [{"type": "mlp", "hidden": [16]}, {"type": "gru", "hidden_size": 8}, LSTM],
    },
}
MEMORY = [name for name, config in BLOCKS.items() if config.get("memory")]
RECURRENT = {"gru", "lstm", "mixture-gru", "sequential", *MEMORY}
EXPORTED_STATE = {
    "gru": ["block.h"],
    "lstm": ["block.h", "block.c"],
    "transformer-memory": ["block.memory", "block.filled"],
    "mixture-gru": ["block.experts.0.h", "block.experts.1.h", "block.experts.2.h"],
    "sequential": ["block.blocks.1.h", "block.blocks.2.h", "block.blocks.2.c"],
}
"""Exported state names. Those of gru, lstm and transformer predate namespacing."""


def initial(block: Block, *batch: int) -> State:
    return {key: torch.zeros(*batch, *shape) for key, shape in block.state_shapes.items()}


@pytest.mark.parametrize("name", BLOCKS)
def test_shapes_and_state(name: str) -> None:
    block = blocks.build(BLOCKS[name], in_sizes=IN_SIZES)
    assert bool(block.state_shapes) == (name in RECURRENT)
    y, next_state = block(torch.randn(5, 3, sum(IN_SIZES)), initial(block, 5, 3))
    assert y.shape == (5, 3, block.out_size)
    assert {key: tuple(value.shape) for key, value in next_state.items()} == {
        key: (5, 3, *shape) for key, shape in block.state_shapes.items()
    }


@pytest.mark.parametrize("name", MEMORY)
def test_memory_ignores_unfilled_steps(name: str) -> None:
    block = blocks.build(BLOCKS[name], in_sizes=IN_SIZES).eval()
    x = torch.randn(2, sum(IN_SIZES))
    fresh = initial(block, 2)
    stale = torch.randn_like(fresh["memory"])
    one = torch.ones(2, 1)
    newest = torch.zeros_like(stale)
    newest[:, -1] = stale[:, -1]
    with torch.no_grad():
        clean, _ = block(x, fresh)
        reset, _ = block(x, dict(fresh, memory=stale))
        partial, _ = block(x, {"memory": stale, "filled": one})
        expected, _ = block(x, {"memory": newest, "filled": one})
    torch.testing.assert_close(clean, reset)
    torch.testing.assert_close(partial, expected)


@pytest.mark.parametrize("name", MEMORY)
def test_memory_influences_output(name: str) -> None:
    block = blocks.build(BLOCKS[name], in_sizes=IN_SIZES).eval()
    state = initial(block, 1)
    with torch.no_grad():
        _, after_one = block(torch.ones(1, sum(IN_SIZES)), state)
        first, _ = block(torch.zeros(1, sum(IN_SIZES)), state)
        second, _ = block(torch.zeros(1, sum(IN_SIZES)), after_one)
    assert after_one["filled"].item() == 1
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("causal", [False, True])
def test_causal_hides_later_steps(causal: bool) -> None:
    config = {**TRANSFORMER, "memory": 2, "causal": causal}
    block = blocks.build(config, in_sizes=IN_SIZES).eval()
    hidden: list[Tensor] = []
    first_layer = block.get_submodule("transformer.layers.0")
    first_layer.register_forward_hook(lambda module, inputs, output: hidden.append(output))
    state = {"memory": torch.randn(1, 2, 2, 16), "filled": torch.full((1, 1), 2.0)}
    with torch.no_grad():
        block(torch.zeros(1, sum(IN_SIZES)), state)
        block(torch.ones(1, sum(IN_SIZES)), state)
    oldest = slice(0, len(IN_SIZES))
    assert torch.allclose(hidden[0][:, oldest], hidden[1][:, oldest]) == causal


def test_mixture_top_k_routes_to_the_best_experts() -> None:
    torch.manual_seed(0)
    dense = blocks.build({**MIXTURE, "top_k": 3}, in_sizes=IN_SIZES).eval()
    torch.manual_seed(0)
    reference = blocks.build(MIXTURE, in_sizes=IN_SIZES).eval()
    single = blocks.build({**MIXTURE, "top_k": 1}, in_sizes=IN_SIZES).eval()
    assert isinstance(single, MixtureBlock)
    x = torch.randn(4, sum(IN_SIZES))
    with torch.no_grad():
        torch.testing.assert_close(dense(x, {})[0], reference(x, {})[0])
        best = single.gate(x).argmax(-1)
        experts = torch.stack([expert(x, {})[0] for expert in single.experts], 1)
        torch.testing.assert_close(single(x, {})[0], experts[torch.arange(4), best])


def test_mixture_rejects_top_k_above_experts() -> None:
    with pytest.raises(ValueError):
        blocks.build({**MIXTURE, "top_k": 4}, in_sizes=IN_SIZES)


def test_state_names_are_bare_and_namespaced_by_containers() -> None:
    assert blocks.build(BLOCKS["gru"], in_sizes=IN_SIZES).state_shapes == {"h": (1, 16)}
    sequential = blocks.build(BLOCKS["sequential"], in_sizes=IN_SIZES)
    assert sequential.state_shapes == {
        "blocks.1.h": (1, 8),
        "blocks.2.h": (1, 8),
        "blocks.2.c": (1, 8),
    }
    nested = blocks.build({**MIXTURE, "expert": BLOCKS["sequential"]}, in_sizes=IN_SIZES)
    assert "experts.2.blocks.2.c" in nested.state_shapes


def test_sequential_chains_recurrent_blocks() -> None:
    block = blocks.build(BLOCKS["sequential"], in_sizes=IN_SIZES).eval()
    x = torch.randn(3, sum(IN_SIZES))
    state = initial(block, 3)
    with torch.no_grad():
        first, after = block(x, state)
        second, _ = block(x, after)
        alone = block.get_submodule("blocks.0")(x, {})[0]
        gru, _ = block.get_submodule("blocks.1")(alone, {"h": state["blocks.1.h"]})
        expected, _ = block.get_submodule("blocks.2")(
            gru, {"h": state["blocks.2.h"], "c": state["blocks.2.c"]}
        )
    assert block.out_size == 8
    torch.testing.assert_close(first, expected)
    assert not torch.allclose(first, second)


def test_mixture_advances_every_stateful_expert() -> None:
    block = blocks.build(BLOCKS["mixture-gru"], in_sizes=IN_SIZES).eval()
    with torch.no_grad():
        _, after = block(torch.randn(2, sum(IN_SIZES)), initial(block, 2))
    assert all(value.abs().sum() > 0 for value in after.values())


@pytest.mark.parametrize("name", BLOCKS)
def test_trains_exports_and_matches_onnx(name: str, tmp_path: Path) -> None:
    learner = train_export_and_match({"block": BLOCKS[name]}, tmp_path)
    assert learner.recurrent == (name in RECURRENT)
    if name in EXPORTED_STATE:
        assert exported_state(tmp_path) == EXPORTED_STATE[name]
