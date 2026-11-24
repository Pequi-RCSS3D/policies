"""Native sequence forwards match stepping `forward` through time, resets included."""

from typing import Any

import pytest
import torch
from torch import Tensor

from policies.core.config import Config
from policies.models import AuxiliaryLosses, Block, Model, State, blocks, models
from policies.models.blocks import MixtureBlock
from policies.models.episodes import Episodes
from policies.models.sequence import step_through
from tests.models.test_blocks import BLOCKS, IN_SIZES

TIME, BATCH = 7, (3, 2)
DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


def resets(*shape: int) -> Tensor:
    """Resets at the first step, mid-sequence, on consecutive steps and at the last step."""
    reset = torch.zeros(*shape, dtype=torch.bool)
    flat = reset.reshape(shape[0], -1)
    flat[0, 0] = flat[3, 0] = flat[4, 0] = flat[-1, 1] = flat[2, -1] = True
    return reset


def random_state(component: Block, *batch: int) -> State:
    """Random state, with transformer `filled` counts that are valid integers."""
    state = {key: torch.randn(*batch, *shape) for key, shape in component.state_shapes.items()}
    memory = getattr(component, "memory", 0)
    for key in state:
        if key.endswith("filled"):
            state[key] = torch.randint(0, memory + 1, (*batch, 1)).float()
    return state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", BLOCKS)
def test_block_sequence_matches_stepping(name: str, device: str) -> None:
    torch.manual_seed(0)
    block = blocks.build(BLOCKS[name], in_sizes=IN_SIZES).to(device)
    x = torch.randn(TIME, *BATCH, sum(IN_SIZES), device=device, requires_grad=True)
    state = {key: value.to(device) for key, value in random_state(block, *BATCH).items()}
    reset = resets(TIME, *BATCH).to(device)
    outputs, gradients = [], []
    for run in (block.forward_sequence, lambda *args: step_through(block, *args)):
        y = run(x, state, reset)
        (gradient,) = torch.autograd.grad(y.square().sum(), x)
        outputs.append(y)
        gradients.append(gradient)
    assert outputs[0].shape == (TIME, *BATCH, block.out_size)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-5, rtol=1e-4)


def test_episodes_split_at_resets() -> None:
    reset = torch.tensor([[False, True], [True, False], [False, False], [False, True]])
    episodes = Episodes(reset)
    assert (episodes.count, episodes.length) == (4, 3)
    x = torch.arange(8.0).reshape(4, 2, 1)
    padded = episodes.pad(x)
    assert padded[..., 0].tolist() == [[0, 0, 0], [2, 4, 6], [1, 3, 5], [7, 0, 0]]
    torch.testing.assert_close(episodes.unpad(padded), x)
    initial = episodes.initial(torch.tensor([[1.0], [2.0]]))
    assert initial[:, 0].tolist() == [1, 0, 0, 0]


def test_mixture_records_the_same_balance() -> None:
    block = blocks.build(BLOCKS["mixture-top-k"], in_sizes=IN_SIZES)
    assert isinstance(block, MixtureBlock)
    x = torch.randn(TIME, 4, sum(IN_SIZES))
    with AuxiliaryLosses(block) as collection:
        block.forward_sequence(x, {}, torch.zeros(TIME, 4, dtype=torch.bool))
    native = collection.losses()["balance"]
    with AuxiliaryLosses(block) as collection:
        for t in range(TIME):
            block(x[t], {})
    torch.testing.assert_close(native, collection.losses()["balance"])


@pytest.mark.parametrize("name", ["gru", "transformer-causal", "mixture-gru", "sequential"])
def test_parts_model_sequence_matches_stepping(name: str) -> None:
    torch.manual_seed(0)
    config: dict[str, Any] = {
        "type": "parts",
        "block": BLOCKS[name],
        "encoders": {"a": {"type": "block", "block": {"type": "lstm", "hidden_size": 4}}},
    }
    model = models.build(config, in_shapes={"a": (2, 3), "b": (4,)}, out_sizes={"y": 3, "z": 1})
    parts = {"a": torch.randn(TIME, 5, 2, 3), "b": torch.randn(TIME, 5, 4)}
    state = {key: torch.randn(5, *shape) for key, shape in model.state_shapes.items()}
    state = {key: value.abs().round() if "filled" in key else value for key, value in state.items()}
    reset = resets(TIME, 5)
    native = model.forward_sequence(parts, state, reset)
    stepped = Model.forward_sequence(model, parts, state, reset)
    assert native.keys() == stepped.keys() == {"y", "z"}
    for key in native:
        torch.testing.assert_close(native[key], stepped[key], atol=1e-5, rtol=1e-4)


class CounterConfig(Config):
    pass


class Counter(Block):
    """A stateful block without a native sequence forward: counts steps since a reset."""

    def __init__(self, config: CounterConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.out_size = 1
        self.state_shapes = {"count": (1,)}

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        count = state["count"] + 1
        return count, {"count": count}


def test_blocks_without_a_native_sequence_step_through() -> None:
    block = Counter(CounterConfig(), in_sizes=(1,))
    reset = torch.tensor([[False], [False], [True], [False]])
    y = block.forward_sequence(torch.zeros(4, 1, 1), {"count": torch.full((1, 1), 5.0)}, reset)
    assert y[:, 0, 0].tolist() == [6, 7, 1, 2]
