from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from policies.core.config import Config
from policies.models import (
    AuxiliaryLosses,
    Model,
    PartsModel,
    State,
    encoders,
    heads,
    models,
    record,
    recording,
)
from policies.models.heads import MLPHead
from policies.spec import PartsModelConfig
from tests.models.parity import exported_state, train_export_and_match

IN_SHAPES = {"state": (2,), "target": (1,)}
OUT_SIZES = {"force": 1}
GRU_ENCODER: dict[str, Any] = {"type": "block", "block": {"type": "gru", "hidden_size": 8}}
MIXTURE: dict[str, Any] = {
    "type": "mixture",
    "experts": 3,
    "expert": {"type": "mlp", "hidden": [8]},
}


def build(**model: Any) -> PartsModel:
    built = models.build({"type": "parts", **model}, in_shapes=IN_SHAPES, out_sizes=OUT_SIZES)
    assert isinstance(built, PartsModel)
    return built


def parts(*batch: int) -> dict[str, Tensor]:
    return {name: torch.randn(*batch, *shape) for name, shape in IN_SHAPES.items()}


def test_models_registry_builds_parts_by_default_config() -> None:
    model = build()
    direct = PartsModel(PartsModelConfig(), IN_SHAPES, OUT_SIZES)
    assert type(model) is type(direct)
    assert {name for name, _ in model.named_parameters()} == {
        name for name, _ in direct.named_parameters()
    }


def test_models_registry_takes_custom_architectures() -> None:
    class SumConfig(Config):
        pass

    class Sum(Model):
        def __init__(
            self,
            config: SumConfig,
            in_shapes: Mapping[str, tuple[int, ...]],
            out_sizes: Mapping[str, int],
        ) -> None:
            super().__init__()
            self.out_sizes = out_sizes
            self.state_shapes = {"total": (1,)}

        def forward(
            self, parts: Mapping[str, Tensor], state: State
        ) -> tuple[dict[str, Tensor], State]:
            total = state["total"] + sum(value.sum(-1, keepdim=True) for value in parts.values())
            return {name: total.expand(-1, size) for name, size in self.out_sizes.items()}, {
                "total": total
            }

    models.register("tests.sum", SumConfig, replace=True)(Sum)
    model = models.build({"type": "tests.sum"}, in_shapes=IN_SHAPES, out_sizes=OUT_SIZES)
    state = model.initial_state((2,))
    _, state = model({name: torch.ones(2, *shape) for name, shape in IN_SHAPES.items()}, state)
    torch.testing.assert_close(state["total"], torch.full((2, 1), 3.0))


def test_stateful_encoder_is_namespaced_and_recurrent() -> None:
    model = build(encoders={"state": GRU_ENCODER}, block={"type": "gru", "hidden_size": 4}).eval()
    assert model.state_shapes == {"encoders.state.block.h": (1, 8), "block.h": (1, 4)}
    x = parts(3)
    with torch.no_grad():
        first, after = model(x, model.initial_state((3,)))
        second, _ = model(x, after)
    assert set(after) == set(model.state_shapes)
    assert not torch.allclose(first["force"], second["force"])


def test_stateful_encoder_exports(tmp_path: Path) -> None:
    learner = train_export_and_match({"encoders": {"state": GRU_ENCODER}}, tmp_path)
    assert learner.recurrent
    assert exported_state(tmp_path) == ["encoders.state.block.h"]


def test_mlp_head_has_hidden_layers_and_small_outputs() -> None:
    head = heads.build({"type": "mlp", "hidden": [16, 8]}, in_size=6, out_size=3)
    assert isinstance(head, MLPHead)
    linear = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(6, 16), (16, 8), (8, 3)]
    y = head(torch.randn(5, 6))
    assert y.shape == (5, 3)
    assert y.abs().max() < 0.1
    model = build(heads={"force": {"type": "mlp", "hidden": [8]}})
    assert model(parts(4), {})[0]["force"].shape == (4, 1)


def test_flatten_encoder_keeps_the_state_protocol() -> None:
    encoder = encoders.build({"type": "flatten"}, shape=(3, 2))
    y, state = encoder(torch.randn(4, 3, 2), {})
    assert y.shape == (4, 6)
    assert state == {} == encoder.state_shapes


def test_auxiliary_losses_are_collected_by_module_path() -> None:
    model = build(block={"type": "sequential", "blocks": [MIXTURE, {"type": "mlp"}]})
    mixture = model.get_submodule("block.blocks.0")
    assert not recording(mixture)
    with AuxiliaryLosses(model) as auxiliary:
        assert recording(mixture)
        state: State = {}
        for _ in range(3):
            model(parts(16), state)
    losses = auxiliary.losses()
    assert set(losses) == {"block.blocks.0.balance"}
    loss = losses["block.blocks.0.balance"]
    # Dense gates give experts times the summed squared mean probabilities, 1 at best.
    assert loss.item() >= 1 - 1e-6
    loss.backward()  # type: ignore[no-untyped-call]  # torch Tensor.backward is unannotated
    gate = mixture.get_submodule("gate")
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())


def test_auxiliary_losses_ignore_other_models_and_closed_collections() -> None:
    model, other = build(block=MIXTURE), build(block=MIXTURE)
    with AuxiliaryLosses(model) as auxiliary:
        other(parts(4), {})
    model(parts(4), {})
    assert auxiliary.losses() == {}
    with AuxiliaryLosses(model) as auxiliary:
        record(model, "custom", torch.tensor(2.0))
    torch.testing.assert_close(auxiliary.losses(), {"custom": torch.tensor(2.0)})
    with auxiliary:
        pass
    assert auxiliary.losses() == {}


def test_top_k_balance_peaks_when_one_expert_takes_everything() -> None:
    model = build(block={**MIXTURE, "top_k": 1})
    gate = model.get_submodule("block.gate.0")
    assert isinstance(gate, torch.nn.Linear)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([20.0, 0.0, 0.0]))
    with torch.no_grad(), AuxiliaryLosses(model) as auxiliary:
        model(parts(6), {})
    # All dispatch and almost all probability on one of 3 experts. Balanced routing gives 1.
    torch.testing.assert_close(auxiliary.losses()["block.balance"], torch.tensor(3.0))
