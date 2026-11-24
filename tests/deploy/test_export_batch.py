from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from policies.deploy.export import ExportSource, export_artifact  # noqa: E402  # needs the torch extra
from policies.deploy.inference import Policy, validate_artifact  # noqa: E402  # same
from policies.learning import Actor, Learner  # noqa: E402  # same
from policies.spec import PolicySpec  # noqa: E402  # same

SPEC = PolicySpec.model_validate(
    {
        "name": "batch",
        "inputs": {
            "parts": [
                {"name": "state", "source": "state", "shape": [2], "normalize": True},
                {"name": "grid", "source": "grid", "shape": [2, 3]},
            ]
        },
        "outputs": {
            "parts": [
                {"name": "force", "kind": {"type": "continuous", "size": 2}},
                {"name": "choice", "kind": {"type": "categorical", "choices": ["a", "b", "c"]}},
            ]
        },
    }
)


@pytest.mark.parametrize(
    "block", [{"type": "mlp", "hidden": [16]}, {"type": "gru", "hidden_size": 8}]
)
def test_batch_axis_is_dynamic(block: dict[str, object], tmp_path: Path) -> None:
    spec = PolicySpec.model_validate({**SPEC.model_dump(), "model": {"block": block}})
    learner = Learner(spec, num_envs=3)
    learner.act({"state": torch.randn(3, 2), "grid": torch.randn(3, 2, 3)})
    export_artifact(learner, tmp_path)
    validate_artifact(tmp_path, samples=5)
    policy = Policy.load(tmp_path)
    tensors = [*policy.session.get_inputs(), *policy.session.get_outputs()]
    assert {value.shape[0] for value in tensors} == {"batch"}
    outputs = policy.step({"state": np.zeros(2, np.float32), "grid": np.ones((2, 3), np.float32)})
    assert outputs["force"].shape == (2,)
    assert outputs["choice"].shape == ()


OUTPUTS: dict[str, tuple[dict[str, Any], tuple[int, ...], str]] = {
    "normal": ({"kind": {"type": "continuous", "shape": [2, 3]}}, (2, 3), "float32"),
    "normal_state_std": (
        {"kind": {"type": "scalar"}, "distribution": {"type": "normal_state_std"}},
        (),
        "float32",
    ),
    "categorical": ({"kind": {"type": "categorical", "choices": ["a", "b", "c"]}}, (), "int64"),
    "bernoulli": ({"kind": {"type": "binary", "size": 3}}, (3,), "float32"),
    "multi_categorical": (
        {"kind": {"type": "multi_categorical", "choices": [["a", "b"], ["c", "d", "e"]]}},
        (2,),
        "int64",
    ),
}


def distribution_spec(case: str) -> PolicySpec:
    """An output of `case`, whose last action is also an input, so its shape and dtype flow."""
    part, shape, dtype = OUTPUTS[case]
    return PolicySpec.model_validate(
        {
            "name": case.replace("_", "-"),
            "inputs": {
                "parts": [
                    {"name": "state", "source": "state", "shape": [2], "normalize": True},
                    {"name": "previous", "source": "last.out", "shape": shape, "dtype": dtype},
                ]
            },
            "outputs": {"parts": [{"name": "out", **part}]},
            "bookkeeping": [{"type": "last", "source": "out"}],
            "model": {"block": {"type": "mlp", "hidden": [16]}},
        }
    )


@pytest.mark.parametrize("case", OUTPUTS)
def test_every_distribution_exports_with_parity(case: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    learner = Learner(distribution_spec(case), num_envs=4)
    for _ in range(3):
        learner.act({"state": torch.randn(4, 2)})
    export_artifact(learner, tmp_path)
    validate_artifact(tmp_path)
    policy = Policy.load(tmp_path)
    actor = learner.actor.eval()
    size = actor.decoder.distributions["out"].raw_size
    with torch.no_grad():
        previous = actor.head({"out": torch.randn(6, size)}).sample()["out"]
        parts = {"state": torch.randn(6, 2), "previous": previous}
        expected = actor(parts, actor.model.initial_state((6,)))[0].mode()
    raw, _ = policy.run({k: v.numpy() for k, v in parts.items()}, policy.initial_state(6))
    np.testing.assert_allclose(policy.decoder.mode(raw)["out"], expected["out"].numpy(), atol=1e-5)
    _, shape, dtype = OUTPUTS[case]
    policy.reset()
    for _ in range(2):
        out = policy.step({"state": np.zeros(2, np.float32)})["out"]
        assert np.shape(out) == shape
    assert policy.bookkeeping.observe({})["last.out"].dtype == np.dtype(dtype)


def test_int_and_bool_inputs_through_onnx(tmp_path: Path) -> None:
    spec = PolicySpec.model_validate(
        {
            "name": "typed",
            "inputs": {
                "parts": [
                    {
                        "name": "counts",
                        "source": "history.counts",
                        "shape": [3, 2],
                        "dtype": "int64",
                    },
                    {"name": "flag", "source": "flag", "shape": [1], "dtype": "bool"},
                    {"name": "level", "source": "level", "shape": [], "dtype": "int32"},
                ]
            },
            "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 2}}]},
            "bookkeeping": [{"type": "history", "source": "counts", "length": 3}],
            "model": {"block": {"type": "gru", "hidden_size": 8}},
        }
    )
    learner = Learner(spec, num_envs=3)
    learner.act(
        {
            "counts": torch.randint(0, 5, (3, 2)),
            "flag": torch.ones(3, 1, dtype=torch.bool),
            "level": torch.arange(3, dtype=torch.int32),
        }
    )
    export_artifact(learner, tmp_path)
    validate_artifact(tmp_path)
    policy = Policy.load(tmp_path)
    types = {value.name: value.type for value in policy.session.get_inputs()}
    assert types["input.counts"] == "tensor(int64)"
    assert types["input.flag"] == "tensor(bool)"
    assert types["input.level"] == "tensor(int32)"
    observation = {
        "counts": np.array([1, 2], np.int64),
        "flag": np.array([True]),
        "level": np.int32(2),
    }
    policy.step(observation)
    assert policy.bookkeeping.observe(observation, update=False)["history.counts"].dtype == np.int64
    actor = learner.actor.cpu().eval()
    parts = {
        "counts": torch.tensor([[[1, 2], [1, 2], [3, 4]]]),
        "flag": torch.tensor([[False]]),
        "level": torch.tensor([7], dtype=torch.int32),
    }
    with torch.no_grad():
        expected, _ = actor.raw(parts, actor.model.initial_state((1,)))
    raw, _ = policy.run({k: v.numpy() for k, v in parts.items()}, policy.initial_state(1))
    np.testing.assert_allclose(raw["force"], expected["force"].numpy(), atol=1e-5)


def test_export_without_a_learner(tmp_path: Path) -> None:
    spec = distribution_spec("categorical")
    actor = Actor(spec, initial_std=0.3, normalizer_clip=1.0)
    with torch.no_grad():
        actor.normalizer.update({"state": torch.randn(64, 2) * 3 + 10})
    actor.train()
    export_artifact(ExportSource(actor, {"state": (2,)}), tmp_path)
    assert actor.training
    # Inputs far outside the fitted statistics are clipped, so a wrong clip would show.
    validate_artifact(tmp_path)
    saved = torch.load(tmp_path / "weights.pt", weights_only=True)
    assert saved["settings"] == {"initial_std": 0.3, "normalizer_clip": 1.0}
    out = Policy.load(tmp_path).step({"state": np.zeros(2, np.float32)})
    assert out["out"].dtype == np.int64


def test_export_keeps_the_learner_training(tmp_path: Path) -> None:
    learner = Learner(SPEC, num_envs=2)
    learner.act({"state": torch.randn(2, 2), "grid": torch.randn(2, 2, 3)})
    export_artifact(learner, tmp_path)
    assert learner.actor.training


def test_export_embeds_the_canonical_spec(tmp_path: Path) -> None:
    # `model_copy(update=...)` skips validation, so this spec lacks the block's defaults.
    spec = SPEC.model_copy(update={"model": {**SPEC.model, "block": {"type": "mlp"}}})
    manifest = export_artifact(ExportSource(Actor(spec)), tmp_path)
    policy = Policy.load(tmp_path)
    assert policy.fingerprint == manifest.fingerprint == policy.spec.fingerprint()
