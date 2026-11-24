from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from policies.io.outputs import KindWithExtras, OutputDecoder, Outputs, kinds
from policies.io.outputs.kinds import ShapedConfig

type Make = Callable[..., Any]

OUTPUTS = Outputs.model_validate(
    {
        "parts": [
            {"name": "move", "kind": {"type": "continuous", "size": 2, "low": -1, "high": 1}},
            {"name": "progress", "kind": {"type": "scalar", "low": 0, "high": 1}},
            {
                "name": "skill",
                "kind": {"type": "categorical", "choices": ["walk", "kick", "stand"]},
            },
        ]
    }
)


def test_decode(make: Make) -> None:
    decoder = OutputDecoder(OUTPUTS)
    raw = {
        "move": make((4, 2), fill=3.0),
        "progress": make((4, 1), fill=-0.5),
        "skill": make((4, 3)),
    }
    raw["skill"][:, 1] = 1.0
    values = decoder(raw)
    assert float(values["move"][0, 0]) == 1.0
    assert values["progress"].shape == (4,)
    assert float(values["progress"][0]) == 0.0
    assert int(values["skill"][0]) == 1


def test_shapes_and_default_distributions() -> None:
    decoder = OutputDecoder(OUTPUTS)
    assert decoder.shapes == {"move": (2,), "progress": (), "skill": ()}
    assert decoder.dtypes == {"move": "float32", "progress": "float32", "skill": "int64"}
    sizes = {name: part.raw_size for name, part in decoder.distributions.items()}
    assert sizes == {"move": 2, "progress": 1, "skill": 3}


def test_multi_dimensional_continuous() -> None:
    outputs = Outputs.model_validate(
        {"parts": [{"name": "grid", "kind": {"type": "continuous", "shape": [2, 3]}}]}
    )
    decoder = OutputDecoder(outputs)
    assert decoder.shapes == {"grid": (2, 3)}
    values = decoder({"grid": np.arange(12, dtype=np.float32).reshape(2, 6)})
    assert values["grid"].shape == (2, 2, 3)


def test_size_or_shape() -> None:
    with pytest.raises(ValidationError, match="either"):
        Outputs.model_validate(
            {"parts": [{"name": "x", "kind": {"type": "continuous", "size": 2, "shape": [2]}}]}
        )


def test_distribution_override() -> None:
    outputs = Outputs.model_validate(
        {
            "parts": [
                {"name": "a", "kind": {"type": "scalar"}, "distribution": {"initial_std": 0.5}},
                {
                    "name": "b",
                    "kind": {"type": "continuous", "size": 2},
                    "distribution": {"type": "normal_state_std"},
                },
                {
                    "name": "c",
                    "kind": {"type": "categorical", "choices": ["x", "y"]},
                    "distribution": {"type": "categorical"},
                },
            ]
        }
    )
    decoder = OutputDecoder(outputs)
    np.testing.assert_allclose(np.exp(decoder.distributions["a"].learned["log_std"]), 0.5)
    assert decoder.distributions["b"].raw_size == 4
    assert decoder.distributions["c"].raw_size == 2


def test_distribution_is_canonical() -> None:
    part = OUTPUTS.parts[2]
    assert part.distribution == {"type": "categorical", "choices": 3}
    assert Outputs.model_validate(OUTPUTS.model_dump(mode="json")) == OUTPUTS


def test_unknown_distribution() -> None:
    with pytest.raises(ValidationError, match="unknown distribution 'nope'"):
        Outputs.model_validate(
            {"parts": [{"name": "x", "kind": {"type": "scalar"}, "distribution": {"type": "nope"}}]}
        )


def test_unknown_kind() -> None:
    with pytest.raises(Exception, match="unknown output kind"):
        Outputs.model_validate({"parts": [{"name": "x", "kind": {"type": "nope"}}]})


def test_duplicate_parts() -> None:
    kind = {"type": "scalar"}
    with pytest.raises(ValidationError, match="duplicate output parts"):
        Outputs.model_validate(
            {"parts": [{"name": "x", "kind": kind}, {"name": "x", "kind": kind}]}
        )


def test_numpy_decode_keeps_dtype() -> None:
    values = OutputDecoder(OUTPUTS)(
        {
            "move": np.zeros(2, np.float32),
            "progress": np.zeros(1, np.float32),
            "skill": np.zeros(3, np.float32),
        }
    )
    assert values["move"].dtype == np.float32


class GainedConfig(ShapedConfig):
    kp: float = 1.0
    kd: float = 0.1


class Gained:
    """Targets with controller gains as extras, like a joint position controller."""

    distribution: Mapping[str, Any] = {"type": "normal"}

    def __init__(self, config: GainedConfig, **_: Any) -> None:
        self.config = config
        self.shape = config.event_shape

    def decode(self, action: Any, xp: Any) -> Any:
        return action

    def extras(self, action: Any, xp: Any) -> dict[str, Any]:
        return {
            "kp": xp.full(action.shape, self.config.kp, dtype=action.dtype),
            "kd": xp.full(action.shape, self.config.kd, dtype=action.dtype),
        }


GAINED = {"type": "tests.gained", "size": 2, "kp": 40.0}


@contextmanager
def gained() -> Iterator[None]:
    with kinds.scoped():
        kinds.register("tests.gained", GainedConfig)(Gained)
        yield


def test_kinds_may_have_extras() -> None:
    with gained():
        outputs = Outputs.model_validate(
            {"parts": [*OUTPUTS.model_dump()["parts"], {"name": "joints", "kind": GAINED}]}
        )
        decoder = OutputDecoder(outputs)
        assert isinstance(decoder.kinds["joints"], KindWithExtras)
        assert not isinstance(decoder.kinds["move"], KindWithExtras)
        actions = {name: np.zeros(shape, np.float32) for name, shape in decoder.shapes.items()}
        extras = decoder.extras(actions)
    assert list(extras) == ["joints"]
    np.testing.assert_array_equal(extras["joints"]["kp"], [40.0, 40.0])
    np.testing.assert_array_equal(extras["joints"]["kd"], np.array([0.1, 0.1], np.float32))


def test_parts_by_name_and_kind_type() -> None:
    assert OUTPUTS["skill"].kind_type == "categorical"
    with pytest.raises(KeyError, match="no output part 'nope'"):
        OUTPUTS["nope"]
