from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from policies.core.parts import PartError
from policies.io.inputs import Inputs

type Make = Callable[..., Any]

INPUTS = Inputs.model_validate(
    {
        "parts": [
            {
                "name": "speed",
                "source": "velocity",
                "shape": [2],
                "transforms": [{"type": "scale", "factor": 2.0}],
            },
            {
                "name": "command",
                "source": "command",
                "shape": [3],
                "transforms": [{"type": "clip", "low": -1, "high": 1}],
            },
            {"name": "previous", "source": "last.action", "shape": [1]},
        ]
    }
)


def test_parts_from_observation_and_slots(make: Make) -> None:
    pipeline = INPUTS.compile()
    parts = pipeline(
        {"velocity": make((4, 2), fill=1.0), "command": make((4, 3), fill=5.0)},
        {"last.action": make((4, 1))},
    )
    assert list(parts) == ["speed", "command", "previous"]
    assert float(parts["speed"][0, 0]) == 2.0
    assert float(parts["command"][0, 0]) == 1.0


def test_missing_source(make: Make) -> None:
    with pytest.raises(PartError, match=r"no source 'last\.action'"):
        INPUTS.compile()({"velocity": make((2,)), "command": make((3,))})


def test_unknown_transform() -> None:
    with pytest.raises(Exception, match="unknown transform 'sacle'"):
        Inputs.model_validate(
            {"parts": [{"name": "x", "source": "x", "transforms": [{"type": "sacle"}]}]}
        )


def test_duplicate_parts() -> None:
    with pytest.raises(ValidationError, match="duplicate input parts"):
        Inputs.model_validate(
            {"parts": [{"name": "x", "source": "a"}, {"name": "x", "source": "b"}]}
        )
