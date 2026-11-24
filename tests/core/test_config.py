from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from policies.core.config import Config, canonical, dump_yaml, load_yaml


class Optimizer(Config):
    type: str = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)


class Run(Config):
    name: str
    iterations: int = 1000
    optimizer: Optimizer = Optimizer()


def test_yaml_round_trip(tmp_path: Path) -> None:
    run = Run(name="walk", optimizer=Optimizer(lr=1e-4))
    path = tmp_path / "run.yaml"
    dump_yaml(run, path)
    assert load_yaml(path, Run) == run


def test_canonical_fills_defaults() -> None:
    assert canonical(Run(name="walk")) == {
        "name": "walk",
        "iterations": 1000,
        "optimizer": {"type": "adamw", "lr": 3e-4, "betas": [0.9, 0.999]},
    }


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ValidationError, match="iteration"):
        Run.model_validate({"name": "walk", "iteration": 5})


def test_wrong_type_is_an_error() -> None:
    with pytest.raises(ValidationError, match="iterations"):
        Run.model_validate({"name": "walk", "iterations": "many"})


def test_configs_are_immutable() -> None:
    run = Run(name="walk")
    with pytest.raises(ValidationError):
        run.name = "kick"  # type: ignore[misc]  # assigning a frozen field on purpose


def test_empty_yaml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_yaml(path, Optimizer) == Optimizer()


class Returns(Config):
    lambda_: float = Field(0.95, alias="lambda")


def test_canonical_round_trips_aliases() -> None:
    assert canonical(Returns()) == {"lambda": 0.95}
    assert Returns.model_validate(canonical(Returns())) == Returns()


def test_defaults_are_validated() -> None:
    class Bad(Config):
        size: int = Field("many")  # type: ignore[assignment]  # an invalid default on purpose

    with pytest.raises(ValidationError, match="size"):
        Bad()
