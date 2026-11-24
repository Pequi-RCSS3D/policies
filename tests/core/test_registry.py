import sys
import types
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import Field, ValidationError

from policies.core.config import Config, canonical
from policies.core.registry import Entry, PluginGroup, Registry, UnknownComponentError


class Shape:
    def area(self) -> float:
        raise NotImplementedError


class SquareConfig(Config):
    side: float = 1.0


class Square(Shape):
    def __init__(self, config: SquareConfig, scale: float = 1.0) -> None:
        self.side = config.side * scale

    def area(self) -> float:
        return self.side**2


@pytest.fixture
def shapes() -> Registry[Shape]:
    # An empty plugin group, so unknown names do not import installed plugins.
    registry: Registry[Shape] = Registry("shape", installed=PluginGroup("policies.tests.none"))
    registry.register("square", SquareConfig)(Square)
    return registry


def test_build_with_defaults(shapes: Registry[Shape]) -> None:
    assert shapes.build({"type": "square"}).area() == 1.0


def test_build_passes_context(shapes: Registry[Shape]) -> None:
    assert shapes.build({"type": "square", "side": 2.0}, scale=3.0).area() == 36.0


def test_scoped_undoes_registrations_and_replacements(shapes: Registry[Shape]) -> None:
    with shapes.scoped():
        shapes.register("triangle", SquareConfig)(Square)
        shapes.register("square", SquareConfig, replace=True)(lambda config: Square(config, 2.0))
        assert shapes.build({"type": "square"}).area() == 4.0
    assert "triangle" not in shapes
    assert shapes.build({"type": "square"}).area() == 1.0


def test_register_from_outside(shapes: Registry[Shape]) -> None:
    class CircleConfig(Config):
        radius: float

    @shapes.register("circle", CircleConfig)
    class Circle(Shape):
        def __init__(self, config: CircleConfig, **_: Any) -> None:
            self.radius = config.radius

        def area(self) -> float:
            return 3.0 * self.radius**2

    assert "circle" in shapes
    assert shapes.build({"type": "circle", "radius": 2.0}).area() == 12.0


def test_unknown_name_suggests_close_match(shapes: Registry[Shape]) -> None:
    with pytest.raises(
        UnknownComponentError, match=r"unknown shape 'sqare'\. Did you mean 'square'"
    ):
        shapes.build({"type": "sqare"})


def test_missing_type(shapes: Registry[Shape]) -> None:
    with pytest.raises(ValueError, match="needs a 'type' key"):
        shapes.build({"side": 2.0})


def test_bad_field(shapes: Registry[Shape]) -> None:
    with pytest.raises(ValidationError, match="sides"):
        shapes.build({"type": "square", "sides": 4})


def test_duplicate_name(shapes: Registry[Shape]) -> None:
    with pytest.raises(ValueError, match="already registered"):
        shapes.register("square", SquareConfig)(Square)


def test_canonical(shapes: Registry[Shape]) -> None:
    assert shapes.canonical({"type": "square"}) == {"type": "square", "side": 1.0}


def test_config(shapes: Registry[Shape]) -> None:
    assert shapes.config("square") is SquareConfig


def test_entries(shapes: Registry[Shape]) -> None:
    assert shapes.entries() == [Entry("square", SquareConfig, None)]


LAZY = """
from tests_lazy_host import Square
"""


def test_lazy_factory_imports_on_first_build(
    shapes: Registry[Shape], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tests_lazy_factory.py").write_text(LAZY)
    host = types.ModuleType("tests_lazy_host")
    vars(host).update(Square=Square)
    monkeypatch.setitem(sys.modules, "tests_lazy_host", host)
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, "tests_lazy_factory", raising=False)
    shapes.register_lazy("lazy", SquareConfig, "tests_lazy_factory:Square")
    assert shapes.canonical({"type": "lazy"}) == {"type": "lazy", "side": 1.0}
    assert "tests_lazy_factory" not in sys.modules
    assert shapes.build({"type": "lazy", "side": 2.0}, scale=3.0).area() == 36.0
    assert "tests_lazy_factory" in sys.modules


def test_lazy_factory_needs_module_and_attribute(shapes: Registry[Shape]) -> None:
    with pytest.raises(ValueError, match="'module:attribute'"):
        shapes.register_lazy("lazy", SquareConfig, "tests_lazy_factory.make")


def test_lazy_names_conflict(shapes: Registry[Shape]) -> None:
    with pytest.raises(ValueError, match="already registered"):
        shapes.register_lazy("square", SquareConfig, "math:sqrt")


def test_field_validates_to_canonical(shapes: Registry[Shape]) -> None:
    class Drawing(Config):
        shape: Annotated[dict[str, Any], shapes] = Field(default_factory=lambda: {"type": "square"})
        more: tuple[Annotated[dict[str, Any], shapes], ...] = ()

    assert Drawing().shape == {"type": "square", "side": 1.0}
    drawing = Drawing(more=({"type": "square", "side": 2},))
    assert drawing.more == ({"type": "square", "side": 2.0},)
    assert Drawing.model_validate(canonical(drawing)) == drawing


def test_field_reports_errors_at_the_field(shapes: Registry[Shape]) -> None:
    class Drawing(Config):
        shape: Annotated[dict[str, Any], shapes]

    with pytest.raises(ValidationError, match=r"shape\n.*unknown shape 'sqare'"):
        Drawing(shape={"type": "sqare"})
    with pytest.raises(ValidationError, match=r"shape\.sides\n"):
        Drawing(shape={"type": "square", "sides": 4})


def test_nested_components_canonicalize(shapes: Registry[Shape]) -> None:
    class GroupConfig(Config):
        inner: Annotated[dict[str, Any], shapes] = Field(default_factory=lambda: {"type": "square"})

    shapes.register("group", GroupConfig)(lambda config: Square(SquareConfig()))
    assert shapes.canonical({"type": "group"}) == {
        "type": "group",
        "inner": {"type": "square", "side": 1.0},
    }
