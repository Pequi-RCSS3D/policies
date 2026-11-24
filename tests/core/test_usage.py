from typing import Annotated, Any

from pydantic import Field

from policies.core.config import Config
from policies.core.registry import PluginGroup, Registry, used_components

NONE = PluginGroup("policies.tests.none")

shapes: Registry[None] = Registry("shape", installed=NONE)
colors: Registry[None] = Registry("color", installed=NONE)

type ShapeConfig = Annotated[dict[str, Any], shapes]
type ColorConfig = Annotated[dict[str, Any], colors]


class Plain(Config):
    pass


class GroupConfig(Config):
    """A component whose config holds other components."""

    members: tuple[ShapeConfig, ...] = ()


class Layer(Config):
    fill: ColorConfig


class Scene(Config):
    main: ShapeConfig
    outline: ColorConfig | None = None
    palette: dict[str, ColorConfig] = Field(default_factory=dict)
    layers: list[Layer] = Field(default_factory=list)
    pair: tuple[ShapeConfig, ColorConfig] | None = None
    count: int = 0


shapes.register("square", Plain)(lambda config: None)
shapes.register("group", GroupConfig)(lambda config: None)
colors.register("red", Plain)(lambda config: None)
colors.register("blue", Plain)(lambda config: None)
colors.register("green", Plain)(lambda config: None)


def test_finds_components_wherever_annotated() -> None:
    scene = Scene.model_validate(
        {
            "main": {"type": "group", "members": [{"type": "square"}]},
            "palette": {"sky": {"type": "blue"}},
            "layers": [{"fill": {"type": "green"}}],
            "pair": [{"type": "square"}, {"type": "red"}],
        }
    )
    assert set(used_components(scene)) == {
        (shapes, "group"),
        (shapes, "square"),
        (colors, "blue"),
        (colors, "green"),
        (colors, "red"),
    }


def test_skips_empty_optionals() -> None:
    assert list(used_components(Scene.model_validate({"main": {"type": "square"}}))) == [
        (shapes, "square")
    ]
