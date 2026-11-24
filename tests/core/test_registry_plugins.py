import sys
import types
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from policies.core.config import Config
from policies.core.registry import (
    DuplicateComponentError,
    Entry,
    Origin,
    PluginError,
    PluginGroup,
    Registry,
    UnknownComponentError,
)
from tests.core import fake_distribution

GROUP = "policies.tests.plugins"

CIRCLES = """
from shapes_host import SizeConfig, shapes

shapes.names()  # looking up names while loading must not recurse

@shapes.register("circles.circle", SizeConfig)
def circle(config):
    return 3.0 * config.size**2
"""

Install = Callable[[str, str], None]


class SizeConfig(Config):
    size: float = 1.0


def square(config: SizeConfig) -> float:
    return config.size**2


@pytest.fixture
def shapes(monkeypatch: pytest.MonkeyPatch) -> Registry[float]:
    """A registry with one built-in, importable by plugins as `shapes_host`."""
    registry: Registry[float] = Registry("shape", installed=PluginGroup(GROUP))
    registry.register("square", SizeConfig)(square)
    host = types.ModuleType("shapes_host")
    vars(host).update(shapes=registry, SizeConfig=SizeConfig)
    monkeypatch.setitem(sys.modules, "shapes_host", host)
    return registry


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Install]:
    """Installs a distribution whose one module is its plugin entry point."""
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(site)
    packages: list[str] = []

    def install(name: str, source: str) -> None:
        fake_distribution.install(site, name, source, f"[{GROUP}]\n{name} = {name}\n")
        packages.append(name)

    yield install
    for package in packages:
        sys.modules.pop(package, None)


def test_plugins_load_lazily(shapes: Registry[float], install: Install) -> None:
    install("circles", CIRCLES)
    assert shapes.build({"type": "square"}) == 1.0
    assert "circles" not in sys.modules
    assert shapes.build({"type": "circles.circle", "size": 2.0}) == 12.0


def test_names_load_plugins(shapes: Registry[float], install: Install) -> None:
    install("circles", CIRCLES)
    assert shapes.names() == ["circles.circle", "square"]


def test_plugins_are_discovered_once(
    shapes: Registry[float], install: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    install("circles", CIRCLES)
    calls: list[dict[str, Any]] = []
    discover = metadata.entry_points

    def spy(**selection: Any) -> metadata.EntryPoints:
        calls.append(selection)
        return discover(**selection)

    monkeypatch.setattr(metadata, "entry_points", spy)
    assert "missing" not in shapes
    assert "circles.circle" in shapes
    shapes.names()
    assert calls == [{"group": GROUP}]


def test_origin(shapes: Registry[float], install: Install) -> None:
    install("circles", CIRCLES)
    assert shapes.origin("circles.circle") == Origin("circles", "1.2.0")
    assert shapes.origin("square") is None


def test_origin_of_builtins() -> None:
    from policies.io.inputs.transforms import transforms

    assert transforms.origin("scale") == Origin("policies", metadata.version("policies"))


def test_origin_of_unknown_name(shapes: Registry[float]) -> None:
    with pytest.raises(UnknownComponentError, match="unknown shape 'circle'"):
        shapes.origin("circle")


def test_close_match_includes_plugins(shapes: Registry[float], install: Install) -> None:
    install("circles", CIRCLES)
    with pytest.raises(UnknownComponentError, match=r"Did you mean 'circles\.circle'"):
        shapes.build({"type": "circles.circel"})


def test_duplicate_names_the_registrant(shapes: Registry[float]) -> None:
    with pytest.raises(DuplicateComponentError, match=r"'square' is already registered by .*"):
        shapes.register("square", SizeConfig)(square)


def test_replace(shapes: Registry[float]) -> None:
    shapes.register("square", SizeConfig, replace=True)(lambda config: 0.0)
    assert shapes.build({"type": "square"}) == 0.0


def test_plugin_conflict(shapes: Registry[float], install: Install) -> None:
    install(
        "squares",
        "from shapes_host import SizeConfig, shapes\n"
        "shapes.register('square', SizeConfig)(lambda config: 0.0)\n",
    )
    with pytest.raises(PluginError) as caught:
        shapes.names()
    assert "entry point 'squares' = 'squares' from squares 1.2.0" in str(caught.value)
    assert "DuplicateComponentError: shape 'square' is already registered" in str(caught.value)
    assert shapes.build({"type": "square"}) == 1.0


def test_plugins_cannot_replace(shapes: Registry[float], install: Install) -> None:
    install(
        "squares",
        "from shapes_host import SizeConfig, shapes\n"
        "shapes.register('square', SizeConfig, replace=True)(lambda config: 0.0)\n",
    )
    with pytest.raises(PluginError, match="installed plugins load lazily and cannot replace"):
        shapes.names()
    assert shapes.build({"type": "square"}) == 1.0


def test_broken_plugin(shapes: Registry[float], install: Install) -> None:
    install("broken", "raise ImportError('missing dependency')\n")
    install("circles", CIRCLES)
    message = r"'broken' = 'broken' from broken 1\.2\.0: ImportError: missing dependency"
    with pytest.raises(PluginError, match=message):
        shapes.build({"type": "triangle"})
    with pytest.raises(PluginError, match=message):
        shapes.build({"type": "triangle"})
    assert shapes.build({"type": "square"}) == 1.0
    assert shapes.build({"type": "circles.circle"}) == 3.0


def test_entries_carry_origins(shapes: Registry[float], install: Install) -> None:
    install("circles", CIRCLES)
    assert shapes.entries() == [
        Entry("circles.circle", SizeConfig, Origin("circles", "1.2.0")),
        Entry("square", SizeConfig, None),
    ]


def test_builtins_need_no_entry_points() -> None:
    """Built-ins register on import, so only other distributions declare plugins."""
    points = metadata.distribution("policies").entry_points.select(group="policies.plugins")
    assert not list(points)
