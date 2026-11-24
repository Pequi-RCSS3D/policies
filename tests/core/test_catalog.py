import importlib
import sys
from pathlib import Path

import pytest

import policies
from policies.core.registry import (
    Catalog,
    PluginError,
    Registry,
    UnknownComponentError,
    registries,
)
from tests.core.fake_distribution import install

GROUP = "policies.tests.registries"

SHAPES = """
from policies.core.registry import Registry

shapes = Registry("shape")
size = 1
"""


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, "shapes_module", raising=False)
    points = f"[{GROUP}]\nshapes = shapes_module:shapes\nbroken = shapes_module:size\n"
    install(tmp_path, "shapes_module", SHAPES, points)
    return Catalog(GROUP)


def test_names_and_lookup(catalog: Catalog) -> None:
    assert catalog.names() == ["broken", "shapes"]
    assert catalog["shapes"].kind == "shape"


def test_unknown_registry(catalog: Catalog) -> None:
    with pytest.raises(UnknownComponentError, match=r"unknown registry 'shape'\. Did you mean"):
        catalog["shape"]


def test_entry_point_must_be_a_registry(catalog: Catalog) -> None:
    with pytest.raises(PluginError, match="is not a Registry"):
        catalog["broken"]


def _modules() -> list[str]:
    root = Path(policies.__file__).parent
    paths = (path.relative_to(root).with_suffix("") for path in root.rglob("*.py"))
    return [".".join(("policies", *path.parts)).removesuffix(".__init__") for path in paths]


def test_every_registry_is_in_the_catalog() -> None:
    pytest.importorskip("torch")
    defined = {
        id(value): f"{name}.{key}"
        for name in _modules()
        for key, value in vars(importlib.import_module(name)).items()
        if isinstance(value, Registry)
    }
    declared = {id(registries[name]) for name in registries.names()}
    assert sorted(defined[key] for key in defined.keys() - declared) == []
    assert declared <= defined.keys()
