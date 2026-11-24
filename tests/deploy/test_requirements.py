from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from policies.deploy.inference.requirements import requirements, unmet
from policies.spec import PolicySpec
from tests.core.fake_distribution import VERSION, installed_acme

POLICIES = metadata.version("policies")


def spec(*transforms: dict[str, Any]) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "name": "walker",
            "inputs": {"parts": [{"name": "x", "source": "x", "transforms": list(transforms)}]},
            "outputs": {"parts": [{"name": "y", "kind": {"type": "continuous", "size": 1}}]},
        }
    )


def test_builtins_require_policies() -> None:
    assert requirements(spec({"type": "scale", "factor": 2.0})) == {"policies": POLICIES}


def test_plugin_distribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with installed_acme(tmp_path, monkeypatch):
        required = requirements(spec({"type": "acme.mirror"}))
    assert required == {"acme": VERSION, "policies": POLICIES}


def test_unmet() -> None:
    assert unmet({"policies": POLICIES}) == {}
    assert unmet({"policies": "0.0.1", "no-such-distribution": "1.0"}) == {
        "policies": POLICIES,
        "no-such-distribution": None,
    }
