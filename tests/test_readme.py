"""The README examples run as written. Each runs the code blocks marked `<!-- test: name -->`."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from policies.core.config import load_yaml
from policies.io.inputs import transforms

README = Path(__file__).parents[1] / "README.md"
MARKED = re.compile(r"<!-- test: (?P<name>\S+) -->\n```\w+\n(?P<code>.*?)\n```", re.DOTALL)
BLOCKS = {match["name"]: match["code"] for match in MARKED.finditer(README.read_text("utf-8"))}


def run(name: str, namespace: dict[str, Any]) -> dict[str, Any]:
    exec(compile(BLOCKS[name], f"README.md <{name}>", "exec"), namespace)
    return namespace


def test_every_example_is_marked() -> None:
    assert set(BLOCKS) == {
        "cartpole.yaml",
        "train",
        "checkpoint",
        "deploy",
        "config",
        "transform",
        "block",
    }


def test_quickstart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from policies.training.loop import RunConfig

    monkeypatch.chdir(tmp_path)
    Path("cartpole.yaml").write_text(BLOCKS["cartpole.yaml"], encoding="utf-8")
    namespace = run("train", {"__name__": "readme"})
    assert namespace["trainer"].agent.iteration == 10
    assert namespace["manifest"].name == "cartpole"
    namespace = run("checkpoint", namespace)
    assert not namespace["learner"].training
    assert namespace["outputs"]["action"].shape == (16,)
    namespace = run("deploy", namespace)
    assert namespace["done"]
    config = run("config", {"__name__": "readme"})["config"]
    assert config == load_yaml("cartpole.yaml", RunConfig)


def test_custom_transform() -> None:
    with transforms.scoped():
        run("transform", {"__name__": "readme"})
        mirror = transforms.build({"type": "my_package.mirror", "axis": 0})
        assert mirror(np.arange(3), np).tolist() == [2, 1, 0]


def test_custom_block() -> None:
    torch = pytest.importorskip("torch")
    from policies.learning import Actor
    from policies.models import blocks
    from policies.spec import PolicySpec

    with blocks.scoped():
        run("block", {"__name__": "readme"})
        spec = PolicySpec.model_validate(
            {
                "name": "residual",
                "inputs": {"parts": [{"name": "x", "source": "x", "shape": [3]}]},
                "outputs": {"parts": [{"name": "y", "kind": {"type": "continuous", "size": 2}}]},
                "model": {"block": {"type": "my_package.residual", "hidden": 8}},
            }
        )
        raw, _ = Actor(spec).raw({"x": torch.zeros(4, 3)}, {})
        assert raw["y"].shape == (4, 2)
