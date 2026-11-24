import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import policies

ROOT = Path(policies.__file__).parent

TORCH = ("models",)
"""Subpackages that need the torch extra, along with everything inside them."""


def _subpackages() -> list[str]:
    """Every directory of Python files under `policies`, as dotted names."""
    directories = {path.parent for path in ROOT.rglob("*.py")} - {ROOT}
    return sorted(".".join(directory.relative_to(ROOT).parts) for directory in directories)


def _needs_torch(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in TORCH)


SUBPACKAGES = _subpackages()
LIGHT = [name for name in SUBPACKAGES if not _needs_torch(name)]
HEAVY = [name for name in SUBPACKAGES if _needs_torch(name)]


def test_torch_subpackages_exist() -> None:
    assert set(TORCH) <= set(SUBPACKAGES)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_has_init(name: str) -> None:
    assert (ROOT.joinpath(*name.split(".")) / "__init__.py").is_file()


@pytest.mark.parametrize("name", LIGHT)
def test_light_subpackage_imports(name: str) -> None:
    importlib.import_module(f"policies.{name}")


@pytest.mark.parametrize("name", HEAVY)
def test_torch_subpackage_imports(name: str) -> None:
    pytest.importorskip("torch")
    importlib.import_module(f"policies.{name}")


def test_light_subpackages_do_not_import_torch() -> None:
    code = (
        "import importlib, sys\n"
        f"for name in {LIGHT!r}:\n"
        "    importlib.import_module('policies.' + name)\n"
        "    assert 'torch' not in sys.modules, 'policies.' + name + ' imported torch'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_torch_extra() -> None:
    torch = pytest.importorskip("torch")
    assert torch.ones(2).sum().item() == 2
