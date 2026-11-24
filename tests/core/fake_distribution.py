"""Fake installed distributions, written to a directory on `sys.path`."""

import importlib
import runpy
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from policies.io.inputs.transforms import transforms

VERSION = "1.2.0"

ACME = """
from policies.core.config import Config
from policies.io.inputs.transforms import transforms


class MirrorConfig(Config):
    axis: int = 0


@transforms.register("acme.mirror", MirrorConfig)
def mirror(config, **_):
    return lambda x, xp: xp.flip(x, axis=config.axis)
"""
"""A plugin distribution that registers transform `acme.mirror`."""


def install(site: Path, name: str, source: str, entry_points: str = "") -> None:
    """Distribution `name` at `VERSION`, whose one module is `name`."""
    (site / f"{name}.py").write_text(source)
    info = site / f"{name}-{VERSION}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {VERSION}\n")
    (info / "entry_points.txt").write_text(entry_points)
    importlib.invalidate_caches()


@contextmanager
def installed_acme(site: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Installs and runs `acme`, without caching it. Its registration lasts until exit."""
    monkeypatch.syspath_prepend(site)
    with transforms.scoped():
        install(site, "acme", ACME)
        runpy.run_path(str(site / "acme.py"), run_name="acme")
        yield
