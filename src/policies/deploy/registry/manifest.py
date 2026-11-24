"""The manifest that describes an artifact directory."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from policies.core.config import Config, canonical
from policies.core.fingerprint import short

MANIFEST = "manifest.json"


def artifact_id(name: str, fingerprint: str) -> str:
    return f"{name}-{short(fingerprint)}"


def now() -> str:
    """Current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class Manifest(Config):
    """What an artifact is, where it came from and how it scored."""

    id: str
    name: str
    fingerprint: str
    created: str = Field(default_factory=now)
    spec: dict[str, Any] = Field(default_factory=dict)
    requirements: dict[str, str] = Field(default_factory=dict)
    """Distribution versions that provided the spec's components at export."""
    metrics: dict[str, float] = Field(default_factory=dict)
    parents: tuple[str, ...] = ()
    """Ids of the artifacts it derives from, such as the teachers it was distilled from."""
    files: tuple[str, ...] = ()
    run: dict[str, Any] = Field(default_factory=dict)


def write_manifest(directory: str | Path, manifest: Manifest) -> Path:
    path = Path(directory) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(directory: str | Path) -> Manifest:
    path = Path(directory) / MANIFEST
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
