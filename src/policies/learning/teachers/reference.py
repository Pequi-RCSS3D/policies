"""Where a teacher artifact comes from: a local path or a registry entry."""

from pathlib import Path
from typing import Self

from pydantic import model_validator

from policies.core.config import Config
from policies.deploy.registry import ArtifactRegistry


class ArtifactReference(Config):
    """An artifact in a registry, by `id` or by policy `name` and `tag`."""

    root: str
    """Registry root: a local path or an fsspec URL such as `hf://`."""
    id: str | None = None
    name: str | None = None
    tag: str | None = None
    cache_dir: str = "~/.cache/policies/artifacts"
    """Local directory that fetched artifacts are kept in."""

    @model_validator(mode="after")
    def _addressed(self) -> Self:
        tagged = (self.name, self.tag)
        if (self.id is None) == (tagged == (None, None)) or (self.id is None and None in tagged):
            raise ValueError("an artifact reference needs either `id`, or `name` and `tag`")
        return self

    def fetch(self) -> Path:
        """A local copy of the artifact directory."""
        registry = ArtifactRegistry(self.root)
        id = self.id
        if id is None:
            assert self.name is not None and self.tag is not None
            id = registry.tagged(self.name, self.tag).id
        return registry.fetch(id, Path(self.cache_dir).expanduser())


type TeacherSource = str | ArtifactReference
"""A local artifact path, or a registry reference."""


def artifact_path(source: TeacherSource) -> Path:
    return source.fetch() if isinstance(source, ArtifactReference) else Path(source)
