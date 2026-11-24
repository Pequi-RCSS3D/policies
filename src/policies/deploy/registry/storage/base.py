"""The storage contract and the per-protocol storage registry."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from policies.core.config import Config
from policies.core.registry import Registry


class StorageConflictError(Exception):
    """The root moved past the revision a write expected."""


class Storage(Protocol):
    """Files under one root. Paths are relative to it and use `/`.

    Backends with revisions support optimistic writes: read at `revision()`, then write with
    `expected` set to it. Backends without revisions return None and write unconditionally.
    """

    url: str

    def revision(self) -> str | None:
        """The current revision, or None when the backend has none."""
        ...

    def read(self, path: str, revision: str | None = None) -> bytes:
        """The file at `revision`, else the latest. Raises `FileNotFoundError` when missing."""
        ...

    def read_each(self, name: str) -> dict[str, bytes]:
        """The file `name` inside each top-level directory that has one, keyed by directory."""
        ...

    def download(self, files: Mapping[str, Path]) -> None:
        """Copy each stored path to its local path."""
        ...

    def write(
        self, files: Mapping[str, Path | bytes], message: str, expected: str | None = None
    ) -> str | None:
        """Add or replace files in order, as one commit where the backend has commits.

        Raises `StorageConflictError` if the root moved past `expected`. Returns the new revision.
        """
        ...


class StorageConfig(Config):
    url: str


storages: Registry[Storage] = Registry("storage")
"""Storages by URL protocol. Protocols without one use fsspec."""
