"""Storage on any fsspec filesystem."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from fsspec import AbstractFileSystem  # type: ignore[import-untyped]  # fsspec ships no type hints
from fsspec.core import url_to_fs  # type: ignore[import-untyped]  # fsspec ships no type hints


class FilesystemStorage:
    """Plain file writes. Bulk reads and downloads are concurrent on async filesystems."""

    def __init__(self, fs: AbstractFileSystem, root: str, url: str) -> None:
        self.fs, self.root, self.url = fs, root.rstrip("/"), url

    @classmethod
    def from_url(cls, url: str, **options: Any) -> "FilesystemStorage":
        """`options` go to the fsspec filesystem."""
        target = url2pathname(urlparse(url).path) if url.startswith("file://") else url
        fs, root = url_to_fs(target, **options)
        return cls(fs, str(root), url)

    def _path(self, path: str) -> str:
        return f"{self.root}/{path}"

    def revision(self) -> None:
        """Plain filesystems have no revisions."""

    def read(self, path: str, revision: str | None = None) -> bytes:
        data: bytes = self.fs.cat_file(self._path(path))
        return data

    def read_each(self, name: str) -> dict[str, bytes]:
        start = len(self.root) + 1
        paths = [
            path
            for path in self.fs.find(self.root, maxdepth=2)
            if path[start:].partition("/")[2] == name
        ]
        if not paths:
            return {}
        contents: dict[str, bytes] = self.fs.cat(paths)
        return {path[start:].partition("/")[0]: data for path, data in contents.items()}

    def download(self, files: Mapping[str, Path]) -> None:
        for local in files.values():
            local.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get([self._path(path) for path in files], [str(local) for local in files.values()])

    def write(
        self, files: Mapping[str, Path | bytes], message: str, expected: str | None = None
    ) -> None:
        """Best effort: no commits or revisions, so `message` and `expected` are unused."""
        for path, source in files.items():
            target = self._path(path)
            self.fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
            if isinstance(source, bytes):
                self.fs.pipe_file(target, source)
            else:
                self.fs.put_file(str(source), target)
