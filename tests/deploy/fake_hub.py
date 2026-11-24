"""An in-memory stand-in for the parts of `HfApi` the hub storage uses."""

from collections.abc import Callable, Iterable
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
from huggingface_hub import CommitOperation, CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError


def _response(status: int, url: str) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", url))


class FakeHubApi:
    """One repository with history. Records commits and counts reads."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.history: list[dict[str, bytes]] = [{}]
        self.commits: list[tuple[str, list[str]]] = []
        self.downloads = 0
        self.snapshots = 0
        # One runs before each commit lands, e.g. to slip in a concurrent commit.
        self.before_commit: list[Callable[[], None]] = []

    @property
    def files(self) -> dict[str, bytes]:
        return self.history[-1]

    @property
    def head(self) -> str:
        return f"{len(self.history) - 1:040x}"

    def _at(self, revision: str | None) -> dict[str, bytes]:
        if revision is None or revision == "main":
            return self.files
        return self.history[int(revision, 16)]

    def repo_info(self, repo_id: str, **options: Any) -> SimpleNamespace:
        return SimpleNamespace(sha=self.head)

    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        parent_commit: str | None = None,
        **options: Any,
    ) -> SimpleNamespace:
        if self.before_commit:
            self.before_commit.pop(0)()
        if parent_commit is not None and parent_commit != self.head:
            url = f"https://hub.test/api/{repo_id}/commit/main"
            raise HfHubHTTPError("A commit has happened since", response=_response(412, url))
        files, paths = dict(self.files), []
        for operation in operations:
            assert isinstance(operation, CommitOperationAdd)
            source = operation.path_or_fileobj
            data = source if isinstance(source, bytes) else Path(str(source)).read_bytes()
            files[operation.path_in_repo] = data
            paths.append(operation.path_in_repo)
        self.history.append(files)
        self.commits.append((options["commit_message"], paths))
        return SimpleNamespace(oid=self.head)

    def _copy(self, files: dict[str, bytes], path: str, folder: Path) -> Path:
        local = folder / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(files[path])
        return local

    def hf_hub_download(
        self, repo_id: str, filename: str, revision: str | None = None, **options: Any
    ) -> str:
        self.downloads += 1
        files = self._at(revision)
        if filename not in files:
            url = f"https://hub.test/{repo_id}/{filename}"
            raise RemoteEntryNotFoundError(f"{filename} not found", response=_response(404, url))
        return str(self._copy(files, filename, self.cache / "files" / str(self.downloads)))

    def snapshot_download(self, repo_id: str, allow_patterns: list[str], **options: Any) -> str:
        self.snapshots += 1
        folder = self.cache / "snapshots" / self.head
        folder.mkdir(parents=True, exist_ok=True)
        # The Hub matches `allow_patterns` with fnmatch, where `*` also crosses `/`.
        for path in self.files:
            if any(fnmatch(path, pattern) for pattern in allow_patterns):
                self._copy(self.files, path, folder)
        return str(folder)
