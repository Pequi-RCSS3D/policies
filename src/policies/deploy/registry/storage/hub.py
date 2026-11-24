"""Storage in a Hugging Face Hub repository. Needs the `hub` extra."""

import glob
import shutil
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, HfUri, parse_hf_uri
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError

from policies.deploy.registry.storage.base import Storage, StorageConflictError
from policies.deploy.registry.storage.filesystem import FilesystemStorage


class HubStorage:
    """One commit per write. Reads go through the local Hugging Face cache, pinned to a commit."""

    def __init__(self, api: HfApi, location: HfUri) -> None:
        self.api, self.url = api, location.to_uri()
        self.repo_id, self.repo_type, self.reference = location.id, location.type, location.revision
        folder = location.path_in_repo.strip("/")
        self.prefix = f"{folder}/" if folder else ""

    @classmethod
    def from_url(cls, url: str, **options: Any) -> Storage:
        """`options` go to `HfApi`. Buckets have no commits, so they use fsspec."""
        location = parse_hf_uri(url)
        if location.is_bucket:
            return FilesystemStorage.from_url(url, **options)
        return cls(HfApi(**options), location)

    def _snapshot(self, patterns: list[str]) -> Path:
        """Local folder with the matching files at the current commit. Cached files are reused."""
        folder: str = self.api.snapshot_download(
            self.repo_id,
            repo_type=self.repo_type,
            revision=self.reference,
            allow_patterns=patterns,
        )
        return Path(folder) / self.prefix

    def revision(self) -> str | None:
        """The commit the root's branch or tag points at."""
        return self.api.repo_info(
            self.repo_id, repo_type=self.repo_type, revision=self.reference
        ).sha

    def read(self, path: str, revision: str | None = None) -> bytes:
        """At a commit, cached files need no network round trip."""
        try:
            local = self.api.hf_hub_download(
                self.repo_id,
                self.prefix + path,
                repo_type=self.repo_type,
                revision=revision or self.reference,
            )
        except RemoteEntryNotFoundError as error:
            raise FileNotFoundError(f"{self.url}/{path}") from error
        return Path(local).read_bytes()

    def read_each(self, name: str) -> dict[str, bytes]:
        escaped = glob.escape(name)
        folder = self._snapshot([f"{glob.escape(self.prefix)}*/{escaped}"])
        return {path.parent.name: path.read_bytes() for path in folder.glob(f"*/{escaped}")}

    def download(self, files: Mapping[str, Path]) -> None:
        folder = self._snapshot([glob.escape(self.prefix + path) for path in files])
        for path, local in files.items():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / path, local)

    def write(
        self, files: Mapping[str, Path | bytes], message: str, expected: str | None = None
    ) -> str | None:
        operations = [
            CommitOperationAdd(path_in_repo=self.prefix + path, path_or_fileobj=source)
            for path, source in files.items()
        ]
        try:
            commit = self.api.create_commit(
                self.repo_id,
                operations,
                commit_message=message,
                repo_type=self.repo_type,
                revision=self.reference,
                parent_commit=expected,
            )
        except HfHubHTTPError as error:
            # The Hub refuses a stale `parent_commit` with 412 Precondition Failed (checked live).
            if error.response.status_code == HTTPStatus.PRECONDITION_FAILED:
                raise StorageConflictError(f"{self.url} moved past {expected}") from error
            raise
        return commit.oid
