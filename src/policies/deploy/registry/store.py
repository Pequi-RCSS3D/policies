"""Artifact registry on any storage."""

import json
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from policies.core.config import canonical
from policies.core.logs import get_logger
from policies.deploy.registry.manifest import MANIFEST, Manifest, read_manifest, write_manifest
from policies.deploy.registry.storage import Storage, StorageConflictError, open_storage

TAGS = "tags.json"
TAG_ATTEMPTS = 5
"""Tries at moving a tag while others change the tags concurrently."""

log = get_logger("policies.deploy")


class ArtifactError(ValueError):
    """An artifact operation was refused."""


def _named(value: Any) -> bool:
    """A list of mappings with distinct `name`s, such as spec parts."""
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        return False
    names = [item.get("name") if isinstance(item, Mapping) else None for item in value]
    return all(isinstance(name, str) for name in names) and len(set(names)) == len(names)


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    """Dotted paths to leaves. Items of named lists go under `prefix[name]`."""
    out: dict[str, Any] = {}
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}"))
        return out
    if _named(value):
        for item in value:
            out.update(_flatten(item, f"{prefix}[{item['name']}]"))
        return out
    return {prefix: value}


def _json(data: Any) -> bytes:
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8")


class ArtifactRegistry:
    """Immutable artifacts under `<root>/<id>/` and moving tags in `<root>/tags.json`."""

    def __init__(self, root: str | Path | Storage, **options: Any) -> None:
        """`root` is a local path, a URL such as `file://`, `hf://` or `s3://`, or a storage.

        `options` go to the storage backend.
        """
        self.storage = open_storage(str(root), **options) if isinstance(root, str | Path) else root

    def register(self, local_dir: str | Path) -> Manifest:
        """Upload an artifact directory. Re-registering identical content is a no-op."""
        local = Path(local_dir)
        manifest = read_manifest(local)
        missing = [name for name in manifest.files if not (local / name).is_file()]
        if missing:
            raise ArtifactError(f"{manifest.id}: files missing from {local}: {missing}")
        try:
            stored = self.get(manifest.id)
        except KeyError:
            files: dict[str, Path | bytes] = {
                f"{manifest.id}/{name}": local / name for name in manifest.files
            }
            # Written last, so an artifact is only listed once complete.
            files[f"{manifest.id}/{MANIFEST}"] = _json(canonical(manifest))
            self.storage.write(files, f"Register {manifest.id}")
            log.info("artifact registered", id=manifest.id, root=self.storage.url)
            return manifest
        self._check_same(stored, manifest, local)
        return manifest

    def _check_same(self, stored: Manifest, manifest: Manifest, local: Path) -> None:
        if canonical(stored) != canonical(manifest):
            raise ArtifactError(
                f"{manifest.id} is already registered with a different manifest. "
                "Artifacts are immutable: give it a new fingerprint."
            )
        for name in manifest.files:
            if (local / name).read_bytes() != self.storage.read(f"{manifest.id}/{name}"):
                raise ArtifactError(
                    f"{manifest.id} is already registered with a different {name}. "
                    "Artifacts are immutable: give it a new fingerprint."
                )

    def get(self, id: str) -> Manifest:
        try:
            data = self.storage.read(f"{id}/{MANIFEST}")
        except FileNotFoundError:
            raise KeyError(f"no artifact {id!r} in {self.storage.url}") from None
        return Manifest.model_validate_json(data)

    def tags(self, revision: str | None = None) -> dict[str, dict[str, str]]:
        """`{tag: {policy name: id}}`, at the storage `revision` or the latest."""
        try:
            data: dict[str, dict[str, str]] = json.loads(self.storage.read(TAGS, revision))
        except FileNotFoundError:
            return {}
        return data

    def list(self, name: str | None = None, tag: str | None = None) -> list[Manifest]:
        """Manifests, oldest first, optionally only one policy name or one tag."""
        manifests = [
            Manifest.model_validate_json(data) for data in self.storage.read_each(MANIFEST).values()
        ]
        if tag is not None:
            tagged = set(self.tags().get(tag, {}).values())
            manifests = [manifest for manifest in manifests if manifest.id in tagged]
        if name is not None:
            manifests = [manifest for manifest in manifests if manifest.name == name]
        return sorted(manifests, key=lambda manifest: (manifest.created, manifest.id))

    def fetch(self, id: str, cache_dir: str | Path) -> Path:
        """Local copy at `cache_dir/<id>`, downloaded if missing."""
        target = Path(cache_dir) / id
        if (target / MANIFEST).is_file():
            return target
        manifest = self.get(id)
        partial = Path(cache_dir) / f".{id}.{uuid.uuid4().hex[:8]}"
        try:
            self.storage.download({f"{id}/{name}": partial / name for name in manifest.files})
            write_manifest(partial, manifest)
            if target.exists():
                shutil.rmtree(target)
            partial.rename(target)
        finally:
            shutil.rmtree(partial, ignore_errors=True)
        return target

    def _edit_tags(self, edit: Callable[[dict[str, dict[str, str]]], None], action: str) -> None:
        """Read, edit and write the tags at one revision, again if someone else wrote between.

        `action`, such as "tag ID as TAG", names the commit and the error.
        """
        for attempt in range(1, TAG_ATTEMPTS + 1):
            revision = self.storage.revision()
            tags = self.tags(revision)
            edit(tags)
            try:
                self.storage.write(
                    {TAGS: _json(tags)}, action[:1].upper() + action[1:], expected=revision
                )
            except StorageConflictError:
                log.debug("tags changed concurrently, retrying", attempt=attempt)
                continue
            return
        raise ArtifactError(
            f"could not {action}: the tags in {self.storage.url} "
            f"changed concurrently {TAG_ATTEMPTS} times in a row. Try again."
        )

    def tag(self, id: str, tag: str) -> Manifest:
        """Point `tag` at this artifact for its policy name."""
        manifest = self.get(id)

        def edit(tags: dict[str, dict[str, str]]) -> None:
            tags.setdefault(tag, {})[manifest.name] = id

        self._edit_tags(edit, f"tag {id} as {tag}")
        log.info("artifact tagged", id=id, tag=tag, root=self.storage.url)
        return manifest

    def untag(self, name: str, tag: str) -> str:
        """Remove `tag` from the policy `name`. Returns the id it pointed at."""
        id = ""

        def edit(tags: dict[str, dict[str, str]]) -> None:
            nonlocal id
            removed = tags.get(tag, {}).pop(name, None)
            if removed is None:
                raise KeyError(f"no {name!r} artifact tagged {tag!r}")
            if not tags[tag]:
                del tags[tag]
            id = removed

        self._edit_tags(edit, f"untag {name} as {tag}")
        log.info("artifact untagged", id=id, tag=tag, root=self.storage.url)
        return id

    def _tagged_id(self, name: str, tag: str) -> str:
        id = self.tags().get(tag, {}).get(name)
        if id is None:
            raise KeyError(f"no {name!r} artifact tagged {tag!r}")
        return id

    def tagged(self, name: str, tag: str) -> Manifest:
        return self.get(self._tagged_id(name, tag))

    def fetch_tagged(self, name: str, tag: str, cache_dir: str | Path) -> Path:
        """`fetch` the artifact `tag` points at for the policy `name`."""
        return self.fetch(self._tagged_id(name, tag), cache_dir)

    def diff(self, id_a: str, id_b: str) -> dict[str, tuple[Any, Any]]:
        """Dotted spec, requirements and metrics paths and `parents` whose values differ.

        Missing are None.
        """
        a, b = self.get(id_a), self.get(id_b)
        flat_a, flat_b = (
            {
                **_flatten(manifest.spec, "spec"),
                **_flatten(manifest.requirements, "requirements"),
                **_flatten(manifest.metrics, "metrics"),
                "parents": list(manifest.parents),
            }
            for manifest in (a, b)
        )
        return {
            key: (flat_a.get(key), flat_b.get(key))
            for key in sorted(set(flat_a) | set(flat_b))
            if flat_a.get(key) != flat_b.get(key)
        }
