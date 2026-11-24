import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from huggingface_hub import HfApi, parse_hf_uri

from policies.deploy.registry import MANIFEST, ArtifactError, ArtifactRegistry
from policies.deploy.registry.storage import FilesystemStorage, open_storage
from policies.deploy.registry.storage.hub import HubStorage
from policies.deploy.registry.store import TAG_ATTEMPTS
from tests.deploy.fake_hub import FakeHubApi
from tests.deploy.test_artifact_registry import make_artifact


@pytest.fixture
def api(tmp_path: Path) -> FakeHubApi:
    return FakeHubApi(tmp_path / "hub-cache")


def hub_registry(api: FakeHubApi, url: str = "hf://org/repo") -> ArtifactRegistry:
    return ArtifactRegistry(HubStorage(cast(HfApi, api), parse_hf_uri(url)))


def test_register_and_tag_are_one_commit_each(tmp_path: Path, api: FakeHubApi) -> None:
    registry = hub_registry(api)
    manifest = registry.register(make_artifact(tmp_path / "a"))
    registry.tag(manifest.id, "prod")
    assert api.commits == [
        (
            f"Register {manifest.id}",
            [
                f"{manifest.id}/policy.onnx",
                f"{manifest.id}/nested/weights.pt",
                f"{manifest.id}/{MANIFEST}",
            ],
        ),
        (f"Tag {manifest.id} as prod", ["tags.json"]),
    ]


def test_registering_again_makes_no_commit(tmp_path: Path, api: FakeHubApi) -> None:
    registry = hub_registry(api)
    registry.register(make_artifact(tmp_path / "a"))
    registry.register(make_artifact(tmp_path / "same"))
    assert len(api.commits) == 1


def test_list_is_one_snapshot(tmp_path: Path, api: FakeHubApi) -> None:
    registry = hub_registry(api)
    for index in range(3):
        registry.register(make_artifact(tmp_path / str(index), fingerprint=str(index) * 64))
    api.downloads = 0
    assert len(registry.list()) == 3
    assert (api.snapshots, api.downloads) == (1, 0)


def concurrent_tag(api: FakeHubApi, tag: str, name: str, id: str) -> Callable[[], None]:
    """Someone else's commit to the tags, landing between our read and our write."""

    def commit() -> None:
        tags = json.loads(api.files.get("tags.json", b"{}"))
        tags.setdefault(tag, {})[name] = id
        api.history.append({**api.files, "tags.json": json.dumps(tags).encode("utf-8")})

    return commit


def test_concurrent_tag_is_not_lost(tmp_path: Path, api: FakeHubApi) -> None:
    registry = hub_registry(api)
    walker = registry.register(make_artifact(tmp_path / "a"))
    kicker = registry.register(make_artifact(tmp_path / "b", name="kicker"))
    api.before_commit = [concurrent_tag(api, "prod", kicker.name, kicker.id)]
    registry.tag(walker.id, "prod")
    assert registry.tags() == {"prod": {"kicker": kicker.id, "walker": walker.id}}
    assert [message for message, _ in api.commits].count(f"Tag {walker.id} as prod") == 1


def test_tag_gives_up_after_repeated_conflicts(tmp_path: Path, api: FakeHubApi) -> None:
    registry = hub_registry(api)
    walker = registry.register(make_artifact(tmp_path / "a"))
    api.before_commit = [
        concurrent_tag(api, f"other-{index}", "kicker", "kicker-1") for index in range(TAG_ATTEMPTS)
    ]
    with pytest.raises(ArtifactError, match="changed concurrently"):
        registry.tag(walker.id, "prod")
    assert "prod" not in registry.tags()


def test_folder_root(tmp_path: Path, api: FakeHubApi) -> None:
    registry = hub_registry(api, "hf://datasets/org/repo@main/team/policies")
    manifest = registry.register(make_artifact(tmp_path / "a"))
    registry.tag(manifest.id, "prod")
    assert f"team/policies/{manifest.id}/{MANIFEST}" in api.files
    assert "team/policies/tags.json" in api.files
    assert [listed.id for listed in registry.list(tag="prod")] == [manifest.id]
    local = registry.fetch(manifest.id, tmp_path / "cache")
    assert (local / "nested" / "weights.pt").read_bytes() == b"weights"


def test_open_storage_picks_by_protocol(tmp_path: Path) -> None:
    hub = open_storage("hf://org/repo@main/folder")
    assert isinstance(hub, HubStorage)
    assert (hub.repo_id, hub.reference, hub.prefix) == ("org/repo", "main", "folder/")
    assert isinstance(open_storage("hf://buckets/org/bucket"), FilesystemStorage)
    assert isinstance(open_storage(str(tmp_path)), FilesystemStorage)
