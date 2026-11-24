from pathlib import Path
from typing import Any, cast

import pytest
from huggingface_hub import HfApi, parse_hf_uri

from policies.deploy.registry import (
    ArtifactError,
    ArtifactRegistry,
    Manifest,
    artifact_id,
    read_manifest,
    write_manifest,
)
from policies.deploy.registry.storage.hub import HubStorage
from tests.deploy.fake_hub import FakeHubApi


def make_artifact(
    base: Path,
    name: str = "walker",
    fingerprint: str = "a" * 64,
    created: str = "2026-01-01T00:00:00+00:00",
    weights: bytes = b"weights",
    **fields: Any,
) -> Path:
    id = artifact_id(name, fingerprint)
    directory = base / id
    (directory / "nested").mkdir(parents=True)
    (directory / "policy.onnx").write_bytes(b"onnx")
    (directory / "nested" / "weights.pt").write_bytes(weights)
    manifest = Manifest(
        id=id,
        name=name,
        fingerprint=fingerprint,
        created=created,
        files=("policy.onnx", "nested/weights.pt"),
        **fields,
    )
    write_manifest(directory, manifest)
    return directory


@pytest.fixture(params=["local", "hub"])
def registry(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactRegistry:
    """The same behavior on a local directory and on a faked Hub repository."""
    if request.param == "hub":
        api = FakeHubApi(tmp_path / "hub-cache")
        return ArtifactRegistry(HubStorage(cast(HfApi, api), parse_hf_uri("hf://org/repo")))
    return ArtifactRegistry(tmp_path / "registry")


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(id="x-1", name="x", fingerprint="1", metrics={"reward": 2.0})
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path) == manifest
    assert manifest.created.startswith("20")


def test_artifact_id() -> None:
    assert artifact_id("walker", "0123456789abcdef") == "walker-0123456789ab"


def test_register_get_and_list(tmp_path: Path, registry: ArtifactRegistry) -> None:
    late = make_artifact(tmp_path / "a", fingerprint="b" * 64, created="2026-02-01T00:00:00")
    early = make_artifact(tmp_path / "b", fingerprint="c" * 64, created="2026-01-01T00:00:00")
    other = make_artifact(tmp_path / "c", name="kicker", created="2026-03-01T00:00:00")
    for directory in (late, early, other):
        registry.register(directory)
    assert [manifest.id for manifest in registry.list()] == [early.name, late.name, other.name]
    assert [manifest.id for manifest in registry.list(name="walker")] == [early.name, late.name]
    assert registry.get(other.name).name == "kicker"
    with pytest.raises(KeyError):
        registry.get("missing")


def test_empty_registry(registry: ArtifactRegistry) -> None:
    assert registry.list() == []
    assert registry.tags() == {}


def test_file_url_root(tmp_path: Path) -> None:
    registry = ArtifactRegistry((tmp_path / "root").as_uri())
    registry.register(make_artifact(tmp_path / "a"))
    assert len(registry.list()) == 1
    assert (tmp_path / "root" / artifact_id("walker", "a" * 64) / "nested" / "weights.pt").exists()


def test_artifacts_are_immutable(tmp_path: Path, registry: ArtifactRegistry) -> None:
    original = make_artifact(tmp_path / "a")
    registry.register(original)
    registry.register(make_artifact(tmp_path / "same"))
    with pytest.raises(ArtifactError, match=r"weights\.pt"):
        registry.register(make_artifact(tmp_path / "files", weights=b"other"))
    with pytest.raises(ArtifactError, match="manifest"):
        registry.register(make_artifact(tmp_path / "meta", metrics={"reward": 1.0}))
    assert registry.get(original.name).metrics == {}


def test_register_needs_all_files(tmp_path: Path, registry: ArtifactRegistry) -> None:
    directory = make_artifact(tmp_path / "a")
    (directory / "policy.onnx").unlink()
    with pytest.raises(ArtifactError, match=r"policy\.onnx"):
        registry.register(directory)
    assert registry.list() == []


def test_tags_move_per_policy_name(tmp_path: Path, registry: ArtifactRegistry) -> None:
    first = registry.register(make_artifact(tmp_path / "a", fingerprint="1" * 64))
    second = registry.register(make_artifact(tmp_path / "b", fingerprint="2" * 64))
    kicker = registry.register(make_artifact(tmp_path / "c", name="kicker", created="2026-05"))
    registry.tag(first.id, "prod")
    registry.tag(kicker.id, "prod")
    assert registry.tagged("walker", "prod").id == first.id
    registry.tag(second.id, "prod")
    assert registry.tagged("walker", "prod").id == second.id
    assert registry.tagged("kicker", "prod").id == kicker.id
    assert [manifest.id for manifest in registry.list(tag="prod")] == [second.id, kicker.id]
    assert [manifest.id for manifest in registry.list(name="walker", tag="prod")] == [second.id]
    with pytest.raises(KeyError):
        registry.tagged("walker", "staging")
    with pytest.raises(KeyError):
        registry.tag("missing", "prod")


def test_diff(tmp_path: Path, registry: ArtifactRegistry) -> None:
    a = registry.register(
        make_artifact(
            tmp_path / "a",
            fingerprint="1" * 64,
            spec={"obs": {"size": 10, "names": ["x"]}, "act": 3},
            metrics={"reward": 1.0},
        )
    )
    b = registry.register(
        make_artifact(
            tmp_path / "b",
            fingerprint="2" * 64,
            spec={"obs": {"size": 12, "names": ["x"]}, "act": 3, "extra": True},
            metrics={"reward": 2.0},
            parents=(a.id, "teacher"),
        )
    )
    assert registry.get(b.id).parents == (a.id, "teacher")
    assert registry.diff(a.id, b.id) == {
        "metrics.reward": (1.0, 2.0),
        "parents": ([], [a.id, "teacher"]),
        "spec.extra": (None, True),
        "spec.obs.size": (10, 12),
    }
    assert registry.diff(a.id, a.id) == {}


def test_fetch_downloads_once(tmp_path: Path, registry: ArtifactRegistry) -> None:
    manifest = registry.register(make_artifact(tmp_path / "a"))
    cache = tmp_path / "cache"
    local = registry.fetch(manifest.id, cache)
    assert local == cache / manifest.id
    assert (local / "nested" / "weights.pt").read_bytes() == b"weights"
    assert read_manifest(local) == manifest
    (local / "policy.onnx").write_bytes(b"cached")
    assert registry.fetch(manifest.id, cache) == local
    assert (local / "policy.onnx").read_bytes() == b"cached"
    assert sorted(path.name for path in cache.iterdir()) == [manifest.id]


def test_untag_and_fetch_tagged(tmp_path: Path, registry: ArtifactRegistry) -> None:
    walker = registry.register(make_artifact(tmp_path / "a"))
    kicker = registry.register(make_artifact(tmp_path / "b", name="kicker"))
    registry.tag(walker.id, "prod")
    registry.tag(kicker.id, "prod")
    local = registry.fetch_tagged("walker", "prod", tmp_path / "cache")
    assert local == tmp_path / "cache" / walker.id
    assert read_manifest(local) == walker
    assert registry.untag("walker", "prod") == walker.id
    assert registry.tags() == {"prod": {"kicker": kicker.id}}
    assert registry.untag("kicker", "prod") == kicker.id
    assert registry.tags() == {}
    with pytest.raises(KeyError, match="no 'walker' artifact tagged 'prod'"):
        registry.untag("walker", "prod")
    with pytest.raises(KeyError, match="no 'walker' artifact tagged 'prod'"):
        registry.fetch_tagged("walker", "prod", tmp_path / "cache")


def test_diff_matches_named_items(tmp_path: Path, registry: ArtifactRegistry) -> None:
    def parts(*items: dict[str, Any]) -> dict[str, Any]:
        return {"inputs": {"parts": list(items)}, "shape": [2, 3]}

    a = registry.register(
        make_artifact(
            tmp_path / "a",
            fingerprint="1" * 64,
            spec=parts({"name": "state", "size": 2}, {"name": "target", "size": 1}),
        )
    )
    b = registry.register(
        make_artifact(
            tmp_path / "b",
            fingerprint="2" * 64,
            spec={
                **parts({"name": "target", "size": 1}, {"name": "state", "size": 3}),
                "shape": [2, 4],
            },
        )
    )
    assert registry.diff(a.id, b.id) == {
        "spec.inputs.parts[state].size": (2, 3),
        "spec.shape": ([2, 3], [2, 4]),
    }
