"""Browse, fetch and tag artifacts in a registry: `list`, `show`, `diff`, `fetch` and tags."""

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policies.cli.base import CliError, Subcommands
from policies.cli.console import console, print_table, table

ENV_ROOT = "POLICIES_REGISTRY"


def _registry(arguments: argparse.Namespace) -> Any:
    from policies.deploy.registry import ArtifactRegistry

    root = arguments.root or os.environ.get(ENV_ROOT)
    if not root:
        raise CliError(f"no registry root: pass --root or set {ENV_ROOT}")
    return ArtifactRegistry(root)


def _number(value: float) -> str:
    return f"{value:.4g}"


def _metrics(metrics: Mapping[str, float]) -> str:
    return "  ".join(f"{key}={_number(value)}" for key, value in sorted(metrics.items()))


def _requirements(requirements: Mapping[str, str]) -> str:
    return ", ".join(f"{name} {version}" for name, version in sorted(requirements.items()))


def _tags(registry: Any) -> dict[str, list[str]]:
    """Tag names per artifact id."""
    names: dict[str, list[str]] = {}
    for tag, ids in sorted(registry.tags().items()):
        for id in ids.values():
            names.setdefault(id, []).append(tag)
    return names


def _list(arguments: argparse.Namespace) -> None:
    registry = _registry(arguments)
    manifests = registry.list(name=arguments.name, tag=arguments.tag)
    if not manifests:
        console().print("no artifacts")
        return
    tags = _tags(registry)
    metrics = sorted({key for manifest in manifests for key in manifest.metrics})
    rows = table("id", "created", "tags")
    for key in metrics:
        rows.add_column(key, justify="right")
    for manifest in manifests:
        values = [
            _number(manifest.metrics[key]) if key in manifest.metrics else "-" for key in metrics
        ]
        rows.add_row(
            manifest.id, manifest.created, ",".join(tags.get(manifest.id, [])) or "-", *values
        )
    print_table(rows)


def _show(arguments: argparse.Namespace) -> None:
    registry = _registry(arguments)
    manifest = registry.get(arguments.id)
    tags = _tags(registry).get(manifest.id, [])
    rows = table("field", "value", header=False)
    rows.columns[0].style = "bold"
    for key, value in [
        ("id", manifest.id),
        ("name", manifest.name),
        ("fingerprint", manifest.fingerprint),
        ("created", manifest.created),
        ("parents", ", ".join(manifest.parents) or "-"),
        ("tags", ", ".join(tags) or "-"),
        ("files", ", ".join(manifest.files) or "-"),
        ("requirements", _requirements(manifest.requirements) or "-"),
        ("metrics", _metrics(manifest.metrics) or "-"),
        ("run", json.dumps(manifest.run, sort_keys=True) if manifest.run else "-"),
    ]:
        rows.add_row(key, value)
    print_table(rows)


def _diff(arguments: argparse.Namespace) -> None:
    changes = _registry(arguments).diff(arguments.a, arguments.b)
    if not changes:
        console().print("no differences")
        return
    rows = table("path", arguments.a, arguments.b)
    for key, (left, right) in changes.items():
        rows.add_row(key, json.dumps(left), json.dumps(right))
    print_table(rows)


def _fetch(arguments: argparse.Namespace) -> None:
    registry = _registry(arguments)
    name, separator, tag = arguments.artifact.partition(":")
    if separator:
        local = registry.fetch_tagged(name, tag, Path(arguments.destination))
    else:
        local = registry.fetch(arguments.artifact, Path(arguments.destination))
    console().print(str(local))


def _tag(arguments: argparse.Namespace) -> None:
    manifest = _registry(arguments).tag(arguments.id, arguments.tag)
    console().print(f"{arguments.tag} -> {manifest.id} ({manifest.name})")


def _untag(arguments: argparse.Namespace) -> None:
    id = _registry(arguments).untag(arguments.name, arguments.tag)
    console().print(f"removed {arguments.tag} from {id} ({arguments.name})")


def add_commands(commands: Subcommands) -> None:
    def registry_parser(name: str, help: str) -> argparse.ArgumentParser:
        subparser = commands.add_parser(name, help=help)
        subparser.add_argument("--root", help=f"registry root, defaults to ${ENV_ROOT}")
        return subparser

    subparser = registry_parser("list", "list artifacts")
    subparser.add_argument("--name")
    subparser.add_argument("--tag")
    subparser.set_defaults(handler=_list)

    subparser = registry_parser("show", "show an artifact")
    subparser.add_argument("id")
    subparser.set_defaults(handler=_show)

    subparser = registry_parser("diff", "compare two artifacts")
    subparser.add_argument("a")
    subparser.add_argument("b")
    subparser.set_defaults(handler=_diff)

    subparser = registry_parser("fetch", "download an artifact, unless already there")
    subparser.add_argument("artifact", metavar="ID|NAME:TAG")
    subparser.add_argument("destination", metavar="DEST", help="the copy goes in DEST/ID")
    subparser.set_defaults(handler=_fetch)

    subparser = registry_parser("tag", "point a tag at an artifact")
    subparser.add_argument("id")
    subparser.add_argument("tag")
    subparser.set_defaults(handler=_tag)

    subparser = registry_parser("untag", "remove a tag from a policy name")
    subparser.add_argument("name")
    subparser.add_argument("tag")
    subparser.set_defaults(handler=_untag)
