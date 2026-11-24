"""List registered components: `components [REGISTRY]`."""

import argparse
import json
from types import UnionType
from typing import Annotated, Any, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

from policies.cli.base import CliError, Subcommands
from policies.cli.console import console, print_table, table
from policies.core.registry import Registry, registries


def _type(annotation: Any) -> str:
    """Short and readable, e.g. `tuple[TransformConfig, ...]` or `int | None`."""
    if isinstance(annotation, TypeAliasType):
        return annotation.__name__
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _type(arguments[0])
    if origin in (Union, UnionType):
        return " | ".join(map(_type, arguments))
    if origin is Literal:
        return f"Literal[{', '.join(map(repr, arguments))}]"
    if origin is not None:
        return f"{_type(origin)}[{', '.join(map(_type, arguments))}]"
    if annotation is Ellipsis:
        return "..."
    if annotation is type(None):
        return "None"
    return str(getattr(annotation, "__name__", annotation))


def _field(name: str, field: FieldInfo) -> str:
    text = f"{name}: {_type(field.annotation)}"
    if field.is_required():
        return text
    default = field.get_default(call_default_factory=True, validated_data={})
    return f"{text} = {json.dumps(to_jsonable_python(default))}"


def _registry(name: str, single: bool) -> Registry[Any] | None:
    """The registry, or None with a warning when its dependencies are missing."""
    try:
        return registries[name]
    except ImportError as error:
        if single:
            raise CliError(f"{name} is unavailable: {error}") from error
        console(stderr=True).print(f"warning: {name} is unavailable: {error}")
        return None


def _components(arguments: argparse.Namespace) -> None:
    names = [arguments.registry] if arguments.registry else registries.names()
    rows = table("registry", "name", "origin", "fields")
    for name in names:
        registry = _registry(name, single=bool(arguments.registry))
        for entry in registry.entries() if registry else []:
            fields = (_field(key, field) for key, field in entry.config.model_fields.items())
            rows.add_row(name, entry.name, str(entry.origin or "-"), ", ".join(fields) or "-")
    print_table(rows)


def add_commands(commands: Subcommands) -> None:
    subparser = commands.add_parser("components", help="list registered components")
    subparser.add_argument("registry", nargs="?", help="e.g. transforms. All when omitted")
    subparser.set_defaults(handler=_components)
