"""The registered components a config uses, found from its field annotations."""

from collections.abc import Iterator, Mapping
from types import UnionType
from typing import Annotated, Any, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel

from policies.core.registry.base import Registry

type Use = tuple[Registry[Any], str]


def used_components(config: BaseModel) -> Iterator[Use]:
    """`(registry, name)` of every component in `config`, nested ones included.

    Any field annotated with a registry counts, inside containers, unions and other configs.
    """
    for name, field in type(config).model_fields.items():
        yield from _walk(field.annotation, field.metadata, getattr(config, name))


def _unwrap(annotation: Any, metadata: list[Any]) -> tuple[Any, Registry[Any] | None]:
    """The annotation without aliases and `Annotated`, and the registry among its metadata."""
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation, *extra = get_args(annotation)
            metadata = [*metadata, *extra]
        else:
            registry = next((item for item in metadata if isinstance(item, Registry)), None)
            return annotation, registry


def _accepts(annotation: Any, value: Any) -> bool:
    """Whether `value` belongs to this arm of a union."""
    annotation, registry = _unwrap(annotation, [])
    if registry is not None:
        return isinstance(value, Mapping) and value.get("type") in registry
    origin = get_origin(annotation) or annotation
    return isinstance(origin, type) and isinstance(value, origin)


def _items(annotation: Any, count: int) -> list[Any]:
    """The annotation of each item of a sequence."""
    arguments = get_args(annotation)
    if get_origin(annotation) is tuple and Ellipsis not in arguments:
        return list(arguments)
    return [arguments[0] if arguments else Any] * count


def _walk(annotation: Any, metadata: list[Any], value: Any) -> Iterator[Use]:
    annotation, registry = _unwrap(annotation, metadata)
    if registry is not None and isinstance(value, Mapping):
        name, config = registry.parse(value)
        yield registry, name
        yield from used_components(config)
    elif isinstance(value, BaseModel):
        yield from used_components(value)
    elif get_origin(annotation) in (Union, UnionType):
        arm = next((arm for arm in get_args(annotation) if _accepts(arm, value)), None)
        if arm is not None:
            yield from _walk(arm, [], value)
    elif isinstance(value, Mapping):
        arguments = get_args(annotation)
        for item in value.values():
            yield from _walk(arguments[-1] if arguments else Any, [], item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item_annotation, item in zip(_items(annotation, len(value)), value, strict=False):
            yield from _walk(item_annotation, [], item)
