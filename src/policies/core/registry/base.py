"""Typed registries that build components from `{type: name, ...}` configs."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pkgutil import resolve_name
from typing import Any, NamedTuple, Self

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from policies.core.config import Config, canonical
from policies.core.logs import get_logger
from policies.core.registry.entry import Entry
from policies.core.registry.errors import (
    DuplicateComponentError,
    UnknownComponentError,
    unknown,
)
from policies.core.registry.origin import Origin
from policies.core.registry.plugins import PluginGroup, installed_plugins

log = get_logger(__name__)


class _Registration[T](NamedTuple):
    factory: Callable[..., T] | str
    """The factory, or its `'module:attribute'` path until the first build."""
    config: type[Config]
    module: str


class Registry[T]:
    """Components of one kind, each registered with a name and a config class.

    Unknown names load every entry point in `installed`, so installed distributions can
    register too. Built-ins register when their package is imported.

    As pydantic metadata, `Annotated[dict[str, Any], registry]` validates a field to the
    canonical config of one of these components, so fingerprints see every default.
    """

    def __init__(
        self,
        kind: str,
        installed: PluginGroup = installed_plugins,
    ) -> None:
        self.kind = kind
        self._entries: dict[str, _Registration[T]] = {}
        self._installed = installed

    def _load_plugins(self) -> None:
        self._installed.load()

    def _conflict(self, name: str, existing: str) -> DuplicateComponentError:
        origin = self._installed.origin(existing)
        registrant = f"{existing} ({origin})" if origin else existing
        remedy = (
            "since installed plugins load lazily and cannot replace names"
            if self._installed.loading
            else "or pass replace=True to override it"
        )
        return DuplicateComponentError(
            f"{self.kind} {name!r} is already registered by {registrant}. "
            f"Namespace the new one, e.g. '<package>.{name}', {remedy}."
        )

    def _add(self, name: str, registration: _Registration[T], replace: bool) -> None:
        existing = self._entries.get(name)
        if existing is not None:
            if not replace or self._installed.loading:
                raise self._conflict(name, existing.module)
            log.warning("component replaced", kind=self.kind, name=name, by=registration.module)
        self._entries[name] = registration

    def register[F: Callable[..., Any]](
        self, name: str, config: type[Config], *, replace: bool = False
    ) -> Callable[[F], F]:
        """Decorator. The factory is called as `factory(config, **context)`.

        Plugins should namespace their names, e.g. `my_package.mlp`. `replace=True` overrides
        an existing name on purpose, from code imported explicitly. Installed plugins cannot
        replace, since whether the override applied would depend on when they loaded.
        """

        def decorator(factory: F) -> F:
            # The defining module, which `origin` maps to its distribution.
            module: str = getattr(factory, "__module__", None) or "__main__"
            self._add(name, _Registration(factory, config, module), replace)
            return factory

        return decorator

    def register_lazy(
        self, name: str, config: type[Config], factory: str, *, replace: bool = False
    ) -> None:
        """Like `register`, with the factory given as `'module:attribute'` and imported on
        first build, so configs validate without the factory's dependencies.
        """
        module, colon, attribute = factory.partition(":")
        if not (module and colon and attribute):
            raise ValueError(
                f"lazy {self.kind} factory must be 'module:attribute', got {factory!r}"
            )
        self._add(name, _Registration(factory, config, module), replace)

    @contextmanager
    def scoped(self) -> Iterator[Self]:
        """Registrations made inside, replacements included, are undone on exit. For tests."""
        self._load_plugins()
        saved = dict(self._entries)
        try:
            yield self
        finally:
            self._entries = saved

    def names(self) -> list[str]:
        self._load_plugins()
        return sorted(self._entries)

    def entries(self) -> list[Entry]:
        """Every component, sorted by name, with its config class and origin."""
        return [
            Entry(
                name, self._entries[name].config, self._installed.origin(self._entries[name].module)
            )
            for name in self.names()
        ]

    def __contains__(self, name: object) -> bool:
        if name not in self._entries:
            self._load_plugins()
        return name in self._entries

    def _entry(self, name: str) -> _Registration[T]:
        if name not in self:
            raise unknown(self.kind, name, self.names())
        return self._entries[name]

    def config(self, name: str) -> type[Config]:
        """The config class of `name`."""
        return self._entry(name).config

    def origin(self, name: str) -> Origin | None:
        """The distribution that registered `name`, or None for unpackaged code like a script."""
        return self._installed.origin(self._entry(name).module)

    def parse(self, data: Mapping[str, Any]) -> tuple[str, Config]:
        """Split off `type`, look it up and validate the rest."""
        fields = dict(data)
        name = fields.pop("type", None)
        if not isinstance(name, str):
            raise ValueError(f"{self.kind} config needs a 'type' key, got {sorted(data)}")
        return name, self._entry(name).config.model_validate(fields)

    def canonical(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """The config with `type` and every default filled in."""
        name, config = self.parse(data)
        return {"type": name, **canonical(config)}

    def build(self, data: Mapping[str, Any], **context: Any) -> T:
        name, config = self.parse(data)
        return self._factory(name)(config, **context)

    def _factory(self, name: str) -> Callable[..., T]:
        registration = self._entries[name]
        if not isinstance(registration.factory, str):
            return registration.factory
        factory: Callable[..., T] = resolve_name(registration.factory)
        self._entries[name] = registration._replace(factory=factory)
        return factory

    def _validate(self, data: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.canonical(data)
        except UnknownComponentError as error:
            # A KeyError would escape pydantic instead of being reported at the field.
            raise ValueError(str(error)) from None

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(self._validate, handler(source))
