"""Immutable, IO-free Options declarations. All runtime state lives on an instance."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, time
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

from ..errors import ContextError


class InvalidStateDefault(ContextError):
    code = "INVALID_STATE_DEFAULT"


class UnknownContextOption(ContextError):
    code = "UNKNOWN_CONTEXT_OPTION"


class ComputedSideEffectError(ContextError):
    code = "COMPUTED_SIDE_EFFECT"


class AmbiguousPartDeclaration(ContextError):
    code = "AMBIGUOUS_PART_DECLARATION"


class MissingPartProvider(ContextError):
    code = "MISSING_PART_PROVIDER"


class ScopeBindingRequired(ContextError):
    code = "SCOPE_BINDING_REQUIRED"


_UNSET = object()


def _mutable(value):
    # Conservative immutable allow-list; opaque/custom values should use a factory.
    if value is None or isinstance(
        value, (str, int, float, bytes, date, time, Decimal, UUID, PurePath)
    ):
        return False
    if isinstance(value, (tuple, frozenset)):
        return any(_mutable(v) for v in value)
    if isinstance(value, Enum):
        return _mutable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        return (
            parameters is None
            or not parameters.frozen
            or any(_mutable(getattr(value, f.name)) for f in fields(value))
        )
    return True


@dataclass(frozen=True, eq=False)
class PartDeclaration:
    kind: str
    default: Any = _UNSET
    default_factory: Any = None
    function: Any = None
    key: str | None = None
    scope: str = "instance"
    mode: str = "source"
    required: bool = True
    deps: tuple[str, ...] = ()
    schema: Any = Any
    project: Any = None
    expose: bool = True
    version: str = "1"
    revision: Any = None
    timeout: float = 10
    provider: str | None = None
    resource: str | None = None
    options: Mapping = field(default_factory=dict)

    def __post_init__(self):
        from ..core.snapshot import freeze

        if self.default is not _UNSET and self.default_factory is not None:
            raise InvalidStateDefault(detail="default and default_factory are exclusive")
        if self.kind == "state" and self.default is not _UNSET and _mutable(self.default):
            raise InvalidStateDefault(detail="use default_factory for mutable defaults")
        if inspect.iscoroutinefunction(self.default_factory):
            raise InvalidStateDefault(detail="default_factory must be synchronous")
        if self.default_factory is not None and not callable(self.default_factory):
            raise InvalidStateDefault(detail="default_factory must be callable")
        if self.kind == "computed" and inspect.iscoroutinefunction(self.function):
            raise TypeError("computed must be synchronous")
        if self.timeout <= 0 or not self.version:
            raise ValueError("invalid timeout or declaration version")
        if self.mode not in {"source", "derived", "retrieved"}:
            raise ValueError("invalid Part mode")
        object.__setattr__(self, "deps", tuple(self.deps))
        object.__setattr__(self, "options", freeze(self.options))

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        name = instance._dx.name_of(self)
        if self.kind in {"state", "computed"}:
            return instance._dx.read(name)
        return PartHandle(instance, name)

    def __set__(self, instance, value):
        if self.kind != "state":
            raise AttributeError("only state declarations can be assigned")
        instance.update(**{instance._dx.name_of(self): value})


@dataclass(frozen=True)
class PartHandle:
    context: Any = field(repr=False, compare=False)
    key: str

    async def resolve(self):
        return await self.context.resolve(self.key)

    def explain(self):
        return self.context.explain(self.key)


def state(default=_UNSET, *, default_factory=None, **options) -> Any:
    return PartDeclaration("state", default=default, default_factory=default_factory, **options)


def computed(function=None, **options) -> Any:
    def decorate(fn):
        return PartDeclaration("computed", function=fn, mode="derived", **options)

    return decorate(function) if function is not None else decorate


def part(function=None, **options) -> Any:
    def decorate(fn):
        return PartDeclaration("part", function=fn, **options)

    return decorate(function) if function is not None else decorate


def _factory(provider, resource, **options):
    scope = options.pop("scope", "instance")
    required = options.pop("required", True)
    key = options.pop("key", None)
    if not isinstance(resource, str) or not resource:
        raise ValueError("resource must be a nonempty string")
    if "limit" in options and (type(options["limit"]) is not int or options["limit"] < 1):
        raise ValueError("limit must be a positive integer")
    return PartDeclaration(
        "provider",
        provider=provider,
        resource=resource,
        scope=scope,
        key=key,
        required=required,
        options=options,
        mode="retrieved" if provider == "memory" else "source",
    )


def skill(resource, *, channel="stable", **options):
    return _factory("skill", resource, channel=channel, **options)


def memory(resource, *, limit=5, **options):
    return _factory("memory", resource, limit=limit, **options)


def intent(resource, **options):
    return _factory("intent", resource, **options)


def specification(resource, **options):
    return _factory("specification", resource, **options)


def collect(cls):
    """Respect ordinary shadows and reject ambiguous declarations before instance binding."""
    own = vars(cls)
    candidates = {}
    owners = {}
    for base in cls.__bases__:
        for name, declaration in getattr(base, "__parts__", {}).items():
            prior = candidates.get(name)
            if (
                prior is not None
                and prior is not declaration
                and name not in own
                and not issubclass(owners[name], base)
                and not issubclass(base, owners[name])
            ):
                # An explicit subclass override is required, including in diamonds.
                raise AmbiguousPartDeclaration(name)
            candidates[name] = declaration
            owners[name] = base
    merged = {}
    for base in reversed(cls.__mro__):
        for name, value in vars(base).items():
            if isinstance(value, PartDeclaration):
                merged[name] = value
            else:
                merged.pop(name, None)
    if len({id(d) for d in merged.values()}) != len(merged):
        raise AmbiguousPartDeclaration(detail="declare separate descriptors for separate fields")
    keys = [d.key or name for name, d in merged.items()]
    if len(set(keys)) != len(keys):
        raise AmbiguousPartDeclaration(detail="duplicate explicit logical key")
    reserved = set(getattr(cls, "__runtime_fields__", ()))
    for base in cls.__mro__[1:]:
        if not getattr(base, "__parts__", {}):
            reserved.update(vars(base))
    for name in merged:
        if name.startswith("_") or name in reserved:
            raise AmbiguousPartDeclaration(name, "reserved Context API name")
    from types import MappingProxyType

    return MappingProxyType(merged)
