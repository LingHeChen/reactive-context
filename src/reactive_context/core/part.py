"""Public typed part factories; plugin hooks always execute under host policy."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import TypeAdapter

from .snapshot import PartSnapshot

T = TypeVar("T")
Hook = Callable[..., Any]


@dataclass(frozen=True)
class ChangeSignal:
    source_id: str
    hinted_revision: str | None = None
    event_time: float | None = None
    revoked: bool = False


@dataclass(frozen=True)
class ResolveContext:
    scope_id: str
    part_id: str
    configuration: Mapping[str, Any]
    dependencies: Mapping[str, PartSnapshot]
    revision: str | None = None


class SourcePartSpec[T](Protocol):
    key: str
    schema: Any

    def resolve(self, ctx: ResolveContext) -> Awaitable[T]: ...
    def revision(self, ctx: ResolveContext) -> Awaitable[str]: ...


@dataclass(frozen=True)
class PartDefinition[T]:
    key: str
    schema: Any
    resolve: Hook
    revision: Hook | None = None
    deps: tuple[str, ...] = ()
    project: Hook | None = None
    subscribe: Hook | None = None
    update: Hook | None = None
    invalidate: Hook | None = None
    on_error: Hook | None = None
    dispose: Hook | None = None
    acl_check: Hook | None = None
    resource_key: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    ttl: float = 0
    timeout: float = 10
    memoize: bool = True
    computed: bool = False
    required: bool = True
    kind: str = "resource"
    schema_version: int = 1
    algorithm_version: str = "1"
    projection_version: str = "1"
    expose: bool = True
    adapter: TypeAdapter = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        from .snapshot import freeze

        if not self.key or self.timeout <= 0 or self.ttl < 0:
            raise ValueError("invalid part key, timeout or ttl")
        object.__setattr__(self, "adapter", TypeAdapter(self.schema))
        object.__setattr__(self, "configuration", freeze(self.configuration))
        object.__setattr__(self, "deps", tuple(self.deps))


def source_part[T](
    *,
    key: str,
    schema: Any,
    resolve: Callable[..., Awaitable[T]],
    revision: Hook | None = None,
    **kwargs,
) -> PartDefinition[T]:
    return PartDefinition(key=key, schema=schema, resolve=resolve, revision=revision, **kwargs)


def computed_part[T](
    *,
    key: str,
    schema: Any,
    dependencies: list[str] | tuple[str, ...],
    compute: Callable[..., Awaitable[T]],
    **kwargs,
) -> PartDefinition[T]:
    return PartDefinition(
        key=key, schema=schema, resolve=compute, deps=tuple(dependencies), computed=True, **kwargs
    )


@dataclass
class PartInstance:
    definition: PartDefinition
    scope_id: str
    generation: int = 0
    state: str = "UNINITIALIZED"
    snapshot: PartSnapshot | None = None
    dep_vector: tuple = ()
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    unsubscribe: Hook | None = None
    subscribed: bool = False
    event_time: float | None = None
