"""Options entry points, delegating snapshots and rendering to the existing Context."""

import asyncio
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..context import Context

from ..core.snapshot import freeze
from .declarations import PartDeclaration, collect
from .runtime import OptionsRuntime, PartsView


class ContextType(type):
    def __setattr__(cls, name, value):
        if (
            name in getattr(cls, "__parts__", {})
            or isinstance(value, PartDeclaration)
            or (name == "__parts__" and name in vars(cls))
        ):
            raise TypeError("Context declarations are immutable; use Composition register")
        super().__setattr__(name, value)

    def __delattr__(cls, name):
        if name in getattr(cls, "__parts__", {}) or name == "__parts__":
            raise TypeError("Context declarations are immutable")
        super().__delattr__(name)


class OptionsAPI:
    __parts__ = MappingProxyType({})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__parts__ = collect(cls)

    @property
    def _host(self) -> "Context":
        return cast("Context", self)

    def _bind_options(self, options, scope_bindings, providers, services):
        self._dx = (
            OptionsRuntime(
                self,
                self.__parts__,
                options,
                scope_bindings,
                providers,
                services,
            )
            if self.__parts__
            else None
        )
        if self._dx is None and options:
            from .declarations import UnknownContextOption

            raise UnknownContextOption(sorted(options)[0])

    @classmethod
    def describe(cls):
        return freeze(
            {
                name: {
                    "key": d.key or name,
                    "kind": d.kind,
                    "scope": d.scope,
                    "mode": d.mode,
                    "required": d.required,
                    "static_dependencies": d.deps,
                    "schema": str(d.schema),
                    "dynamic_dependencies": "not_resolved",
                    "declaration_version": d.version,
                }
                for name, d in cls.__parts__.items()
            }
        )

    @property
    def parts(self):
        if self._dx is None:
            raise AttributeError("no Options declarations")
        return PartsView(self._dx)

    @property
    def state_revision(self):
        return self._dx.revision if self._dx is not None else None

    def update(self, *, expected_revision=None, **values):
        self._host._ensure_open()
        if self._dx is None:
            from .declarations import UnknownContextOption

            raise UnknownContextOption(next(iter(values), ""))
        return self._dx.update(values, expected_revision)

    async def resolve(self, key):
        self._host._ensure_open()
        if self._dx is None or key not in {*self._dx.declarations, *self._dx.names}:
            snapshot = await self._host.next_context(selected=[key])
            return next(p.value for p in snapshot.parts if p.part_id == key)
        name = self._dx.name(key)
        if any(
            dep not in self._dx.declarations and dep not in self._dx.names
            for dep in self.__parts__[name].deps
        ):
            snapshot = await self._host.next_context(selected=[self._dx.keys[name]])
            return snapshot.part(self._dx.keys[name]).value
        return await self._dx.resolve(key)

    async def flush(self):
        self._host._ensure_open()
        # Local mutation/dirty propagation commits synchronously; do not evaluate parts.
        await asyncio.sleep(0)

    async def aclose(self):
        await self._host.close()


def create_context(*, parts=(), **options):
    from ..context import Context

    return Context(parts, **options)
