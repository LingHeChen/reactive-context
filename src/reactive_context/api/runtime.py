"""Instance-owned local mutation and dependency tracking; no alternative message compiler."""

import asyncio
import inspect
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, get_type_hints
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from ..core.part import PartDefinition
from ..core.scheduler import invoke
from ..core.snapshot import digest, freeze
from ..errors import CircularDependency, ConsistencyUnavailable, SchemaInvalid, SourceUnavailable
from .declarations import (
    _UNSET,
    ComputedSideEffectError,
    MissingPartProvider,
    PartHandle,
    ScopeBindingRequired,
    UnknownContextOption,
)
from .protocol import PartMaterialization

# Synchronous computed evaluation never awaits; a Task cannot inherit an active mutable frame.
_computations: ContextVar[tuple] = ContextVar("reactive_computations", default=())
_resolving: ContextVar[tuple] = ContextVar("reactive_resolving", default=())
_core_reads: ContextVar[tuple] = ContextVar("reactive_core_reads", default=())


@dataclass
class Node:
    value: Any = None
    resolved: bool = False
    dirty: bool = True
    version: int = 0
    dependencies: dict = field(default_factory=dict)
    materialization: PartMaterialization | None = None
    hits: int = 0
    computations: int = 0
    error: str | None = None
    token_key: tuple = ()
    revision_token: str = ""


class ReadContext:
    """A scoped, read-only view. Services are explicit host-provided capability ports."""

    __slots__ = ("_runtime", "_reads", "_tokens", "scope_bindings", "services")

    def __init__(self, runtime, reads):
        self._runtime = runtime
        self._reads = reads
        self._tokens = {}
        self.scope_bindings = runtime.scope_bindings
        self.services = runtime.services

    def read(self, key):
        if key not in self._runtime.declarations and key not in self._runtime.names:
            snapshot = self._runtime.legacy(key)
            self._reads[key] = snapshot.revision
            return snapshot.value
        name = self._runtime.name(key)
        value = self._runtime.read(name)
        node = self._runtime.nodes[name]
        self._reads[name] = node.version
        if node.materialization is not None:
            self._tokens[name] = node.materialization.revision
        # Readers must not mutate state through a container reference.
        return freeze(value)

    async def resolve(self, key):
        if key not in self._runtime.declarations and key not in self._runtime.names:
            return self.read(key)
        name = self._runtime.name(key)
        value = await self._runtime.resolve(name)
        node = self._runtime.nodes[name]
        self._reads[name] = node.version
        if node.materialization is not None:
            self._tokens[name] = node.materialization.revision
        return freeze(value)

    def __getattr__(self, key):
        try:
            return self.services[key]
        except KeyError:
            raise AttributeError(key) from None


class PartsView:
    def __init__(self, runtime):
        self._runtime = runtime

    def __getattr__(self, key):
        return PartHandle(self._runtime.context, self._runtime.name(key))


class OptionsRuntime:
    def __init__(self, context, declarations, options, scope_bindings, providers, services):
        self.context = context
        self.declarations = declarations
        self.scope_bindings = MappingProxyType(dict(scope_bindings or {}))
        if any(
            not isinstance(k, str) or not isinstance(v, str) or not v
            for k, v in self.scope_bindings.items()
        ):
            raise ScopeBindingRequired(detail="scope bindings must be nonempty string IDs")
        self.providers = MappingProxyType(dict(providers or {}))
        self.services = MappingProxyType(dict(services or {}))
        self.identity = uuid4().hex
        self.bindings = {}
        self.nodes = {name: Node() for name in declarations}
        self.keys = {name: d.key or name for name, d in declarations.items()}
        self.names = {key: name for name, key in self.keys.items()}
        self.adapters = {}
        self.locks = {name: asyncio.Lock() for name in declarations}
        self.tasks = {}
        self.epoch = 0
        self.strict = ContextVar(f"reactive_strict_{self.identity}", default=False)
        self.closed = False
        self.boundary_lock = asyncio.Lock()
        self.boundary_task: asyncio.Task | None = None
        self.diagnostics = []
        try:
            hints = get_type_hints(type(context))
        except (NameError, TypeError):
            hints = {}
        for name, declaration in declarations.items():
            if declaration.scope != "instance" and not self.scope_bindings.get(declaration.scope):
                raise ScopeBindingRequired(name, declaration.scope)
            schema = (
                hints.get(name, declaration.schema)
                if declaration.kind == "state"
                else (declaration.schema)
            )
            self.adapters[name] = TypeAdapter(schema)
        unknown = set(options) - {n for n, d in declarations.items() if d.kind == "state"}
        if unknown:
            raise UnknownContextOption(sorted(unknown)[0])
        initial = {}
        for name, declaration in declarations.items():
            if declaration.kind == "state":
                value = options.get(name, _UNSET)
                if value is _UNSET:
                    value = (
                        declaration.default_factory()
                        if declaration.default_factory is not None
                        else declaration.default
                    )
                if value is _UNSET:
                    raise UnknownContextOption(name, "required state value missing")
                initial[name] = value
        self.update(initial)

    def legacy(self, key):
        for identity, parts in reversed(_core_reads.get()):
            if identity == self.identity and key in parts:
                return parts[key]
        raise SourceUnavailable(key, "declare legacy dependencies explicitly in deps")

    def dependency_version(self, name):
        return self.nodes[name].version if name in self.nodes else self.legacy(name).revision

    def name(self, key):
        if key in self.declarations:
            return key
        if key in self.names:
            return self.names[key]
        raise KeyError(key)

    def name_of(self, declaration):
        return next(n for n, d in self.declarations.items() if d is declaration)

    def ensure_open(self):
        if self.closed or self.context._closed:
            raise RuntimeError("context is closed")

    def _record(self, name):
        stack = _computations.get()
        if stack and stack[-1][0] is self:
            stack[-1][2][name] = self.nodes[name].version

    def read(self, name):
        self.ensure_open()
        declaration, node = self.declarations[name], self.nodes[name]
        if declaration.kind not in {"state", "computed"}:
            if not node.resolved or node.dirty:
                raise SourceUnavailable(name, "use await reader.resolve for unresolved async Part")
            self._record(name)
            return node.value
        if declaration.kind == "computed":
            stack = _computations.get()
            path = [n for runtime, n, _ in stack if runtime is self]
            if name in path:
                raise CircularDependency(name, " -> ".join([*path, name]))
            if node.dirty or not node.resolved:
                reads = {}
                token = _computations.set((*stack, (self, name, reads)))
                try:
                    value = declaration.function(self.context)
                    if inspect.isawaitable(value):
                        if inspect.iscoroutine(value):
                            value.close()
                        raise TypeError("computed returned an awaitable")
                    value = self.adapters[name].validate_python(value)
                except BaseException as exc:
                    node.error = getattr(exc, "code", "COMPUTATION_FAILED")
                    node.dirty = True
                    raise
                finally:
                    _computations.reset(token)
                node.value, node.dependencies = value, reads
                node.resolved, node.dirty = True, False
                node.version += 1
                node.computations += 1
                node.error = None
            else:
                node.hits += 1
        self._record(name)
        return node.value

    def invalidate(self, name):
        affected, queue = {name}, [name]
        while queue:
            dependency = queue.pop()
            for key, node in self.nodes.items():
                if key not in affected and (
                    dependency in node.dependencies or dependency in self.declarations[key].deps
                ):
                    affected.add(key)
                    queue.append(key)
        for key in affected:
            node = self.nodes[key]
            if self.declarations[key].kind != "state":
                node.dirty = True
            # Include legacy dependents, and work without a running event loop.
            if hasattr(self.context, "registry"):
                for scope, registered in list(self.context.registry.instances):
                    if registered == self.keys[key]:
                        self.context._mark_dirty(scope, registered)

    def update(self, values, expected_revision=None):
        self.ensure_open()
        if _computations.get():
            raise ComputedSideEffectError()
        if expected_revision is not None and expected_revision != self.revision:
            raise ConsistencyUnavailable(detail="local state compare-and-swap failed")
        validated = {}
        for name, value in values.items():
            if name not in self.declarations or self.declarations[name].kind != "state":
                raise UnknownContextOption(name)
            try:
                validated[name] = deepcopy(self.adapters[name].validate_python(value))
            except (ValueError, TypeError):
                raise SchemaInvalid(name) from None
        changed = []
        for name, value in validated.items():
            node = self.nodes[name]
            try:
                equal = type(value) is type(node.value) and bool(value == node.value)
            except Exception:
                equal = False
            if not node.resolved or not equal:
                node.value = value
                node.version += 1
                node.resolved, node.dirty = True, False
                changed.append(name)
        for name in changed:
            self.invalidate(name)
        return self.revision

    @property
    def revision(self):
        return digest(
            {n: v.version for n, v in self.nodes.items() if self.declarations[n].kind == "state"}
        )

    def binding(self, name):
        declaration = self.declarations[name]
        scope = (
            self.identity
            if declaration.scope == "instance"
            else self.scope_bindings[declaration.scope]
        )
        if name not in self.bindings:
            identity = digest((declaration.scope, scope, self.keys[name]))
            self.bindings[name] = f"{declaration.scope}:{identity}:{self.keys[name]}"
        return self.bindings[name]

    def _provider(self, name):
        d = self.declarations[name]
        if d.kind != "provider":
            return None
        provider = self.providers.get(d.provider)
        if provider is None:
            raise MissingPartProvider(name, d.provider or "")
        return provider

    async def external_revision(self, name):
        d = self.declarations[name]
        provider = self._provider(name)
        hook = getattr(provider, "revision", None) if provider is not None else d.revision
        if hook is None:
            if self.strict.get():
                raise ConsistencyUnavailable(name, "provider is best_effort")
            return None
        reader = ReadContext(self, {})
        async with asyncio.timeout(d.timeout):
            if provider is not None:
                result = hook(reader, d.resource, **dict(d.options))
                result = await result if inspect.isawaitable(result) else result
                if result is None and self.strict.get():
                    raise ConsistencyUnavailable(name, "provider returned no verifiable revision")
                return result
            result = await invoke(hook, reader)
            if result is None and self.strict.get():
                raise ConsistencyUnavailable(name, "provider returned no verifiable revision")
            return result

    async def token(self, name, trail=()):
        if name not in self.nodes:
            return self.legacy(name).revision
        if name in trail:
            raise CircularDependency(name, " -> ".join((*trail, name)))
        d, node = self.declarations[name], self.nodes[name]
        if d.kind in {"state", "computed"}:
            self.read(name)
            key = (node.version, self.context.compiler.policy.materialization_fingerprint)
            if node.token_key != key:
                node.revision_token = digest((self.binding(name), d.version, *key))
                node.token_key = key
            return node.revision_token
        external = await self.external_revision(name)
        dependencies = {}
        for dep in dict.fromkeys((*d.deps, *node.dependencies)):
            name_or_key = self.names.get(dep, dep)
            dependencies[dep] = await self.token(name_or_key, (*trail, name))
        return digest(
            (
                self.binding(name),
                d.version,
                self.context.compiler.policy.materialization_fingerprint,
                external,
                self.epoch if external is None else None,
                dependencies,
            )
        )

    async def resolve(self, name):
        self.ensure_open()
        name = self.name(name)
        d = self.declarations[name]
        if d.kind in {"state", "computed"}:
            return self.read(name)
        stack = _resolving.get()
        if (self.identity, name) in stack:
            raise CircularDependency(name, " -> ".join([*(n for _, n in stack), name]))
        token = _resolving.set((*stack, (self.identity, name)))
        task = asyncio.current_task()
        self.tasks[task] = self.tasks.get(task, 0) + 1
        try:
            async with self.locks[name]:
                node = self.nodes[name]
                for _ in range(self.context.retries):
                    current_token = await self.token(name)
                    if (
                        node.resolved
                        and not node.dirty
                        and node.materialization is not None
                        and (node.materialization.revision == current_token)
                    ):
                        node.hits += 1
                        return node.value
                    reads = {}
                    reader = ReadContext(self, reads)
                    for dep in dict.fromkeys((*d.deps, *node.dependencies)):
                        await reader.resolve(dep)
                    before = await self.external_revision(name)
                    provider = self._provider(name)
                    try:
                        async with asyncio.timeout(d.timeout):
                            value = (
                                await provider.resolve(reader, d.resource, **dict(d.options))
                                if provider is not None
                                else await invoke(d.function, self.context, reader)
                            )
                        if isinstance(value, PartMaterialization):
                            sources = value.sources
                            value = value.value
                        else:
                            sources = (d.resource or self.binding(name),)
                        try:
                            value = self.adapters[name].validate_python(value)
                        except (ValidationError, TypeError, ValueError):
                            raise SchemaInvalid(name) from None
                        if self.closed:
                            raise asyncio.CancelledError()
                        after = await self.external_revision(name)
                        for dep in reads:
                            if dep in self.declarations and self.declarations[dep].kind in {
                                "state",
                                "computed",
                            }:
                                self.read(dep)
                        tokens_stale = False
                        for dep, observed in reader._tokens.items():
                            if await self.token(dep) != observed:
                                self.invalidate(dep)
                                tokens_stale = True
                        if (
                            before != after
                            or tokens_stale
                            or any(
                                self.dependency_version(dep) != rev for dep, rev in reads.items()
                            )
                        ):
                            node.dirty = True
                            continue
                    except asyncio.CancelledError:
                        node.dirty = True
                        raise
                    except Exception as exc:
                        node.error = getattr(exc, "code", "MATERIALIZATION_FAILED")
                        node.dirty = True
                        raise
                    value = freeze(value)
                    node.value = value
                    node.dependencies = reads
                    node.version += 1
                    node.resolved, node.dirty = True, False
                    revision = await self.token(name)
                    node.materialization = PartMaterialization(
                        value,
                        revision,
                        sources,
                        reads,
                        consistency="versioned" if after is not None else "best_effort",
                    )
                    node.computations += 1
                    node.error = None
                    return value
                raise ConsistencyUnavailable(name, "async dependencies changed during resolution")
        finally:
            _resolving.reset(token)
            self.tasks[task] -= 1
            if not self.tasks[task]:
                del self.tasks[task]

    def definitions(self):
        result = []
        for name, d in self.declarations.items():

            async def resolve(ctx, name=name):
                token = _core_reads.set((*_core_reads.get(), (self.identity, ctx.dependencies)))
                try:
                    return await self.resolve(name)
                finally:
                    _core_reads.reset(token)

            async def revision(ctx, name=name):
                token = _core_reads.set((*_core_reads.get(), (self.identity, ctx.dependencies)))
                try:
                    return await self.token(name)
                finally:
                    _core_reads.reset(token)

            definition = PartDefinition(
                self.keys[name],
                d.schema,
                resolve,
                revision=revision,
                deps=tuple(self.keys.get(dep, dep) for dep in d.deps),
                resource_key=self.binding(name),
                required=d.required,
                timeout=d.timeout,
                algorithm_version=d.version,
                project=d.project,
                expose=d.expose,
            )
            if d.provider == "skill":
                role = self.context.compiler.policy.grants.get(self.keys[name])
                if role is not None:
                    from ..sources.skills import skill_part

                    definition = skill_part(definition, role=role)
            result.append(definition)
        return result

    async def begin(self, consistency, selected=None):
        self.ensure_open()
        mode = getattr(consistency, "mode", consistency)
        self.epoch += 1
        from ..core.graph import topological

        keys = set(topological(self.context.registry.definitions, selected))
        for name, d in self.declarations.items():
            if self.keys[name] not in keys or d.kind in {"state", "computed"}:
                continue
            provider = self.providers.get(d.provider) if d.kind == "provider" else None
            versioned = getattr(provider, "revision", None) if provider is not None else d.revision
            if mode == "strict_fresh" and versioned is None:
                raise ConsistencyUnavailable(name, "provider is best_effort")
            if versioned is None:
                self.invalidate(name)

    async def verify(self, snapshot):
        view = {p.part_id: p for p in snapshot.parts}
        token = _core_reads.set((*_core_reads.get(), (self.identity, view)))
        try:
            for part in snapshot.parts:
                if part.part_id in self.names:
                    name = self.names[part.part_id]
                    if await self.token(name) != part.revision:
                        return False
            return True
        finally:
            _core_reads.reset(token)

    def metadata(self, parts):
        result = {}
        for key in parts:
            if key not in self.names:
                continue
            name = self.names[key]
            node = self.nodes[name]
            d = self.declarations[name]
            result[key] = {
                "declaration_version": d.version,
                "mode": d.mode,
                "resource_id": self.binding(name),
                "dependencies": {self.keys.get(n, n): v for n, v in node.dependencies.items()},
                "sources": node.materialization.sources if node.materialization else (),
                "consistency": node.materialization.consistency
                if node.materialization
                else ("versioned"),
            }
        return result

    def explain(self, name):
        name = self.name(name)
        node = self.nodes[name]
        return freeze(
            {
                "key": self.keys[name],
                "resource_id": self.binding(name),
                "status": "not_resolved"
                if not node.resolved
                else ("dirty" if node.dirty else "clean"),
                "revision": node.version,
                "dependencies": dict(node.dependencies),
                "cache_hits": node.hits,
                "computations": node.computations,
                "sources": node.materialization.sources if node.materialization else (),
                "consistency": node.materialization.consistency if node.materialization else None,
                "diagnostic": node.error,
            }
        )

    async def close(self):
        self.closed = True
        pending = set(self.tasks)
        if self.boundary_task is not None:
            pending.add(self.boundary_task)
        tasks = [task for task in pending if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
