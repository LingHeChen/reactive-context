"""Reactive parts, consistency boundaries and immutable compilation."""

import asyncio
from collections import OrderedDict
from collections.abc import Coroutine
from dataclasses import replace
from functools import partial
from time import perf_counter, time
from types import MappingProxyType
from typing import Any, overload

from pydantic import ValidationError

from .api.context import ContextType, OptionsAPI
from .api.declarations import MissingPartProvider
from .api.protocol import normalize_part
from .compiler.compiler import Compiler
from .compiler.ir import CompiledContext, ContextIRSegment, TraceDecision
from .compiler.policy import CompilerPolicy
from .core.graph import downstream, topological
from .core.part import ChangeSignal, PartDefinition, ResolveContext
from .core.registry import Registry
from .core.scheduler import Scheduler
from .core.scope import ConsistencyPolicy, ContextScope
from .core.snapshot import ContextSnapshot, PartSnapshot, digest, freeze, thaw
from .errors import (
    AccessRevoked,
    AuthorityViolation,
    ConsistencyUnavailable,
    ContextError,
    SchemaInvalid,
    SourceUnavailable,
)
from .observability.events import EventLog


class Context(OptionsAPI, metaclass=ContextType):
    __runtime_fields__ = frozenset({"registry", "compiler", "scheduler", "retries", "events"})

    def __init__(
        self,
        parts=(),
        *,
        compiler_policy: CompilerPolicy | None = None,
        retries: int = 3,
        concurrency: int = 16,
        event_log: EventLog | None = None,
        scope_bindings=None,
        providers=None,
        services=None,
        **options,
    ):
        if retries < 1:
            raise ValueError("retries must be positive")
        self._closed = False
        self.compiler = Compiler(compiler_policy)
        self._bind_options(options, scope_bindings, providers, services)
        definitions = [normalize_part(p) for p in parts]
        if self._dx is not None:
            definitions.extend(self._dx.definitions())
        self.registry = Registry(definitions)
        self.scheduler = Scheduler(concurrency)
        self.retries = retries
        self.events = event_log or EventLog()
        self._pending: set[tuple[str, str]] = set()
        self._disabled: set[str] = set()
        self._revoked: set[tuple[str | None, str]] = set()
        self._pinned: dict[str, ContextSnapshot] = {}
        self._traces: OrderedDict[str, tuple] = OrderedDict()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing_scopes: set[str] = set()

    def register(self, part):
        self._ensure_open()
        self.registry.register(normalize_part(part))
        return self

    def extend(self, *, parts):
        """Return an independent context; live instances/subscriptions are never shared."""
        if self._dx is not None:
            from copy import deepcopy

            legacy = [p for k, p in self.registry.definitions.items() if k not in self._dx.names]
            options = {
                n: deepcopy(node.value)
                for n, node in self._dx.nodes.items()
                if self.__parts__[n].kind == "state"
            }
            extension_options = self._extension_options()
            rebound = {
                value.part.key
                for value in extension_options.values()
                if isinstance(getattr(value, "part", None), PartDefinition)
            }
            legacy = [definition for definition in legacy if definition.key not in rebound]
            extended = type(self)(
                [*legacy, *parts],
                compiler_policy=self.compiler.policy,
                retries=self.retries,
                scope_bindings=self._dx.scope_bindings,
                providers=self._dx.providers,
                services=self._dx.services,
                **options,
                **extension_options,
            )
        else:
            extended = Context(
                [*self.registry.definitions.values(), *parts],
                compiler_policy=self.compiler.policy,
                retries=self.retries,
            )
        extended._disabled = self._disabled.copy()
        extended._revoked = self._revoked.copy()
        return extended

    def _extension_options(self):
        return {}

    def _mark_dirty(self, scope, key, event_time=None):
        for affected in downstream(self.registry.definitions, key):
            target = self.registry.instance(scope, affected)
            if target.state != "REVOKED":
                target.generation += 1
                target.event_time = event_time
                target.state = "DIRTY"

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("context is closed")

    def notify(self, signal: ChangeSignal | str, *, scope: str | None = None):
        if self._closed:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if self._loop is not None and running is not self._loop:
            self._loop.call_soon_threadsafe(partial(self.notify, signal, scope=scope))
            return
        if isinstance(signal, str):
            signal = ChangeSignal(signal)
        key = signal.source_id
        if key not in self.registry.definitions:
            raise KeyError(key)
        if signal.revoked:
            self.revoke(key, scope=scope)
            return
        for (sid, name), instance in list(self.registry.instances.items()):
            if (scope is not None and sid != scope) or name != key:
                continue
            if instance.definition.invalidate:
                decision = instance.definition.invalidate(signal, instance.snapshot)
                if decision == "revoke":
                    self.revoke(key, scope=sid)
                    continue
                if decision is False or decision == "ignore":
                    continue
            pair = (sid, key)
            if pair in self._pending:
                continue
            self._pending.add(pair)
            asyncio.get_running_loop().call_soon(self._pending.discard, pair)
            self._mark_dirty(sid, key, signal.event_time)
            self.events.emit(
                "dirty",
                scope_id=sid,
                part_id=key,
                generation=instance.generation,
                reason="change_signal",
            )

    def revoke(self, key: str, *, scope: str | None = None):
        self._revoked.add((scope, key))
        for (sid, name), _instance in list(self.registry.instances.items()):
            if name == key and (scope is None or sid == scope):
                for affected in downstream(self.registry.definitions, key):
                    target = self.registry.instance(sid, affected)
                    target.generation += 1
                    target.state = "REVOKED"
                    target.snapshot = None
                self.events.emit("revoked", scope_id=sid, part_id=key)

    async def _authorize(self, instance, ctx):
        if (
            instance.state == "REVOKED"
            or (None, ctx.part_id) in self._revoked
            or (ctx.scope_id, ctx.part_id) in self._revoked
        ):
            raise AccessRevoked(ctx.part_id)
        if instance.definition.acl_check:
            try:
                allowed = await self.scheduler.call(
                    instance.definition.acl_check, instance.definition.timeout, ctx
                )
            except AccessRevoked:
                self.revoke(ctx.part_id, scope=ctx.scope_id)
                raise
            except ContextError:
                raise
            except Exception:
                raise AccessRevoked(ctx.part_id, "authorization status unavailable") from None
            if not allowed:
                self.revoke(ctx.part_id, scope=ctx.scope_id)
                raise AccessRevoked(ctx.part_id)

    async def _resolve(self, scope, key, dependencies, policy):
        instance = self.registry.instance(scope, key)
        definition = instance.definition
        ctx = ResolveContext(scope, key, definition.configuration, MappingProxyType(dependencies))
        async with instance.lock:
            await self._authorize(instance, ctx)
            if policy.mode == "pinned" and instance.snapshot is not None:
                pinned = replace(instance.snapshot, generation=instance.generation)
                instance.snapshot = pinned
                instance.state = "CLEAN"
                return pinned
            if not instance.subscribed and definition.subscribe:
                instance.unsubscribe = await self.scheduler.call(
                    definition.subscribe,
                    definition.timeout,
                    ctx,
                    lambda change=None: self.notify(change or key, scope=scope),
                )
                instance.subscribed = True
            for _attempt in range(self.retries):
                started = perf_counter()
                generation = instance.generation
                previous = instance.snapshot
                vector = tuple((k, p.revision, p.status) for k, p in dependencies.items())
                try:
                    factor = None
                    await self._authorize(instance, ctx)
                    if definition.computed:
                        vector = (
                            *vector,
                            ("algorithm", definition.algorithm_version),
                            ("configuration", digest(definition.configuration)),
                        )
                        if definition.revision:
                            factor = await self.scheduler.call(
                                definition.revision, definition.timeout, ctx
                            )
                            vector = (*vector, ("external_factor", factor))
                        revision = previous.revision if previous else None
                        fresh = (
                            definition.memoize
                            and previous is not None
                            and instance.dep_vector == vector
                        )
                    else:
                        within_age = (
                            policy.mode == "bounded_staleness"
                            and previous is not None
                            and instance.state == "CLEAN"
                            and time() - previous.verified_at <= policy.max_age
                        )
                        if within_age and previous is not None:
                            self.events.emit(
                                "resolve",
                                scope_id=scope,
                                part_id=key,
                                revision=previous.revision,
                                cache_hit=True,
                                freshness_status=previous.status,
                            )
                            return previous
                        revision = (
                            (
                                await self.scheduler.call(
                                    definition.revision, definition.timeout, ctx
                                )
                            )
                            if definition.revision
                            else None
                        )
                        if (
                            revision is not None
                            and previous is not None
                            and revision != previous.revision
                            and generation == previous.generation
                        ):
                            # Polling discovers changes even when every watch event is lost.
                            self._mark_dirty(scope, key)
                            generation = instance.generation
                        fresh = (
                            previous is not None
                            and revision is not None
                            and previous.revision == revision
                        )
                        if revision is None and previous and definition.ttl:
                            fresh = (
                                instance.state == "CLEAN"
                                and time() - previous.verified_at < definition.ttl
                            )
                    instance.state = "RESOLVING"
                    if fresh and previous:
                        value = previous.value
                    else:
                        incoming = await self.scheduler.call(
                            definition.resolve, definition.timeout, replace(ctx, revision=revision)
                        )
                        if definition.update and previous:
                            incoming = await self.scheduler.call(
                                definition.update, definition.timeout, previous.value, incoming, ctx
                            )
                        try:
                            value = freeze(definition.adapter.validate_python(incoming))
                        except (ValidationError, TypeError, ValueError):
                            raise SchemaInvalid(key) from None
                        if definition.computed:
                            revision = digest((vector, value))
                        if definition.computed and definition.revision:
                            after_factor = await self.scheduler.call(
                                definition.revision, definition.timeout, ctx
                            )
                            if after_factor != factor:
                                self._mark_dirty(scope, key)
                                continue
                        if not definition.computed and definition.revision:
                            after = await self.scheduler.call(
                                definition.revision, definition.timeout, ctx
                            )
                            if after != revision:
                                instance.state = "DIRTY"
                                continue
                    if self._closed or scope in self._closing_scopes:
                        raise asyncio.CancelledError()
                    await self._authorize(instance, ctx)
                    if instance.generation != generation:
                        instance.state = "DIRTY"
                        continue
                    status = "CURRENT" if revision is not None else "UNKNOWN"
                    if any(p.status != "CURRENT" for p in dependencies.values()):
                        status = "UNKNOWN"
                    if policy.mode == "strict_fresh" and status != "CURRENT":
                        raise ConsistencyUnavailable(key, "source has no verifiable revision")
                    now = time()
                    snapshot = PartSnapshot(
                        key,
                        scope,
                        definition.resource_key or key,
                        str(revision) if revision is not None else digest(value),
                        generation,
                        value,
                        previous.observed_at if fresh and previous else now,
                        (previous.verified_at if fresh and revision is None and previous else now),
                        status,
                        (
                            ("source", definition.resource_key or key),
                            ("kind", definition.kind),
                            ("schema_version", str(definition.schema_version)),
                        ),
                        event_time=instance.event_time,
                    )
                    instance.snapshot = snapshot
                    instance.dep_vector = vector
                    instance.state = "CLEAN"
                    self.events.emit(
                        "resolve",
                        scope_id=scope,
                        part_id=key,
                        revision=snapshot.revision,
                        generation=generation,
                        duration_ms=(perf_counter() - started) * 1000,
                        cache_hit=bool(fresh),
                        freshness_status=status,
                    )
                    return snapshot
                except asyncio.CancelledError:
                    if instance.state != "REVOKED":
                        instance.state = "DIRTY"
                    if instance.unsubscribe:
                        try:
                            await self.scheduler.call(instance.unsubscribe, definition.timeout)
                        except Exception:
                            self.events.emit(
                                "dispose_error",
                                scope_id=scope,
                                part_id=key,
                                reason="subscription disposal failed",
                            )
                        finally:
                            instance.unsubscribe = None
                            instance.subscribed = False
                    raise
                except AccessRevoked:
                    self.revoke(key, scope=scope)
                    raise
                except ConsistencyUnavailable:
                    instance.state = "DIRTY"
                    if _attempt + 1 == self.retries:
                        raise
                    await asyncio.sleep(0)
                    continue
                except Exception as exc:
                    instance.state = "ERROR"
                    self.events.emit(
                        "error",
                        scope_id=scope,
                        part_id=key,
                        reason=exc.code if isinstance(exc, ContextError) else "source_unavailable",
                    )
                    if definition.on_error:
                        await self.scheduler.call(
                            definition.on_error, definition.timeout, SourceUnavailable(key), ctx
                        )
                    # Schema violations and consistency failures cannot silently reuse old
                    # values.
                    if isinstance(exc, ContextError) and not isinstance(exc, SourceUnavailable):
                        raise
                    if previous and policy.on_unavailable != "block":
                        await self._authorize(instance, ctx)
                        if instance.generation != generation:
                            raise ConsistencyUnavailable(key) from None
                        return replace(
                            previous,
                            status=("STALE" if policy.on_unavailable == "cached" else "UNKNOWN"),
                        )
                    raise SourceUnavailable(key) from None
            instance.state = "DIRTY"
            raise ConsistencyUnavailable(key, "retry budget exhausted")

    async def next_context(
        self,
        scope="default",
        consistency="live",
        *,
        selected=None,
        max_age=0,
        on_unavailable="block",
        part_policies=None,
    ) -> ContextSnapshot:
        self._ensure_open()
        if self._dx is not None and self._dx.boundary_task is not asyncio.current_task():
            async with self._dx.boundary_lock:
                self._dx.boundary_task = asyncio.current_task()
                strict_token = self._dx.strict.set(
                    getattr(consistency, "mode", consistency) == "strict_fresh"
                )
                try:
                    if not (consistency == "pinned" and scope in self._pinned):
                        await self._dx.begin(consistency, selected)
                    for _ in range(self.retries):
                        snapshot = await Context.next_context(
                            self,
                            scope,
                            consistency,
                            selected=selected,
                            max_age=max_age,
                            on_unavailable=on_unavailable,
                            part_policies=part_policies,
                        )
                        if snapshot.consistency == "pinned" or await self._dx.verify(snapshot):
                            return snapshot
                    raise ConsistencyUnavailable(detail="declaration snapshot changed")
                finally:
                    self._dx.boundary_task = None
                    self._dx.strict.reset(strict_token)
        running = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not running:
            raise RuntimeError("a Context must be used on one event loop")
        self._loop = running
        if isinstance(scope, ContextScope):
            consistency, max_age, scope = scope.consistency, scope.max_age, scope.id
        policy = (
            consistency
            if isinstance(consistency, ConsistencyPolicy)
            else ConsistencyPolicy(consistency, max_age, on_unavailable)
        )
        if scope in self._closing_scopes:
            raise RuntimeError("scope is closing")
        definitions = self.registry.definitions
        requested = list(definitions) if selected is None else list(selected)
        roots = [k for k in requested if k not in self._disabled]
        order = topological(definitions, roots)
        if any(k in self._disabled for k in order):
            raise AccessRevoked(detail="disabled dependency")
        if policy.mode == "pinned" and scope in self._pinned:
            pinned = self._pinned[scope]
            if set(pinned.revision_vector) != set(order):
                raise ValueError("repin required after selection changes")
            for key in order:
                instance = self.registry.instance(scope, key)
                await self._authorize(
                    instance,
                    ResolveContext(
                        scope, key, instance.definition.configuration, MappingProxyType({})
                    ),
                )
            return pinned
        for _ in range(self.retries):
            parts = {}
            omissions = []
            for key in order:
                definition = definitions[key]
                if any(dep not in parts for dep in definition.deps):
                    if definition.required:
                        raise SourceUnavailable(key, "required dependency unavailable")
                    omissions.append({"part_key": key, "code": "DEPENDENCY_UNAVAILABLE"})
                    continue
                try:
                    parts[key] = await self._resolve(
                        scope,
                        key,
                        {dep: parts[dep] for dep in definition.deps},
                        (part_policies or {}).get(
                            key, ConsistencyPolicy("live") if policy.mode == "pinned" else policy
                        ),
                    )
                except (SourceUnavailable, MissingPartProvider) as exc:
                    if definition.required:
                        raise
                    omissions.append({"part_key": key, "code": exc.code})
            if any(
                self.registry.instance(scope, key).generation != snap.generation
                for key, snap in parts.items()
            ):
                continue
            segments = []
            for key, snapshot in parts.items():
                definition = definitions[key]
                if not definition.expose:
                    continue
                if definition.project:
                    hook_ctx = ResolveContext(
                        scope,
                        key,
                        definition.configuration,
                        MappingProxyType({dep: parts[dep] for dep in definition.deps}),
                        snapshot.revision,
                    )
                    projected = await self.scheduler.call(
                        definition.project, definition.timeout, snapshot.value, hook_ctx
                    )
                else:
                    projected = [
                        ContextIRSegment(
                            key,
                            key,
                            snapshot.revision,
                            snapshot.value,
                            required=definition.required,
                        )
                    ]
                if definition.required and not projected:
                    raise SourceUnavailable(key, "required part projected no segments")
                for segment in projected:
                    if not isinstance(segment, ContextIRSegment):
                        raise TypeError("project must return ContextIRSegment objects")
                    if (
                        definition.kind == "fact" or definition.kind.startswith("fact.")
                    ) and segment.requested_role in {
                        "system",
                        "developer",
                    }:
                        raise AuthorityViolation(key, "facts cannot acquire instruction authority")
                    if definition.kind == "fact" or definition.kind.startswith("fact."):
                        segment = replace(
                            segment,
                            kind="fact",
                            trust_class="agent_fact",
                            payload={"verification": "unverified", "value": segment.payload},
                        )
                    # Plugin claims cannot forge another part's provenance or host grants.
                    if segment.part_id != key:
                        raise ValueError("project cannot impersonate another part")
                    segments.append(
                        replace(
                            segment,
                            scope_id=scope,
                            source_revision=snapshot.revision,
                            resource_id=snapshot.resource_id,
                            status=snapshot.status,
                            required=definition.required or segment.required,
                        )
                    )
            if any(
                self.registry.instance(scope, key).generation != snap.generation
                for key, snap in parts.items()
            ):
                continue
            config_revision = self.compiler.policy.fingerprint
            policy_revision = digest(
                {
                    "part_policies": {
                        k: (v.mode, v.max_age, v.on_unavailable)
                        for k, v in (part_policies or {}).items()
                    },
                    "mode": policy.mode,
                    "max_age": policy.max_age,
                    "fallback": policy.on_unavailable,
                    "revision": policy.revision,
                }
            )
            metadata = freeze(
                {
                    "scope_bindings": dict(self._dx.scope_bindings) if self._dx else {},
                    "parts": self._dx.metadata(parts) if self._dx else {},
                    "omissions": omissions,
                }
            )
            identity = digest(
                (
                    scope,
                    [(k, p.revision, p.status) for k, p in parts.items()],
                    [
                        (
                            k,
                            definitions[k].kind,
                            definitions[k].schema_version,
                            definitions[k].algorithm_version,
                            definitions[k].projection_version,
                            definitions[k].expose,
                            definitions[k].deps,
                            digest(definitions[k].configuration),
                        )
                        for k in parts
                    ],
                    config_revision,
                    policy_revision,
                    metadata,
                )
            )
            snapshot = ContextSnapshot(
                identity,
                scope,
                tuple(parts.values()),
                tuple(segments),
                config_revision,
                policy_revision,
                policy.mode,
                metadata,
            )
            self.events.emit(
                "snapshot",
                scope_id=scope,
                snapshot_id=identity,
                revision=digest(snapshot.revision_vector),
            )
            self.events.record_manifest(self.export_snapshot(snapshot))
            if policy.mode == "pinned":
                self._pinned[scope] = snapshot
            return snapshot
        raise ConsistencyUnavailable(detail="boundary changed during compilation")

    @overload
    def compile(
        self,
        snapshot: ContextSnapshot,
        *,
        target="generic",
        history_tokens=0,
        scope="default",
        consistency="live",
        evidence=None,
    ) -> CompiledContext: ...

    @overload
    def compile(
        self,
        snapshot: None = None,
        *,
        target="generic",
        history_tokens=0,
        scope="default",
        consistency="live",
        evidence=None,
    ) -> Coroutine[Any, Any, CompiledContext]: ...

    def compile(
        self,
        snapshot: ContextSnapshot | None = None,
        *,
        target="generic",
        history_tokens=0,
        scope="default",
        consistency="live",
        evidence=None,
    ):
        if snapshot is None:
            return self._compile_current(
                scope=scope,
                consistency=consistency,
                target=target,
                history_tokens=history_tokens,
                evidence=evidence,
            )
        if target not in {"generic", "pydantic_ai"}:
            raise ValueError("unknown target")
        self._ensure_open()
        if snapshot.compiler_config_revision != self.compiler.policy.fingerprint:
            raise ConsistencyUnavailable(detail="compiler policy changed; take a new snapshot")
        for part in snapshot.parts:
            instance = self.registry.instances.get((snapshot.scope_id, part.part_id))
            if instance is None or instance.state == "REVOKED" or part.part_id in self._disabled:
                raise AccessRevoked(part.part_id)
        start = perf_counter()
        try:
            compiled = self.compiler.compile(
                snapshot, history_tokens=history_tokens, evidence=evidence
            )
        except ContextError as exc:
            self._traces[snapshot.id] = (
                TraceDecision(
                    exc.part_id,
                    "",
                    "",
                    "ERROR",
                    False,
                    exc.code,
                    0,
                    self.compiler.policy.fingerprint,
                ),
            )
            self.events.emit(
                "compile_error",
                scope_id=snapshot.scope_id,
                snapshot_id=snapshot.id,
                part_id=exc.part_id,
                reason=exc.code,
            )
            raise
        self._traces[snapshot.id] = compiled.trace
        if len(self._traces) > 1000:
            self._traces.popitem(last=False)
        self.events.emit(
            "compile",
            scope_id=snapshot.scope_id,
            snapshot_id=snapshot.id,
            duration_ms=(perf_counter() - start) * 1000,
        )
        for decision in compiled.trace:
            self.events.emit(
                "segment_decision",
                scope_id=snapshot.scope_id,
                snapshot_id=snapshot.id,
                part_id=decision.part_id,
                revision=decision.revision,
                reason=decision.inclusion_reason,
                freshness_status=decision.status,
                resource_id=decision.resource_id,
                token_estimate=decision.token_estimate,
                compiler_config_revision=decision.compiler_config_revision,
            )
        return compiled

    async def _compile_current(
        self, *, scope, consistency, target, history_tokens, evidence=None
    ) -> CompiledContext:
        snapshot = await self.next_context(scope, consistency)
        return self.compile(
            snapshot,
            target=target,
            history_tokens=history_tokens,
            **({"evidence": evidence} if evidence is not None else {}),
        )

    def repin(self, scope="default"):
        self._pinned.pop(scope, None)

    async def guard(self, snapshot: ContextSnapshot):
        """Recheck access and versions immediately before a host-controlled side effect.

        This is a precondition check, not a distributed transaction with the action.
        """
        current = await self.next_context(
            snapshot.scope_id, "strict_fresh", selected=list(snapshot.revision_vector)
        )
        if current.revision_vector != snapshot.revision_vector:
            raise ConsistencyUnavailable(detail="action precondition changed")
        return current

    def inspect(self, part_id, *, scope="default"):
        instance = self.registry.instance(scope, part_id)
        return freeze(
            {
                "part_id": part_id,
                "scope_id": scope,
                "state": instance.state,
                "generation": instance.generation,
                "dependencies": instance.definition.deps,
                "revision": instance.snapshot.revision if instance.snapshot else None,
                "last_verified": instance.snapshot.verified_at if instance.snapshot else None,
            }
        )

    def explain(self, snapshot_id):
        if self._dx is not None and snapshot_id in {*self._dx.declarations, *self._dx.names}:
            return self._dx.explain(snapshot_id)
        return self._traces[snapshot_id]

    def export_trace(self):
        return self.events.export()

    def export_snapshot(self, snapshot, *, include_content=False):
        """Metadata by default; content export is an explicit host decision."""
        result = {
            "snapshot_id": snapshot.id,
            "scope_id": snapshot.scope_id,
            "revision_vector": thaw(snapshot.revision_vector),
            "policy_revision": snapshot.policy_revision,
            "compiler_config_revision": snapshot.compiler_config_revision,
        }
        if include_content:
            result["parts"] = {p.part_id: thaw(p.value) for p in snapshot.parts}
        return result

    async def disable(self, key):
        dependents = downstream(self.registry.definitions, key)
        if len(dependents) > 1:
            raise ValueError("cannot disable a part with dependents")
        self._disabled.add(key)
        await self._dispose_instances(key=key)
        self._pinned.clear()

    def enable(self, key):
        if key not in self.registry.definitions:
            raise KeyError(key)
        self._disabled.discard(key)

    async def unregister(self, key):
        await self.disable(key)
        del self.registry.definitions[key]
        self._disabled.discard(key)

    async def _dispose_instances(self, *, scope=None, key=None):
        errors = []
        for pair, instance in list(self.registry.instances.items()):
            sid, name = pair
            if (scope is not None and sid != scope) or (key is not None and name != key):
                continue
            instance.generation += 1
            instance.state = "REVOKED"
            async with instance.lock:
                for hook in (instance.unsubscribe, instance.definition.dispose):
                    if hook:
                        try:
                            await self.scheduler.call(hook, instance.definition.timeout)
                        except Exception:
                            errors.append(name)
                self.registry.instances.pop(pair, None)
        if errors:
            raise SourceUnavailable(detail="subscription disposal failed")

    async def close_scope(self, scope="default"):
        self._closing_scopes.add(scope)
        try:
            await self._dispose_instances(scope=scope)
        finally:
            self._pinned.pop(scope, None)
            self._closing_scopes.discard(scope)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._dx is not None:
            await self._dx.close()
        try:
            await self._dispose_instances()
        finally:
            self.events.close()
            self._pinned.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
