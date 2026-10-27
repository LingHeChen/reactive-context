"""Host registry + agent-selected persistent references, composed over the existing core."""

import weakref
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from time import time
from typing import Any, overload

from .compiler.compiler import Compiler
from .compiler.ir import CompiledContext, ContextIRSegment
from .context import Context
from .core.graph import topological
from .core.part import ResolveContext, source_part
from .core.scope import ConsistencyPolicy
from .core.snapshot import ContextSnapshot, PartSnapshot, freeze, thaw
from .errors import (
    AccessRevoked,
    CompilerBudget,
    ConsistencyUnavailable,
    ContextBudgetExceeded,
    ResourceConflict,
)
from .history import HistoryManager
from .store import ContextStore, Principal, VersionConflict


@dataclass(frozen=True)
class LoadedPart:
    id: str
    snapshot: PartSnapshot
    evidence_id: str


class ContextManager(Context):
    __runtime_fields__ = Context.__runtime_fields__ | {
        "store",
        "principal",
        "objective",
        "work_state",
        "history",
        "mandatory",
    }

    def __init__(
        self,
        parts=(),
        *,
        store: ContextStore,
        principal: Principal,
        mandatory=(),
        history=None,
        objective=None,
        work_state=None,
        **kwargs,
    ):
        if work_state is not None:
            if work_state.store is not store or work_state.principal != principal:
                raise ValueError("work state must share the manager's store and identity")
            parts = (*parts, work_state.part)
            mandatory = (*mandatory, work_state.key)
        if objective is not None:
            if objective.store is not store or objective.principal != principal:
                raise ValueError("objective must share the manager's store and identity")
            parts = (*parts, objective.part)
            mandatory = (*mandatory, objective.key)
        super().__init__(parts, **kwargs)
        self.store, self.principal = store, principal
        self.objective = objective
        self.work_state = work_state
        self.history = HistoryManager(store, principal) if history is None else history
        if self.history is not False and (
            self.history.store is not store or self.history.principal != principal
        ):
            raise ValueError("history must share the manager's store and identity")
        if self.history is not False:
            self.history.authorize_sources = self._authorize_history_sources
        self._declaration_keys = tuple(self._dx.keys.values()) if self._dx else ()
        self.mandatory = tuple(dict.fromkeys((*mandatory, *self._declaration_keys)))
        if not set(self.mandatory) <= self.registry.definitions.keys():
            raise ValueError("mandatory parts must be registered by the host")
        self._resource_keys = set(self.registry.definitions)
        self._snapshot_leases: dict[int, tuple] = {}
        self._fact_refs: dict[tuple[str, str], Any] = {}

    def _extension_options(self):
        history = self.history
        if history is not False:
            history = HistoryManager(
                self.store,
                self.principal,
                compactor=history.compactor,
                validator=history.validator,
                policy=history.policy,
                estimator=history.estimator,
                summary_model=history.summary_model,
                final_projector=history.final_projector,
            )
        return {
            "store": self.store,
            "principal": self.principal,
            "mandatory": tuple(k for k in self.mandatory if k not in self._declaration_keys),
            "history": history,
            "objective": self.objective,
            "work_state": self.work_state,
        }

    async def save_fact(
        self, scope, schema_id, key, value, evidence, *, schema_version=1, expected_revision=None
    ):
        """Check first; create/update only durable changes. Never implicitly activate.

        Existing changed keys require an explicit expected revision. Exact replays,
        skipped transient claims and rejected claims create no proposal or Fact.
        """
        from dataclasses import asdict

        result = self.store.check_fact(
            self.principal, scope, schema_id, key, value, evidence, schema_version=schema_version
        )
        if expected_revision is not None and expected_revision != result.revision:
            raise VersionConflict(result.part_id or "")
        if result.decision != "propose":
            return {**asdict(result), "verification": "unverified"}
        if result.part_id and expected_revision is None:
            return {**asdict(result), "decision": "revision_required"}
        proposal = self.store.propose_fact(
            self.principal,
            scope,
            schema_id,
            key,
            value,
            evidence,
            schema_version=schema_version,
            expected_revision=expected_revision,
        )
        record = self.store.commit(self.principal, proposal)
        return {
            "decision": "updated" if result.part_id else "created",
            "part_id": record.id,
            "revision": record.revision,
            "validation": result.validation,
            "reasons": result.reasons,
            "verification": record.verification,
        }

    def publish_catalog(self, host: Principal, scope):
        """Explicit host provisioning, never exposed as an agent tool."""
        for key in sorted(self._resource_keys - set(self.mandatory)):
            definition = self.registry.definitions[key]
            if definition.kind == "fact" or definition.kind.startswith("fact."):
                continue
            part_id = self.store.register_resource(
                host,
                scope,
                key,
                kind=definition.kind,
                metadata={
                    "schema_version": definition.schema_version,
                    "resource_key": definition.resource_key or key,
                },
            )
            self.store.grant_part(host, part_id, self.principal.id)

    async def query(self, scope, intent="", *, kind=None, limit=20, after_id=None):
        rows = self.store.query(
            self.principal, scope, intent, kind=kind, limit=limit, after_id=after_id
        )
        visible = []
        for row in rows:
            if row["kind"] != "fact":
                try:
                    await self._authorize_resource(scope, row["key"])
                except AccessRevoked:
                    continue
            visible.append(row)
        return visible

    async def _authorize_history_sources(self, scope, sources):
        for source in sources:
            if source["kind"] == "fact":
                self.store.metadata(self.principal, source["id"])
            else:
                await self._authorize_resource(scope, source["key"])

    async def search_history(
        self, scope, intent, *, kind="all", limit=20, exclude_event_ids=(), preview_chars=2000
    ):
        if self.history is False:
            raise ValueError("history disabled")
        hits = self.history.search(
            scope,
            intent,
            kind=kind,
            limit=limit,
            exclude_event_ids=exclude_event_ids,
            preview_chars=preview_chars,
        )
        visible = []
        for hit in hits:
            try:
                await self._authorize_history_sources(scope, hit["sources"])
            except AccessRevoked:
                continue
            visible.append(hit)
        return visible

    async def search_facts(self, scope, intent, *, limit=20):
        """Relevant authorized Fact metadata; independent of ID-paginated catalogs."""
        return self.store.search_facts(self.principal, scope, intent, limit=limit)

    async def load_history(self, scope, event_ids):
        if self.history is False:
            raise ValueError("history disabled")
        if len(event_ids) > 50:
            raise ValueError("history lookup limit exceeded")
        await self._authorize_history_sources(scope, self.history.event_sources(scope, event_ids))
        return self.history.retrieve(scope, event_ids)

    async def prepare_history(self, scope, groups, **budget):
        if self.history is False:
            raise ValueError("history disabled")
        return await self.history.prepare(scope, groups, **budget)

    async def compact_history(self, scope, groups, *, revisions=None):
        if self.history is False:
            raise ValueError("history disabled")
        groups = tuple(groups)
        self.history.observe(scope, groups)
        ids = [self.history.event_id(e) for g in groups for e in g.events]
        await self._authorize_history_sources(scope, self.history.event_sources(scope, ids))
        return await self.history.archive(scope, groups, revisions=revisions)

    async def refresh_objective_if_needed(self, scope, groups, *, used_tokens):
        if self.objective is None or not self.objective.policy.auto_update:
            return False
        if used_tokens <= self.compiler.policy.max_tokens * self.objective.policy.context_ratio:
            return False
        groups = tuple(groups)
        if self.history is not False:
            ids = [
                self.history.event_id(e)
                for g in groups[-self.objective.policy.max_update_groups :]
                for e in g.events
            ]
            await self._authorize_history_sources(scope, self.history.event_sources(scope, ids))
        before = self.objective.read(scope)
        after = await self.objective.refresh_at_threshold(
            scope,
            groups,
            used_tokens=used_tokens,
            total_tokens=self.compiler.policy.max_tokens,
            estimator=self.compiler.policy.estimator,
        )
        return (before or {}).get("revision") != (after or {}).get("revision")

    async def _authorize_resource(self, scope, key):
        if key not in self._resource_keys:
            raise AccessRevoked(detail="resource definition unavailable")
        instance = self.registry.instance(scope, key)
        await self._authorize(
            instance, ResolveContext(scope, key, instance.definition.configuration, {})
        )

    def _is_managed_fact(self, key):
        # Host parts may legitimately use names such as fact.schemas. Only generated
        # persistent Fact keys use numeric store revisions and acquire store leases.
        return key.startswith("fact.") and key not in self._resource_keys

    def _fact_key(self, scope, part_id):
        key = "fact." + part_id
        if key in self._resource_keys:
            raise ResourceConflict(key, "persistent Fact key collides with a host part")
        if key not in self.registry.definitions:

            def record(ctx):
                ref = self._fact_refs.get((ctx.scope_id, part_id))
                revision = ref.revision if ref and ref.policy == "pinned" else None
                return self.store.load(self.principal, part_id, revision=revision)

            async def revision(ctx):
                meta = self.store.metadata(self.principal, part_id)
                if meta["scope_id"] != ctx.scope_id:
                    raise AccessRevoked(detail="fact scope mismatch")
                ref = self._fact_refs.get((ctx.scope_id, part_id))
                return str(ref.revision if ref and ref.policy == "pinned" else meta["revision"])

            async def resolve(ctx):
                fact = record(ctx)
                return {
                    "value": thaw(fact.value),
                    "verification": fact.verification,
                    "schema_id": fact.schema_id,
                    "schema_version": fact.schema_version,
                    "evidence": list(fact.evidence),
                }

            async def allowed(ctx):
                return self.store.metadata(self.principal, part_id)["scope_id"] == ctx.scope_id

            def project(value, ctx):
                return [
                    ContextIRSegment(
                        key,
                        key,
                        ctx.revision,
                        value,
                        kind="fact",
                        required=True,
                        requested_role="user",
                    )
                ]

            self.register(
                source_part(
                    key=key,
                    schema=dict[str, Any],
                    resolve=resolve,
                    revision=revision,
                    project=project,
                    acl_check=allowed,
                    kind="fact",
                    resource_key=part_id,
                )
            )
        return key

    def _authorize_keys(self, scope, roots):
        with self.store._lock:
            for key in topological(self.registry.definitions, roots):
                if key in self.mandatory:
                    continue
                if self._is_managed_fact(key):
                    meta = self.store.metadata(self.principal, key[5:])
                else:
                    row = self.store.db.execute(
                        "SELECT id FROM parts WHERE scope_id=? AND key=? AND schema_id=''",
                        (scope, key),
                    ).fetchone()
                    if row is None:
                        raise AccessRevoked(key)
                    meta = self.store.metadata(self.principal, row[0])
                if meta["scope_id"] != scope:
                    raise AccessRevoked(key)

    async def _read_part(self, scope, part_id):
        meta = self.store.metadata(self.principal, part_id)
        if meta["scope_id"] != scope:
            raise AccessRevoked(detail="part scope mismatch")
        if meta["kind"] == "fact":
            fact = self.store.load(self.principal, part_id)
            now = time()
            # A one-off Fact read must not create a permanent registry definition/instance.
            # Active Facts are registered lazily by next_context from persistent references.
            part = PartSnapshot(
                "fact." + part_id,
                scope,
                part_id,
                str(fact.revision),
                0,
                freeze(
                    {
                        "value": thaw(fact.value),
                        "verification": fact.verification,
                        "schema_id": fact.schema_id,
                        "schema_version": fact.schema_version,
                        "evidence": list(fact.evidence),
                    }
                ),
                now,
                now,
                provenance=(
                    ("source", part_id),
                    ("kind", "fact"),
                    ("schema_version", str(fact.schema_version)),
                ),
            )
            return part
        key = meta["key"]
        self._authorize_keys(scope, [key])
        snapshot = await super().next_context(scope, selected=[key])
        part = snapshot.part(key)
        return part

    async def validate_read(self, scope, part_id, resource_id, revision):
        """Recheck a transient read's authorization and current source revision."""
        part = await self._read_part(scope, part_id)
        return part.resource_id == resource_id and part.revision == revision

    async def load(self, scope, part_id):
        part = await self._read_part(scope, part_id)
        evidence = self.store.observe_evidence(
            self.principal, scope, part.resource_id, part.revision
        )
        # Loading alone never adds an Active Ref.
        return LoadedPart(part_id, part, evidence)

    async def activate(self, scope, part_id, *, policy="live", max_age=0):
        meta = self.store.metadata(self.principal, part_id)
        revision = None
        if policy == "pinned" and meta["kind"] != "fact":
            revision = (await self.load(scope, part_id)).snapshot.revision
        previous = next(
            (r for r in self.store.active_refs(self.principal, scope) if r.part_id == part_id), None
        )
        ref = self.store.activate(
            self.principal, scope, part_id, policy=policy, revision=revision, max_age=max_age
        )
        try:
            self.compile(await self.next_context(scope))
        except BaseException:
            if previous:
                self.store.activate(
                    self.principal,
                    scope,
                    part_id,
                    policy=previous.policy,
                    revision=previous.revision,
                    max_age=previous.max_age,
                )
            else:
                self.store.deactivate(self.principal, scope, part_id)
            raise
        return ref

    async def deactivate(self, scope, part_id):
        if part_id in self.mandatory:
            raise AccessRevoked(detail="mandatory parts cannot be deactivated")
        meta = self.store.metadata(self.principal, part_id)
        if meta["key"] in self.mandatory:
            raise AccessRevoked(detail="mandatory parts cannot be deactivated")
        target = "fact." + part_id if meta["kind"] == "fact" else meta["key"]
        other_roots = list(self.mandatory)
        for active in self.store.active_refs(self.principal, scope):
            if active.part_id == part_id:
                continue
            other = self.store.metadata(self.principal, active.part_id)
            # Persisted refs can outlive this manager's in-memory registry. Rehydrate
            # fact definitions before checking dependencies, as next_context does.
            other_roots.append(
                self._fact_key(scope, other["id"]) if other["kind"] == "fact" else other["key"]
            )
        if other_roots and target in topological(self.registry.definitions, other_roots):
            raise ConsistencyUnavailable(target, "deactivate dependent parts first")
        self.store.deactivate(self.principal, scope, part_id)
        self.store.cache.clear(scope)
        if meta["kind"] == "fact":
            instance = self.registry.instances.get((scope, "fact." + part_id))
            if instance:
                instance.snapshot = None
            self.registry.instances.pop((scope, "fact." + part_id), None)
            self._fact_refs.pop((scope, part_id), None)
            self._prune_fact_definitions()

    def _prune_fact_definitions(self):
        needed = {d for p in self.registry.definitions.values() for d in p.deps}
        live = {key for _, key in self.registry.instances}
        for key in list(self.registry.definitions):
            if (
                key.startswith("fact.")
                and key not in self._resource_keys
                and key not in needed
                and key not in live
            ):
                del self.registry.definitions[key]

    async def next_context(self, scope="default", consistency="live", *, selected=None, **kwargs):
        self.store._scope_access(self.principal, scope)
        refs = self.store.active_refs(self.principal, scope)
        keys, policies = list(self.mandatory), {}
        pinned_resources = {}
        for ref in refs:
            meta = self.store.metadata(self.principal, ref.part_id)
            key = meta["key"]
            if meta["kind"] == "fact":
                self._fact_refs[scope, ref.part_id] = ref
                key = self._fact_key(scope, ref.part_id)
                policies[key] = ConsistencyPolicy("live")
            else:
                await self._authorize_resource(scope, key)
                mode = "live" if ref.policy == "validate_before_action" else ref.policy
                policies[key] = ConsistencyPolicy(mode, ref.max_age)
                if ref.policy == "pinned":
                    pinned_resources[key] = str(ref.revision)
            keys.append(key)
        # Explicit selected can narrow optional refs, never add unactivated resources or drop
        # mandatory.
        if selected is not None:
            keys = [k for k in keys if k in self.mandatory or k in selected]
        self._authorize_keys(scope, keys)
        snapshot = await super().next_context(
            scope, consistency, selected=list(dict.fromkeys(keys)), part_policies=policies, **kwargs
        )
        for key, revision in pinned_resources.items():
            if key in snapshot.revision_vector and snapshot.part(key).revision != revision:
                raise VersionConflict(
                    key, "pinned source revision unavailable; explicit replan required"
                )
        snapshot = replace(
            snapshot,
            segments=tuple(
                replace(segment, required=True)
                if segment.part_id in keys and segment.part_id not in self._declaration_keys
                else segment
                for segment in snapshot.segments
            ),
        )

        tokens = []
        try:
            for part in snapshot.parts:
                if self._is_managed_fact(part.part_id):
                    tokens.append(
                        self.store.lease(
                            self.principal, part.resource_id, int(part.revision), seconds=300
                        )
                    )
        except BaseException:
            for token in tokens:
                self.store.release(token)
            raise
        finally:
            for part in snapshot.parts:
                if self._is_managed_fact(part.part_id):
                    self.registry.instance(scope, part.part_id).snapshot = None
        if tokens:
            key = id(snapshot)
            self._snapshot_leases[key] = (snapshot.scope_id, tuple(tokens), time() + 300)
            weakref.finalize(snapshot, ContextManager._release_snapshot, weakref.ref(self), key)
        return snapshot

    @staticmethod
    def _release_snapshot(manager_ref, key):
        manager = manager_ref()
        if manager is not None:
            lease = manager._snapshot_leases.pop(key, None)
            with manager.store._lock:
                if lease and not manager.store.closed:
                    for token in lease[1]:
                        manager.store.release(token)

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
        snapshot=None,
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
        self.store._scope_access(self.principal, snapshot.scope_id)
        lease = self._snapshot_leases.get(id(snapshot))
        if lease and lease[2] < time():
            raise ConsistencyUnavailable(detail="snapshot lease expired; refresh before compiling")
        for part in snapshot.parts:
            if self._is_managed_fact(part.part_id):
                self.store.metadata(self.principal, part.resource_id)
            elif part.part_id not in self.mandatory:
                rows = self.store.db.execute(
                    "SELECT id FROM parts WHERE scope_id=? AND key=? AND schema_id=''",
                    (snapshot.scope_id, part.part_id),
                )
                row = rows.fetchone()
                if row is None:
                    raise AccessRevoked(part.part_id)
                self.store.metadata(self.principal, row[0])
        try:
            return super().compile(
                snapshot, target=target, history_tokens=history_tokens, evidence=evidence
            )
        except CompilerBudget as error:
            raise ContextBudgetExceeded(
                [p.part_id for p in snapshot.parts],
                Compiler(replace(self.compiler.policy, max_tokens=2**63 - 1))
                .compile(snapshot, history_tokens=history_tokens)
                .token_estimate,
                ["activate a narrower resource", "deactivate unrelated parts"],
            ) from error
        finally:
            # The returned immutable snapshot owns invocation payloads. Active refs stay
            # lightweight.
            for part in snapshot.parts:
                if self._is_managed_fact(part.part_id):
                    self.registry.instance(snapshot.scope_id, part.part_id).snapshot = None

    async def guard(self, snapshot):
        for part in snapshot.parts:
            if self._is_managed_fact(part.part_id):
                meta = self.store.metadata(self.principal, part.resource_id)
                if str(meta["revision"]) != part.revision:
                    raise ConsistencyUnavailable(part.part_id, "fact action precondition changed")
        self._authorize_keys(snapshot.scope_id, list(snapshot.revision_vector))
        current = await Context.next_context(
            self, snapshot.scope_id, "strict_fresh", selected=list(snapshot.revision_vector)
        )
        if current.revision_vector != snapshot.revision_vector:
            raise ConsistencyUnavailable(detail="action precondition changed")
        return current

    async def close_scope(self, scope="default"):
        try:
            await super().close_scope(scope)
        finally:
            for key, lease in list(self._snapshot_leases.items()):
                if lease[0] == scope:
                    self._release_snapshot(weakref.ref(self), key)
            self.store.cache.clear(scope)
            for pair in list(self._fact_refs):
                if pair[0] == scope:
                    del self._fact_refs[pair]
            self._prune_fact_definitions()
            if self.history is not False:
                self.history.close_scope(scope)
            if self.objective is not None:
                self.objective.close_scope(scope)
            if self.work_state is not None:
                self.work_state.close_scope(scope)

    async def close(self):
        try:
            await super().close()
        finally:
            for key in list(self._snapshot_leases):
                self._release_snapshot(weakref.ref(self), key)
