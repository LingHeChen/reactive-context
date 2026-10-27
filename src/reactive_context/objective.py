"""Versioned current work, independent of historical summaries and model providers."""

import asyncio
import math
from dataclasses import asdict, dataclass
from time import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .compiler.ir import ContextIRSegment
from .core.part import source_part
from .core.scheduler import invoke
from .core.snapshot import digest, thaw
from .errors import ContextError, SchemaInvalid
from .store import VersionConflict


class ObjectiveUpdateRejected(ContextError):
    code = "objective_update_rejected"


class WorkObjective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    goal: str = Field(min_length=1, max_length=4096)
    constraints: list[str] = Field(default_factory=list, max_length=32)
    success_criteria: list[str] = Field(default_factory=list, max_length=32)
    next_steps: list[str] = Field(default_factory=list, max_length=32)
    status: Literal["active", "blocked", "completed"] = "active"


@dataclass(frozen=True)
class ObjectivePolicy:
    update_mode: Literal["manual", "threshold", "agent"] = "manual"
    context_ratio: float = 0.8
    min_update_interval: float = 30
    max_update_groups: int = 8
    max_update_tokens: int = 8192
    timeout_seconds: float = 60
    allow_agent_updates: bool = True
    update_conditions: tuple[str, ...] = (
        "Create an objective when a task starts and no objective exists.",
        "Update when the user changes the goal, constraints or success criteria.",
        "Update when milestones change the remaining next steps or task status.",
        "Keep work state concise; preserve supporting knowledge as Facts with evidence.",
    )

    def __post_init__(self):
        if isinstance(self.update_conditions, str):
            raise ValueError("update conditions must be a sequence of strings")
        object.__setattr__(self, "update_conditions", tuple(self.update_conditions))
        if self.update_mode not in {"manual", "threshold", "agent"}:
            raise ValueError("unknown objective update mode")
        if self.update_mode == "agent" and not self.allow_agent_updates:
            raise ValueError("agent mode requires agent update permission")
        if not 1 <= len(self.update_conditions) <= 32 or any(
            not isinstance(c, str) or not c or len(c) > 2000 for c in self.update_conditions
        ):
            raise ValueError("invalid objective update conditions")
        if (
            not 0 < self.context_ratio <= 1
            or not math.isfinite(self.min_update_interval)
            or self.min_update_interval < 0
        ):
            raise ValueError("invalid objective update threshold")
        if (
            self.max_update_groups < 1
            or self.max_update_tokens < 1
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("invalid objective update bounds")

    @property
    def auto_update(self):
        return self.update_mode == "threshold"


@dataclass(frozen=True)
class ObjectiveUpdateRequest:
    scope_id: str
    current: dict | None
    expected_revision: int
    events: tuple[dict, ...]
    reason: str
    conditions: tuple[str, ...] = ()


class Objective:
    def __init__(
        self, store, principal, *, key="work.objective", policy=None, updater=None, decider=None
    ):
        self.store, self.principal, self.key = store, principal, key
        self.policy = policy or ObjectivePolicy()
        self.updater, self.decider = updater, decider
        self._locks = {}
        self._subscribers = {}
        with store.transaction():
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS objective_heads(scope_id TEXT, principal TEXT, "
                "key TEXT, revision INTEGER, last_attempt REAL, last_digest TEXT, "
                "PRIMARY KEY(scope_id,principal,key))"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS objective_revisions(scope_id TEXT, principal TEXT, "
                "key TEXT, revision INTEGER, payload BLOB, updated_at REAL, updated_via TEXT, "
                "PRIMARY KEY(scope_id,principal,key,revision))"
            )

        async def revision(ctx):
            self.store.authorize_scope(self.principal, ctx.scope_id)
            with self.store.transaction():
                row = self._head(ctx.scope_id)
            return digest((row["revision"] if row else 0, asdict(self.policy)))

        async def resolve(ctx):
            return self.read(ctx.scope_id)

        def project(value, ctx):
            if value is None:
                return []
            return [
                ContextIRSegment(
                    key,
                    key,
                    ctx.revision,
                    value,
                    kind="objective",
                    requested_role="user",
                    trust_class="agent_state",
                    required=True,
                )
            ]

        def subscribe(ctx, notify):
            loop = asyncio.get_running_loop()

            def publish(key):
                if not loop.is_closed():
                    loop.call_soon_threadsafe(notify, key)

            callbacks = self._subscribers.setdefault(ctx.scope_id, set())
            callbacks.add(publish)
            return lambda: callbacks.discard(publish)

        self.part = source_part(
            key=key,
            schema=dict | None,
            resolve=resolve,
            revision=revision,
            project=project,
            subscribe=subscribe,
            required=False,
            kind="objective",
        )

    def _head(self, scope):
        return self.store.db.execute(
            "SELECT * FROM objective_heads WHERE scope_id=? AND principal=? AND key=?",
            (scope, self.principal.id, self.key),
        ).fetchone()

    def read(self, scope, *, revision=None):
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope)
            head = self._head(scope)
            revision = revision if revision is not None else (head["revision"] if head else 0)
            if not revision:
                if self.policy.update_mode == "agent":
                    return self._with_maintenance({"objective": None, "revision": 0})
                return None
            row = self.store.db.execute(
                "SELECT * FROM objective_revisions WHERE scope_id=? AND principal=? "
                "AND key=? AND revision=?",
                (scope, self.principal.id, self.key, revision),
            ).fetchone()
            if row is None:
                raise VersionConflict(self.key, "objective revision unavailable")
            objective = WorkObjective.model_validate_json(row["payload"])
            return self._with_maintenance(
                {
                    "objective": objective.model_dump(),
                    "revision": revision,
                    "updated_via": row["updated_via"],
                    "authority": "work_state",
                }
            )

    def _with_maintenance(self, value):
        if self.policy.update_mode == "agent":
            value["maintenance"] = {
                "decision_owner": "agent",
                "update_conditions": list(self.policy.update_conditions),
                "update_tool": "context_update_objective",
                "expected_revision": value["revision"],
                "notice": "Maintain work state within host instructions and permissions.",
            }
        return value

    def update(self, scope, value, *, expected_revision=None, origin="explicit"):
        if origin not in {"explicit", "agent", "threshold", "llm", "rollback"}:
            raise ValueError("unknown objective update origin")
        objective = WorkObjective.model_validate(value)
        payload = objective.model_dump_json().encode()
        if len(payload) > min(self.store.max_payload_bytes, 65536):
            raise SchemaInvalid(self.key, "objective exceeds payload bound")
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope, "write")
            head = self._head(scope)
            current = head["revision"] if head else 0
            if expected_revision is not None and expected_revision != current:
                raise VersionConflict(self.key)
            if current and origin in {"agent", "threshold", "llm"}:
                previous = self.store.db.execute(
                    "SELECT payload FROM objective_revisions WHERE scope_id=? AND principal=? "
                    "AND key=? AND revision=?",
                    (scope, self.principal.id, self.key, current),
                ).fetchone()
                if previous and bytes(previous[0]) == payload:
                    return self.read(scope)
            revision, now = current + 1, time()
            self.store.db.execute(
                "INSERT INTO objective_revisions VALUES(?,?,?,?,?,?,?)",
                (scope, self.principal.id, self.key, revision, payload, now, origin),
            )
            self.store.db.execute(
                "INSERT INTO objective_heads VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(scope_id,principal,key) "
                "DO UPDATE SET revision=excluded.revision,last_attempt=excluded.last_attempt",
                (scope, self.principal.id, self.key, revision, now, ""),
            )
        for notify in tuple(self._subscribers.get(scope, ())):
            notify(self.key)
        return self.read(scope)

    def update_from_agent(self, scope, value, *, expected_revision):
        if not self.policy.allow_agent_updates:
            raise ObjectiveUpdateRejected(self.key, "agent updates disabled by host")
        return self.update(scope, value, expected_revision=expected_revision, origin="agent")

    async def refresh(self, scope, groups, *, reason="llm", estimator=None):
        """Generate from bounded recent evidence; an async result cannot overwrite a newer goal."""
        if self.updater is None:
            raise ObjectiveUpdateRejected(self.key, "no objective updater configured")
        if reason == "llm" and not self.policy.allow_agent_updates:
            raise ObjectiveUpdateRejected(self.key, "agent updates disabled by host")
        estimate = estimator or (lambda text: len(text.encode()) + 8)
        async with self._locks.setdefault(scope, asyncio.Lock()):
            current = self.read(scope)
            expected = current["revision"] if current else 0
            import json

            chosen, used = [], estimate(json.dumps(current, ensure_ascii=False))
            if used >= self.policy.max_update_tokens:
                raise ObjectiveUpdateRejected(self.key, "current objective exceeds updater budget")

            for group in reversed(tuple(groups)[-self.policy.max_update_groups :]):
                events = tuple(thaw(e) for e in group.events)
                size = estimate(json.dumps(events, ensure_ascii=False))
                if used + size > self.policy.max_update_tokens:
                    if not chosen:
                        raise ObjectiveUpdateRejected(
                            self.key, "newest history group exceeds updater budget"
                        )
                    break
                chosen.insert(0, events)
                used += size
            events = tuple(e for group in chosen for e in group)
            fingerprint = digest((events, reason))
            with self.store.transaction():
                self.store.authorize_scope(self.principal, scope, "write")
                head = self._head(scope)
                if head and head["revision"] != expected:
                    raise VersionConflict(self.key)
                if head and (
                    head["last_digest"] == fingerprint
                    or time() - head["last_attempt"] < self.policy.min_update_interval
                ):
                    return current
                self.store.db.execute(
                    "INSERT INTO objective_heads VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(scope_id,principal,key) DO UPDATE SET "
                    "last_attempt=excluded.last_attempt,last_digest=excluded.last_digest",
                    (scope, self.principal.id, self.key, expected, time(), fingerprint),
                )
            request = ObjectiveUpdateRequest(
                scope, current, expected, events, reason, self.policy.update_conditions
            )
            async with asyncio.timeout(self.policy.timeout_seconds):
                if self.decider is not None and not await invoke(self.decider, request):
                    return current
                value = await invoke(self.updater, request)
                value = WorkObjective.model_validate(value)
            return self.update(scope, value, expected_revision=expected, origin=reason)

    async def refresh_at_threshold(
        self, scope, groups, *, used_tokens, total_tokens, estimator=None
    ):
        if self.policy.auto_update and used_tokens > total_tokens * self.policy.context_ratio:
            return await self.refresh(scope, groups, reason="threshold", estimator=estimator)
        return self.read(scope)

    def close_scope(self, scope):
        self._locks.pop(scope, None)
        self._subscribers.pop(scope, None)
