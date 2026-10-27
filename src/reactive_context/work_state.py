"""Structured current work with exact, host-authorized changes and durable revisions.

The library checks structure and retention and invokes host-defined value checks.
It does not interpret natural language. Only a host/domain adapter can authorize
a changed field; citing an arbitrary message cannot grant an agent permission.
"""

import asyncio
import inspect
import json
import re
from collections import Counter
from dataclasses import dataclass
from time import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, model_validator

from .compiler.ir import ContextIRSegment
from .core.part import source_part
from .core.snapshot import digest, freeze, thaw
from .errors import AccessRevoked, ContextError, SchemaInvalid
from .state_validation import StateIssue, StateValidation, StateValidationRequest
from .store import StoreCorruption, VersionConflict

Action = Literal["add", "replace", "remove", "clear"]


def _tokens(path):
    if not isinstance(path, str) or not path.startswith("/") or len(path) > 2048:
        raise ValueError("a non-root JSON Pointer is required")
    if re.search(r"~(?:[^01]|$)", path):
        raise ValueError("invalid JSON Pointer escape")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
    if len(parts) > 32:
        raise ValueError("state path exceeds depth limit")
    return parts


def _pointer(parent, key):
    return parent + "/" + key.replace("~", "~0").replace("/", "~1")


class StateOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    op: Literal["set", "remove"]
    path: str
    value: JsonValue = None

    @model_validator(mode="after")
    def valid_operation(self):
        _tokens(self.path)
        if self.op == "set" and "value" not in self.model_fields_set:
            raise ValueError("set requires an explicit value")
        if self.op == "remove" and "value" in self.model_fields_set:
            raise ValueError("remove cannot carry a value")
        return self


class StateGrant(BaseModel):
    """Exact leaf path and action, supplied by trusted host code only."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    action: Action

    @model_validator(mode="after")
    def valid_path(self):
        _tokens(self.path)
        return self


@dataclass(frozen=True)
class StateChange:
    path: str
    action: Action
    before: Any = None
    after: Any = None


class StateUpdateRejected(ContextError):
    code = "state_update_rejected"

    def __init__(self, reasons, changes=(), validation=None):
        self.reasons, self.changes = tuple(reasons), tuple(changes)
        self.validation = validation
        super().__init__(detail="structured state change was not authorized")


def state_diff(before, after, *, empty_values=(None, "", [], {})):
    """Diff objects by stable keys. Lists are atomic; use keyed objects for entities.

    Removing/replacing an object exposes every removed descendant. Permission to
    replace a parent never implicitly permits clearing or removing its children.
    """
    changes = []
    empty = {digest(value) for value in empty_values}

    def visit(old, new, path, old_exists=True, new_exists=True):
        if old_exists and new_exists and digest(old) == digest(new):
            return
        if isinstance(old, dict) and isinstance(new, dict) and old_exists and new_exists:
            for key in sorted(old.keys() | new.keys()):
                visit(old.get(key), new.get(key), _pointer(path, key), key in old, key in new)
        elif not new_exists:
            if isinstance(old, dict) and old:
                for key, value in sorted(old.items()):
                    visit(value, None, _pointer(path, key), True, False)
            else:
                changes.append(StateChange(path, "remove", freeze(old)))
        elif not old_exists:
            if isinstance(new, dict) and new:
                for key, value in sorted(new.items()):
                    visit(None, value, _pointer(path, key), False, True)
            else:
                changes.append(StateChange(path, "add", after=freeze(new)))
        elif not isinstance(old, dict) and digest(new) in empty and digest(old) not in empty:
            changes.append(StateChange(path, "clear", freeze(old), freeze(new)))
        elif isinstance(old, dict) != isinstance(new, dict):
            # Structural replacement includes removal of all old leaves, even if
            # the caller submitted just one set operation at the parent path.
            visit(old, None, path, True, False)
            visit(None, new, path, False, True)
        else:
            action = "clear" if digest(new) in empty and digest(old) not in empty else "replace"
            if isinstance(old, list) and isinstance(new, list) and new:
                # Reordering is just replacement. Losing even one repeated item
                # also needs deletion permission; list replacement is no loophole.
                if Counter(map(digest, old)) - Counter(map(digest, new)):
                    changes.append(StateChange(path, "remove", freeze(old), freeze(new)))
            changes.append(StateChange(path, action, freeze(old), freeze(new)))

    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("work state must be a JSON object")
    visit(before, after, "")
    return tuple(changes)


def apply_state_operations(value, operations):
    """Apply an explicit patch to a detached copy; omission always means retain."""
    operations = [StateOperation.model_validate(op) for op in operations]
    if not 1 <= len(operations) <= 256:
        raise ValueError("provide between 1 and 256 state operations")
    paths = [_tokens(op.path) for op in operations]
    for i, path in enumerate(paths):
        if any(path[: len(other)] == other or other[: len(path)] == path for other in paths[:i]):
            raise ValueError("overlapping state operations are ambiguous")
    candidate = json.loads(json.dumps(value, allow_nan=False))
    for op, tokens in zip(operations, paths, strict=True):
        parent = candidate
        for token in tokens[:-1]:
            if token not in parent:
                if op.op == "remove":
                    raise ValueError("remove target does not exist")
                parent[token] = {}
            parent = parent[token]
            if not isinstance(parent, dict):
                raise ValueError("state paths cannot traverse scalars or lists")
        key = tokens[-1]
        if op.op == "remove":
            if key not in parent:
                raise ValueError("remove target does not exist")
            del parent[key]
        else:
            parent[key] = json.loads(json.dumps(op.value, allow_nan=False))
    return candidate


def state_value_at(value, path):
    """Read an object field by the same stable JSON Pointer used in state patches."""
    from collections.abc import Mapping

    for token in _tokens(path):
        if not isinstance(value, Mapping) or token not in value:
            raise KeyError(path)
        value = value[token]
    return value


class WorkState:
    """Scope-owned state; schema/policy are host-defined and immutable for a key.

    Permits bind exact changes to a principal, base revision and scoped evidence.
    All writes recheck under the store transaction. Old permits cannot authorize
    a later revision. The separate state tables cannot be modified by Fact tools.
    """

    def __init__(
        self,
        store,
        principal,
        *,
        key="work.state",
        schema: Any = dict[str, JsonValue],
        empty_values=(None, "", [], {}),
        check=None,
        check_id=None,
    ):
        self.store, self.principal, self.key = store, principal, key
        self.adapter = TypeAdapter(schema)
        self.empty_values = tuple(json.loads(json.dumps(empty_values, allow_nan=False)))
        self._subscribers = {}
        self.check, self.check_id = check, check_id
        self._candidates = None
        if (check is None) != (check_id is None):
            raise ValueError("state checker and stable check_id must be supplied together")
        if check is not None and (
            not callable(check)
            or inspect.iscoroutinefunction(check)
            or inspect.iscoroutinefunction(type(check).__call__)
        ):
            raise ValueError("state checks must be synchronous host callbacks")
        if check_id is not None and (
            not isinstance(check_id, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", check_id)
        ):
            raise ValueError("invalid state checker id")
        fingerprint = digest((self.adapter.json_schema(), self.empty_values))
        with store.transaction():
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_specs(key TEXT PRIMARY KEY,fingerprint TEXT)"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_revisions("
                "scope TEXT,key TEXT,revision INTEGER,payload BLOB,digest TEXT,"
                "changes TEXT,evidence TEXT,created_at REAL,PRIMARY KEY(scope,key,revision))"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_permits(id TEXT PRIMARY KEY,"
                "scope TEXT,key TEXT,principal TEXT,base_revision INTEGER,grants TEXT,"
                "evidence TEXT,active INTEGER NOT NULL DEFAULT 1)"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_audit(seq INTEGER PRIMARY KEY,"
                "scope TEXT,key_hash TEXT,principal TEXT,event TEXT,revision INTEGER,"
                "metadata TEXT,created_at REAL)"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_checks(key TEXT PRIMARY KEY,check_id TEXT)"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_requirements(scope TEXT,key TEXT,"
                "revision INTEGER,payload BLOB,digest TEXT,evidence TEXT,"
                "PRIMARY KEY(scope,key,revision))"
            )
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_deltas(scope TEXT,key TEXT,"
                "revision INTEGER,payload BLOB,digest TEXT,PRIMARY KEY(scope,key,revision))"
            )
            required = store.db.execute(
                "SELECT check_id FROM work_state_checks WHERE key=?", (key,)
            ).fetchone()
            if check_id is not None:
                if required and required[0] != check_id:
                    raise ValueError("state checker identity changed; use a new key and migrate")
                store.db.execute(
                    "INSERT OR IGNORE INTO work_state_checks VALUES(?,?)", (key, check_id)
                )
            previous = store.db.execute(
                "SELECT fingerprint FROM work_state_specs WHERE key=?", (key,)
            ).fetchone()
            if previous and previous[0] != fingerprint:
                raise ValueError(
                    "state schema/policy changed; use a new key and explicit migration"
                )
            store.db.execute(
                "INSERT OR IGNORE INTO work_state_specs VALUES(?,?)", (key, fingerprint)
            )

        async def revision(ctx):
            return digest(self.read(ctx.scope_id))

        async def resolve(ctx):
            return self.read(ctx.scope_id)

        def project(value, ctx):
            return [
                ContextIRSegment(
                    key,
                    key,
                    ctx.revision,
                    value,
                    kind="work_state",
                    requested_role="user",
                    trust_class="agent_state",
                    required=True,
                )
            ]

        def subscribe(ctx, notify):
            loop = asyncio.get_running_loop()

            def publish():
                if not loop.is_closed():
                    loop.call_soon_threadsafe(notify, key)

            listeners = self._subscribers.setdefault(ctx.scope_id, set())
            listeners.add(publish)
            return lambda: listeners.discard(publish)

        self.part = source_part(
            key=key,
            schema=dict,
            resolve=resolve,
            revision=revision,
            project=project,
            subscribe=subscribe,
            kind="work_state",
        )

    def _notify(self, scope):
        for notify in tuple(self._subscribers.get(scope, ())):
            notify()

    def close_scope(self, scope):
        self._subscribers.pop(scope, None)

    @property
    def candidates(self):
        from .state_candidates import StateCandidates

        if self._candidates is None:
            self._candidates = StateCandidates(self)
        return self._candidates

    def requirements(self, scope):
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope)
            row = self.store.db.execute(
                "SELECT * FROM work_state_requirements WHERE scope=? AND key=? "
                "ORDER BY revision DESC LIMIT 1",
                (scope, self.key),
            ).fetchone()
            value = self._payload(row) if row else {}
            return {
                "revision": row["revision"] if row else 0,
                "value": value,
                "evidence": json.loads(row["evidence"]) if row else [],
            }

    def set_requirements(self, scope, value, evidence, *, expected_revision, host):
        """Bind checked current requirements; invalidate earlier field permits atomically."""
        if not host.admin:
            raise AccessRevoked(detail="requirements are host-bound")
        raw = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        if (
            not isinstance(raw, dict)
            or len(json.dumps(raw).encode()) > self.store.max_payload_bytes
        ):
            raise SchemaInvalid(detail="invalid state requirements")
        with self.store.transaction():
            self.store.authorize_scope(host, scope, "write")
            self.store.authorize_scope(self.principal, scope, "write")
            self._evidence(scope, evidence)
            current = self.requirements(scope)
            if current["revision"] != expected_revision:
                raise VersionConflict(self.key)
            if (
                current["revision"]
                and digest(current["value"]) == digest(raw)
                and (set(current["evidence"]) == set(evidence))
            ):
                return current
            self.store.db.execute(
                "INSERT INTO work_state_requirements VALUES(?,?,?,?,?,?)",
                (
                    scope,
                    self.key,
                    expected_revision + 1,
                    json.dumps(raw).encode(),
                    digest(raw),
                    json.dumps(list(evidence)),
                ),
            )
            self.store.db.execute(
                "UPDATE work_state_permits SET active=0 WHERE scope=? AND key=?",
                (scope, self.key),
            )
            self._audit(scope, "requirements_updated", expected_revision + 1)
        self._notify(scope)
        return self.requirements(scope)

    def _check_values(self, scope, candidate, changes, evidence, revision):
        required = self.store.db.execute(
            "SELECT check_id FROM work_state_checks WHERE key=?", (self.key,)
        ).fetchone()
        if required is None:
            if self.requirements(scope)["revision"]:
                return {
                    **StateValidation(
                        "pending", (StateIssue("", "requirements_have_no_checker"),)
                    ).as_dict(),
                    "check_id": None,
                }
            return {"status": "structural_only", "issues": [], "check_id": None}
        if self.check is None or self.check_id != required[0]:
            raise SchemaInvalid(detail="restore required state value checker before writing")
        requirements = self.requirements(scope)
        request = StateValidationRequest(
            scope,
            self.key,
            revision,
            requirements["revision"],
            freeze(self._payload(self._head(scope))),
            freeze(candidate),
            tuple(changes),
            tuple(evidence),
            freeze(requirements["value"]),
        )
        result = self.check(request)
        if not isinstance(result, StateValidation):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("state checker must return StateValidation synchronously")
        return {
            **result.as_dict(),
            "check_id": required[0],
            "requirements_revision": requirements["revision"],
        }

    def deltas(self, scope, *, after_revision=0, limit=50):
        """Read committed structured memory increments; never includes rejected candidates."""
        if after_revision < 0 or not 1 <= limit <= 200:
            raise ValueError("invalid delta pagination")
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope)
            rows = self.store.db.execute(
                "SELECT * FROM work_state_deltas WHERE scope=? AND key=? AND revision>? "
                "ORDER BY revision LIMIT ?",
                (scope, self.key, after_revision, limit),
            ).fetchall()
            return [self._payload(row) for row in rows]

    def _audit(self, scope, event, revision, **metadata):
        self.store.db.execute(
            "INSERT INTO work_state_audit VALUES(NULL,?,?,?,?,?,?,?)",
            (
                scope,
                digest(self.key),
                self.principal.id,
                event,
                revision,
                json.dumps(metadata),
                time(),
            ),
        )

    def _head(self, scope):
        return self.store.db.execute(
            "SELECT * FROM work_state_revisions WHERE scope=? AND key=? "
            "ORDER BY revision DESC LIMIT 1",
            (scope, self.key),
        ).fetchone()

    def _payload(self, row):
        if row is None:
            return {}
        value = json.loads(row["payload"])
        if digest(value) != row["digest"]:
            raise StoreCorruption()
        return value

    def _validate(self, value):
        try:
            # Normalization must not introduce schema defaults or drop fields.
            raw = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
            parsed = self.adapter.validate_python(raw)
            result = json.loads(self.adapter.dump_json(parsed))
            if not isinstance(result, dict) or digest(raw) != digest(result):
                raise ValueError("state schema must preserve explicit JSON fields and types")
            if len(json.dumps(result).encode()) > self.store.max_payload_bytes:
                raise ValueError("state payload too large")
            return result
        except (ValueError, TypeError):
            raise SchemaInvalid(detail="work state schema validation failed") from None

    def _evidence(self, scope, evidence):
        if not evidence or len(evidence) > 128:
            raise SchemaInvalid(detail="scoped evidence required")
        for identity in evidence:
            if not self.store.db.execute(
                "SELECT 1 FROM evidence WHERE id=? AND scope_id=?", (identity, scope)
            ).fetchone():
                raise AccessRevoked(detail="state evidence is not visible in this scope")

    def read(self, scope):
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope)
            head = self._head(scope)
            revision = head["revision"] if head else 0
            permits = self.store.db.execute(
                "SELECT id,grants,evidence FROM work_state_permits WHERE scope=? AND key=? "
                "AND principal=? AND base_revision=? AND active=1",
                (scope, self.key, self.principal.id, revision),
            ).fetchall()
            required = self.store.db.execute(
                "SELECT check_id FROM work_state_checks WHERE key=?", (self.key,)
            ).fetchone()
            return {
                "revision": revision,
                "value": self._payload(head),
                "requirements": self.requirements(scope),
                "permits": [
                    {
                        "permit_id": p["id"],
                        "grants": json.loads(p["grants"]),
                        "evidence": json.loads(p["evidence"]),
                    }
                    for p in permits
                ],
                "maintenance": {
                    "update_tool": "context_update_work_state",
                    "preview_tool": "context_preview_work_state",
                    "rule": "Patch only permitted fields; omitted fields stay valid. "
                    "Removal and clearing need separate host grants.",
                    "value_checker": required[0] if required else None,
                    "candidate_tool": "context_propose_state_candidate",
                    "commit_candidate_tool": "context_commit_state_candidate",
                },
            }

    def initialize(self, scope, value, evidence, *, host):
        if not host.admin:
            raise AccessRevoked(detail="state initialization is host-only")
        value = self._validate(value)
        with self.store.transaction():
            self.store.authorize_scope(host, scope, "write")
            self.store.authorize_scope(self.principal, scope)
            self.store.authorize_scope(self.principal, scope, "write")
            self._evidence(scope, evidence)
            if self._head(scope):
                raise VersionConflict(self.key)
            self._write(scope, value, (), evidence, 1)
        self._notify(scope)
        return self.read(scope)

    def authorize(self, scope, grants, evidence, *, expected_revision, host):
        """Host-only grant. Deciding which requirements justify it belongs to the adapter."""
        if not host.admin:
            raise AccessRevoked(detail="state change grants are host-only")
        grants = [StateGrant.model_validate(grant) for grant in grants]
        if not 1 <= len(grants) <= 256:
            raise ValueError("provide between 1 and 256 field grants")
        with self.store.transaction():
            self.store.authorize_scope(host, scope, "write")
            self.store.authorize_scope(self.principal, scope)
            self.store.authorize_scope(self.principal, scope, "write")
            head = self._head(scope)
            if head is None or expected_revision != head["revision"]:
                raise VersionConflict(self.key)
            self._evidence(scope, evidence)
            identity = uuid4().hex
            self.store.db.execute(
                "INSERT INTO work_state_permits VALUES(?,?,?,?,?,?,?,1)",
                (
                    identity,
                    scope,
                    self.key,
                    self.principal.id,
                    expected_revision,
                    json.dumps([g.model_dump() for g in grants]),
                    json.dumps(list(evidence)),
                ),
            )
            self._audit(scope, "authorized", expected_revision, grant_count=len(grants))
        self._notify(scope)
        return identity

    def revoke_permit(self, scope, permit_id, *, host):
        if not host.admin:
            raise AccessRevoked()
        with self.store.transaction():
            self.store.authorize_scope(host, scope, "write")
            self.store.db.execute(
                "UPDATE work_state_permits SET active=0 WHERE scope=? AND key=? AND id=?",
                (scope, self.key, permit_id),
            )
            head = self._head(scope)
            self._audit(scope, "permit_revoked", head["revision"] if head else 0)
        self._notify(scope)

    def _prepare(self, scope, candidate, expected_revision, permit_id):
        self.store.authorize_scope(self.principal, scope)
        self.store.authorize_scope(self.principal, scope, "write")
        head = self._head(scope)
        if head is None or head["revision"] != expected_revision:
            raise VersionConflict(self.key)
        candidate = self._validate(candidate)
        changes = state_diff(self._payload(head), candidate, empty_values=self.empty_values)
        if not changes:
            return candidate, changes, (), ()
        permit = self.store.db.execute(
            "SELECT * FROM work_state_permits WHERE id=? AND scope=? AND key=? AND principal=? "
            "AND base_revision=? AND active=1",
            (permit_id, scope, self.key, self.principal.id, expected_revision),
        ).fetchone()
        if permit is None:
            return candidate, changes, ("missing_or_stale_permit",), ()
        evidence = tuple(json.loads(permit["evidence"]))
        self._evidence(scope, evidence)
        grants = {(g["path"], g["action"]) for g in json.loads(permit["grants"])}
        blocked = tuple(change for change in changes if (change.path, change.action) not in grants)
        return (
            candidate,
            changes,
            tuple(sorted({f"unpermitted_{c.action}" for c in blocked})),
            evidence,
        )

    def _write(self, scope, value, changes, evidence, revision, validation=None):
        summary = [{"path_hash": digest(c.path), "action": c.action} for c in changes]
        self.store.db.execute(
            "INSERT INTO work_state_revisions VALUES(?,?,?,?,?,?,?,?)",
            (
                scope,
                self.key,
                revision,
                json.dumps(value, ensure_ascii=False).encode(),
                digest(value),
                json.dumps(summary),
                json.dumps(list(evidence)),
                time(),
            ),
        )
        self._audit(scope, "committed", revision, changes=summary)
        if changes:
            delta = {
                "state_key": self.key,
                "base_revision": revision - 1,
                "revision": revision,
                "requirements_revision": self.requirements(scope)["revision"],
                "validation": validation or {"status": "structural_only"},
                "evidence": list(evidence),
                "changes": [
                    {
                        "path": c.path,
                        "action": c.action,
                        "before": thaw(c.before),
                        "after": thaw(c.after),
                    }
                    for c in changes
                ],
            }
            self.store.db.execute(
                "INSERT INTO work_state_deltas VALUES(?,?,?,?,?)",
                (
                    scope,
                    self.key,
                    revision,
                    json.dumps(delta, ensure_ascii=False).encode(),
                    digest(delta),
                ),
            )

    def replace(
        self, scope, value, *, expected_revision, permit_id=None, preview=False, evidence=()
    ):
        """Check an entire candidate too; missing keys are explicit removal differences."""
        with self.store.transaction():
            candidate, changes, reasons, permit_evidence = self._prepare(
                scope, value, expected_revision, permit_id
            )
            evidence = tuple(dict.fromkeys((*permit_evidence, *evidence)))
            if evidence:
                self._evidence(scope, evidence)
            validation = (
                {"status": "not_run", "issues": []}
                if reasons
                else self._check_values(scope, candidate, changes, evidence, expected_revision)
            )
            if validation["status"] in {"failed", "pending"}:
                reasons = (*reasons, "value_validation_" + validation["status"])
            receipt = {
                "decision": "reject" if reasons else "ready" if changes else "unchanged",
                "revision": expected_revision,
                "reasons": list(reasons),
                "validation": validation,
                "changes": [
                    {
                        "path": c.path,
                        "action": c.action,
                        "before": thaw(c.before),
                        "after": thaw(c.after),
                    }
                    for c in changes
                ],
            }
            if preview:
                return receipt
            if reasons:
                self._audit(
                    scope, "rejected", expected_revision, reasons=reasons, change_count=len(changes)
                )
            elif changes:
                self._write(scope, candidate, changes, evidence, expected_revision + 1, validation)
                receipt.update(decision="updated", revision=expected_revision + 1)
        if reasons:
            raise StateUpdateRejected(reasons, changes, validation)
        if changes:
            self._notify(scope)
        return receipt

    def patch(
        self, scope, operations, *, expected_revision, permit_id=None, preview=False, evidence=()
    ):
        # Candidate creation and CAS verification share one transaction, even when
        # two WorkState objects or processes update this store concurrently.
        rejected = None
        result = None
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope)
            self.store.authorize_scope(self.principal, scope, "write")
            value = self._payload(self._head(scope))
            candidate = apply_state_operations(value, operations)
            try:
                result = self.replace(
                    scope,
                    candidate,
                    expected_revision=expected_revision,
                    permit_id=permit_id,
                    preview=preview,
                    evidence=evidence,
                )
            except StateUpdateRejected as error:
                rejected = error
        if rejected is not None:
            raise rejected
        assert result is not None
        return result
