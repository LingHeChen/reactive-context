"""SQLite WAL metadata, scoped CAS, validated facts and persistent lightweight active refs.

The store does not execute agent tools, infer semantic identity, or grant instruction authority.
Principal objects and schema registration are host capabilities, never model-supplied arguments.
"""

import inspect
import json
import os
import re
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from .cache import WeightedLRU
from .core.snapshot import digest as value_digest
from .core.snapshot import freeze
from .errors import AccessRevoked, ContextError, SchemaInvalid
from .fact_checks import FactAdmissionRejected, FactCheckRequest, FactCheckResult, FactVerdict


class VersionConflict(ContextError):
    code = "version_conflict"


class StoreCorruption(ContextError):
    code = "store_corruption"


@dataclass(frozen=True)
class Principal:
    id: str
    admin: bool = False


@dataclass(frozen=True)
class FactRecord:
    id: str
    scope_id: str
    key: str
    kind: str
    schema_id: str
    schema_version: int
    revision: int
    status: str
    verification: str
    value: Any
    evidence: tuple[str, ...]
    digest: str

    @property
    def lifecycle(self):
        if self.status != "active":
            return self.status
        return "confirmed" if self.verification == "confirmed" else "candidate"


@dataclass(frozen=True)
class ActiveRef:
    scope_id: str
    part_id: str
    policy: str = "live"
    revision: int | None = None
    max_age: float = 0


DDL = """
CREATE TABLE IF NOT EXISTS scopes(id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent_id TEXT);
CREATE TABLE IF NOT EXISTS scope_acl(scope_id TEXT, principal TEXT, action TEXT,
 PRIMARY KEY(scope_id,principal,action));
CREATE TABLE IF NOT EXISTS parts(id TEXT PRIMARY KEY, scope_id TEXT NOT NULL, key TEXT NOT NULL,
 kind TEXT NOT NULL, schema_id TEXT NOT NULL, schema_version INTEGER NOT NULL,
 revision INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active',
 verification TEXT NOT NULL DEFAULT 'unverified', metadata TEXT NOT NULL DEFAULT '{}',
 UNIQUE(scope_id,schema_id,key));
CREATE TABLE IF NOT EXISTS part_acl(part_id TEXT, principal TEXT, action TEXT,
 PRIMARY KEY(part_id,principal,action));
CREATE TABLE IF NOT EXISTS revisions(part_id TEXT, rev INTEGER, digest TEXT NOT NULL,
 inline BLOB, object_path TEXT, observed_at REAL NOT NULL, source_ref TEXT,
 schema_version INTEGER NOT NULL,
 PRIMARY KEY(part_id,rev));
CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, scope_id TEXT NOT NULL,
 source_ref TEXT NOT NULL, revision TEXT NOT NULL, observed_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_support(id INTEGER PRIMARY KEY AUTOINCREMENT,
 evidence_id TEXT NOT NULL, span_start INTEGER, span_end INTEGER, content_hash TEXT NOT NULL,
 principal_scope TEXT NOT NULL, support_status TEXT NOT NULL, rule_ref TEXT,
 observed_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS fact_confirmations(part_id TEXT, rev INTEGER, confirmation_ref TEXT,
 principal TEXT, PRIMARY KEY(part_id,rev));
CREATE TABLE IF NOT EXISTS fact_evidence(part_id TEXT, rev INTEGER, evidence_id TEXT,
 PRIMARY KEY(part_id,rev,evidence_id));
CREATE TABLE IF NOT EXISTS schema_versions(id TEXT, version INTEGER, fingerprint TEXT,
 PRIMARY KEY(id,version));
CREATE TABLE IF NOT EXISTS schema_fact_checks(id TEXT,version INTEGER,PRIMARY KEY(id,version));
CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY, scope_id TEXT, schema_id TEXT,
 schema_version INTEGER, key TEXT, payload BLOB, evidence TEXT, author TEXT,
 expected_revision INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS active_refs(scope_id TEXT, principal TEXT, part_id TEXT,
 policy TEXT, revision INTEGER, max_age REAL, PRIMARY KEY(scope_id,principal,part_id));
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
 type TEXT NOT NULL, scope_id TEXT NOT NULL, part_id TEXT, payload TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS relations(from_id TEXT, to_id TEXT, kind TEXT, source_revision TEXT,
 PRIMARY KEY(from_id,to_id,kind));
CREATE TABLE IF NOT EXISTS leases(token TEXT PRIMARY KEY, part_id TEXT, rev INTEGER, expires REAL);
CREATE TABLE IF NOT EXISTS history_events(scope_id TEXT, event_id TEXT, seq INTEGER,
 payload BLOB, protected INTEGER NOT NULL DEFAULT 0, digest TEXT, object_path TEXT,
 PRIMARY KEY(scope_id,event_id), UNIQUE(scope_id,seq));
CREATE TABLE IF NOT EXISTS history_search_entries(scope_id TEXT, kind TEXT, record_id TEXT,
 PRIMARY KEY(scope_id,kind,record_id));
CREATE VIRTUAL TABLE IF NOT EXISTS history_search USING fts5(body);
CREATE TABLE IF NOT EXISTS history_refs(scope_id TEXT, event_id TEXT, part_id TEXT, rev INTEGER,
 PRIMARY KEY(scope_id,event_id,part_id));
CREATE TABLE IF NOT EXISTS compaction_records(id TEXT PRIMARY KEY, scope_id TEXT,
 first_seq INTEGER, last_seq INTEGER, source_ids TEXT, checksum TEXT, payload BLOB,
 model TEXT, validation TEXT, status TEXT, created_at REAL, usage TEXT);
CREATE TABLE IF NOT EXISTS compaction_refs(compaction_id TEXT, part_id TEXT, rev TEXT,
 PRIMARY KEY(compaction_id,part_id));
CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(part_id UNINDEXED, body);
CREATE INDEX IF NOT EXISTS parts_scope ON parts(scope_id,status,kind);
CREATE INDEX IF NOT EXISTS events_scope ON events(scope_id,seq);
"""


class ContextStore:
    def __init__(
        self,
        directory,
        *,
        cache_bytes=4 * 1024 * 1024,
        scope_cache_bytes=None,
        inline_bytes=4096,
        max_payload_bytes=8 * 1024 * 1024,
    ):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.objects = self.directory / "objects/sha256"
        self.objects.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(
            self.directory / "index.sqlite",
            timeout=10,
            isolation_level=None,
            check_same_thread=False,
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("PRAGMA secure_delete=ON")
        self._savepoint = 0
        self.db.executescript(DDL)
        self.cache = WeightedLRU(cache_bytes, scope_bytes=scope_cache_bytes)
        self.inline_bytes, self.max_payload_bytes = inline_bytes, max_payload_bytes
        self.schemas: dict[tuple[str, int], TypeAdapter] = {}
        self.fact_checks = {}
        self._seen_seq = self.event_sequence
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            if not self.closed:
                self.cache.clear()
                self.db.close()
                self.closed = True

    @property
    def event_sequence(self):
        return self.db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]

    @contextmanager
    def transaction(self):
        with self._lock:
            nested = self.db.in_transaction
            self._savepoint += 1
            name = f"context_savepoint_{self._savepoint}"
            self.db.execute(f"SAVEPOINT {name}" if nested else "BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute(f"RELEASE SAVEPOINT {name}" if nested else "COMMIT")
            except BaseException:
                if nested:
                    self.db.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    self.db.execute(f"RELEASE SAVEPOINT {name}")
                else:
                    self.db.execute("ROLLBACK")
                self.cache.clear()
                raise

    def batch(self):
        """Host-owned atomic bulk ingestion; inner operations use SQL savepoints."""
        return self.transaction()

    def _purge_bridge(self, scope, part_id=None, principal=None):
        exists = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='bridge_authorizations'"
        ).fetchone()
        if not exists:
            return
        if part_id:
            self.db.execute(
                "UPDATE bridge_requests SET response=NULL,state='revoked' WHERE scope=? "
                "AND key IN (SELECT key FROM bridge_authorizations "
                "WHERE scope=? AND part_id=?)",
                (scope, scope, part_id),
            )
        elif principal:
            self.db.execute(
                "UPDATE bridge_requests SET response=NULL,state='revoked' "
                "WHERE scope=? AND principal=?",
                (scope, principal),
            )

    def _emit(self, kind, scope, part_id="", **metadata):
        # Intentionally no user payloads, keys, query strings or credentials in audit events.
        self.db.execute(
            "INSERT INTO events(type,scope_id,part_id,payload,created_at) VALUES(?,?,?,?,?)",
            (kind, scope, part_id, json.dumps(metadata), time()),
        )
        self.cache.clear(scope)

    def _reconcile(self):
        seq = self.event_sequence
        if seq != self._seen_seq:
            self.cache.clear()
            self._seen_seq = seq

    def register_schema(self, schema_id, schema, *, version=1, check=None):
        if not schema_id or version < 1:
            raise ValueError("schema id and positive version required")
        if check is not None and (
            not callable(check)
            or inspect.iscoroutinefunction(check)
            or inspect.iscoroutinefunction(type(check).__call__)
        ):
            raise ValueError("Fact checks must be synchronous host callbacks")
        adapter = TypeAdapter(schema)
        old = self.schemas.get((schema_id, version))
        if old and old.json_schema() != adapter.json_schema():
            raise ValueError("schema version is immutable; register a new version and migrate")
        fingerprint = sha256(json.dumps(adapter.json_schema(), sort_keys=True).encode()).hexdigest()
        with self.transaction():
            saved = self.db.execute(
                "SELECT fingerprint FROM schema_versions WHERE id=? AND version=?",
                (schema_id, version),
            ).fetchone()
            if saved and saved[0] != fingerprint:
                raise ValueError("persisted schema version differs; explicit migration required")
            self.db.execute(
                "INSERT OR IGNORE INTO schema_versions VALUES(?,?,?)",
                (schema_id, version, fingerprint),
            )
            if check is not None:
                self.db.execute(
                    "INSERT OR IGNORE INTO schema_fact_checks VALUES(?,?)", (schema_id, version)
                )
        self.schemas[schema_id, version] = adapter
        if check is not None:
            self.fact_checks[schema_id, version] = check

    def _scope_access(self, actor, scope, action="read"):
        if not self.db.execute("SELECT 1 FROM scopes WHERE id=?", (scope,)).fetchone():
            raise AccessRevoked(detail="scope unavailable")
        if actor.admin:
            return
        if not self.db.execute(
            "SELECT 1 FROM scope_acl WHERE scope_id=? AND principal=? AND action=?",
            (scope, actor.id, action),
        ).fetchone():
            raise AccessRevoked(detail="scope access denied")

    def _part_access(self, actor, part_id, action="read"):
        row = self.db.execute("SELECT * FROM parts WHERE id=?", (part_id,)).fetchone()
        if row is None:
            raise AccessRevoked(detail="part unavailable")
        self._scope_access(actor, row["scope_id"], "read" if action == "read" else "write")
        if row["status"] != "active":
            raise AccessRevoked(detail="part unavailable")
        if (
            not actor.admin
            and not self.db.execute(
                "SELECT 1 FROM part_acl WHERE part_id=? AND principal=? AND action=?",
                (part_id, actor.id, action),
            ).fetchone()
        ):
            raise AccessRevoked(detail="part access denied")
        return row

    def create_scope(self, actor, scope, *, kind="task", parent_id=None):
        if not actor.admin:
            raise AccessRevoked(detail="scope creation is host-only")
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO scopes VALUES(?,?,?)", (scope, kind, parent_id))

    def retrieval_revision(self, actor, scope, *, exclude_event_ids=()):
        """Fingerprint searchable content and permissions, excluding current-run events.

        Observations, leases, active refs and protected flags do not invalidate this key.
        The caller still rechecks source ACL on every read; this is no permission grant.
        """
        with self._lock:
            self._scope_access(actor, scope)
            excluded = set(exclude_event_ids)
            events = [
                tuple(r)
                for r in self.db.execute(
                    "SELECT event_id,digest FROM history_events WHERE scope_id=? ORDER BY event_id",
                    (scope,),
                )
                if r[0] not in excluded
            ]
            indexed = []
            if self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='history_search_entries'"
            ).fetchone():
                indexed = [
                    tuple(r)
                    for r in self.db.execute(
                        "SELECT e.kind,e.record_id,s.body FROM history_search_entries e "
                        "JOIN history_search s ON s.rowid=e.rowid WHERE e.scope_id=? "
                        "ORDER BY e.kind,e.record_id",
                        (scope,),
                    )
                    if not (r[0] == "event" and r[1] in excluded)
                ]
            facts = [
                tuple(r)
                for r in self.db.execute(
                    "SELECT id,revision,status,verification FROM parts "
                    "WHERE scope_id=? ORDER BY id",
                    (scope,),
                )
            ]
            summaries = [
                tuple(r)
                for r in self.db.execute(
                    "SELECT id,checksum,status FROM compaction_records "
                    "WHERE scope_id=? ORDER BY id",
                    (scope,),
                )
            ]
            acl = [
                tuple(r)
                for r in self.db.execute(
                    "SELECT principal,action FROM scope_acl "
                    "WHERE scope_id=? ORDER BY principal,action",
                    (scope,),
                )
            ]
            part_acl = [
                tuple(r)
                for r in self.db.execute(
                    "SELECT a.part_id,a.principal,a.action FROM part_acl a "
                    "JOIN parts p ON p.id=a.part_id "
                    "WHERE p.scope_id=? ORDER BY a.part_id,a.principal,a.action",
                    (scope,),
                )
            ]
            return value_digest(
                (actor.id, actor.admin, scope, events, indexed, facts, summaries, acl, part_acl)
            )

    def authorize_scope(self, actor, scope, action="read"):
        """Check a bound component's scope permission without exposing ACL internals."""
        with self._lock:
            self._scope_access(actor, scope, action)

    def grant_scope(self, actor, scope, principal, actions=("read", "write")):
        if not actor.admin:
            raise AccessRevoked(detail="scope grants are host-only")
        if not set(actions) <= {"read", "write", "manage"}:
            raise ValueError("unknown permission")
        with self.transaction():
            self._scope_access(actor, scope)
            self.db.executemany(
                "INSERT OR IGNORE INTO scope_acl VALUES(?,?,?)",
                [(scope, principal, action) for action in actions],
            )
            self._emit("scope_acl_changed", scope)

    def revoke_scope(self, actor, scope, principal):
        if not actor.admin:
            raise AccessRevoked(detail="scope grants are host-only")
        with self.transaction():
            self.db.execute(
                "DELETE FROM scope_acl WHERE scope_id=? AND principal=?", (scope, principal)
            )
            self._purge_bridge(scope, principal=principal)
            self._emit("scope_acl_changed", scope)

    def grant_part(self, actor, part_id, principal, actions=("read",)):
        if not set(actions) <= {"read", "write", "manage"}:
            raise ValueError("unknown permission")
        with self.transaction():
            row = self._part_access(actor, part_id, "manage")
            self.db.executemany(
                "INSERT OR IGNORE INTO part_acl VALUES(?,?,?)",
                [(part_id, principal, action) for action in actions],
            )
            self._emit("part_acl_changed", row["scope_id"], part_id)

    def revoke_part_access(self, actor, part_id, principal):
        with self.transaction():
            row = self._part_access(actor, part_id, "manage")
            self.db.execute(
                "DELETE FROM part_acl WHERE part_id=? AND principal=?", (part_id, principal)
            )
            self._purge_bridge(row["scope_id"], part_id=part_id)
            self._emit("part_acl_changed", row["scope_id"], part_id)

    def observe_evidence(self, actor, scope, source_ref, revision):
        if not source_ref or not revision:
            raise ValueError("evidence requires a source and revision")
        with self.transaction():
            self._scope_access(actor, scope, "write")
            identity = uuid4().hex
            self.db.execute(
                "INSERT INTO evidence VALUES(?,?,?,?,?)",
                (identity, scope, source_ref, str(revision), time()),
            )
            self._emit("evidence_observed", scope)
            return identity

    def record_evidence_support(
        self, actor, evidence_id, *, span, content_hash, support_status="observed", rule_ref=None
    ):
        """Append support assessments; semantic validation is a host capability."""
        if support_status not in {"observed", "validated", "conflicted", "rejected"}:
            raise ValueError("unknown support status")
        if not content_hash or not (len(span) == 2 and 0 <= span[0] < span[1]):
            raise ValueError("source hash and nonempty span required")
        if support_status != "observed" and (not actor.admin or not rule_ref):
            raise AccessRevoked(detail="support validation requires trusted rule/source")
        with self.transaction():
            row = self.db.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone()
            if row is None:
                raise AccessRevoked()
            self._scope_access(actor, row["scope_id"], "write")
            self.db.execute(
                "INSERT INTO evidence_support(evidence_id,span_start,span_end,content_hash,"
                "principal_scope,support_status,rule_ref,observed_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    evidence_id,
                    *span,
                    content_hash,
                    actor.id + ":" + row["scope_id"],
                    support_status,
                    rule_ref,
                    time(),
                ),
            )
            self._emit(
                "evidence_support_recorded",
                row["scope_id"],
                evidence_id=evidence_id,
                support_status=support_status,
            )

    def evidence_assessments(self, actor, evidence_id):
        """Read append-only assessments, independently of Fact confirmation state."""
        from .evidence.ledger import EvidenceRecord

        with self._lock:
            row = self.db.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone()
            if row is None:
                raise AccessRevoked()
            self._scope_access(actor, row["scope_id"])
            return tuple(
                EvidenceRecord(
                    row["source_ref"],
                    row["revision"],
                    (a["span_start"], a["span_end"]),
                    a["content_hash"],
                    a["observed_at"],
                    a["principal_scope"],
                    a["support_status"],
                    a["rule_ref"],
                )
                for a in self.db.execute(
                    "SELECT * FROM evidence_support WHERE evidence_id=? ORDER BY id", (evidence_id,)
                )
            )

    def confirm_fact(self, actor, part_id, *, expected_revision, confirmation_ref):
        """Host-only confirmation, append a revision; never expose as an agent tool."""
        if not actor.admin or not confirmation_ref:
            raise AccessRevoked(detail="explicit trusted confirmation required")
        with self.transaction():
            row = self._part_access(actor, part_id, "write")
            if row["kind"] != "fact":
                raise ValueError("confirmation is only valid for facts")
            if row["revision"] != expected_revision:
                raise VersionConflict(part_id)
            rev = expected_revision + 1
            self.db.execute(
                "INSERT INTO revisions SELECT part_id,?,digest,inline,object_path,?,?,"
                "schema_version FROM revisions WHERE part_id=? AND rev=?",
                (rev, time(), confirmation_ref, part_id, expected_revision),
            )
            self.db.execute(
                "INSERT INTO fact_evidence SELECT part_id,?,evidence_id "
                "FROM fact_evidence WHERE part_id=? AND rev=?",
                (rev, part_id, expected_revision),
            )
            self.db.execute(
                "INSERT INTO fact_confirmations VALUES(?,?,?,?)",
                (part_id, rev, confirmation_ref, actor.id),
            )
            self.db.execute(
                "UPDATE parts SET revision=?,verification='confirmed' WHERE id=?", (rev, part_id)
            )
            self._invalidate_summaries(part_id)
            self._emit(
                "fact_confirmed",
                row["scope_id"],
                part_id,
                revision=rev,
                confirmation_ref=confirmation_ref,
            )
        return self.load(actor, part_id)

    def _validate(self, schema_id, version, value):
        adapter = self.schemas.get((schema_id, version))
        if adapter is None:
            raise SchemaInvalid(detail="schema version not registered by host")
        try:
            parsed = adapter.validate_python(value)
            payload = adapter.dump_json(parsed)
            if len(payload) > self.max_payload_bytes:
                raise ValueError("payload limit")
            return payload
        except Exception:
            raise SchemaInvalid(detail="fact schema validation failed") from None

    def check_fact(self, actor, scope, schema_id, key, value, evidence, *, schema_version=1):
        """Preflight structure, scoped evidence and host rules; never creates a Fact/proposal.

        Host checks are synchronous and deterministic. Without a registered check,
        validation is explicitly structural_only, not a semantic truth guarantee.
        """
        payload = self._validate(schema_id, schema_version, value)
        with self.transaction():
            self._scope_access(actor, scope, "write")
            if not key or not evidence:
                raise SchemaInvalid(detail="fact key and observed evidence required")
            for evidence_id in evidence:
                if not self.db.execute(
                    "SELECT 1 FROM evidence WHERE id=? AND scope_id=?", (evidence_id, scope)
                ).fetchone():
                    raise AccessRevoked(detail="evidence is not visible in this scope")
            previous = self.db.execute(
                "SELECT p.*,r.digest,r.inline,r.object_path FROM parts p JOIN revisions r "
                "ON r.part_id=p.id AND r.rev=p.revision "
                "WHERE p.scope_id=? AND p.schema_id=? AND p.key=?",
                (scope, schema_id, key),
            ).fetchone()
            if previous:
                self._part_access(actor, previous["id"], "write")
            identical = False
            if previous and previous["schema_version"] == schema_version:
                try:
                    old_payload = (
                        bytes(previous["inline"])
                        if previous["inline"] is not None
                        else (self.directory / previous["object_path"]).read_bytes()
                    )
                    if sha256(old_payload).hexdigest() != previous["digest"]:
                        raise ValueError("checksum mismatch")
                    identical = value_digest(json.loads(old_payload)) == value_digest(
                        json.loads(payload)
                    )
                except (OSError, ValueError, TypeError):
                    raise StoreCorruption(previous["id"]) from None
            request = FactCheckRequest(
                scope,
                schema_id,
                schema_version,
                key,
                freeze(json.loads(payload)),
                tuple(evidence),
                previous["id"] if previous else None,
                previous["revision"] if previous else None,
            )
            check = self.fact_checks.get((schema_id, schema_version))
            if (
                check is None
                and self.db.execute(
                    "SELECT 1 FROM schema_fact_checks WHERE id=? AND version=?",
                    (schema_id, schema_version),
                ).fetchone()
            ):
                raise SchemaInvalid(detail="register the required host Fact checker before writing")
            verdict = check(request) if check else FactVerdict()
            if not isinstance(verdict, FactVerdict):
                if inspect.iscoroutine(verdict):
                    verdict.close()
                raise TypeError("Fact checker must return FactVerdict synchronously")
            validation = "host_checks_passed" if check else "structural_only"
            if verdict.action in {"skip", "reject"}:
                result = FactCheckResult(
                    "reject" if verdict.action == "reject" else "skip",
                    "host_checks_rejected" if verdict.action == "reject" else "host_checks_skipped",
                    verdict.reasons,
                )
            elif identical:
                result = FactCheckResult(
                    "unchanged",
                    validation,
                    ("identical_payload", *verdict.reasons),
                    previous["id"],
                    previous["revision"],
                )
            else:
                result = FactCheckResult(
                    "propose",
                    validation,
                    (*verdict.reasons, "value_changed" if previous else "new_key"),
                    previous["id"] if previous else None,
                    previous["revision"] if previous else None,
                )
            self._emit(
                "fact_checked",
                scope,
                result.part_id or "",
                decision=result.decision,
                validation=result.validation,
                reasons=result.reasons,
                schema_id=schema_id,
                key_hash=sha256(key.encode()).hexdigest(),
            )
            return result

    def propose_fact(
        self,
        actor,
        scope,
        schema_id,
        key,
        value,
        evidence,
        *,
        schema_version=1,
        expected_revision=None,
    ):
        result = self.check_fact(
            actor, scope, schema_id, key, value, evidence, schema_version=schema_version
        )
        if result.decision in {"skip", "reject"}:
            raise FactAdmissionRejected(result)
        payload = self._validate(schema_id, schema_version, value)
        with self.transaction():
            self._scope_access(actor, scope, "write")
            if not key or not evidence:
                raise SchemaInvalid(detail="fact key and observed evidence required")
            for evidence_id in evidence:
                if not self.db.execute(
                    "SELECT 1 FROM evidence WHERE id=? AND scope_id=?", (evidence_id, scope)
                ).fetchone():
                    raise AccessRevoked(detail="evidence is not visible in this scope")
            identity = uuid4().hex
            self.db.execute(
                "INSERT INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    scope,
                    schema_id,
                    schema_version,
                    key,
                    payload,
                    json.dumps(list(evidence)),
                    actor.id,
                    expected_revision,
                    time(),
                ),
            )
            self._emit("fact_proposed", scope)
            return identity

    def _publish(self, scope, payload, *, force_object=False):
        digest = sha256(payload).hexdigest()
        if not force_object and len(payload) <= self.inline_bytes:
            return digest, payload, None
        tenant = sha256(scope.encode()).hexdigest()
        target = self.objects / tenant / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, name = tempfile.mkstemp(dir=target.parent, prefix=".publish-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, target)
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                Path(name).unlink(missing_ok=True)
        return digest, None, str(target.relative_to(self.directory))

    def _fts_write(self, part_id, body=None):
        rowid = self.db.execute("SELECT rowid FROM parts WHERE id=?", (part_id,)).fetchone()[0]
        self.db.execute("DELETE FROM fact_fts WHERE rowid=?", (rowid,))
        if body is not None:
            self.db.execute(
                "INSERT INTO fact_fts(rowid,part_id,body) VALUES(?,?,?)", (rowid, part_id, body)
            )

    def commit(self, actor, proposal_id):
        with self._lock:
            proposal = self.db.execute(
                "SELECT * FROM proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if proposal is None or (proposal["author"] != actor.id and not actor.admin):
                raise AccessRevoked(detail="proposal unavailable")
            result = self.check_fact(
                actor,
                proposal["scope_id"],
                proposal["schema_id"],
                proposal["key"],
                json.loads(proposal["payload"]),
                json.loads(proposal["evidence"]),
                schema_version=proposal["schema_version"],
            )
            if result.decision in {"skip", "reject"}:
                raise FactAdmissionRejected(result)
            return self._commit_checked(actor, proposal_id, result)

    def _commit_checked(self, actor, proposal_id, admission):
        with self.transaction():
            proposal = self.db.execute(
                "SELECT * FROM proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if proposal is None or (proposal["author"] != actor.id and not actor.admin):
                raise AccessRevoked(detail="proposal unavailable")
            scope = proposal["scope_id"]
            self._scope_access(actor, scope, "write")
            self._validate(
                proposal["schema_id"], proposal["schema_version"], json.loads(proposal["payload"])
            )
            row = self.db.execute(
                "SELECT * FROM parts WHERE scope_id=? AND schema_id=? AND key=?",
                (scope, proposal["schema_id"], proposal["key"]),
            ).fetchone()
            if row:
                self._part_access(actor, row["id"], "write")
                if proposal["expected_revision"] != row["revision"]:
                    raise VersionConflict(row["id"])
                if proposal["schema_version"] != row["schema_version"] and not actor.admin:
                    raise SchemaInvalid(detail="schema migration is host-only")
                if proposal["schema_version"] < row["schema_version"]:
                    raise SchemaInvalid(detail="schema version cannot be downgraded")
                if admission.decision == "unchanged":
                    self.db.execute("DELETE FROM proposals WHERE id=?", (proposal_id,))
                    self._emit("fact_unchanged", scope, row["id"], revision=row["revision"])
                    return self.load(actor, row["id"])
                part_id, rev = row["id"], row["revision"] + 1
            else:
                if proposal["expected_revision"] is not None:
                    raise VersionConflict(detail="fact does not exist")
                part_id, rev = uuid4().hex, 1
                self.db.execute(
                    "INSERT INTO parts(id,scope_id,key,kind,schema_id"
                    ",schema_version) VALUES(?,?,?,?,?,?)",
                    (
                        part_id,
                        scope,
                        proposal["key"],
                        "fact",
                        proposal["schema_id"],
                        proposal["schema_version"],
                    ),
                )
                self.db.executemany(
                    "INSERT INTO part_acl VALUES(?,?,?)",
                    [(part_id, actor.id, a) for a in ("read", "write", "manage")],
                )
            digest, inline, object_path = self._publish(scope, bytes(proposal["payload"]))
            self.db.execute(
                "INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?)",
                (
                    part_id,
                    rev,
                    digest,
                    inline,
                    object_path,
                    time(),
                    "agent_proposal",
                    proposal["schema_version"],
                ),
            )
            self.db.execute(
                "UPDATE parts SET revision=?,schema_version=?,verification='unverified' WHERE id=?",
                (rev, proposal["schema_version"], part_id),
            )
            self.db.executemany(
                "INSERT INTO fact_evidence VALUES(?,?,?)",
                [(part_id, rev, e) for e in json.loads(proposal["evidence"])],
            )
            self._fts_write(part_id, proposal["key"] + " " + bytes(proposal["payload"]).decode())
            self.db.execute("DELETE FROM proposals WHERE id=?", (proposal_id,))
            self._invalidate_summaries(part_id)
            self._emit("fact_committed", scope, part_id, revision=rev)
        return self.load(actor, part_id)

    def load(self, actor, part_id, *, revision=None):
        with self._lock:
            self._reconcile()
            row = self._part_access(actor, part_id)
            rev = row["revision"] if revision is None else revision
            revision_row = self.db.execute(
                "SELECT * FROM revisions WHERE part_id=? AND rev=?", (part_id, rev)
            ).fetchone()
            if revision_row is None:
                raise VersionConflict(part_id, "revision unavailable")
            key = (row["scope_id"], part_id, rev)
            value = self.cache.get(key)
            if value is None:
                try:
                    payload = (
                        bytes(revision_row["inline"])
                        if revision_row["inline"] is not None
                        else (self.directory / revision_row["object_path"]).read_bytes()
                    )
                    if sha256(payload).hexdigest() != revision_row["digest"]:
                        raise ValueError("checksum mismatch")
                    adapter = self.schemas.get((row["schema_id"], revision_row["schema_version"]))
                    if adapter is None:
                        raise SchemaInvalid(
                            part_id, "register the stored schema version before loading"
                        )
                    value = freeze(adapter.validate_json(payload))
                except (OSError, ValueError, TypeError):
                    raise StoreCorruption(part_id) from None
                self.cache.put(key, value, len(payload))
            evidence = tuple(
                r[0]
                for r in self.db.execute(
                    "SELECT evidence_id FROM fact_evidence WHERE part_id=? AND rev=?",
                    (part_id, rev),
                )
            )
            return FactRecord(
                part_id,
                row["scope_id"],
                row["key"],
                row["kind"],
                row["schema_id"],
                revision_row["schema_version"],
                rev,
                row["status"],
                (
                    "confirmed"
                    if self.db.execute(
                        "SELECT 1 FROM fact_confirmations WHERE part_id=? AND rev=?", (part_id, rev)
                    ).fetchone()
                    else "unverified"
                ),
                value,
                evidence,
                revision_row["digest"],
            )

    def metadata(self, actor, part_id):
        with self._lock:
            row = self._part_access(actor, part_id)
            return dict(row)

    def query(self, actor, scope, intent="", *, kind=None, limit=20, after_id=None):
        return self._query(actor, scope, intent, kind=kind, limit=limit, after_id=after_id)

    def search_facts(self, actor, scope, intent, *, limit=20):
        """Rank authorized Fact matches before truncating; catalog ID cursors stay unchanged."""
        return self._query(actor, scope, intent, kind="fact", limit=limit, ranked=True)

    def _query(self, actor, scope, intent, *, kind=None, limit=20, after_id=None, ranked=False):
        if not 1 <= limit <= 100:
            raise ValueError("query limit must be between 1 and 100")
        with self._lock:
            self._scope_access(actor, scope)
            clauses = ["p.scope_id=?", "p.status='active'"]
            args: list[Any] = [scope]
            if not actor.admin:
                clauses.append(
                    "EXISTS(SELECT 1 FROM part_acl a WHERE a.part_id=p.id "
                    "AND a.principal=? AND a.action='read')"
                )
                args.append(actor.id)
            if after_id:
                clauses.append("p.id>?")
                args.append(after_id)
            if kind:
                clauses.append("p.kind=?")
                args.append(kind)
            words = re.findall(r"[\w]+", intent, re.UNICODE)[:32]
            source, order = "parts p", "p.id"
            if words:
                expression = " OR ".join('"' + word + '"' for word in words)
                if ranked:
                    source += " JOIN fact_fts ON fact_fts.part_id=p.id"
                    clauses.append("fact_fts MATCH ?")
                else:
                    clauses.append("p.id IN (SELECT part_id FROM fact_fts WHERE fact_fts MATCH ?)")
                args.append(expression)
                if ranked:
                    normalized = (
                        "' '||lower(replace(replace(replace(p.key,'/',' '),'-',' '),'_',' '))||' '"
                    )
                    matches = [
                        f"CASE WHEN instr({normalized},?)>0 THEN 1 ELSE 0 END" for _ in words
                    ]
                    order = "(" + "+".join(matches) + ") DESC, bm25(fact_fts), p.id"
                    args.extend(" " + word.lower() + " " for word in words)
            args.append(limit)
            sql = (
                "SELECT p.* FROM "
                + source
                + " WHERE "
                + " AND ".join(clauses)
                + " ORDER BY "
                + order
                + " LIMIT ?"
            )
            return [dict(r) for r in self.db.execute(sql, args)]

    def register_resource(self, actor, scope, key, *, kind="resource", metadata=None):
        if not actor.admin:
            raise AccessRevoked(detail="resource registration is host-only")
        with self.transaction():
            self._scope_access(actor, scope)
            row = self.db.execute(
                "SELECT id FROM parts WHERE scope_id=? AND schema_id='' AND key=?", (scope, key)
            ).fetchone()
            if row:
                return row[0]
            part_id = uuid4().hex
            self.db.execute(
                "INSERT INTO parts(id,scope_id,key,kind,schema_id,schema_version,metadata) "
                "VALUES(?,?,?,?,?,?,?)",
                (part_id, scope, key, kind, "", 1, json.dumps(metadata or {})),
            )
            self._fts_write(part_id, key + " " + json.dumps(metadata or {}))
            self._emit("resource_registered", scope, part_id)
            return part_id

    def activate(self, actor, scope, part_id, *, policy="live", revision=None, max_age=0):
        if policy not in {"live", "pinned", "bounded_staleness", "validate_before_action"}:
            raise ValueError("unknown version policy")
        if max_age < 0:
            raise ValueError("max_age must be nonnegative")
        with self.transaction():
            self._scope_access(actor, scope, "write")
            row = self._part_access(actor, part_id)
            if row["scope_id"] != scope:
                raise AccessRevoked(detail="cross-scope activation is forbidden")
            if policy == "pinned" and row["kind"] == "fact":
                revision = row["revision"] if revision is None else revision
                if not self.db.execute(
                    "SELECT 1 FROM revisions WHERE part_id=? AND rev=?", (part_id, revision)
                ).fetchone():
                    raise VersionConflict(part_id)
            self.db.execute(
                "INSERT OR REPLACE INTO active_refs VALUES(?,?,?,?,?,?)",
                (scope, actor.id, part_id, policy, revision, max_age),
            )
            self._emit("activated", scope, part_id)
            return ActiveRef(scope, part_id, policy, revision, max_age)

    def deactivate(self, actor, scope, part_id):
        with self.transaction():
            self._scope_access(actor, scope, "write")
            self.db.execute(
                "DELETE FROM active_refs WHERE scope_id=? AND principal=? AND part_id=?",
                (scope, actor.id, part_id),
            )
            self._emit("deactivated", scope, part_id)

    def active_refs(self, actor, scope):
        with self._lock:
            self._scope_access(actor, scope)
            rows = self.db.execute(
                "SELECT * FROM active_refs WHERE scope_id=? AND principal=?", (scope, actor.id)
            ).fetchall()
            refs = []
            for r in rows:
                self._part_access(actor, r["part_id"])
                refs.append(
                    ActiveRef(scope, r["part_id"], r["policy"], r["revision"], r["max_age"])
                )
            return refs

    def _invalidate_summaries(self, part_id):
        self.db.execute(
            "UPDATE compaction_records SET status='invalidated' WHERE id IN "
            "(SELECT compaction_id FROM compaction_refs WHERE part_id=?)",
            (part_id,),
        )

    def retire(self, actor, part_id, *, status="revoked", superseded_by=None):
        if status not in {"revoked", "superseded", "forgotten"}:
            raise ValueError("unknown retirement status")
        with self.transaction():
            row = self._part_access(actor, part_id, "write")
            if superseded_by:
                replacement = self._part_access(actor, superseded_by)
                if replacement["scope_id"] != row["scope_id"]:
                    raise AccessRevoked(detail="cross-scope supersession is forbidden")
                self.db.execute(
                    "INSERT OR REPLACE INTO relations VALUES(?,?,?,?)",
                    (part_id, superseded_by, "superseded_by", str(row["revision"])),
                )
            self.db.execute("UPDATE parts SET status=? WHERE id=?", (status, part_id))
            self.db.execute("DELETE FROM active_refs WHERE part_id=?", (part_id,))
            self._fts_write(part_id)
            self._invalidate_summaries(part_id)
            if status == "forgotten":
                self.db.execute("DELETE FROM fact_confirmations WHERE part_id=?", (part_id,))
                self.db.execute("DELETE FROM revisions WHERE part_id=?", (part_id,))
                self.db.execute("DELETE FROM leases WHERE part_id=?", (part_id,))
                self.db.execute("DELETE FROM fact_evidence WHERE part_id=?", (part_id,))
                self.db.execute(
                    "DELETE FROM proposals WHERE scope_id=? AND schema_id=? AND key=?",
                    (row["scope_id"], row["schema_id"], row["key"]),
                )
                self.db.execute(
                    "DELETE FROM history_search WHERE rowid IN "
                    "(SELECT e.rowid FROM history_search_entries e JOIN compaction_refs r "
                    "ON r.compaction_id=e.record_id WHERE e.kind='summary' AND r.part_id=?)",
                    (part_id,),
                )
                self.db.execute(
                    "DELETE FROM history_search_entries WHERE kind='summary' AND record_id IN "
                    "(SELECT compaction_id FROM compaction_refs WHERE part_id=?)",
                    (part_id,),
                )
                self.db.execute(
                    "UPDATE compaction_records SET payload=?,status='withdrawn' WHERE id IN "
                    "(SELECT compaction_id FROM compaction_refs WHERE part_id=?)",
                    (b"{}", part_id),
                )
            self._purge_bridge(row["scope_id"], part_id=part_id)
            self._emit("fact_" + status, row["scope_id"], part_id)

    def lease(self, actor, part_id, revision, *, seconds=60):
        if not 0 < seconds <= 3600:
            raise ValueError("lease duration must be in (0,3600]")
        with self.transaction():
            self._part_access(actor, part_id)
            if not self.db.execute(
                "SELECT 1 FROM revisions WHERE part_id=? AND rev=?", (part_id, revision)
            ).fetchone():
                raise VersionConflict(part_id)
            token = uuid4().hex
            self.db.execute(
                "INSERT INTO leases VALUES(?,?,?,?)", (token, part_id, revision, time() + seconds)
            )
            return token

    def release(self, token):
        with self.transaction():
            self.db.execute("DELETE FROM leases WHERE token=?", (token,))

    def gc(self, actor, *, prune_revisions=False):
        if not actor.admin:
            raise AccessRevoked(detail="garbage collection is host-only")
        with self.transaction():
            self.db.execute("DELETE FROM leases WHERE expires<?", (time(),))
            if prune_revisions:
                self.db.execute("""DELETE FROM revisions AS r WHERE
                    r.rev < (SELECT revision FROM parts WHERE id=r.part_id)
                    AND NOT EXISTS(SELECT 1 FROM leases l WHERE l.part_id=r.part_id AND l.rev=r.rev)
                    AND NOT EXISTS(SELECT 1 FROM active_refs a WHERE a.part_id=r.part_id
                                   AND a.policy='pinned' AND a.revision=r.rev)
                    AND NOT EXISTS(SELECT 1 FROM history_refs h
                                   WHERE h.part_id=r.part_id AND h.rev=r.rev)
                    AND NOT EXISTS(SELECT 1 FROM compaction_refs c
                                   WHERE c.part_id=r.part_id AND c.rev=CAST(r.rev AS TEXT))""")
            referenced = {
                r[0]
                for r in self.db.execute(
                    "SELECT object_path FROM revisions WHERE object_path IS NOT NULL"
                )
            }
            referenced.update(
                r[0]
                for r in self.db.execute(
                    "SELECT object_path FROM history_events WHERE object_path IS NOT NULL"
                )
            )
            removed = 0
            for path in self.objects.rglob("*"):
                if path.is_file() and str(path.relative_to(self.directory)) not in referenced:
                    path.unlink()
                    removed += 1
            self.cache.clear()
            return {"orphan_objects_removed": removed}

    def integrity(self, actor):
        if not actor.admin:
            raise AccessRevoked(detail="integrity inspection is host-only")
        with self._lock:
            problems = []
            for row in self.db.execute(
                "SELECT part_id,rev,digest,inline,object_path FROM revisions"
            ):
                try:
                    body = (
                        bytes(row["inline"])
                        if row["inline"] is not None
                        else (self.directory / row["object_path"]).read_bytes()
                    )
                    if sha256(body).hexdigest() != row["digest"]:
                        raise ValueError()
                except (OSError, ValueError):
                    problems.append({"part_id": row["part_id"], "revision": row["rev"]})
            return {
                "sqlite": self.db.execute("PRAGMA integrity_check").fetchone()[0],
                "bad_revisions": problems,
            }

    def events_since(self, actor, scope, seq=0, *, limit=100):
        with self._lock:
            self._scope_access(actor, scope)
            return [
                dict(r)
                for r in self.db.execute(
                    "SELECT * FROM events WHERE scope_id=? AND seq>? ORDER BY seq LIMIT ?",
                    (scope, seq, min(max(limit, 1), 1000)),
                )
            ]
