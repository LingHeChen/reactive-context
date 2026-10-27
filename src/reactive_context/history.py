"""Immutable event archive and incremental, auditable history-only compaction.

Compactors are host-supplied async hooks. Schema/anchor validation is structural; a semantic
validator is a separate explicit hook, never an implied guarantee that an LLM summary is lossless.
"""

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from time import monotonic, time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .errors import ContextError
from .history_content import event_text, matching_preview


class HistoryBudgetExceeded(ContextError):
    code = "history_budget_exceeded"


class CompactionRejected(ContextError):
    code = "compaction_rejected"


class SummaryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    constraints: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    task_state: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str]
    current_resource_claims: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class CompactionOutput:
    """Usage metadata must come from the host callback, never the model's JSON output."""

    summary: SummaryDraft
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HistoryPolicy:
    recent_groups: int = 1
    segment_groups: int = 4
    trigger_ratio: float = 1.0
    max_compactions_per_boundary: int = 4
    min_compaction_interval: float = 0
    timeout_seconds: float = 60
    required_constraints: tuple[str, ...] = ()
    host_compaction: bool = False
    require_semantic_validation: bool = True
    max_summary_tokens: int = 2048
    context_ratio: float = 0.8
    projection: Literal["recent", "compact"] = "recent"
    recent_turns: int = 1
    recent_context_ratio: float = 0.01

    def __post_init__(self):
        if type(self.recent_turns) is not int or self.recent_turns < 0:
            raise ValueError("recent_turns must be a nonnegative integer")
        if not 0 <= self.recent_context_ratio <= 1:
            raise ValueError("recent_context_ratio must be between zero and one")
        if self.projection not in {"recent", "compact"}:
            raise ValueError("unknown history projection")
        if self.recent_groups < 1 or self.segment_groups < 1:
            raise ValueError("history windows must be positive")
        if (
            not 0 < self.trigger_ratio <= 1
            or not 0 < self.context_ratio <= 1
            or self.max_compactions_per_boundary < 1
        ):
            raise ValueError("invalid compaction threshold or bound")
        if (
            self.min_compaction_interval < 0
            or self.timeout_seconds <= 0
            or self.max_summary_tokens < 1
        ):
            raise ValueError("invalid compaction time bounds")

    def history_limits(self, total_tokens: int, reserved_tokens: int = 0):
        """Return (hard history capacity, soft trigger), counting all request reservations.

        Reservations include current resources, instructions, tools and output allowance.
        context_ratio applies to the whole configured window, not only its free remainder.
        """
        if type(total_tokens) is not int or total_tokens <= 0:
            raise ValueError("total context budget must be a positive integer")
        if type(reserved_tokens) is not int or reserved_tokens < 0:
            raise ValueError("reserved tokens must be a nonnegative integer")
        hard = max(0, total_tokens - reserved_tokens)
        soft = max(
            0,
            min(
                hard * self.trigger_ratio, int(total_tokens * self.context_ratio) - reserved_tokens
            ),
        )
        return hard, soft


@dataclass(frozen=True)
class HistoryGroup:
    events: tuple[dict[str, Any], ...]
    protected: bool = False
    revisions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HistoryProjection:
    summaries: tuple[dict, ...]
    raw_groups: tuple[HistoryGroup, ...]
    estimated_tokens: int
    compacted_groups: int
    archive_events: int


class HistoryManager:
    def __init__(
        self,
        store,
        principal,
        *,
        compactor=None,
        validator=None,
        policy=None,
        estimator=None,
        summary_model="host-hook",
        authorize_sources=None,
        final_projector=None,
    ):
        self.store, self.principal = store, principal
        self.compactor, self.validator = compactor, validator
        self.policy = policy or HistoryPolicy()
        self.estimator = estimator or (lambda text: len(text.encode()) + 8)
        self.summary_model = summary_model
        with store.transaction():
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS history_results("
                "scope_id TEXT,event_id TEXT,PRIMARY KEY(scope_id,event_id))"
            )
        if final_projector is not None and (
            not callable(final_projector)
            or inspect.iscoroutinefunction(final_projector)
            or inspect.iscoroutinefunction(type(final_projector).__call__)
        ):
            raise ValueError("final_projector must be a synchronous host callback")
        self.final_projector = final_projector
        self.authorize_sources = authorize_sources
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_compaction: dict[str, float] = {}
        self._requested: set[str] = set()

    @staticmethod
    def event_id(event):
        return sha256(
            json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()

    def observe(self, scope, groups):
        with self.store.transaction():
            self.store._scope_access(self.principal, scope, "write")
            seq = self.store.db.execute(
                "SELECT COALESCE(MAX(seq),0) FROM history_events WHERE scope_id=?", (scope,)
            ).fetchone()[0]
            for group in groups:
                for event in group.events:
                    event_id = self.event_id(event)
                    existing = self.store.db.execute(
                        "SELECT payload FROM history_events WHERE scope_id=? AND event_id=?",
                        (scope, event_id),
                    ).fetchone()
                    if existing is not None:
                        if existing["payload"] is not None:
                            _, _, path = self.store._publish(
                                scope, bytes(existing["payload"]), force_object=True
                            )
                            self.store.db.execute(
                                "UPDATE history_events SET payload=NULL,object_path=? "
                                "WHERE scope_id=? AND event_id=?",
                                (path, scope, event_id),
                            )
                        self._index(
                            scope, "event", event_id, self._event_text(scope, event_id, event)
                        )
                        if group.protected:
                            self.store.db.execute(
                                "UPDATE history_events SET protected=1 WHERE "
                                "scope_id=? AND event_id=?",
                                (scope, event_id),
                            )
                        for part_id, revision in group.revisions.items():
                            self.store.db.execute(
                                "INSERT OR IGNORE INTO history_refs VALUES(?,?,?,?)",
                                (scope, event_id, part_id, revision),
                            )
                        continue
                    seq += 1
                    body = json.dumps(event, sort_keys=True, ensure_ascii=False).encode()
                    digest, inline, object_path = self.store._publish(
                        scope, body, force_object=True
                    )
                    self.store.db.execute(
                        "INSERT INTO history_events VALUES(?,?,?,?,?,?,?)",
                        (scope, event_id, seq, inline, int(group.protected), digest, object_path),
                    )
                    self._index(scope, "event", event_id, self._event_text(scope, event_id, event))
                    for part_id, revision in group.revisions.items():
                        self.store.db.execute(
                            "INSERT OR IGNORE INTO history_refs VALUES(?,?,?,?)",
                            (scope, event_id, part_id, revision),
                        )
            return seq

    def _event_text(self, scope, event_id, event):
        final = (
            self.store.db.execute(
                "SELECT 1 FROM history_results WHERE scope_id=? AND event_id=?", (scope, event_id)
            ).fetchone()
            is not None
        )
        return event_text(event, self.final_projector, final=final)

    def record_result(self, scope, event_id):
        """Host/runtime marks an archived model-visible response as a successful final result.

        Only the raw event is projected; callback return objects and host secrets are never indexed.
        """
        with self.store.transaction():
            self.store._scope_access(self.principal, scope, "write")
            event = self.retrieve(scope, [event_id])[0]
            self.store.db.execute(
                "INSERT OR IGNORE INTO history_results VALUES(?,?)", (scope, event_id)
            )
            self._index(scope, "event", event_id, self._event_text(scope, event_id, event))

    def _index(self, scope, kind, record_id, body):
        row = self.store.db.execute(
            "SELECT rowid FROM history_search_entries WHERE scope_id=? AND kind=? AND record_id=?",
            (scope, kind, record_id),
        ).fetchone()
        if not body.strip():
            if row:
                self.store.db.execute("DELETE FROM history_search WHERE rowid=?", (row[0],))
                self.store.db.execute("DELETE FROM history_search_entries WHERE rowid=?", (row[0],))
            return
        body = body[:65536]
        if row:
            previous = self.store.db.execute(
                "SELECT body FROM history_search WHERE rowid=?", (row[0],)
            ).fetchone()
            if previous is None or previous[0] != body:
                self.store.db.execute("DELETE FROM history_search WHERE rowid=?", (row[0],))
                self.store.db.execute(
                    "INSERT INTO history_search(rowid,body) VALUES(?,?)", (row[0], body)
                )
        else:
            cursor = self.store.db.execute(
                "INSERT INTO history_search_entries VALUES(?,?,?)", (scope, kind, record_id)
            )
            self.store.db.execute(
                "INSERT INTO history_search(rowid,body) VALUES(?,?)", (cursor.lastrowid, body)
            )

    def _check_event_access(self, scope, event_ids):
        self.store._scope_access(self.principal, scope)
        for event_id in event_ids:
            if not self.store.db.execute(
                "SELECT 1 FROM history_events WHERE scope_id=? AND event_id=?", (scope, event_id)
            ).fetchone():
                raise KeyError(event_id)
            for ref in self.store.db.execute(
                "SELECT part_id FROM history_refs WHERE scope_id=? AND event_id=?",
                (scope, event_id),
            ):
                for match in self.store.db.execute(
                    "SELECT id FROM parts WHERE scope_id=? AND (id=? OR key=? "
                    "OR json_extract(metadata,'$.resource_key')=?)",
                    (scope, ref[0], ref[0], ref[0]),
                ):
                    self.store.metadata(self.principal, match[0])

    def search(
        self, scope, intent, *, kind="all", limit=20, exclude_event_ids=(), preview_chars=2000
    ):
        """Search archives on demand; hits are historical evidence, never current instructions."""
        if kind not in {"summary", "event", "all"} or not 1 <= limit <= 50:
            raise ValueError("invalid history search kind or limit")
        if type(preview_chars) is not int or not 1 <= preview_chars <= 2000:
            raise ValueError("invalid history preview length")
        words = re.findall(r"[\w]+", intent, re.UNICODE)[:32]
        if not words or len(intent) > 2000:
            raise ValueError("history search requires a bounded nonempty query")
        expression = " OR ".join('"' + word + '"' for word in words)
        excluded = tuple(dict.fromkeys(exclude_event_ids))
        if len(excluded) > 2000:
            raise ValueError("too many excluded history events")
        exclusion = (
            (
                " AND (e.kind!='event' OR e.record_id NOT IN ("
                + ",".join("?" for _ in excluded)
                + "))"
            )
            if excluded
            else ""
        )
        with self.store._lock:
            self.store._scope_access(self.principal, scope)
            if any("\u3400" <= c <= "\u9fff" for c in intent):
                # unicode61 does not segment Chinese phrases. Preserve lexical substring
                # search without pretending this local index is a semantic retriever.
                terms = ["%" + w.replace("_", "\\_") + "%" for w in words]
                rows = self.store.db.execute(
                    "SELECT e.* FROM history_search h JOIN history_search_entries e "
                    "ON e.rowid=h.rowid WHERE e.scope_id=? AND (?='all' OR e.kind=?) AND ("
                    + " OR ".join("h.body LIKE ? ESCAPE '\\'" for _ in terms)
                    + ")"
                    + exclusion
                    + " ORDER BY e.rowid DESC LIMIT ?",
                    (scope, kind, kind, *terms, *excluded, limit * 4),
                ).fetchall()
            else:
                rows = self.store.db.execute(
                    "SELECT e.* FROM history_search JOIN history_search_entries e "
                    "ON e.rowid=history_search.rowid WHERE history_search MATCH ? "
                    "AND e.scope_id=? AND (?='all' OR e.kind=?)"
                    + exclusion
                    + " ORDER BY rank LIMIT ?",
                    (expression, scope, kind, kind, *excluded, limit * 4),
                ).fetchall()
            result = []
            from .errors import AccessRevoked

            for row in rows:
                try:
                    if row["kind"] == "summary":
                        summary = self.store.db.execute(
                            "SELECT * FROM compaction_records WHERE scope_id=? AND id=? "
                            "AND status='current'",
                            (scope, row["record_id"]),
                        ).fetchone()
                        if summary is None:
                            continue
                        ids = json.loads(summary["source_ids"])
                        self._check_event_access(scope, ids)
                        text = json.loads(summary["payload"])["text"]
                    else:
                        ids = [row["record_id"]]
                        if ids[0] in excluded:
                            continue
                        text = self._event_text(scope, ids[0], self.retrieve(scope, ids)[0])
                        if not text.strip():
                            continue
                    result.append(
                        {
                            "id": row["record_id"],
                            "kind": row["kind"],
                            "text": matching_preview(text, words, preview_chars),
                            "event_ids": ids[:50],
                            "sources": self.event_sources(scope, ids),
                            "source_event_count": len(ids),
                            "authority": "historical_evidence",
                        }
                    )
                except (AccessRevoked, KeyError):
                    continue
                if len(result) == limit:
                    break
            return result

    def event_sources(self, scope, event_ids):
        """Authorized source metadata for host adapters to enforce external plugin ACLs."""
        with self.store._lock:
            self._check_event_access(scope, event_ids)
            result = {}
            for event_id in event_ids:
                for ref in self.store.db.execute(
                    "SELECT part_id FROM history_refs WHERE scope_id=? AND event_id=?",
                    (scope, event_id),
                ):
                    for row in self.store.db.execute(
                        "SELECT id FROM parts WHERE scope_id=? AND (id=? OR key=? "
                        "OR json_extract(metadata,'$.resource_key')=?)",
                        (scope, ref[0], ref[0], ref[0]),
                    ):
                        meta = self.store.metadata(self.principal, row[0])
                        result[meta["id"]] = {k: meta[k] for k in ("id", "key", "kind")}
            return list(result.values())

    def tail_ids(self, scope, *, limit=16):
        if not 1 <= limit <= 50:
            raise ValueError("invalid history tail limit")
        with self.store._lock:
            self.store._scope_access(self.principal, scope)
            return [
                r[0]
                for r in reversed(
                    self.store.db.execute(
                        "SELECT event_id FROM history_events WHERE scope_id=? "
                        "ORDER BY seq DESC LIMIT ?",
                        (scope, limit),
                    ).fetchall()
                )
            ]

    async def archive(self, scope, groups, *, revisions=None):
        """Compress one older segment for retrieval, without projecting it into a request."""
        groups = tuple(groups)
        async with self._locks.setdefault(scope, asyncio.Lock()):
            self.observe(scope, groups)
            records = self._records(scope, dict(revisions or {}))
            covered = {e for record in records for e in record["source_ids"]}
            candidates = [
                g
                for g in groups[: -self.policy.recent_groups]
                if not g.protected and not all(self.event_id(e) in covered for e in g.events)
            ]
            if not candidates:
                return None
            return await self._compact(
                scope, candidates[: self.policy.segment_groups], dict(revisions or {})
            )

    def reindex(self, scope, *, after_seq=0, limit=100):
        """Index and materialize legacy events; continue with the returned sequence."""
        if not 1 <= limit <= 1000:
            raise ValueError("invalid history reindex limit")
        with self.store.transaction():
            self.store._scope_access(self.principal, scope, "write")
            rows = self.store.db.execute(
                "SELECT seq,event_id,payload FROM history_events WHERE scope_id=? AND seq>? "
                "ORDER BY seq LIMIT ?",
                (scope, after_seq, limit),
            ).fetchall()
            for row in rows:
                event = self.retrieve(scope, [row["event_id"]])[0]
                if row["payload"] is not None:
                    _, _, path = self.store._publish(
                        scope, bytes(row["payload"]), force_object=True
                    )
                    self.store.db.execute(
                        "UPDATE history_events SET payload=NULL,object_path=? "
                        "WHERE scope_id=? AND event_id=?",
                        (path, scope, row["event_id"]),
                    )
                self._index(
                    scope, "event", row["event_id"], self._event_text(scope, row["event_id"], event)
                )
            for summary in self.store.db.execute(
                "SELECT id,payload FROM compaction_records WHERE scope_id=? AND status='current' "
                "AND id NOT IN (SELECT record_id FROM history_search_entries "
                "WHERE scope_id=? AND kind='summary') LIMIT ?",
                (scope, scope, limit),
            ).fetchall():
                self._index(scope, "summary", summary["id"], bytes(summary["payload"]).decode())
            return rows[-1]["seq"] if rows else after_seq

    async def prepare(
        self,
        scope,
        groups,
        *,
        total_tokens=None,
        max_tokens=None,
        reserved_tokens=0,
        revisions=None,
        measure=None,
        projection_groups=None,
    ):
        if self.policy.projection == "compact":
            return await self.project(
                scope,
                groups,
                total_tokens=total_tokens,
                max_tokens=max_tokens,
                reserved_tokens=reserved_tokens,
                revisions=revisions,
                measure=measure,
            )
        groups = tuple(groups)
        if scope in self._requested:
            self._requested.discard(scope)
            await self.archive(scope, groups, revisions=revisions)
        async with self._locks.setdefault(scope, asyncio.Lock()):
            archived = self.observe(scope, groups)
            if projection_groups is not None:
                groups = tuple(projection_groups)
            recent = tuple(
                g
                for i, g in enumerate(groups)
                if g.protected or i >= len(groups) - self.policy.recent_groups
            )
            size = (
                measure((), recent)
                if measure
                else self.estimator(json.dumps([g.events for g in recent], ensure_ascii=False))
            )
            if total_tokens is None:
                if max_tokens is None or reserved_tokens:
                    raise ValueError("supply total context or an explicit local budget")
                hard = max_tokens
            else:
                hard, _ = self.policy.history_limits(total_tokens, reserved_tokens)
                if max_tokens is not None:
                    hard = min(hard, max_tokens)
            if size > hard:
                raise HistoryBudgetExceeded(
                    detail="recent/protected history exceeds request budget"
                )
            return HistoryProjection((), recent, size, 0, archived)

    def retrieve(self, scope, event_ids):
        event_ids = tuple(event_ids)
        with self.store._lock:
            self._check_event_access(scope, event_ids)
            result = []
            for event_id in event_ids:
                row = self.store.db.execute(
                    "SELECT * FROM history_events WHERE scope_id=? AND event_id=?",
                    (scope, event_id),
                ).fetchone()
                if row is None:
                    raise KeyError(event_id)
                from .store import StoreCorruption

                try:
                    body = (
                        bytes(row["payload"])
                        if row["payload"] is not None
                        else (self.store.directory / row["object_path"]).read_bytes()
                    )
                    if sha256(body).hexdigest() != row["digest"]:
                        raise ValueError("archive checksum mismatch")
                    result.append(json.loads(body))
                except (OSError, ValueError):
                    raise StoreCorruption(detail="history archive object unavailable") from None
            return result

    def request_compaction(self, scope, reason="agent request"):
        self.store._scope_access(self.principal, scope, "write")
        if self.policy.host_compaction:
            raise CompactionRejected(detail="host owns compaction; negotiate through its adapter")
        self._requested.add(scope)
        with self.store.transaction():
            self.store._emit("compaction_requested", scope)

    def _records(self, scope, revisions):
        self.store._scope_access(self.principal, scope)
        rows = self.store.db.execute(
            "SELECT * FROM compaction_records WHERE scope_id=? AND "
            "status != 'absorbed' ORDER BY first_seq",
            (scope,),
        ).fetchall()
        result = []
        for row in rows:
            refs = dict(
                self.store.db.execute(
                    "SELECT part_id,rev FROM compaction_refs WHERE compaction_id=?", (row["id"],)
                )
            )
            invalid = row["status"] != "current" or any(
                revisions.get(key) != value for key, value in refs.items()
            )
            result.append(
                {
                    "id": row["id"],
                    "source_ids": json.loads(row["source_ids"]),
                    "checksum": row["checksum"],
                    "model": row["model"],
                    "validation": row["validation"],
                    "status": "WITHDRAWN_SOURCE_CHANGED" if invalid else "HISTORICAL",
                    "summary": None if invalid else json.loads(row["payload"]),
                    "first_seq": row["first_seq"],
                    "last_seq": row["last_seq"],
                }
            )
        return result

    async def _compact(self, scope, groups, revisions, parents=()):
        if self.policy.require_semantic_validation and self.validator is None:
            raise CompactionRejected(detail="host semantic validator is required by policy")
        if self.compactor is None:
            raise HistoryBudgetExceeded(
                detail="history exceeds budget and no compactor is configured"
            )
        if (
            monotonic() - self._last_compaction.get(scope, float("-inf"))
            < self.policy.min_compaction_interval
        ):
            raise CompactionRejected(detail="compaction rate limit")
        source = (
            [
                {
                    "kind": "historical_summary",
                    "summary_id": parent["id"],
                    "summary": parent["summary"],
                }
                for parent in parents
            ]
            if parents
            else [event for group in groups for event in group.events]
        )
        ids = (
            list(dict.fromkeys(e for parent in parents for e in parent["source_ids"]))
            if parents
            else [self.event_id(event) for event in source]
        )
        originals = self.retrieve(scope, ids) if parents else source
        sources = self.event_sources(scope, ids)
        if self.authorize_sources is not None:
            from .core.scheduler import invoke

            await invoke(self.authorize_sources, scope, sources)
        digest = sha256(
            json.dumps(originals, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        async with asyncio.timeout(self.policy.timeout_seconds):
            result = self.compactor(tuple(source))
            if inspect.isawaitable(result):
                result = await result
            model = self.summary_model
            usage = {"reported": False}
            if isinstance(result, CompactionOutput):
                model, usage, result = (
                    result.model,
                    {"reported": True, **result.usage},
                    result.summary,
                )
            draft = SummaryDraft.model_validate(result)
            if self.estimator(draft.model_dump_json()) > self.policy.max_summary_tokens:
                raise CompactionRejected(detail="summary exceeds its configured output budget")
            if not draft.evidence_ids or not set(draft.evidence_ids) <= set(ids):
                raise CompactionRejected(
                    detail="summary evidence must reference its archived sources"
                )
            if any(c not in draft.constraints for c in self.policy.required_constraints):
                raise CompactionRejected(detail="summary dropped a host-protected task constraint")
            if any(revisions.get(k) != v for k, v in draft.current_resource_claims.items()):
                raise CompactionRejected(
                    detail="summary claims an invalid current resource revision"
                )
            validation = "structural_only"
            if self.validator:
                valid = self.validator(tuple(source), draft)
                if inspect.isawaitable(valid):
                    valid = await valid
                if not valid:
                    raise CompactionRejected(detail="host semantic validator rejected summary")
                validation = "host_semantic_validator_passed"
        with self.store.transaction():
            self.store._scope_access(self.principal, scope, "write")
            # Concurrent scopes/processes may already have compacted this exact source segment.
            existing = self.store.db.execute(
                "SELECT id FROM compaction_records WHERE scope_id=? AND checksum=?", (scope, digest)
            ).fetchone()
            if existing:
                return existing[0]
            positions = [
                self.store.db.execute(
                    "SELECT seq FROM history_events WHERE scope_id=? AND event_id=?", (scope, e)
                ).fetchone()[0]
                for e in ids
            ]
            identity = uuid4().hex
            self.store.db.execute(
                "INSERT INTO compaction_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    scope,
                    min(positions),
                    max(positions),
                    json.dumps(ids),
                    digest,
                    draft.model_dump_json().encode(),
                    model,
                    validation,
                    "current",
                    time(),
                    json.dumps(usage),
                ),
            )
            self._index(scope, "summary", identity, draft.model_dump_json())
            refs = {k: v for group in groups for k, v in group.revisions.items()}
            for parent in parents:
                refs.update(
                    dict(
                        self.store.db.execute(
                            "SELECT part_id,rev FROM compaction_refs WHERE compaction_id=?",
                            (parent["id"],),
                        )
                    )
                )
            refs.update(draft.current_resource_claims)
            for parent in parents:
                self.store.db.execute(
                    "UPDATE compaction_records SET status='absorbed' WHERE id=?", (parent["id"],)
                )
            self.store.db.executemany(
                "INSERT INTO compaction_refs VALUES(?,?,?)",
                [(identity, k, v) for k, v in refs.items()],
            )
            self.store._emit(
                "history_compacted",
                scope,
                segment_id=identity,
                source_count=len(ids),
                model=model,
                usage=usage,
                validation=validation,
            )
        self._last_compaction[scope] = monotonic()
        return identity

    async def project(
        self,
        scope,
        groups,
        *,
        max_tokens=None,
        total_tokens=None,
        reserved_tokens=0,
        revisions=None,
        measure=None,
    ):
        if total_tokens is not None:
            hard, trigger = self.policy.history_limits(total_tokens, reserved_tokens)
            if max_tokens is not None:
                hard = min(hard, max_tokens)
                trigger = min(trigger, hard * self.policy.trigger_ratio)
            max_tokens = hard
        else:
            if max_tokens is None or reserved_tokens:
                raise ValueError("supply total_tokens, or an explicit legacy max_tokens budget")
            trigger = max_tokens * self.policy.trigger_ratio
        if max_tokens <= 0:
            raise HistoryBudgetExceeded(detail="no history budget remains")
        revisions = dict(revisions or {})
        groups = tuple(groups)
        async with self._locks.setdefault(scope, asyncio.Lock()):
            archive_events = self.observe(scope, groups)
            force = scope in self._requested
            if self.policy.host_compaction:
                size = (
                    measure((), groups)
                    if measure
                    else self.estimator(json.dumps([g.events for g in groups], ensure_ascii=False))
                )
                if size > max_tokens:
                    raise HistoryBudgetExceeded(detail="host must compact before this model call")
                return HistoryProjection((), groups, size, 0, archive_events)
            compacted = 0
            for _ in range(self.policy.max_compactions_per_boundary + 1):
                with self.store._lock:
                    records = self._records(scope, revisions)
                source_ids = {e for record in records for e in record["source_ids"]}
                raw = [
                    g
                    for g in groups
                    if g.protected or not all(self.event_id(e) in source_ids for e in g.events)
                ]
                size = (
                    measure(tuple(records), tuple(raw))
                    if measure
                    else self.estimator(
                        json.dumps(
                            [{k: v for k, v in r.items() if k != "source_ids"} for r in records],
                            ensure_ascii=False,
                        )
                    )
                    + sum(self.estimator(json.dumps(g.events, ensure_ascii=False)) for g in raw)
                )
                if size <= trigger and not force:
                    return HistoryProjection(
                        tuple(records), tuple(raw), size, compacted, archive_events
                    )
                candidates = [g for g in raw[: -self.policy.recent_groups] if not g.protected]
                mergeable = [r for r in records if r["status"] == "HISTORICAL"]
                if (
                    not candidates and len(mergeable) < 2
                ) or compacted >= self.policy.max_compactions_per_boundary:
                    if size <= max_tokens:
                        return HistoryProjection(
                            tuple(records), tuple(raw), size, compacted, archive_events
                        )
                    raise HistoryBudgetExceeded(
                        detail="protected/recent history or summaries exceed budget"
                    )
                segment = candidates[: self.policy.segment_groups]
                before = size
                parents = mergeable[: max(2, self.policy.segment_groups)] if not segment else ()
                await self._compact(scope, segment, revisions, parents=parents)
                compacted += 1
                force = False
                self._requested.discard(scope)
                # Non-shrinking summaries cannot cause an infinite compaction loop.
                if before <= 0:
                    raise CompactionRejected(detail="invalid input measurement")
            raise HistoryBudgetExceeded(detail="compaction bound reached")

    def close_scope(self, scope):
        self._locks.pop(scope, None)
        self._last_compaction.pop(scope, None)
        self._requested.discard(scope)
