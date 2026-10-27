"""Metadata-only append-only audit log. Part payloads and exceptions are never logged."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from time import time
from uuid import uuid4


@dataclass(frozen=True)
class Event:
    trace_id: str
    event: str
    scope_id: str = ""
    part_id: str = ""
    revision: str = ""
    generation: int = 0
    reason: str = ""
    duration_ms: float = 0
    cache_hit: bool = False
    freshness_status: str = ""
    snapshot_id: str = ""
    invocation_id: str = ""
    observed_at: float = 0
    prefix_hash: str = ""
    prefix_bytes: int = 0
    prefix_blocks: int = 0
    reused_prefix_bytes: int = 0
    reused_prefix_blocks: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usage_reported: bool = False
    usage_source: str = ""
    resource_id: str = ""
    token_estimate: int = 0
    compiler_config_revision: str = ""


class EventLog:
    def __init__(self, sqlite_path: str | None = None, max_events: int = 10000):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.events: list[Event] = []
        self.max_events = max_events
        self.db = sqlite3.connect(sqlite_path) if sqlite_path else None
        if self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, data TEXT)")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS manifests (snapshot_id TEXT PRIMARY KEY, "
                "data TEXT NOT NULL)"
            )
            self.db.commit()

    def emit(self, event: str, **kwargs):
        item = Event(trace_id=str(uuid4()), event=event, observed_at=time(), **kwargs)
        self.events.append(item)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        if self.db:
            self.db.execute("INSERT INTO events(data) VALUES (?)", (json.dumps(asdict(item)),))
            self.db.commit()
        return item

    def export(self) -> str:
        return json.dumps([asdict(e) for e in self.events], sort_keys=True)

    def record_manifest(self, manifest):
        if self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO manifests VALUES (?, ?)",
                (manifest["snapshot_id"], json.dumps(manifest, sort_keys=True)),
            )
            self.db.commit()

    def read_manifest(self, snapshot_id):
        if not self.db:
            raise KeyError(snapshot_id)
        row = self.db.execute(
            "SELECT data FROM manifests WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return json.loads(row[0])

    def read_persisted(self):
        if not self.db:
            return []
        return [
            json.loads(row[0]) for row in self.db.execute("SELECT data FROM events ORDER BY id")
        ]

    def cache_summary(self, scope=None):
        events = [
            event
            for event in self.events
            if event.event == "model_cache_usage" and (scope is None or event.scope_id == scope)
        ]
        inputs = sum(event.input_tokens for event in events)
        reads = sum(event.cache_read_tokens for event in events)
        return {
            "window": "retained_events",
            "recorded_requests": len(events),
            "input_tokens": inputs,
            "cache_read_tokens": reads,
            "cache_write_tokens": sum(event.cache_write_tokens for event in events),
            "cache_read_fraction": reads / inputs if inputs else None,
            "usage_sources": sorted({event.usage_source for event in events}),
        }

    def close(self):
        if self.db:
            self.db.close()
            self.db = None
