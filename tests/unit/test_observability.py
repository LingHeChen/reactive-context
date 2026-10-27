import json

from reactive_context import Context
from reactive_context.observability.events import EventLog
from reactive_context.sources.config import ConfigSource


async def test_sqlite_persistence_and_metadata_only_export(tmp_path):
    path = str(tmp_path / "audit.sqlite3")
    log = EventLog(path)
    async with Context([ConfigSource("a", "secret-payload").part], event_log=log) as context:
        snapshot = await context.next_context()
        context.compile(snapshot)
        persisted = log.read_persisted()
        assert any(item["snapshot_id"] == snapshot.id for item in persisted)
        assert "secret-payload" not in json.dumps(persisted)
        assert "secret-payload" not in context.export_trace()
        assert context.inspect("a")["last_verified"] > 0
    reopened = EventLog(path)
    try:
        assert reopened.read_persisted() == persisted
    finally:
        reopened.close()
