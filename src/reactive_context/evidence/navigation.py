"""Navigation adapters reuse authorized archives; source content is never rewritten."""

import json

from ..core.snapshot import digest
from .model import SourceSnapshot


class HistoryNavigator:
    """Bind raw event spans with a trusted, synchronous host schema adapter.

    bind(event, canonical_text) returns SourceBinding entries. The host is responsible
    for entity/date/field semantics. Summary hits navigate to originals; they never
    become direct evidence. Search I/O and source reads are exposed separately.
    """

    def __init__(self, manager, bind, *, indexed_sources=()):
        self.manager, self.bind = manager, bind
        self.index = {(s.scope, s.id): s.hits() for s in indexed_sources}

    async def search(self, requirements, *, depth, limit):
        # One scope/query per navigation operation. Researcher budgets each operation.
        scopes = {r.scope for r in requirements}
        if len(scopes) != 1:
            raise ValueError("history research requires a single explicit scope")
        scope = next(iter(scopes))
        query = " ".join(r.subject.replace(".", " ") for r in requirements)[:2000]
        kind = "summary" if depth == "summary" else "event"
        rows = await self.manager.search_history(scope, query, kind=kind, limit=min(limit, 50))
        # Search metadata is only a locator. Binding occurs on the explicit read path.
        hits = []
        for row in rows:
            for event_id in row["event_ids"]:
                hits.extend(self.index.get((scope, event_id), ()))
        return tuple(dict.fromkeys(hits))[:limit]

    async def read(self, source_id, revision, scope):
        events = await self.manager.load_history(scope, [source_id])
        event = events[0]
        text = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if digest(text) != revision:
            raise KeyError("source revision changed")
        return SourceSnapshot(source_id, revision, scope, text, tuple(self.bind(event, text)))

    async def locate(self, scope, event_id):
        """Host indexing operation, outside a research invocation; budget it separately."""
        event = (await self.manager.load_history(scope, [event_id]))[0]
        text = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        source = SourceSnapshot(event_id, digest(text), scope, text, tuple(self.bind(event, text)))
        self.index[(scope, event_id)] = source.hits()
        return source


class IndexedNavigator:
    """Small deterministic structured index over immutable sources.

    authorize(scope, source_id) runs on every search/read. False/ACL exceptions refuse
    access. Replacing this local index never replaces the persistent raw history.
    """

    def __init__(self, sources, *, authorize):
        sources = tuple(sources)
        self._sources = {(s.scope, s.id, s.revision): s for s in sources}
        if len(self._sources) != len(sources):
            raise ValueError("duplicate immutable source identity")
        self._authorize = authorize

    async def read(self, source_id, revision, scope):
        from ..errors import AccessRevoked

        if not self._authorize(scope, source_id):
            raise AccessRevoked(source_id)
        return self._sources[(scope, source_id, revision)]

    async def search(self, requirements, *, depth, limit):
        from ..errors import AccessRevoked

        slots = {(r.scope, r.subject) for r in requirements}
        hits = []
        for _key, source in sorted(self._sources.items()):
            try:
                if not self._authorize(source.scope, source.id):
                    continue
            except AccessRevoked:
                continue
            if not source.current:
                continue
            hits.extend(h for h in source.hits() if (h.scope, h.subject) in slots)
        return tuple(hits[:limit])
