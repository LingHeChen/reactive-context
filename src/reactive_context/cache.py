"""Byte-weighted, scope-aware hot cache. Active references never pin payloads here."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheStats:
    bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int


class WeightedLRU:
    def __init__(self, max_bytes=4 * 1024 * 1024, *, scope_bytes=None, max_entries=2048):
        if max_bytes < 0 or max_entries < 1 or (scope_bytes is not None and scope_bytes < 0):
            raise ValueError("invalid cache bounds")
        self.max_bytes = max_bytes
        self.scope_bytes = max_bytes if scope_bytes is None else scope_bytes
        self.max_entries = max_entries
        self._items: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self._scope_bytes: dict[str, int] = {}
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self._items:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(key)
        return self._items[key][0]

    def _remove(self, key):
        _, size = self._items.pop(key)
        self.bytes -= size
        scope = key[0]
        remaining = self._scope_bytes[scope] - size
        if remaining:
            self._scope_bytes[scope] = remaining
        else:
            del self._scope_bytes[scope]

    def put(self, key, value, size):
        size = max(1, size)
        if key in self._items:
            self._remove(key)
        if size > min(self.max_bytes, self.scope_bytes):
            return False
        while self._scope_bytes.get(key[0], 0) + size > self.scope_bytes:
            victim = next(k for k in self._items if k[0] == key[0])
            self._remove(victim)
            self.evictions += 1
        while self._items and (
            self.bytes + size > self.max_bytes or len(self._items) >= self.max_entries
        ):
            self._remove(next(iter(self._items)))
            self.evictions += 1
        self._items[key] = (value, size)
        self.bytes += size
        self._scope_bytes[key[0]] = self._scope_bytes.get(key[0], 0) + size
        return True

    def clear(self, scope=None):
        for key in list(self._items):
            if scope is None or key[0] == scope:
                self._remove(key)

    @property
    def stats(self):
        return CacheStats(self.bytes, len(self._items), self.hits, self.misses, self.evictions)
