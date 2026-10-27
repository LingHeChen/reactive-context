"""In-memory configuration with revisioned updates and optional change events."""

from copy import deepcopy

from ..core.part import ChangeSignal, source_part
from ..core.snapshot import digest


class ConfigSource:
    def __init__(self, key, value, *, schema=None, project=None, revision_mode="content", **kwargs):
        if revision_mode not in {"content", "release"}:
            raise ValueError("unknown revision mode")
        self.revision_mode = revision_mode
        self.release = 0
        self.key = key
        self._value = deepcopy(value)
        self._subscribers = set()
        self.part = source_part(
            key=key,
            schema=schema or type(value),
            resolve=self.resolve,
            revision=self.revision,
            subscribe=self.subscribe,
            project=project,
            **kwargs,
        )

    def set(self, value):
        # Reject malformed updates before publishing or notifying observers.
        validated = self.part.adapter.validate_python(value)
        self._value = deepcopy(validated)
        self.release += 1
        for notify in tuple(self._subscribers):
            notify(ChangeSignal(self.key))

    async def revision(self, ctx):
        from ..core.snapshot import freeze

        value = freeze(self._value)
        return digest((self.release, value) if self.revision_mode == "release" else value)

    async def resolve(self, ctx):
        return deepcopy(self._value)

    def subscribe(self, ctx, notify):
        self._subscribers.add(notify)
        return lambda: self._subscribers.discard(notify)
