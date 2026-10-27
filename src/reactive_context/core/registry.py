from .graph import topological
from .part import PartDefinition, PartInstance


class Registry:
    def __init__(self, parts=()):
        self.definitions: dict[str, PartDefinition] = {}
        self.instances: dict[tuple[str, str], PartInstance] = {}
        for part in parts:
            self.register(part)
        topological(self.definitions)

    def register(self, part: PartDefinition):
        if part.key in self.definitions:
            raise ValueError(f"duplicate part: {part.key}")
        self.definitions[part.key] = part
        try:
            # Missing dependencies may be registered later; cycles may not.
            if all(d in self.definitions for p in self.definitions.values() for d in p.deps):
                topological(self.definitions)
        except Exception:
            del self.definitions[part.key]
            raise

    def instance(self, scope: str, key: str) -> PartInstance:
        pair = (scope, key)
        if pair not in self.instances:
            self.instances[pair] = PartInstance(self.definitions[key], scope)
        return self.instances[pair]
