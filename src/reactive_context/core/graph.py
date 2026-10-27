from ..errors import CircularDependency
from .part import PartDefinition


def topological(definitions: dict[str, PartDefinition], selected=None) -> list[str]:
    """Iterative DFS also supports deep chains beyond Python's recursion limit."""
    order, done, active = [], set(), []
    for root in definitions if selected is None else selected:
        stack = [(root, False)]
        while stack:
            key, leaving = stack.pop()
            if leaving:
                active.pop()
                done.add(key)
                order.append(key)
                continue
            if key in done:
                continue
            if key in active:
                start = active.index(key)
                raise CircularDependency(key, " -> ".join([*active[start:], key]))
            if key not in definitions:
                raise ValueError(f"missing dependency: {key}")
            active.append(key)
            stack.append((key, True))
            stack.extend((dep, False) for dep in reversed(definitions[key].deps))
    return order


def downstream(definitions, key):
    found, queue = {key}, [key]
    while queue:
        current = queue.pop()
        for name, definition in definitions.items():
            if current in definition.deps and name not in found:
                found.add(name)
                queue.append(name)
    return found
