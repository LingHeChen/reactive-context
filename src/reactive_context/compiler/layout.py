"""Host-owned render order and canonical JSON. No authority is granted here."""

from collections.abc import Mapping
from typing import Any

from ..core.snapshot import thaw


def canonical_data(value) -> Any:
    value = thaw(value)
    if isinstance(value, Mapping):
        return {key: canonical_data(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_data(item) for item in value]
    return value


def tool_sort_key(name, policy):
    positions = {key: index for index, key in enumerate(policy.tool_order)}
    return (positions.get(name, len(positions)), name)


def canonical_catalog(payload, policy):
    payload = canonical_data(payload)
    if isinstance(payload, list):

        def key(tool):
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                raise ValueError("catalog entries require a tool name")
            return tool_sort_key(tool["name"], policy)

        return sorted(payload, key=key)
    return payload


def render_key(segment, policy, positions=None):
    if positions is None:
        positions = {key: index for index, key in enumerate(policy.part_order)}
    role = {"tools": -1, "instructions": 0, "prefix": 2, "suffix": 3}[placement(segment, policy)]
    if segment.requested_role == "developer":
        role = 1
    return (role, positions.get(segment.part_id, len(positions)), segment.part_id, segment.id)


def placement(segment, policy):
    if segment.kind == "tool_definition":
        return "tools"
    if segment.requested_role in {"system", "developer"}:
        return "instructions"
    if segment.part_id in policy.prefix_parts:
        return "prefix"
    return "suffix"
