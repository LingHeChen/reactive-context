"""Conservative clause-to-Fact relations; no extra recall, gold, or semantic validation."""

import json
import re

from .. import computed_part
from ..core.snapshot import thaw
from . import memoryarena_pilot as base
from .direction_pilot import DirectionIntegration
from .history_regression import append
from .travel_dependencies import _DAY, _FIELD, _ORDINALS, dependencies


def day_number(match):
    return int(match[1] or match[2]) if match[1] or match[2] else _ORDINALS[match[3].lower()]


def relations(query, people):
    """Keep each original clause and repeated references; ambiguous targets stay unknown."""
    people = tuple(sorted(set(people)))
    result = []
    if not people:
        return result
    names = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in people) + r")\b", re.I)
    for clause in query.splitlines():
        clause = clause.strip()
        refs = dependencies(clause, people)
        if not refs:
            continue
        mention = names.search(clause)
        prefix = clause[: mention.start()] if mention else ""
        days = {day_number(m) for m in _DAY.finditer(prefix)}
        fields = {m[0].lower() for m in _FIELD.finditer(prefix)}
        target = (
            {"person": "current", "day": next(iter(days)), "field": next(iter(fields))}
            if len(days) == len(fields) == 1
            else None
        )
        for person, day, field in refs:
            result.append(
                {
                    "source_clause": clause,
                    "target": target,
                    "reference": {"person": person, "day": day, "field": field},
                }
            )
    return result


def quoted_field(fact, day, field):
    """Only day-scoped structured quotes or explicit day blocks; never infer from prose."""
    found = []
    for i, quote in enumerate(fact.get("quotes", ())):
        try:
            obj = json.loads(quote)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            if obj.get("day", obj.get("days")) == day and isinstance(obj.get(field), str):
                found.append(
                    (
                        obj[field],
                        {
                            "quote_path": f"/quotes/{i}",
                            "format": "json-day",
                            "day": day,
                            "field": field,
                        },
                    )
                )
            continue
        blocks = list(re.finditer(r"(?m)^Day\s+(\d+):\s*$", quote))
        matches = [j for j, m in enumerate(blocks) if int(m[1]) == day]
        if len(matches) != 1:
            continue
        j = matches[0]
        body = quote[blocks[j].end() : blocks[j + 1].start() if j + 1 < len(blocks) else len(quote)]
        label = re.escape(field.replace("_", " "))
        values = re.findall(r"(?im)^" + label + r":\s*([^\n]+)$", body)
        if len(values) == 1:
            found.append(
                (
                    values[0].strip(),
                    {
                        "quote_path": f"/quotes/{i}",
                        "format": "day-block",
                        "day": day,
                        "field": field,
                    },
                )
            )
    values = {v for v, _ in found if v.strip()}
    return found[0] if len(values) == 1 else None


def bind_relations(edges, snapshots):
    rows = []
    for edge in edges:
        ref = edge["reference"]
        matching = []
        for snap in snapshots:
            value = thaw(snap.value).get("value", {})
            if value.get("subject") == ref["person"] and value.get("topic") == f"day-{ref['day']}":
                matching.append((snap, value))
        row = {**edge, "evidence": None, "status": "unknown"}
        if len(matching) == 1:
            snap, value = matching[0]
            exact = quoted_field(value, ref["day"], ref["field"])
            row.update(
                status="bound",
                evidence={
                    "part_id": snap.resource_id,
                    "revision": snap.revision,
                    "key": f"{ref['person']}/day-{ref['day']}",
                    "field": ref["field"],
                    "value_status": "exact_quote" if exact else "unknown",
                    **({"value": exact[0], "locator": exact[1]} if exact else {}),
                    "verification": "source_binding_only_not_semantic_validation",
                },
            )
        rows.append(row)
    return {
        "relations": rows,
        "policy": "Each relation belongs to its source_clause and target. Resolve the reference "
        "only from its bound Fact/version; do not substitute baseline or another person. "
        "An unknown target or value is unresolved, not permission to guess. All original "
        "query qualifiers still apply. Bound means source identity, not a verified choice.",
    }


class ReferenceBindingIntegration(DirectionIntegration):
    """Replace flat hints with relations over already selected, versioned Fact snapshots."""

    def __init__(self, directory, *, stop_budget=None):
        super().__init__(directory, "reference-hints", stop_budget=stop_budget)

    def recovery_context(self, records, query):
        context = super().recovery_context(records, query)
        context.pop("comparison_references", None)
        self.edges = relations(query, self.known_people)
        return context

    async def prepare_context(self, manager, query, selected, records, stage):
        await super().prepare_context(manager, query, selected, records, stage)
        snapshot = await manager.next_context(base.SCOPE)
        refs = {(e["reference"]["person"], f"day-{e['reference']['day']}") for e in self.edges}
        bound = []
        for part in snapshot.parts:
            if part.resource_id not in selected:
                continue
            value = thaw(part.value).get("value", {})
            if (value.get("subject"), value.get("topic")) in refs:
                bound.append(part)
        edges = self.edges

        async def compute(ctx):
            result = bind_relations(edges, ctx.dependencies.values())
            append(self.directory / "reference-bindings.jsonl", {"stage": stage, **result})
            return result

        manager.register(
            computed_part(
                key="comparison_bindings",
                schema=dict,
                dependencies=[part.part_id for part in bound],
                compute=compute,
            )
        )
        manager.mandatory = (*manager.mandatory, "comparison_bindings")

    def audit(self):
        return {**super().audit(), "direction": "reference-bindings"}
