"""Independent, opt-in hypotheses on the stable baseline; never combine candidates."""

import re

from . import memoryarena_pilot as base
from .history_regression import append
from .intelligence_pilot import CoverageIntegration
from .travel_dependencies import dependencies

DIRECTIONS = ("control", "fact-state", "reference-hints", "writer-catalog")


def compact_catalog(catalog, source):
    """Keep complete old values for the current subject; host checks keep full catalog."""
    match = re.match(r"I am ([^.\n]+)\.", source.get("query", ""))
    if not match:
        return catalog  # Ambiguous writer scope: preserve the original representation.
    subject = match[1]
    return [
        record
        if record["value"].get("subject") == subject
        else {
            **record,
            "value": {
                key: record["value"][key]
                for key in ("subject", "topic", "record_state")
                if key in record["value"]
            },
            "catalog_note": "Identity only; omitted old content is not a deletion.",
        }
        for record in catalog
    ]


class DirectionIntegration(CoverageIntegration):
    def __init__(self, directory, direction, *, stop_budget=None):
        if direction not in DIRECTIONS:
            raise ValueError("unknown independent direction")
        super().__init__(
            "dx-baseline-shadow", directory, retain_initial_plan=True, stop_budget=stop_budget
        )
        self.direction = direction
        self.preserve_record_state = direction == "fact-state"

    def recovery_context(self, records, query):
        context = super().recovery_context(records, query)
        if self.direction == "reference-hints":
            context["comparison_references"] = {
                "hints": [
                    {"person": person, "day": day, "field": field}
                    for person, day, field in dependencies(query, self.known_people)
                ],
                "policy": "These are best-effort references to OTHER travelers, not the "
                "current traveler edit targets. Resolve their exact person/day/field "
                "from observed Facts. Apply every qualifier in the current query; "
                "a reference day need not be the day you are editing. The query "
                "remains authoritative; hints may be incomplete.",
            }
        return context

    def writer_catalog(self, catalog, source):
        if self.direction != "writer-catalog":
            return catalog
        projected = compact_catalog(catalog, source)
        append(
            self.directory / "writer-catalog-audit.jsonl",
            {
                "original_tokens": base.token_count(base.encoded(catalog)),
                "projected_tokens": base.token_count(base.encoded(projected)),
                "entries": len(catalog),
                "keys_preserved": [r["key"] for r in catalog] == [r["key"] for r in projected],
                "host_checks": "full original catalog retained",
            },
        )
        return projected

    def audit(self):
        return {
            **super().audit(),
            "direction": self.direction,
            "preserve_record_state": self.preserve_record_state,
        }
