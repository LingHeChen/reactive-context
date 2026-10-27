"""Versioned MemoryArena output contract; no gold or task-solving rules."""

import re
from dataclasses import dataclass
from typing import Any

from ..core.snapshot import digest, freeze

PROJECTION_REVISION = "declared-final-plan-v2"
REQUIRED_FIELDS = (
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)
_MARKER = re.compile(r"(?m)^=== .+['’]s Plan ===\s*$")


def final_plan_view(text: str) -> str | None:
    markers = list(_MARKER.finditer(text))
    # Ambiguous multiple final answers must not silently select the first/last.
    return text[markers[0].start() :].strip() if len(markers) == 1 else None


@dataclass(frozen=True)
class PlanProjection:
    raw_plan: Any
    final_plan: Any
    audit: Any


def project_plan(
    text,
    parser,
    *,
    required_fields=REQUIRED_FIELDS,
    expected_days=None,
    strict_slots=False,
    expected_entity=None,
):
    expected_days = tuple(expected_days) if expected_days is not None else None
    required_fields = tuple(required_fields)
    raw = parser(text)
    final = final_plan_view(text)
    parsed = parser(final) if final else None
    errors = []
    if final is None:
        errors.append("missing_or_ambiguous_final_boundary")
    days = [row.get("day") for row in parsed or ()]
    textual_days = [int(n) for n in re.findall(r"Day\s*(\d+):", final or "")]
    if not days:
        errors.append("empty_plan")
    if days != textual_days:
        errors.append("parser_order_mismatch")
    if len(set(days)) != len(days):
        errors.append("duplicate_day")
    if days and days != list(range(1, len(days) + 1)):
        errors.append("invalid_day_sequence")
    if expected_days is not None and days != list(expected_days):
        errors.append("unexpected_days")
    if any(not str(row.get(key, "")).strip() for row in parsed or () for key in required_fields):
        errors.append("missing_required_field")
    if strict_slots:
        blocks = re.split(r"Day\s*\d+:", final or "")[1:]
        for block in blocks:
            labels = [
                label.casefold().replace(" ", "_")
                for label in re.findall(r"(?m)^\s*([A-Za-z_ ]+):", block)
            ]
            if any(labels.count(key) > 1 for key in required_fields):
                errors.append("duplicate_slot")
                break
    if expected_entity is not None:
        match = re.match(r"=== (.+)['’]s Plan ===", final or "")
        if match is None or match[1] != expected_entity:
            errors.append("unexpected_entity")
    return PlanProjection(
        freeze(raw),
        freeze(parsed) if not errors else None,
        freeze(
            {
                "projection_revision": (
                    "declared-final-plan-v3"
                    if strict_slots or expected_entity is not None
                    else PROJECTION_REVISION
                ),
                "expected_days": expected_days,
                "required_fields": required_fields,
                "raw_hash": digest(text),
                "final_hash": digest(final),
                "boundary_found": final is not None,
                "raw_day_count": len(raw or ()),
                "final_day_count": len(parsed or ()),
                "parsed_final": parsed,
                "errors": errors,
                "valid": not errors,
                "changed": raw != parsed,
            }
        ),
    )


def score_tracks(rows, submission, episode_directories, *, projection_revision=PROJECTION_REVISION):
    """Write separate raw and validated official-matcher scores for NEW experiments."""
    import json

    from . import memoryarena as upstream

    raw_records = []
    for row, directory in zip(rows, episode_directories, strict=True):
        state = json.loads((directory / "checkpoint.json").read_text())
        if state.get("projection_revision") != projection_revision:
            raise ValueError("mixed scoring revisions")
        raw_records.append({"id": row["id"], "persons": state["raw_persons"]})
    raw_path = submission.with_name(submission.stem + "-official-raw.jsonl")
    if raw_path.exists():
        raise FileExistsError("raw scoring artifact already exists")
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw_records))
    validated = upstream.score(rows, submission)
    report = {
        "projection_revision": projection_revision,
        "official_raw": upstream.score(rows, raw_path),
        "validated_final": validated,
        "task_success_rate": "NOT_MEASURED",
        "semantic_validation": "NOT_MEASURED",
    }
    submission.with_name(submission.stem + "-score-tracks.json").write_text(
        json.dumps(report, indent=2)
    )
    return validated
