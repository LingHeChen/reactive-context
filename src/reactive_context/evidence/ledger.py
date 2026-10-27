"""Explicit task requirements survive absent or malformed source artifacts."""

from dataclasses import dataclass

from .model import EvidenceRequirement


@dataclass(frozen=True)
class RequirementRecord:
    requirement_id: str
    entity: str
    slot: str
    time: str
    scope: str
    derivation: str
    required: bool = True
    observed_sources: tuple[str, ...] = ()
    source_status: str = "unlocated"
    value_status: str = "unresolved"
    search_status: str = "not_searched"
    allow_unknown: bool = True
    status: str = "active"

    def __post_init__(self):
        if not all((self.requirement_id, self.entity, self.slot, self.scope, self.derivation)):
            raise ValueError("explicit requirement identity and derivation required")
        allowed = {
            "source_status": {"unlocated", "located", "source_artifact_missing"},
            "value_status": {"unresolved", "observed", "validated", "conflicted"},
            "search_status": {"not_searched", "found", "not_found", "budget_exhausted"},
            "status": {"active", "not_applicable", "revoked"},
        }
        for name, choices in allowed.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"invalid {name}")
        object.__setattr__(self, "observed_sources", tuple(self.observed_sources))


class RequirementLedger:
    def __init__(self, records):
        self.records = tuple(records)
        if len({r.requirement_id for r in self.records}) != len(self.records):
            raise ValueError("duplicate requirement")

    def requirements(self):
        return tuple(
            EvidenceRequirement(
                r.requirement_id, r.requirement_id, required=r.required, scope=r.scope
            )
            for r in self.records
            if r.status == "active"
        )

    def metrics(self):
        required = [r for r in self.records if r.required and r.status == "active"]
        return {
            "required_count": len(required),
            "source_coverage": {
                "located": sum(r.source_status == "located" for r in required),
                "total": len(required),
            },
            "evidence_validity": "NOT_MEASURED",
            "requirement_coverage": "NOT_MEASURED",
            "answer_preservation": "NOT_MEASURED",
            "missing_artifacts": [
                r.requirement_id for r in required if r.source_status == "source_artifact_missing"
            ],
        }


@dataclass(frozen=True)
class EvidenceRecord:
    source_ref: str
    source_revision: str
    span: tuple[int, int]
    content_hash: str
    observed_at: float
    principal_scope: str
    support_status: str = "observed"
    validation_ref: str | None = None

    def __post_init__(self):
        from math import isfinite

        if not all(
            (self.source_ref, self.source_revision, self.content_hash, self.principal_scope)
        ):
            raise ValueError("complete provenance required")
        if len(self.span) != 2 or not 0 <= self.span[0] < self.span[1]:
            raise ValueError("invalid source span")
        if not isfinite(self.observed_at) or self.observed_at < 0:
            raise ValueError("invalid observation time")
        if self.support_status not in {"observed", "validated", "conflicted", "rejected"}:
            raise ValueError("invalid support status")
        if self.support_status != "observed" and not self.validation_ref:
            raise ValueError("assessment requires a validation reference")
        object.__setattr__(self, "span", tuple(self.span))
