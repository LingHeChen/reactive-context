"""Host-bound evidence contracts. A source quote proves provenance, not semantic truth."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from ..core.snapshot import digest, freeze
from ..errors import ContextError


class EvidenceInsufficient(ContextError):
    code = "EVIDENCE_INSUFFICIENT"


@dataclass(frozen=True)
class EvidenceRequirement:
    id: str
    subject: str
    predicate: str = "needs_current_value"
    required: bool = True
    origin: Literal["trusted_spec", "current_user", "heuristic"] = "trusted_spec"
    depends_on: tuple[str, ...] = ()
    scope: str = "default"

    def __post_init__(self):
        if not self.id or not self.subject or not self.scope:
            raise ValueError("requirement identity must be nonempty")
        if self.origin not in {"trusted_spec", "current_user", "heuristic"}:
            raise ValueError("unknown requirement origin")
        if self.required and self.origin == "heuristic":
            raise ValueError("heuristic requirements cannot become hard constraints")
        if self.predicate not in {"needs_current_value", "preserve", "update"}:
            raise ValueError("unsupported predicate")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))


@dataclass(frozen=True)
class EvidenceHit:
    source_id: str
    source_revision: str
    span_start: int
    span_end: int
    subject: str
    content_hash: str
    scope: str = "default"
    evidence_kind: Literal["direct", "inferred", "disputed"] = "direct"

    def __post_init__(self):
        if not self.source_id or not self.source_revision or not self.subject or not self.scope:
            raise ValueError("missing evidence identity")
        if self.span_start < 0 or self.span_end <= self.span_start:
            raise ValueError("invalid source span")
        if self.evidence_kind not in {"direct", "inferred", "disputed"}:
            raise ValueError("invalid evidence kind")

    @property
    def identity(self):
        return digest(freeze(self))


@dataclass(frozen=True)
class SourceBinding:
    """Trusted adapter's structural field binding, not a claim supplied by the retriever.

    Canonical value is used only for conflict detection. Validating a source span does
    not validate prices, truth or any other business constraint.
    """

    subject: str
    start: int
    end: int
    value: str
    confirmed: bool = True


@dataclass(frozen=True)
class SourceSnapshot:
    id: str
    revision: str
    scope: str
    text: str
    bindings: tuple[SourceBinding, ...] = ()
    supersedes: tuple[tuple[str, str], ...] = ()
    current: bool = True

    def __post_init__(self):
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "supersedes", tuple(tuple(x) for x in self.supersedes))
        for binding in self.bindings:
            if not 0 <= binding.start < binding.end <= len(self.text):
                raise ValueError("binding outside source")

    def hits(self):
        return tuple(
            EvidenceHit(
                self.id,
                self.revision,
                b.start,
                b.end,
                b.subject,
                digest(self.text[b.start : b.end]),
                self.scope,
                "direct" if b.confirmed else "inferred",
            )
            for b in self.bindings
        )


@dataclass(frozen=True)
class ResearchBudget:
    max_rounds: int = 3
    max_io: int = 32
    max_tokens: int = 8192
    timeout_seconds: float = 5
    initial_k: int = 8
    max_k: int = 32

    def __post_init__(self):
        if any(
            type(n) is not int or n < 0 for n in (self.max_rounds, self.max_io, self.max_tokens)
        ):
            raise ValueError("research limits must be nonnegative integers")
        if (
            not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or type(self.initial_k) is not int
            or type(self.max_k) is not int
            or not 0 < self.initial_k <= self.max_k <= 1000
        ):
            raise ValueError("invalid research bounds")


@dataclass(frozen=True)
class EvidenceBundle:
    requirements: tuple[EvidenceRequirement, ...]
    hits: tuple[EvidenceHit, ...]
    covered_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    sufficiency: str
    stop_reason: str
    trace_id: str
    search_rounds: int = 0
    source_reads: int = 0
    io_count: int = 0
    tokens: int = 0
    latency_ms: float = 0
    trace: tuple = ()
    mode: str = "shadow"
    policy_revision: str = "coverage-v1"
    excerpts: tuple = ()

    def __post_init__(self):
        from ..core.snapshot import freeze

        for name in ("requirements", "hits", "covered_ids", "missing_ids", "conflict_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "trace", freeze(self.trace))
        object.__setattr__(self, "excerpts", freeze(self.excerpts))

    def manifest(self):
        from ..core.snapshot import freeze

        required = {r.id for r in self.requirements if r.required}
        return freeze(
            {
                "evidence_requirements": self.requirements,
                "covered_ids": self.covered_ids,
                "missing_ids": self.missing_ids,
                "conflict_ids": self.conflict_ids,
                "sufficiency": self.sufficiency,
                "stop_reason": self.stop_reason,
                "trace_id": self.trace_id,
                "source_refs": self.hits,
                "mode": self.mode,
                "policy_revision": self.policy_revision,
                "required_count": len(required),
                "covered_count": len(required & set(self.covered_ids)),
                "unresolved_count": len(required & set(self.missing_ids)),
                "conflict_count": len(self.conflict_ids),
                "research_cost": {
                    "search_rounds": self.search_rounds,
                    "source_reads": self.source_reads,
                    "io_count": self.io_count,
                    "tokens": self.tokens,
                    "latency_ms": self.latency_ms,
                    "model_calls": 0,
                },
                "trace": self.trace,
            }
        )


class SlotPlanner:
    """Explicit host/current-user slots only; never mine benchmark gold or free text."""

    def plan(self, slots):
        requirements = tuple(slots)
        by_id = {r.id: r for r in requirements}
        if len(by_id) != len(requirements):
            raise ValueError("duplicate requirement id")
        visiting, done = set(), set()

        def visit(key):
            if key not in by_id:
                raise ValueError("missing requirement dependency")
            if key in visiting:
                raise ValueError("requirement dependency cycle")
            if key in done:
                return
            visiting.add(key)
            for dependency in by_id[key].depends_on:
                if by_id.get(dependency) and by_id[dependency].scope != by_id[key].scope:
                    raise ValueError("cross-scope requirement dependency")
                visit(dependency)
            visiting.remove(key)
            done.add(key)

        for key in by_id:
            visit(key)
        return requirements
