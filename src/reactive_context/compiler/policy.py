from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType

from ..core.snapshot import digest


def estimate_tokens(text: str) -> int:
    # Conservative UTF-8 upper bound; supply a provider tokenizer for tighter budgets.
    return len(text.encode("utf-8")) + 8


@dataclass(frozen=True)
class CompilerPolicy:
    max_tokens: int = 32768
    output_reserve: int = 1024
    grants: Mapping[str, str] = field(default_factory=dict)
    revision: str = "1"
    estimator: Callable[[str], int] = estimate_tokens
    include_uncertain: bool = False
    part_order: tuple[str, ...] = ()
    prefix_parts: frozenset[str] = frozenset()
    tool_order: tuple[str, ...] = ()

    research_mode: str = "off"
    evidence_fail_closed: bool = False
    experiment_id: str | None = None
    tokenizer_revision: str = "utf8-upper-bound-v1"
    history_projection_revision: str = "host-unspecified"

    def __post_init__(self):
        if self.research_mode not in {"off", "shadow", "schema_coverage"}:
            raise ValueError("unknown research mode")
        if self.evidence_fail_closed and self.research_mode != "schema_coverage":
            raise ValueError("fail-closed evidence requires schema_coverage")
        if self.max_tokens <= 0 or not 0 <= self.output_reserve < self.max_tokens:
            raise ValueError("invalid token budget")
        if any(role not in {"system", "developer"} for role in self.grants.values()):
            raise ValueError("invalid authority grant")
        object.__setattr__(self, "grants", MappingProxyType(dict(self.grants)))
        for name in ("part_order", "tool_order"):
            order = tuple(getattr(self, name))
            if len(order) != len(set(order)):
                raise ValueError(f"duplicate entry in {name}")
            object.__setattr__(self, name, order)
        object.__setattr__(self, "prefix_parts", frozenset(self.prefix_parts))

    @cached_property
    def materialization_fingerprint(self):
        return digest(
            {
                "budget": self.max_tokens,
                "reserve": self.output_reserve,
                "grants": self.grants,
                "revision": self.revision,
                "uncertain": self.include_uncertain,
                "part_order": self.part_order,
                "prefix_parts": sorted(self.prefix_parts),
                "tool_order": self.tool_order,
                "estimator": f"{self.estimator.__module__}.{self.estimator.__qualname__}",
            }
        )

    @cached_property
    def fingerprint(self):
        if (
            self.research_mode == "off"
            and not self.evidence_fail_closed
            and self.experiment_id is None
            and self.tokenizer_revision == "utf8-upper-bound-v1"
            and self.history_projection_revision == "host-unspecified"
        ):
            return self.materialization_fingerprint
        return digest(
            {
                "materialization": self.materialization_fingerprint,
                "research_mode": self.research_mode,
                "evidence_fail_closed": self.evidence_fail_closed,
                "experiment_id": self.experiment_id,
                "tokenizer_revision": self.tokenizer_revision,
                "history_projection_revision": self.history_projection_revision,
            }
        )
