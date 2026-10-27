from dataclasses import dataclass


@dataclass(frozen=True)
class ContextScope:
    id: str
    consistency: str = "live"
    max_age: float = 0


@dataclass(frozen=True)
class ConsistencyPolicy:
    mode: str = "live"
    max_age: float = 0
    on_unavailable: str = "block"  # block | cached | unknown
    revision: str = "1"

    def __post_init__(self):
        if self.mode not in {"live", "pinned", "bounded_staleness", "strict_fresh"}:
            raise ValueError("unknown consistency mode")
        if self.on_unavailable not in {"block", "cached", "unknown"} or self.max_age < 0:
            raise ValueError("invalid consistency policy")
        if self.mode == "strict_fresh" and self.on_unavailable != "block":
            raise ValueError("strict_fresh requires block")
