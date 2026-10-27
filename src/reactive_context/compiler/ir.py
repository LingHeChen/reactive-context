from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from ..core.snapshot import freeze


@dataclass(frozen=True)
class ContextIRSegment:
    id: str
    part_id: str
    source_revision: str
    payload: Any
    resource_id: str = ""
    kind: str = "evidence"
    required: bool = True
    trust_class: str = "external_data"
    requested_role: str = "user"
    inclusion_reason: str = "selected dependency"
    token_estimate: int | None = None
    priority: int = 0
    exclusive_namespace: str | None = None
    status: str = "CURRENT"
    scope_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "payload", freeze(self.payload))


ContextSegment = ContextIRSegment


@dataclass(frozen=True)
class ProviderMessage:
    role: str
    content: str
    resource_id: str
    revision: str
    kind: str
    part_id: str
    placement: str = "suffix"


@dataclass(frozen=True)
class TraceDecision:
    part_id: str
    resource_id: str
    revision: str
    status: str
    included: bool
    inclusion_reason: str
    token_estimate: int
    compiler_config_revision: str
    render_index: int | None = None
    placement: str | None = None


@dataclass(frozen=True)
class CompiledContext:
    snapshot_id: str
    provider_messages: tuple[ProviderMessage, ...]
    provider_tools: tuple[Any, ...]
    trace: tuple[TraceDecision, ...]
    token_estimate: int = 0
    render_order: tuple[str, ...] = ()
    compile_id: str = field(default_factory=lambda: uuid4().hex)
    revision_vector: Any = field(default_factory=lambda: MappingProxyType({}))
    manifest: Any = field(default_factory=lambda: MappingProxyType({}))
    diagnostics: tuple[Any, ...] = ()

    @property
    def messages(self):
        return self.provider_messages

    @property
    def usage(self):
        return freeze({"estimated_tokens": self.token_estimate, "actual_tokens": None})
