"""Parts are reactive; contexts are compiled; inference is snapshot-based."""

from . import standard
from .api import (
    AmbiguousPartDeclaration,
    ComputedSideEffectError,
    ContextPart,
    FilePartProvider,
    InvalidStateDefault,
    LegacyPartAdapter,
    MissingPartProvider,
    PartDeclaration,
    PartHandle,
    PartMaterialization,
    PartSpec,
    ReadContext,
    ScopeBindingRequired,
    UnknownContextOption,
    computed,
    create_context,
    intent,
    memory,
    part,
    skill,
    specification,
    state,
)
from .compiler.ir import CompiledContext, ContextIRSegment, ContextSegment
from .compiler.policy import CompilerPolicy
from .context import Context
from .core.part import (
    ChangeSignal,
    PartDefinition,
    PartInstance,
    ResolveContext,
    SourcePartSpec,
    computed_part,
    source_part,
)
from .core.scope import ConsistencyPolicy, ContextScope
from .core.snapshot import ContextSnapshot, PartSnapshot
from .errors import ContextBudgetExceeded
from .fact_checks import FactCheckRequest, FactCheckResult, FactVerdict, fact_schema_part
from .history import CompactionOutput, HistoryGroup, HistoryManager, HistoryPolicy, SummaryDraft
from .history_part import history_part
from .manager import ContextManager
from .objective import Objective, ObjectivePolicy, ObjectiveUpdateRequest, WorkObjective
from .sources.skills import SkillRegistry, skill_part
from .state_candidates import StateCandidates
from .state_validation import StateIssue, StateValidation, StateValidationRequest
from .store import ActiveRef, ContextStore, FactRecord, Principal
from .work_state import (
    StateChange,
    StateGrant,
    StateOperation,
    StateUpdateRejected,
    WorkState,
    apply_state_operations,
    state_diff,
    state_value_at,
)

__all__ = [
    "FilePartProvider",
    "state",
    "computed",
    "part",
    "skill",
    "memory",
    "intent",
    "specification",
    "create_context",
    "PartDeclaration",
    "PartHandle",
    "PartSpec",
    "PartMaterialization",
    "ContextPart",
    "ReadContext",
    "LegacyPartAdapter",
    "InvalidStateDefault",
    "UnknownContextOption",
    "ComputedSideEffectError",
    "AmbiguousPartDeclaration",
    "MissingPartProvider",
    "ScopeBindingRequired",
    "Context",
    "history_part",
    "ContextManager",
    "Objective",
    "ObjectivePolicy",
    "ObjectiveUpdateRequest",
    "WorkObjective",
    "WorkState",
    "StateChange",
    "StateGrant",
    "StateOperation",
    "StateUpdateRejected",
    "StateCandidates",
    "StateIssue",
    "StateValidation",
    "StateValidationRequest",
    "state_value_at",
    "state_diff",
    "apply_state_operations",
    "ContextStore",
    "Principal",
    "FactRecord",
    "FactCheckRequest",
    "FactCheckResult",
    "FactVerdict",
    "fact_schema_part",
    "ActiveRef",
    "HistoryManager",
    "HistoryGroup",
    "HistoryPolicy",
    "SummaryDraft",
    "CompactionOutput",
    "ContextBudgetExceeded",
    "SkillRegistry",
    "skill_part",
    "ChangeSignal",
    "PartDefinition",
    "PartInstance",
    "ResolveContext",
    "SourcePartSpec",
    "source_part",
    "computed_part",
    "ConsistencyPolicy",
    "ContextScope",
    "ContextSnapshot",
    "PartSnapshot",
    "CompiledContext",
    "ContextIRSegment",
    "ContextSegment",
    "CompilerPolicy",
    "standard",
]
