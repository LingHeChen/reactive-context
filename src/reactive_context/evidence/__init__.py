"""Opt-in evidence coverage; import from this namespace without agent dependencies."""

from .model import (
    EvidenceBundle,
    EvidenceHit,
    EvidenceInsufficient,
    EvidenceRequirement,
    ResearchBudget,
    SlotPlanner,
    SourceBinding,
    SourceSnapshot,
)
from .navigation import HistoryNavigator, IndexedNavigator
from .part import EvidencePart
from .research import EvidenceNavigator, EvidenceResearcher, EvidenceResolver, SufficiencyEvaluator

__all__ = [
    "EvidencePart",
    "EvidenceBundle",
    "EvidenceHit",
    "EvidenceInsufficient",
    "EvidenceRequirement",
    "ResearchBudget",
    "SlotPlanner",
    "SourceBinding",
    "SourceSnapshot",
    "HistoryNavigator",
    "IndexedNavigator",
    "EvidenceNavigator",
    "EvidenceResearcher",
    "EvidenceResolver",
    "SufficiencyEvaluator",
]

from .ledger import EvidenceRecord, RequirementLedger, RequirementRecord
from .selection import BudgetUnsatisfiable, CompilationBudget, EvidenceReader, ExecutionSelector
from .session import EvidenceStatus, ResearchSession, ResearchStopped, SessionBudget

__all__ += [
    "EvidenceRecord",
    "RequirementLedger",
    "RequirementRecord",
    "BudgetUnsatisfiable",
    "CompilationBudget",
    "EvidenceReader",
    "ExecutionSelector",
    "EvidenceStatus",
    "ResearchSession",
    "ResearchStopped",
    "SessionBudget",
]
