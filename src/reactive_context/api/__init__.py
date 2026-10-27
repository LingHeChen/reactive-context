"""Declarative and protocol APIs; optional model integrations stay outside core."""

from .context import create_context
from .declarations import (
    AmbiguousPartDeclaration,
    ComputedSideEffectError,
    InvalidStateDefault,
    MissingPartProvider,
    PartDeclaration,
    PartHandle,
    ScopeBindingRequired,
    UnknownContextOption,
    computed,
    intent,
    memory,
    part,
    skill,
    specification,
    state,
)
from .protocol import ContextPart, LegacyPartAdapter, PartMaterialization, PartSpec
from .providers import FilePartProvider
from .runtime import ReadContext

__all__ = [
    "FilePartProvider",
    "AmbiguousPartDeclaration",
    "ComputedSideEffectError",
    "InvalidStateDefault",
    "MissingPartProvider",
    "PartDeclaration",
    "PartHandle",
    "ScopeBindingRequired",
    "UnknownContextOption",
    "computed",
    "intent",
    "memory",
    "part",
    "skill",
    "specification",
    "state",
    "ContextPart",
    "LegacyPartAdapter",
    "PartMaterialization",
    "PartSpec",
    "ReadContext",
    "create_context",
]
