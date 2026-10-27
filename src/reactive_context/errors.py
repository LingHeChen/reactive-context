class ContextError(Exception):
    """Content-free diagnostic safe to record in audit logs."""

    code = "context_error"

    def __init__(self, part_id: str = "", detail: str = ""):
        self.part_id = part_id
        super().__init__(f"{self.code}: {part_id}" + (f" ({detail})" if detail else ""))


class SourceUnavailable(ContextError):
    code = "source_unavailable"


class ConsistencyUnavailable(ContextError):
    code = "revision_mismatch"


class CircularDependency(ContextError):
    code = "circular_dependency"


class SchemaInvalid(ContextError):
    code = "schema_invalid"


class CompilerBudget(ContextError):
    code = "compiler_budget"


class ContextBudgetExceeded(CompilerBudget):
    code = "context_budget_exceeded"

    def __init__(self, parts, estimated_tokens, suggested_scopes=()):
        self.parts = tuple(parts)
        self.estimated_tokens = estimated_tokens
        self.suggested_scopes = tuple(suggested_scopes)
        super().__init__(detail="shrink activation scope or deactivate optional resources")


class ProtocolInvalid(ContextError):
    code = "protocol_invalid"


class AccessRevoked(ContextError):
    code = "access_revoked"


class ResourceConflict(ContextError):
    code = "resource_conflict"


class AuthorityViolation(ContextError):
    code = "authority_violation"
