"""Host-defined validation of candidate values, independent of edit permission."""

import re
from dataclasses import dataclass
from typing import Any, Literal

from .core.snapshot import freeze, thaw


@dataclass(frozen=True)
class StateIssue:
    path: str
    code: str
    expected: Any = None
    actual: Any = None

    def __post_init__(self):
        if not isinstance(self.path, str) or len(self.path) > 2048:
            raise ValueError("invalid issue path")
        if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", self.code):
            raise ValueError("issue code must be content-free")
        object.__setattr__(self, "expected", freeze(self.expected))
        object.__setattr__(self, "actual", freeze(self.actual))

    def as_dict(self):
        return {
            "path": self.path,
            "code": self.code,
            "expected": thaw(self.expected),
            "actual": thaw(self.actual),
        }


@dataclass(frozen=True)
class StateValidation:
    status: Literal["passed", "failed", "pending"]
    issues: tuple[StateIssue, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.status not in {"passed", "failed", "pending"} or len(self.issues) > 256:
            raise ValueError("invalid state validation result")
        if any(not isinstance(issue, StateIssue) for issue in self.issues):
            raise TypeError("state validation requires StateIssue objects")
        if (self.status == "passed") == bool(self.issues):
            raise ValueError("passed checks have no issues; failed/pending checks need an issue")

    def as_dict(self):
        return {"status": self.status, "issues": [issue.as_dict() for issue in self.issues]}


@dataclass(frozen=True)
class StateValidationRequest:
    scope_id: str
    state_key: str
    revision: int
    requirements_revision: int
    before: Any
    after: Any
    changes: tuple
    evidence: tuple[str, ...]
    requirements: Any
