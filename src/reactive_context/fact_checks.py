"""Host-defined Fact admission: no domain rules or model calls are implicit."""

import re
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ContextError


@dataclass(frozen=True)
class FactCheckRequest:
    scope_id: str
    schema_id: str
    schema_version: int
    key: str
    value: Any
    evidence: tuple[str, ...]
    previous_id: str | None = None
    previous_revision: int | None = None


@dataclass(frozen=True)
class FactVerdict:
    action: Literal["accept", "skip", "reject"] = "accept"
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.reasons, str) or len(self.reasons) > 32:
            raise ValueError("Fact reasons must be a bounded sequence of codes")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.action not in {"accept", "skip", "reject"} or (
            self.action != "accept" and not self.reasons
        ):
            raise ValueError("invalid Fact verdict")
        if any(not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", code) for code in self.reasons):
            raise ValueError("Fact reasons must be content-free codes")


@dataclass(frozen=True)
class FactCheckResult:
    decision: Literal["propose", "unchanged", "skip", "reject"]
    validation: Literal[
        "structural_only", "host_checks_passed", "host_checks_rejected", "host_checks_skipped"
    ]
    reasons: tuple[str, ...] = ()
    part_id: str | None = None
    revision: int | None = None


class FactAdmissionRejected(ContextError):
    code = "fact_admission_rejected"

    def __init__(self, result):
        self.decision, self.reasons = result.decision, result.reasons
        super().__init__(detail="Fact admission did not permit persistence")


def fact_schema_part(store, schemas: dict[str, int], *, key="fact.schemas"):
    """Publish explicitly selected schema definitions as ordinary context data.

    No schemas are auto-discovered or granted to an agent. Defaults/examples are
    omitted; the store still performs validation against its complete registered type.
    """
    from .core.part import source_part
    from .core.snapshot import digest
    from .errors import SchemaInvalid

    selected = tuple(sorted(schemas.items()))
    if not selected:
        raise ValueError("select at least one host-registered Fact schema")

    def fields_only(value):
        if not isinstance(value, dict):
            return value
        maps = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
        sequences = {"allOf", "anyOf", "oneOf", "prefixItems"}
        children = {
            "items",
            "contains",
            "additionalProperties",
            "unevaluatedProperties",
            "unevaluatedItems",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
        }
        result = {}
        for name, child in value.items():
            if name in {"default", "examples"}:
                continue  # annotations only, not property names or enum/const literals
            if name in maps and isinstance(child, dict):
                result[name] = {field: fields_only(schema) for field, schema in child.items()}
            elif name in sequences and isinstance(child, list):
                result[name] = [fields_only(schema) for schema in child]
            elif name in children:
                result[name] = (
                    [fields_only(schema) for schema in child]
                    if isinstance(child, list)
                    else fields_only(child)
                )
            else:
                result[name] = child
        return result

    def payload():
        result = []
        for schema_id, version in selected:
            adapter = store.schemas.get((schema_id, version))
            if adapter is None:
                raise SchemaInvalid(detail="schema selected for context is not registered")
            required = (
                store.db.execute(
                    "SELECT 1 FROM schema_fact_checks WHERE id=? AND version=?",
                    (schema_id, version),
                ).fetchone()
                is not None
            )
            result.append(
                {
                    "schema_id": schema_id,
                    "schema_version": version,
                    "json_schema": fields_only(adapter.json_schema()),
                    "requires_host_check": required,
                    "host_check_available": (schema_id, version) in store.fact_checks,
                }
            )
        return {"schemas": result}

    async def resolve(ctx):
        return payload()

    async def revision(ctx):
        return digest(payload())

    return source_part(
        key=key, schema=dict[str, Any], resolve=resolve, revision=revision, kind="schema"
    )
