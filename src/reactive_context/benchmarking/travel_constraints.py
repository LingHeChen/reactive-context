"""Benchmark domain bridge: quoted expressions over actual tool observations.

Expressions come from the current request, never from reference answers. This is
not natural-language verification: a quoted expression may still be mistranslated.
"""

import json
import math
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .. import StateIssue, StateValidation, state_value_at
from ..core.snapshot import digest, thaw
from ..errors import AccessRevoked
from ..store import StoreCorruption

Attribute = Literal["choice", "price", "rating", "cuisines", "room_type", "rules", "capacity"]


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    person: str
    day: int = Field(ge=1)
    field: Literal["breakfast", "lunch", "dinner", "accommodation"]
    attribute: Attribute


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    attribute: Attribute
    op: Literal["ge", "le", "gt", "lt", "eq", "ne", "contains_all", "overlaps"]
    value: JsonValue = None
    reference: Reference | None = None
    scale: float = 1
    offset: float = 0
    quote: str = Field(min_length=4, max_length=1500)

    @model_validator(mode="after")
    def operand(self):
        if (self.reference is None) == (self.value is None):
            raise ValueError("provide exactly one literal value or prior-person reference")
        numeric = self.attribute in {"price", "rating", "capacity"}
        sets = self.attribute in {"cuisines", "rules"}
        if numeric and self.op in {"contains_all", "overlaps"}:
            raise ValueError("numeric attributes require numeric comparisons")
        if sets and self.op not in {"contains_all", "overlaps", "eq", "ne"}:
            raise ValueError("set attributes require set comparisons")
        if self.reference is None:
            if numeric and (
                isinstance(self.value, bool) or not isinstance(self.value, (int, float))
            ):
                raise ValueError("numeric literal required")
            if sets and (
                not isinstance(self.value, list)
                or not self.value
                or not all(isinstance(v, str) for v in self.value)
            ):
                raise ValueError("nonempty string set required")
        if not numeric and (self.scale != 1 or self.offset != 0):
            raise ValueError("scale/offset only apply to numeric targets")
        return self


class SlotConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: int = Field(ge=1)
    field: Literal["breakfast", "lunch", "dinner", "accommodation"]
    clauses: list[Constraint] = Field(min_length=1, max_length=16)


class ConstraintBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slots: list[SlotConstraints] = Field(min_length=1, max_length=100)


def label(value):
    """Normalize only city annotations used by the upstream tools, never fuzzy names."""
    if value == "-":
        return "-"
    if not isinstance(value, str) or ", " not in value:
        return value
    name, city = value.rsplit(", ", 1)
    return name.strip() + ", " + city.split("(", 1)[0].strip()


class ObservedTravelConstraints:
    check_id = "memoryarena.quoted-travel-constraints.v1"

    def __init__(self, store, principal, scope, plans):
        self.store, self.principal, self.scope, self.plans = store, principal, scope, plans
        with store.transaction():
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS checked_travel_options(scope TEXT,kind TEXT,label TEXT,"
                "payload BLOB,digest TEXT,evidence TEXT,PRIMARY KEY(scope,kind,label))"
            )

    def observe(self, kind, records, evidence, *, host):
        if not host.admin:
            raise AccessRevoked()
        grouped = defaultdict(list)
        for row in records:
            if kind == "restaurant":
                item = {
                    "choice": row["Name"] + ", " + row["City"],
                    "price": float(row["Average Cost"]),
                    "rating": float(row["Aggregate Rating"]),
                    "cuisines": [v.strip() for v in row["Cuisines"].split(",")],
                }
            else:
                item = {
                    "choice": row["NAME"] + ", " + row["city"],
                    "price": float(row["price"]),
                    "rating": float(row["review rate number"]),
                    "room_type": row["room type"],
                    "rules": [v.strip() for v in row["house_rules"].split("&")],
                    "capacity": float(row["maximum occupancy"]),
                }
            grouped[label(item["choice"])].append(item)
        with self.store.transaction():
            self.store.authorize_scope(host, self.scope, "write")
            if not self.store.db.execute(
                "SELECT 1 FROM evidence WHERE id=? AND scope_id=?", (evidence, self.scope)
            ).fetchone():
                raise AccessRevoked()
            for name, items in grouped.items():
                payload = items[0] if len({digest(v) for v in items}) == 1 else {"ambiguous": True}
                self.store.db.execute(
                    "INSERT INTO checked_travel_options VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(scope,kind,label) DO UPDATE SET payload=excluded.payload,"
                    "digest=excluded.digest,evidence=excluded.evidence",
                    (
                        self.scope,
                        kind,
                        name,
                        json.dumps(payload).encode(),
                        digest(payload),
                        evidence,
                    ),
                )

    def sources(self):
        self.store.authorize_scope(self.principal, self.scope)
        return [
            r[0]
            for r in self.store.db.execute(
                "SELECT DISTINCT evidence FROM checked_travel_options WHERE scope=?", (self.scope,)
            )
        ]

    def _value(self, choice, field, attribute, evidence):
        if attribute == "choice":
            return label(choice)
        kind = "hotel" if field == "accommodation" else "restaurant"
        row = self.store.db.execute(
            "SELECT * FROM checked_travel_options WHERE scope=? AND kind=? AND label=?",
            (self.scope, kind, label(choice)),
        ).fetchone()
        if row is None:
            raise LookupError("option_not_observed")
        payload = json.loads(row["payload"])
        if digest(payload) != row["digest"]:
            raise StoreCorruption()
        if payload.get("ambiguous"):
            raise LookupError("ambiguous_option")
        if row["evidence"] not in evidence:
            raise LookupError("source_not_bound")
        if attribute not in payload:
            raise LookupError("attribute_not_available")
        return payload[attribute]

    def __call__(self, request):
        self.store.authorize_scope(self.principal, request.scope_id)
        if request.scope_id != self.scope:
            raise AccessRevoked()
        try:
            bundle = ConstraintBundle.model_validate(thaw(request.requirements)["constraints"])
        except (ValueError, KeyError):
            return StateValidation("pending", (StateIssue("", "constraints_missing"),))
        targets = {f"/days/{slot.day}/{slot.field}" for slot in bundle.slots}
        failed, pending = [], []
        for change in request.changes:
            if change.path not in targets:
                pending.append(StateIssue(change.path, "constraint_not_registered"))
        for slot in bundle.slots:
            path = f"/days/{slot.day}/{slot.field}"
            try:
                choice = state_value_at(request.after, path)
            except KeyError:
                failed.append(StateIssue(path, "selection_missing"))
                continue
            for clause in slot.clauses:
                try:
                    left = self._value(choice, slot.field, clause.attribute, request.evidence)
                    if clause.reference:
                        ref = clause.reference
                        if ref.person not in self.plans or self.plans[ref.person] is None:
                            raise LookupError("reference_plan_missing")
                        plan = self.plans[ref.person]
                        day = next(
                            (d for d in plan if d.get("day", d.get("days")) == ref.day), None
                        )
                        if day is None:
                            raise LookupError("reference_day_missing")
                        right = self._value(
                            day[ref.field], ref.field, ref.attribute, request.evidence
                        )
                    else:
                        right = clause.value
                    if isinstance(right, (int, float)) and not isinstance(right, bool):
                        right = right * clause.scale + clause.offset
                    elif clause.scale != 1 or clause.offset != 0:
                        raise LookupError("non_numeric_transform")
                    if (
                        isinstance(left, str)
                        and isinstance(right, str)
                        and clause.attribute != "choice"
                    ):
                        left, right = left.casefold(), right.casefold()
                    if clause.op in {"contains_all", "overlaps"}:
                        if not isinstance(left, list) or not isinstance(right, list):
                            raise LookupError("set_operand_required")
                        a, b = {str(v).casefold() for v in left}, {str(v).casefold() for v in right}
                        passed = b <= a if clause.op == "contains_all" else bool(a & b)
                    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        close = math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9)
                        passed = {
                            "ge": left > right or close,
                            "le": left < right or close,
                            "gt": left > right and not close,
                            "lt": left < right and not close,
                            "eq": close,
                            "ne": not close,
                        }[clause.op]
                    else:
                        if clause.op not in {"eq", "ne"}:
                            raise LookupError("comparison_type_mismatch")
                        passed = (left == right) if clause.op == "eq" else (left != right)
                    if not passed:
                        failed.append(
                            StateIssue(
                                path,
                                "constraint_" + clause.attribute + "_" + clause.op,
                                right,
                                left,
                            )
                        )
                except (LookupError, KeyError, TypeError) as error:
                    code = str(error).strip("'")
                    if not code.replace("_", "").isalnum():
                        code = "constraint_resolution_failed"
                    pending.append(StateIssue(path, code))
        return StateValidation(
            "failed" if failed else "pending" if pending else "passed",
            tuple((failed + pending)[:256]),
        )
