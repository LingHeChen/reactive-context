"""Travel-owned restaurant constraints against trusted, scoped tool observations.

No reference answers, inheritance rules, or natural-language parsing live here.
The application supplies structured requirements and observed source records.
"""

import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.snapshot import digest, thaw
from ..errors import AccessRevoked, SchemaInvalid
from ..state_validation import StateIssue, StateValidation
from ..store import StoreCorruption
from ..work_state import StateGrant, state_value_at


class RestaurantRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    city: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    rating_min: float | None = Field(default=None, ge=0)
    rating_max: float | None = Field(default=None, ge=0)
    cuisines: tuple[str, ...] = ()
    allow_clear: bool = False

    @model_validator(mode="after")
    def ordered_bounds(self):
        for lower, upper in ((self.price_min, self.price_max), (self.rating_min, self.rating_max)):
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("minimum exceeds maximum")
        return self


class RestaurantObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    name: str = Field(min_length=1)
    city: str = Field(min_length=1)
    price: float = Field(ge=0)
    rating: float = Field(ge=0)
    cuisines: tuple[str, ...] = ()

    @property
    def label(self):
        return f"{self.name}, {self.city}"


class RestaurantStateValidator:
    check_id = "travel.restaurant-constraints.v1"

    def __init__(self, store, principal, state_key):
        self.store, self.principal, self.key = store, principal, state_key
        with store.transaction():
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS travel_restaurant_observations("
                "scope TEXT,key TEXT,label TEXT,payload BLOB,digest TEXT,evidence TEXT,"
                "PRIMARY KEY(scope,key,label))"
            )

    def observe(self, scope, observations, evidence_id, *, host):
        """Host adapter calls this after a real source/tool read; never an agent tool."""
        if not host.admin:
            raise AccessRevoked(detail="trusted observations are host-only")
        values = [RestaurantObservation.model_validate(item) for item in observations]
        if len({item.label for item in values}) != len(values):
            raise ValueError("ambiguous duplicate restaurant labels")
        with self.store.transaction():
            self.store.authorize_scope(host, scope, "write")
            self.store.authorize_scope(self.principal, scope)
            if not self.store.db.execute(
                "SELECT 1 FROM evidence WHERE id=? AND scope_id=?", (evidence_id, scope)
            ).fetchone():
                raise AccessRevoked(detail="observation evidence is outside this scope")
            for value in values:
                payload = value.model_dump(mode="json")
                previous = self.store.db.execute(
                    "SELECT digest,evidence FROM travel_restaurant_observations "
                    "WHERE scope=? AND key=? AND label=?",
                    (scope, self.key, value.label),
                ).fetchone()
                if (
                    previous
                    and previous["evidence"] == evidence_id
                    and previous["digest"] != digest(payload)
                ):
                    raise SchemaInvalid(detail="source revision is immutable; observe new evidence")
                self.store.db.execute(
                    "INSERT INTO travel_restaurant_observations VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(scope,key,label) DO UPDATE SET payload=excluded.payload,"
                    "digest=excluded.digest,evidence=excluded.evidence",
                    (
                        scope,
                        self.key,
                        value.label,
                        json.dumps(payload).encode(),
                        digest(payload),
                        evidence_id,
                    ),
                )

    def options(self, scope):
        with self.store.transaction():
            self.store.authorize_scope(self.principal, scope)
            rows = self.store.db.execute(
                "SELECT * FROM travel_restaurant_observations WHERE scope=? AND key=? "
                "ORDER BY label",
                (scope, self.key),
            ).fetchall()
            result = []
            for row in rows:
                payload = json.loads(row["payload"])
                if digest(payload) != row["digest"]:
                    raise StoreCorruption()
                result.append({"label": row["label"], **payload, "evidence_id": row["evidence"]})
            return result

    def bind(self, state, scope, requirements, evidence, *, expected_revision, host):
        if (
            state.key != self.key
            or state.store is not self.store
            or state.principal != self.principal
        ):
            raise ValueError("validator and state must share key/store/identity")
        payload = {}
        for path, requirement in requirements.items():
            StateGrant(path=path, action="replace")
            payload[path] = RestaurantRequirement.model_validate(requirement).model_dump(
                mode="json"
            )
        if not payload:
            raise SchemaInvalid(detail="at least one explicit restaurant requirement is needed")
        return state.set_requirements(
            scope,
            {"restaurants": payload},
            evidence,
            expected_revision=expected_revision,
            host=host,
        )

    def __call__(self, request):
        raw = thaw(request.requirements).get("restaurants", {})
        if not raw:
            return StateValidation("pending", (StateIssue("", "requirements_not_configured"),))
        try:
            requirements = {p: RestaurantRequirement.model_validate(v) for p, v in raw.items()}
        except (ValueError, AttributeError):
            return StateValidation("pending", (StateIssue("", "requirements_invalid"),))
        failed, pending = [], []
        for change in request.changes:
            if change.path not in requirements:
                pending.append(StateIssue(change.path, "changed_field_has_no_validator"))
        catalog = {item["label"]: item for item in self.options(request.scope_id)}
        for path, requirement in requirements.items():
            try:
                label = state_value_at(request.after, path)
            except (KeyError, ValueError):
                failed.append(StateIssue(path, "required_selection_missing"))
                continue
            if label in ("", "-", None) and requirement.allow_clear:
                continue
            if not isinstance(label, str) or label not in catalog:
                pending.append(StateIssue(path, "selection_not_observed", actual=label))
                continue
            observed = catalog[label]
            if observed["evidence_id"] not in request.evidence:
                pending.append(StateIssue(path, "observed_source_not_bound"))
                continue
            if requirement.city is not None and observed["city"] != requirement.city:
                failed.append(StateIssue(path, "city_mismatch", requirement.city, observed["city"]))
            for field, low, high in (
                ("price", requirement.price_min, requirement.price_max),
                ("rating", requirement.rating_min, requirement.rating_max),
            ):
                actual = observed[field]
                if low is not None and actual < low:
                    failed.append(StateIssue(path, field + "_below_minimum", low, actual))
                if high is not None and actual > high:
                    failed.append(StateIssue(path, field + "_above_maximum", high, actual))
            absent = sorted(
                {c.casefold() for c in requirement.cuisines}
                - {c.casefold() for c in observed["cuisines"]}
            )
            if absent:
                failed.append(
                    StateIssue(path, "required_cuisines_missing", absent, observed["cuisines"])
                )
        issues = tuple((failed + pending)[:256])
        return StateValidation("failed" if failed else "pending" if pending else "passed", issues)
