"""Opt-in travel-domain adapter for protected state, independent of answer labels.

The caller supplies an explicit visible base plan and trusted field grants. This
module never guesses inheritance, reads reference plans, or broadens grants.
It is not enabled retroactively in earlier benchmark results.
"""

import re

from pydantic import BaseModel, ConfigDict, field_validator

from .. import WorkState
from ..errors import SchemaInvalid

FIELDS = (
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)


class TravelDay(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    current_city: str
    transportation: str
    breakfast: str
    attraction: str
    lunch: str
    dinner: str
    accommodation: str

    @field_validator("*")
    @classmethod
    def one_line(cls, value):
        if "\n" in value or "\r" in value:
            raise ValueError("plan fields must remain on one line")
        return value


class TravelPlanState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    days: dict[str, TravelDay]


def keyed_plan(plan):
    if not isinstance(plan, list) or not plan:
        raise SchemaInvalid(detail="explicit nonempty base plan required")
    days = {}
    for item in plan:
        if not isinstance(item, dict):
            raise SchemaInvalid(detail="plan day must be an object")
        if "day" in item and "days" in item:
            raise SchemaInvalid(detail="ambiguous day identity")
        day = item.get("day", item.get("days"))
        if isinstance(day, bool) or not isinstance(day, int) or day < 1 or str(day) in days:
            raise SchemaInvalid(detail="duplicate or invalid day in plan")
        # No inferred defaults: omissions, unexpected fields and duplicates fail.
        data = {key: value for key, value in item.items() if key not in {"day", "days"}}
        days[str(day)] = TravelDay.model_validate(data).model_dump()
    return {"days": days}


def final_plan_text(text):
    boundaries = list(re.finditer(r"(?m)^=== .+['’]s Plan ===\s*$", text))
    if len(boundaries) != 1:
        raise SchemaInvalid(detail="one declared final plan is required")
    return text[boundaries[0].start() :].strip()


class TravelStateAdapter:
    def __init__(
        self,
        store,
        principal,
        *,
        key="travel.plan",
        validate_restaurants=False,
        check=None,
        check_id=None,
    ):
        self.restaurant_validator = None
        options = {"check": check, "check_id": check_id}
        if validate_restaurants:
            if check is not None or check_id is not None:
                raise ValueError("choose the restaurant validator or a supplied composite checker")
            from ..adapters.travel_validation import RestaurantStateValidator

            self.restaurant_validator = RestaurantStateValidator(store, principal, key)
            options = {
                "check": self.restaurant_validator,
                "check_id": self.restaurant_validator.check_id,
            }
        self.state = WorkState(
            store,
            principal,
            key=key,
            schema=TravelPlanState,
            empty_values=(None, "", [], {}, "-"),
            **options,
        )

    def initialize(self, scope, visible_plan, evidence, *, host):
        return self.state.initialize(scope, keyed_plan(visible_plan), evidence, host=host)

    def authorize_fields(self, scope, fields, evidence, *, expected_revision, host):
        """fields = [(day, field, action)]; trusted adapter decisions, never gold-derived."""
        grants = []
        for day, field, action in fields:
            if isinstance(day, bool) or not isinstance(day, int) or day < 1 or field not in FIELDS:
                raise ValueError("invalid domain field grant")
            grants.append({"path": f"/days/{day}/{field}", "action": action})
        return self.state.authorize(
            scope, grants, evidence, expected_revision=expected_revision, host=host
        )

    def accept_output(self, scope, text, parser, *, expected_revision, permit_id, preview=False):
        # Projection applies before parsing in this *new* protocol, so an analysis
        # paragraph mentioning Day N cannot shadow a real final-plan day.
        candidate = keyed_plan(parser(final_plan_text(text)))
        return self.state.replace(
            scope,
            candidate,
            expected_revision=expected_revision,
            permit_id=permit_id,
            preview=preview,
        )

    def plan(self, scope):
        value = self.state.read(scope)["value"]
        return [
            {"day": int(day), **data}
            for day, data in sorted(value["days"].items(), key=lambda pair: int(pair[0]))
        ]

    def render(self, scope, traveler):
        if not traveler or "\n" in traveler or "\r" in traveler:
            raise ValueError("one-line traveler name required")
        lines = [f"=== {traveler}'s Plan ==="]
        for day in self.plan(scope):
            lines.append(f"Day {day['day']}:")
            for field in FIELDS:
                label = " ".join(word.capitalize() for word in field.split("_"))
                lines.append(f"{label}: {day[field]}")
            lines.append("")
        return "\n".join(lines).strip()
