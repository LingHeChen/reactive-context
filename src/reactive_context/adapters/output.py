"""Shared final-boundary validation and explicit, budgeted repair (no hidden model)."""

from dataclasses import dataclass
from math import isfinite

from ..core.snapshot import digest, freeze


@dataclass(frozen=True)
class RepairBudget:
    max_attempts: int = 1
    max_cost_usd: float = 0

    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 0:
            raise ValueError("invalid repair attempts")
        if not isfinite(self.max_cost_usd) or self.max_cost_usd < 0:
            raise ValueError("invalid repair cost")


async def validate_output(
    raw, *, final_view, validate, repair=None, budget=None, reserve=None, attempt_cost_upper_bound=0
):
    """Host repair callback uses the SAME task ledger; original text never replaced.

    validate must include semantic unknown/preservation checks when the domain requires
    them. Structural validity alone is not task success. No callback means no repair.
    """
    budget = budget or RepairBudget()
    if not isfinite(attempt_cost_upper_bound) or attempt_cost_upper_bound < 0:
        raise ValueError("invalid cost bound")
    versions = []
    text = raw
    spent = 0
    final, errors = None, ("not_validated",)
    for attempt in range(budget.max_attempts + 1):
        final = final_view(text)
        errors = ("missing_final_boundary",) if final is None else tuple(validate(final))
        versions.append({"text": text, "hash": digest(text), "errors": errors})
        if not errors or repair is None or attempt == budget.max_attempts:
            break
        if spent + attempt_cost_upper_bound > budget.max_cost_usd:
            break
        if reserve is None:
            raise ValueError("repair requires explicit shared task budget reservation")
        await reserve(attempt_cost_upper_bound)
        spent += attempt_cost_upper_bound
        text = await repair(text, errors)
    return freeze(
        {
            "raw": raw,
            "versions": versions,
            "validated_final": final if not errors else None,
            "repair_cost_upper_bound_usd": spent,
            "semantic_success": "NOT_MEASURED",
        }
    )
