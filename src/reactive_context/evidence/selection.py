"""Budgeted execution projection; research traces remain out of model input."""

import json
from dataclasses import dataclass

from ..core.snapshot import digest, freeze, thaw
from ..errors import ContextError


class BudgetUnsatisfiable(ContextError):
    code = "BUDGET_UNSATISFIABLE"


@dataclass(frozen=True)
class CompilationBudget:
    mandatory_tokens: int
    evidence_tokens: int
    optional_tokens: int
    actor_input_token_cap: int

    def __post_init__(self):
        if any(
            type(v) is not int or v < 0
            for v in (
                self.mandatory_tokens,
                self.evidence_tokens,
                self.optional_tokens,
                self.actor_input_token_cap,
            )
        ):
            raise ValueError("nonnegative token budgets required")


class ExecutionSelector:
    def __init__(self, budget, estimator):
        self.budget, self.estimator = budget, estimator
        self.trace = ()

    def cost(self, value):
        n = self.estimator(json.dumps(thaw(value), ensure_ascii=False, sort_keys=True))
        if type(n) is not int or n < 0:
            raise ValueError("invalid estimate")
        return n

    def select(self, bundle, *, mandatory=(), reserved_actor_tokens=0):
        b = self.budget
        if type(reserved_actor_tokens) is not int or reserved_actor_tokens < 0:
            raise ValueError("invalid reserved actor tokens")
        unresolved = sorted(
            set(bundle.missing_ids)
            | set(bundle.conflict_ids)
            | {r.id for r in bundle.requirements if r.required}
        )
        # Required slots always survive, including explicit unknowns and conflicts.
        payload = {
            "requirements": [r.id for r in bundle.requirements if r.required],
            "mandatory": list(mandatory),
            "unresolved": unresolved,
            "evidence": [],
        }
        base_cost = self.cost(payload)
        if (
            base_cost > b.mandatory_tokens
            or base_cost + reserved_actor_tokens > b.actor_input_token_cap
        ):
            raise BudgetUnsatisfiable(detail="mandatory context cannot fit")
        required = {r.subject for r in bundle.requirements if r.required}
        conflicted = {r.subject for r in bundle.requirements if r.id in bundle.conflict_ids}
        candidates, trace, seen = [], [], set()
        for frozen in bundle.excerpts:
            e = thaw(frozen)
            key = (e["scope"], e["subject"], e["content_hash"])
            ref = digest(
                freeze(
                    (
                        e["source_id"],
                        e["source_revision"],
                        e["scope"],
                        e["span_start"],
                        e["span_end"],
                        e["content_hash"],
                    )
                )
            )
            if key in seen or e["subject"] in conflicted:
                trace.append({"ref": ref, "decision": "duplicate" if key in seen else "conflict"})
                continue
            seen.add(key)
            item = {"subject": e["subject"], "value": e["text"], "ref": ref, "support": "observed"}
            candidates.append((e["subject"] not in required, self.cost(item), ref, item))
        evidence_cost = optional_cost = 0
        for optional, cost, ref, item in sorted(candidates, key=lambda c: c[:3]):
            trial = {**payload, "evidence": [*payload["evidence"], item]}
            selected_subjects = {e["subject"] for e in trial["evidence"]}
            trial["unresolved"] = sorted(
                set(bundle.missing_ids)
                | set(bundle.conflict_ids)
                | {
                    r.id
                    for r in bundle.requirements
                    if r.required and r.subject not in selected_subjects
                }
            )
            fits = (
                evidence_cost + cost <= b.evidence_tokens
                and (not optional or optional_cost + cost <= b.optional_tokens)
                and self.cost(trial) + reserved_actor_tokens <= b.actor_input_token_cap
            )
            trace.append(
                {
                    "ref": ref,
                    "decision": "included" if fits else "evidence_budget",
                    "estimated_tokens": cost,
                }
            )
            if fits:
                payload = trial
                evidence_cost += cost
                optional_cost += cost if optional else 0
        self.trace = freeze(trace)
        return freeze(payload)


class EvidenceReader:
    """Re-read a bound reference through the authorized navigator; never serve cached text."""

    def __init__(self, navigator, bundle):
        self.navigator = navigator
        self.refs = {
            digest(
                freeze(
                    (
                        h.source_id,
                        h.source_revision,
                        h.scope,
                        h.span_start,
                        h.span_end,
                        h.content_hash,
                    )
                )
            ): h
            for h in bundle.hits
        }

    async def read(self, ref, scope):
        from ..errors import AccessRevoked
        from .research import EvidenceResolver

        hit = self.refs.get(ref)
        if hit is None or hit.scope != scope:
            raise AccessRevoked()
        source = await self.navigator.read(hit.source_id, hit.source_revision, scope)
        if EvidenceResolver().validate(hit, source) is None:
            raise AccessRevoked(detail="stale or unbound evidence")
        return {
            "ref": ref,
            "text": source.text[hit.span_start : hit.span_end],
            "support": "observed",
        }
