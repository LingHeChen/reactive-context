"""Deterministic coverage with bounded, host-authorized navigation.

Navigator.read must recheck ACL and current revision on every call. Navigation summaries
are hints only; direct evidence must bind to an immutable raw source span.
"""

import asyncio
from dataclasses import replace
from itertools import islice
from time import monotonic
from typing import Protocol

from ..core.snapshot import digest, freeze
from ..errors import AccessRevoked
from .model import (
    EvidenceBundle,
    EvidenceHit,
    EvidenceRequirement,
    ResearchBudget,
    SlotPlanner,
    SourceSnapshot,
)


class EvidenceNavigator(Protocol):
    async def search(
        self, requirements: tuple[EvidenceRequirement, ...], *, depth: str, limit: int
    ) -> tuple[EvidenceHit, ...]: ...

    async def read(self, source_id: str, revision: str, scope: str) -> SourceSnapshot: ...


class EvidenceResolver:
    def validate(self, hit, source):
        if (source.id, source.revision, source.scope) != (
            hit.source_id,
            hit.source_revision,
            hit.scope,
        ) or not source.current:
            return None
        if hit.span_end > len(source.text):
            return None
        if digest(source.text[hit.span_start : hit.span_end]) != hit.content_hash:
            return None
        return next(
            (
                b
                for b in source.bindings
                if (b.subject, b.start, b.end) == (hit.subject, hit.span_start, hit.span_end)
                and b.confirmed
                and hit.evidence_kind == "direct"
            ),
            None,
        )


class SufficiencyEvaluator:
    def evaluate(self, requirements, validated):
        # Explicit replacement edges only, never timestamp wins. Replacements are scoped.
        replaced = {(s.scope, old_id, rev) for _, s, _ in validated for old_id, rev in s.supersedes}
        active = [(h, s, b) for h, s, b in validated if (s.scope, s.id, s.revision) not in replaced]
        covered, conflicts = set(), set()
        for req in requirements:
            values = {
                b.value for h, _, b in active if (h.subject, h.scope) == (req.subject, req.scope)
            }
            if len(values) > 1:
                conflicts.add(req.id)
            elif values:
                covered.add(req.id)
        # A matching field without its dependencies is not covered.
        while True:
            reduced = {
                r.id for r in requirements if r.id in covered and set(r.depends_on) <= covered
            }
            if reduced == covered:
                break
            covered = reduced
        missing = {r.id for r in requirements} - covered - conflicts
        required = {r.id for r in requirements if r.required}
        status = (
            "unknown"
            if not required
            else "conflicting"
            if conflicts
            else "incomplete"
            if required & missing
            else "sufficient"
        )
        return tuple(sorted(covered)), tuple(sorted(missing)), tuple(sorted(conflicts)), status


class EvidenceResearcher:
    def __init__(
        self,
        navigator: EvidenceNavigator,
        *,
        budget=None,
        estimator=None,
        policy_revision="coverage-v1",
        session=None,
        index_revision=None,
    ):
        self.navigator = navigator
        self.budget = budget or ResearchBudget()
        # Conservative bytes upper bound. No hidden model calls.
        self.estimator = estimator or (lambda text: len(text.encode("utf-8")))
        self.policy_revision = policy_revision
        self.session, self.index_revision = session, index_revision

    async def run(self, requirements, candidates=(), *, mode="shadow"):
        requirements = tuple(requirements)
        if self.session is None or mode == "shadow":
            return await self._run(requirements, candidates, mode=mode)
        if not callable(self.index_revision):
            raise ValueError("session research requires live index revision")

        async def operation():
            result = await self._run(requirements, candidates, mode=mode)
            # Session execute reserves one step; count every additional search as well.
            assert self.session is not None
            self.session.steps += max(0, result.search_rounds - 1)
            return result

        async def validate(bundle):
            return tuple(h.identity for h in bundle.hits)

        return await self.session.execute(
            "coverage",
            slots=tuple(r.id for r in requirements),
            scope=tuple(sorted({r.scope for r in requirements})),
            index_revision=str(self.index_revision()),
            policy_revision=self.policy_revision,
            operation=operation,
            validate=validate,
            input_tokens=self.budget.max_tokens,
            coverage=lambda bundle: bundle.covered_ids,
        )

    async def _run(self, requirements, candidates=(), *, mode="shadow"):
        if mode not in {"shadow", "dynamic", "research"}:
            raise ValueError("unknown research mode")
        requirements = SlotPlanner().plan(requirements)
        budget = self.budget
        if self.session is not None and mode != "shadow":
            budget = replace(
                budget,
                max_rounds=min(
                    budget.max_rounds, self.session.budget.max_steps - self.session.steps + 1
                ),
            )
        start = monotonic()
        trace, validated, hits = [], [], []
        seen = set()
        rounds = reads = io = tokens = 0
        stalls = 0
        previous_validated = 0
        previous_covered = set()
        stop = "no_more_candidates"
        resolver, evaluator = EvidenceResolver(), SufficiencyEvaluator()
        covered, missing, conflicts, status = evaluator.evaluate(requirements, validated)
        pending = tuple(islice(iter(candidates), budget.max_k + 1))
        if len(pending) > budget.max_k:
            raise ValueError("initial candidates exceed max_k")
        try:
            async with asyncio.timeout(budget.timeout_seconds):
                while True:
                    for hit in pending:
                        if hit.identity in seen:
                            continue
                        seen.add(hit.identity)
                        if not any(
                            (r.subject, r.scope) == (hit.subject, hit.scope) for r in requirements
                        ):
                            trace.append({"hit": hit.identity, "decision": "unrelated"})
                            continue
                        if io >= budget.max_io:
                            stop = "budget"
                            break
                        io += 1
                        reads += 1
                        try:
                            source = await self.navigator.read(
                                hit.source_id, hit.source_revision, hit.scope
                            )
                        except (AccessRevoked, KeyError):
                            trace.append({"hit": hit.identity, "decision": "unavailable"})
                            continue
                        cost = self.estimator(source.text)
                        if not isinstance(cost, int) or cost < 0:
                            raise ValueError("invalid token estimate")
                        if tokens + cost > budget.max_tokens:
                            stop = "budget"
                            break
                        tokens += cost
                        binding = resolver.validate(hit, source)
                        trace.append(
                            {"hit": hit.identity, "decision": "bound" if binding else "unbound"}
                        )
                        if binding:
                            validated.append((hit, source, binding))
                            hits.append(hit)
                    covered, missing, conflicts, status = evaluator.evaluate(
                        requirements, validated
                    )
                    if self.session is not None and rounds:
                        delta = len(validated) - previous_validated
                        stalls = 0 if delta else stalls + 1
                        trace.append(
                            {
                                "coverage_delta": len(set(covered) - previous_covered),
                                "new_evidence_hashes": [
                                    h.identity for h, _, _ in validated[previous_validated:]
                                ],
                            }
                        )
                        if stalls >= 2:
                            stop = "no_new_validated_evidence"
                            break
                    previous_validated = len(validated)
                    previous_covered = set(covered)
                    if status == "sufficient":
                        stop = "sufficient"
                        break
                    if stop == "budget" or mode == "shadow":
                        break
                    if rounds >= budget.max_rounds or io >= budget.max_io:
                        stop = "budget"
                        break
                    # B2 expands candidates; B3 navigates summary -> lexical -> raw depth.
                    depth = (
                        "lexical"
                        if mode == "dynamic"
                        else ("summary", "lexical", "raw")[min(rounds, 2)]
                    )
                    rounds += 1
                    io += 1
                    unresolved = tuple(r for r in requirements if r.id not in covered)
                    limit = min(budget.max_k, budget.initial_k * 2 ** (rounds - 1))
                    pending = tuple(
                        await self.navigator.search(unresolved, depth=depth, limit=limit)
                    )
                    if len(pending) > limit:
                        raise ValueError("navigator violated candidate limit")
                    trace.append(
                        {
                            "depth": depth,
                            "limit": limit,
                            "queries": tuple(r.id for r in unresolved),
                            "candidates": tuple(h.identity for h in pending),
                        }
                    )
                    if not any(h.identity not in seen for h in pending):
                        if mode == "research" and rounds < min(3, budget.max_rounds):
                            continue
                        stop = "no_more_candidates"
                        break
        except TimeoutError:
            stop = "timeout"
        covered, missing, conflicts, status = evaluator.evaluate(requirements, validated)
        if status == "sufficient":
            stop = "sufficient"
        return EvidenceBundle(
            requirements,
            tuple(hits),
            covered,
            missing,
            conflicts,
            status,
            stop,
            digest(freeze((requirements, trace, self.policy_revision, mode))),
            rounds,
            reads,
            io,
            tokens,
            (monotonic() - start) * 1000,
            tuple(trace),
            mode,
            self.policy_revision,
            tuple(
                {
                    "source_id": h.source_id,
                    "source_revision": h.source_revision,
                    "scope": h.scope,
                    "subject": h.subject,
                    "span_start": h.span_start,
                    "span_end": h.span_end,
                    "content_hash": h.content_hash,
                    "text": s.text[h.span_start : h.span_end],
                }
                for h, s, _ in validated
                if (s.scope, s.id, s.revision)
                not in {
                    (new.scope, old_id, old_rev)
                    for _, new, _ in validated
                    for old_id, old_rev in new.supersedes
                }
            ),
        )

    async def revalidate(self, bundle):
        """Recheck all admitted source spans immediately before compile/action.

        This is optimistic validation, not a transaction with an external side effect.
        No cached ACL decision survives between requests.
        """
        remaining = replace(
            self.budget,
            max_rounds=max(0, self.budget.max_rounds - bundle.search_rounds),
            max_io=max(0, self.budget.max_io - bundle.io_count),
            max_tokens=max(0, self.budget.max_tokens - bundle.tokens),
            timeout_seconds=max(0.000001, self.budget.timeout_seconds - bundle.latency_ms / 1000),
        )
        fresh = await EvidenceResearcher(
            self.navigator,
            budget=remaining,
            estimator=self.estimator,
            policy_revision=self.policy_revision,
        ).run(bundle.requirements, bundle.hits, mode="shadow")
        return replace(
            fresh,
            mode=bundle.mode,
            search_rounds=bundle.search_rounds + fresh.search_rounds,
            source_reads=bundle.source_reads + fresh.source_reads,
            io_count=bundle.io_count + fresh.io_count,
            tokens=bundle.tokens + fresh.tokens,
            latency_ms=bundle.latency_ms + fresh.latency_ms,
            trace=(*bundle.trace, *fresh.trace),
        )
