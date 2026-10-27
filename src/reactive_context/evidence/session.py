"""Task-local stop state. Cache contains decisions, never reusable ACL grants or payloads."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from time import monotonic

from ..core.snapshot import digest, freeze


class EvidenceStatus(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    UNRECOVERABLE = "unrecoverable"
    CONFLICTED = "conflicted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INSUFFICIENT = "insufficient"


class ResearchStopped(Exception):
    """Retrieval stop signal; a bounded finish handler can close the branch safely.

    Intentionally not ContextError: tool error-to-text handlers must not restart the loop.
    """

    def __init__(self, reason, status=EvidenceStatus.INSUFFICIENT):
        self.reason, self.status = reason, status
        super().__init__(reason)


@dataclass(frozen=True)
class SessionBudget:
    max_steps: int = 8
    max_input_tokens: int = 8192
    max_output_tokens: int = 2048
    max_total_tokens: int = 10240
    max_cost_usd: float = 0
    max_wall_time_s: float = 10
    max_same_query_retries: int = 1
    negative_ttl_s: float = 30

    def __post_init__(self):
        for n in (
            self.max_steps,
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_total_tokens,
            self.max_same_query_retries,
        ):
            if type(n) is not int or n < 0:
                raise ValueError("nonnegative integer budget required")
        for n in (self.max_cost_usd, self.max_wall_time_s, self.negative_ttl_s):
            if not isfinite(n) or n < 0:
                raise ValueError("finite nonnegative budget required")


def normalized(value):
    return " ".join(value.casefold().split())


class ResearchSession:
    def __init__(self, task_id, snapshot_id, principal_scope, *, budget=None, clock=monotonic):
        if not task_id or not snapshot_id or not principal_scope:
            raise ValueError("session identity must include task, snapshot and principal scope")
        self.task_id, self.snapshot_id, self.principal_scope = task_id, snapshot_id, principal_scope
        self.budget = budget or SessionBudget()
        self.clock, self.started = clock, clock()
        self.steps = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0
        self.trace = []
        self._negative, self._tries, self._stalls, self._seen = {}, {}, {}, set()
        self._covered = set()
        self._branch_steps = {}
        self.read_steps = self.search_steps = self.research_steps = 0
        self._lock = asyncio.Lock()
        self.finalizing = False
        self.finish_handler_attached = False
        self.stop_feedback_count = 0
        self.unresolved = {}
        self.observed_refs = set()

    def begin_finish(self, reason):
        if not self.finalizing:
            self.trace.append({"decision": "finish_requested", "reason": reason})
        self.finalizing = True

    def stop_feedback(self, error, slots):
        """Report a local stop; only the host task budget can request global finish."""
        self.stop_feedback_count += 1
        for slot in slots:
            self.unresolved[slot] = {"status": error.status.value, "reason": error.reason}
        self.trace.append(
            {
                "decision": "retrieval_stopped",
                "reason": error.reason,
                "slots": tuple(slots),
                "finalizing": self.finalizing,
            }
        )
        return {
            "status": "retrieval_stopped",
            "reason": error.reason,
            "unresolved": dict(self.unresolved),
            "finalize": self.finalizing,
            "instruction": "Do not repeat this retrieval without new evidence or a source change. "
            "Other requirements, source reads and business tools remain available. "
            "Report this dependency as unknown if it cannot be resolved.",
        }

    def handoff(self):
        return {
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "unresolved": dict(self.unresolved),
            "observed_refs": sorted(self.observed_refs),
            "finalizing": self.finalizing,
            "observations_are_committed_decisions": False,
        }

    def fingerprint(self, query, slots, scope, index_revision, policy_revision, filters):
        if not index_revision or not policy_revision:
            raise ValueError("versioned index and retrieval policy required")
        return digest(
            freeze(
                (
                    self.task_id,
                    self.snapshot_id,
                    self.principal_scope,
                    normalized(query),
                    sorted(set(slots)),
                    scope,
                    filters,
                    index_revision,
                    policy_revision,
                )
            )
        )

    def _check(
        self, input_tokens=0, output_tokens=0, cost_usd=0, *, work_kind="research", branch_steps=0
    ):
        b = self.budget
        if (
            (work_kind == "research" and self.research_steps >= b.max_steps)
            or (work_kind == "search" and branch_steps >= b.max_steps)
            or self.clock() - self.started >= b.max_wall_time_s
            or self.input_tokens + input_tokens > b.max_input_tokens
            or self.output_tokens + output_tokens > b.max_output_tokens
            or self.input_tokens + self.output_tokens + input_tokens + output_tokens
            > b.max_total_tokens
            or self.cost_usd + cost_usd > b.max_cost_usd
        ):
            raise ResearchStopped("research_budget", EvidenceStatus.BUDGET_EXHAUSTED)

    async def execute(
        self,
        query,
        *,
        slots,
        scope,
        index_revision,
        policy_revision,
        operation,
        validate,
        filters=(),
        input_tokens=0,
        output_tokens=0,
        cost_usd=0,
        new_clue=False,
        coverage=None,
        progress_kind="support",
        work_kind="research",
    ):
        """Reserve conservative costs BEFORE work; validate returns support hashes.

        Paid operations must declare a strict upper bound and share the host task ledger.
        This session never makes model calls. Changed query text alone is not progress.
        """
        if any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens)):
            raise ValueError("invalid token reservation")
        if not isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("invalid cost reservation")
        if progress_kind not in {"support", "observation"}:
            raise ValueError("invalid progress kind")
        if work_kind not in {"research", "search", "read"}:
            raise ValueError("invalid work kind")
        slots = tuple(sorted(set(slots)))
        key = self.fingerprint(query, slots, scope, index_revision, policy_revision, filters)
        stall_key = (slots, scope, index_revision, policy_revision)
        branch_key = (slots, scope, policy_revision)
        async with self._lock:
            if self.finalizing:
                raise ResearchStopped("finish_requested")
            self._check(
                input_tokens,
                output_tokens,
                cost_usd,
                work_kind=work_kind,
                branch_steps=self._branch_steps.get(branch_key, 0),
            )
            expiry = self._negative.get(key)
            if expiry is not None and expiry > self.clock():
                self.trace.append({"fingerprint": key, "decision": "negative_cache"})
                raise ResearchStopped("negative_cache", EvidenceStatus.NOT_FOUND)
            if expiry is not None:
                self._negative.pop(key, None)
                self._tries.pop(key, None)
                self._stalls.pop(stall_key, None)
            if work_kind == "read" and self._tries.get(key, 0):
                raise ResearchStopped("already_read_current_version")
            if self._tries.get(key, 0) > self.budget.max_same_query_retries:
                raise ResearchStopped("same_query_limit")
            if self._stalls.get(stall_key, 0) >= 2 and not new_clue:
                raise ResearchStopped("no_new_validated_evidence")
            self.steps += 1
            if work_kind == "search":
                self.search_steps += 1
                self._branch_steps[branch_key] = self._branch_steps.get(branch_key, 0) + 1
            elif work_kind == "read":
                self.read_steps += 1
            else:
                self.research_steps += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += cost_usd
            self._tries[key] = self._tries.get(key, 0) + 1
            remaining = max(0, self.budget.max_wall_time_s - (self.clock() - self.started))
            try:
                async with asyncio.timeout(remaining):
                    result = await operation()
                    hashes = set(await validate(result))
            except TimeoutError as exc:
                raise ResearchStopped("deadline", EvidenceStatus.BUDGET_EXHAUSTED) from exc
            covered_slots = set(coverage(result)) if coverage is not None else None
            coverage_delta = (
                len(covered_slots - self._covered) if covered_slots is not None else "NOT_MEASURED"
            )
            if covered_slots is not None:
                self._covered.update(covered_slots)
            new = hashes - self._seen
            self._seen.update(hashes)
            if progress_kind == "observation":
                self.observed_refs.update(hashes)
            if new:
                for slot in slots:
                    if progress_kind == "support":
                        self.unresolved.pop(slot, None)
                    elif slot in self.unresolved:
                        self.unresolved[slot] = {
                            "status": EvidenceStatus.INSUFFICIENT.value,
                            "reason": "new_observation_requires_validation",
                        }
                # Actual progress from a scoped read resets stalls in the same branch.
                for prior in self._stalls:
                    if prior[:2] == (slots, scope):
                        self._stalls[prior] = 0
            self._stalls[stall_key] = 0 if new else self._stalls.get(stall_key, 0) + 1
            if not new:
                self._negative[key] = self.clock() + self.budget.negative_ttl_s
            self.trace.append(
                {
                    "fingerprint": key,
                    "slots": slots,
                    "source_scope": scope,
                    "index_revision": index_revision,
                    "new_evidence_hashes": sorted(new) if progress_kind == "support" else [],
                    "new_observation_hashes": sorted(new) if progress_kind == "observation" else [],
                    "progress_kind": progress_kind,
                    "work_kind": work_kind,
                    "coverage_delta": coverage_delta,
                    "status": (
                        "found"
                        if covered_slots is not None and set(slots) <= covered_slots
                        else "insufficient"
                        if new
                        else "not_found"
                    ),
                }
            )
            return result

    def unrecoverable(self, *, requirement_id, proof_kind, proof_ref):
        if proof_kind not in {"closed_source_scan", "source_artifact_missing"} or not proof_ref:
            raise ValueError("unrecoverable requires auditable host proof")
        self.trace.append(
            {
                "requirement_id": requirement_id,
                "status": "unrecoverable",
                "proof_kind": proof_kind,
                "proof_ref": proof_ref,
            }
        )
        return EvidenceStatus.UNRECOVERABLE

    def usage(self):
        return {
            "steps": self.steps,
            "search_steps": self.search_steps,
            "read_steps": self.read_steps,
            "research_steps": self.research_steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_upper_bound_usd": self.cost_usd,
            "elapsed_s": self.clock() - self.started,
        }
