"""Native tools for a host-bound structured work state. No permission-minting tool."""

from functools import wraps

from pydantic_ai.toolsets import FunctionToolset

from ..errors import ContextError
from ..work_state import StateOperation, StateUpdateRejected
from .pydantic_ai import managed_result


def work_state_tools(manager, scope):
    state = manager.work_state
    if state is None:
        raise ValueError("work state is not configured")

    def guarded(function):
        @wraps(function)
        async def call(*args, **kwargs):
            try:
                return await function(*args, **kwargs)
            except StateUpdateRejected as error:
                return {
                    "error": error.code,
                    "reasons": list(error.reasons),
                    "changes": [{"path": c.path, "action": c.action} for c in error.changes],
                    "validation": error.validation,
                }
            except ContextError as error:
                return {"error": error.code}
            except ValueError:
                return {"error": "invalid_state_operations"}

        return call

    @guarded
    async def context_get_work_state():
        """Read current structured work, its version and host-authorized field changes.
        This is current state, not a history summary. The host alone can grant changes.
        """
        snapshot = await manager.next_context(scope)
        part = snapshot.part(state.key)
        return managed_result(part.value, part)

    @guarded
    async def context_preview_work_state(
        operations: list[StateOperation],
        expected_revision: int,
        permit_id: str | None = None,
        evidence: list[str] | None = None,
    ):
        """Preview exact field differences and blocked removals/clears without writing.
        Patches retain omitted fields. A parent set may delete children and be rejected.
        """
        return state.patch(
            scope,
            operations,
            expected_revision=expected_revision,
            permit_id=permit_id,
            preview=True,
            evidence=evidence or (),
        )

    @guarded
    async def context_update_work_state(
        operations: list[StateOperation],
        expected_revision: int,
        permit_id: str | None = None,
        evidence: list[str] | None = None,
    ):
        """Patch only fields explicitly authorized by a current host-issued permit.
        Use the observed revision. Removal and clearing require separate grants.
        Rejected or stale patches leave every field unchanged. Never reset a field
        merely because the latest message did not mention it. Configured value
        checks also run here: failed or pending validation cannot be bypassed.
        """
        return state.patch(
            scope,
            operations,
            expected_revision=expected_revision,
            permit_id=permit_id,
            evidence=evidence or (),
        )

    @guarded
    async def context_propose_state_candidate(
        operations: list[StateOperation],
        evidence: list[str],
        expected_revision: int,
        permit_id: str | None = None,
    ):
        """Stage a candidate patch and check it without activating it or creating Facts.
        Supply observed source IDs; failed/pending issues include expected and actual values.
        Only a candidate with passed host value checks can be committed.
        """
        return state.candidates.propose(
            scope, operations, evidence, expected_revision=expected_revision, permit_id=permit_id
        )

    @guarded
    async def context_get_state_candidate(candidate_id: str):
        """Read your own scoped candidate, its patch, base versions and last check."""
        return state.candidates.read(scope, candidate_id)

    @guarded
    async def context_check_state_candidate(candidate_id: str):
        """Recheck a staged candidate against current requirements and observed sources.
        A prior ready status is not a permanent validation guarantee.
        """
        return state.candidates.check(scope, candidate_id)

    @guarded
    async def context_commit_state_candidate(candidate_id: str):
        """Recheck then atomically commit a ready candidate; pending/failed/stale stays inactive.
        Same candidate commits once. No full-history extraction is needed for its state delta.
        """
        return state.candidates.commit(scope, candidate_id)

    @guarded
    async def context_discard_state_candidate(candidate_id: str):
        """Discard an uncommitted candidate without deleting current state or prior evidence."""
        return state.candidates.discard(scope, candidate_id)

    @guarded
    async def context_get_state_deltas(after_revision: int = 0, limit: int = 10):
        """Read committed field increments only when past state changes are needed.
        These are state observations with validation scope, not universal factual truth.
        Rejected/pending candidates are never in this memory stream.
        """
        return state.deltas(scope, after_revision=after_revision, limit=limit)

    return FunctionToolset(
        [
            context_get_work_state,
            context_preview_work_state,
            context_update_work_state,
            context_propose_state_candidate,
            context_get_state_candidate,
            context_check_state_candidate,
            context_commit_state_candidate,
            context_discard_state_candidate,
            context_get_state_deltas,
        ],
        sequential=True,
        id="reactive-work-state",
    )
