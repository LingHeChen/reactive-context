"""Explicit context tools with host-bound identity/scope and content-free failure feedback."""

from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from ..core.snapshot import digest, freeze, thaw
from ..errors import ContextBudgetExceeded, ContextError
from ..evidence.session import ResearchStopped
from ..fact_checks import FactAdmissionRejected
from ..history import HistoryGroup
from ..objective import WorkObjective
from .pydantic_ai import managed_result

CONTROL_TOOL_NAMES = frozenset(
    {
        "context_query",
        "context_load",
        "context_activate",
        "context_deactivate",
        "context_save_fact",
        "context_propose_fact",
        "context_revise_fact",
        "context_commit_fact",
        "context_retire_fact",
        "context_search_facts",
        "context_inspect",
        "context_request_compaction",
        "context_history_lookup",
        "context_history_search",
        "context_recent_history",
        "context_get_objective",
        "context_update_objective",
        "context_refresh_objective",
        "context_get_work_state",
        "context_preview_work_state",
        "context_update_work_state",
        "context_propose_state_candidate",
        "context_get_state_candidate",
        "context_check_state_candidate",
        "context_commit_state_candidate",
        "context_discard_state_candidate",
        "context_get_state_deltas",
    }
)


@dataclass(kw_only=True)
class _HistoryFallbackTool(ToolsetTool[Any]):
    source: ToolsetTool[Any]


@dataclass
class HistoryFallbackToolset(WrapperToolset[Any]):
    """Supply two archive readers, respecting explicitly installed host tools."""

    factory: Any = None

    async def get_tools(self, ctx):
        tools = dict(await self.wrapped.get_tools(ctx))
        if {"context_history_search", "context_history_lookup"} <= tools.keys():
            return tools
        fallback = await self.factory(ctx).get_tools(ctx)
        for name in ("context_history_search", "context_history_lookup"):
            if name not in tools:
                tool = fallback[name]
                tools[name] = _HistoryFallbackTool(
                    toolset=tool.toolset,
                    tool_def=tool.tool_def,
                    max_retries=tool.max_retries,
                    args_validator=tool.args_validator,
                    args_validator_func=tool.args_validator_func,
                    source=tool,
                )
        return tools

    async def call_tool(self, name, tool_args, ctx, tool):
        if isinstance(tool, _HistoryFallbackTool):
            return await tool.source.toolset.call_tool(name, tool_args, ctx, tool.source)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)


def context_tools(
    manager,
    scope,
    *,
    history=None,
    research_session=None,
    index_revision=None,
    evidence_reader=None,
    requirement_key=None,
):
    """Bind once per trusted session; neither principal nor scope is an LLM tool argument."""
    history = history if history is not None else getattr(manager, "history", None)
    if history is False:
        history = None

    if research_session is not None and index_revision is None and not hasattr(manager, "store"):
        raise ValueError("guarded retrieval requires store content revisions or a host callback")

    def current_events(ctx):
        if history is None:
            return ()
        from .history import history_groups

        messages = ctx.messages
        starts = [
            i
            for i, message in enumerate(messages)
            if isinstance(message, ModelRequest)
            and any(isinstance(p, UserPromptPart) for p in message.parts)
            and not any(isinstance(p, ToolReturnPart) for p in message.parts)
        ]
        current = messages[starts[-1] :] if starts else messages
        return tuple(
            history.event_id(event) for group in history_groups(current) for event in group.events
        )

    async def retrieve(ctx, kind, arguments, operation, observe, *, intent=None):
        if research_session is None:
            return await operation()
        is_read = kind.endswith("_read")
        # Each read owns its source identity; never inherit the last search's requirement.
        slots = (kind + ":" + digest(freeze(arguments)),) if is_read else ("unresolved_context",)
        if not is_read and intent is not None and requirement_key is not None:
            resolved = tuple(requirement_key(intent))
            if resolved:
                slots = tuple(sorted(set(resolved)))
        excluded = current_events(ctx)
        revision = (
            str(index_revision())
            if index_revision is not None
            else manager.store.retrieval_revision(
                manager.principal, scope, exclude_event_ids=excluded
            )
        )

        async def validate(result):
            return observe(result)

        try:
            return await research_session.execute(
                kind + ":" + digest(freeze(arguments)),
                slots=slots,
                scope=scope,
                index_revision=revision,
                policy_revision="retrieval-recovery-v3",
                filters=(kind,),
                operation=operation,
                validate=validate,
                progress_kind="observation",
                work_kind="read" if is_read else "search",
            )
        except ResearchStopped as error:
            if not research_session.finish_handler_attached:
                # Legacy callers without a bounded finish handler must not enter a retry loop.
                raise
            return research_session.stop_feedback(error, slots)

    def located(rows):
        # Provenance of authorized discovery, never semantic validation of the value.
        return tuple(
            "located:"
            + digest(
                freeze(
                    {
                        k: row[k]
                        for k in (
                            "id",
                            "part_id",
                            "revision",
                            "record_id",
                            "event_id",
                            "sources",
                            "text",
                        )
                        if k in row
                    }
                )
            )
            for row in rows
        )

    def guarded(function):
        @wraps(function)
        async def call(*args, **kwargs):
            try:
                return await function(*args, **kwargs)
            except ContextBudgetExceeded as error:
                return {
                    "error": error.code,
                    "parts": list(error.parts),
                    "estimated_tokens": error.estimated_tokens,
                    "suggested_scopes": list(error.suggested_scopes),
                }
            except FactAdmissionRejected as error:
                return {"error": error.code, "decision": error.decision, "reasons": error.reasons}
            except ContextError as error:
                return {"error": error.code}

        return call

    @guarded
    async def context_query(
        ctx: RunContext, intent: str = "", kind: str | None = None, after_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Enumerate authorized catalog metadata by ID, without loading payloads.
        Use context_search_facts for relevant Fact matches.
        This endpoint supports catalog pagination.
        """
        return await retrieve(
            ctx,
            "catalog",
            (intent, kind, after_id),
            lambda: manager.query(scope, intent, kind=kind, after_id=after_id),
            located,
            intent=intent,
        )

    @guarded
    async def context_load(ctx: RunContext, part_id: str):
        """Read an authorized source for the current user turn, without activating it.
        Call context_activate to keep it available across later user turns.
        """
        loaded = await retrieve(
            ctx,
            "fact_read",
            (part_id,),
            lambda: manager.load(scope, part_id),
            lambda item: (
                "read:"
                + digest(
                    freeze(
                        (
                            item.snapshot.resource_id,
                            item.snapshot.revision,
                            item.snapshot.scope_id,
                            item.snapshot.value,
                        )
                    )
                ),
            ),
        )
        if isinstance(loaded, dict):
            return loaded
        return managed_result(
            {
                "part_id": loaded.id,
                "revision": loaded.snapshot.revision,
                "value": thaw(loaded.snapshot.value),
                "evidence_id": loaded.evidence_id,
            },
            loaded.snapshot,
            read_part_id=loaded.id,
            read_scope=scope,
        )

    @guarded
    async def context_activate(part_id: str, policy: str = "live", max_age: float = 0) -> dict:
        """Activate a resource or fact. Oversized resources return scope-narrowing feedback."""
        return asdict(await manager.activate(scope, part_id, policy=policy, max_age=max_age))

    @guarded
    async def context_deactivate(part_id: str) -> dict:
        """Stop including an optional part in future calls. Mandatory context stays active."""
        await manager.deactivate(scope, part_id)
        return {"deactivated": part_id}

    @guarded
    async def context_save_fact(
        schema_id: str,
        key: str,
        value: dict[str, Any],
        evidence: list[str],
        expected_revision: int | None = None,
        schema_version: int = 1,
    ):
        """Save durable new evidence only when it changes a stable Fact key.
        Host admission checks run before storage. Transient/unsupported claims may be skipped
        or rejected; identical replays are unchanged. To update an existing value, supply its
        current expected_revision. This does not activate the Fact or grant instruction authority.
        """
        return await manager.save_fact(
            scope,
            schema_id,
            key,
            value,
            evidence,
            expected_revision=expected_revision,
            schema_version=schema_version,
        )

    @guarded
    async def context_propose_fact(
        schema_id: str, key: str, value: dict[str, Any], evidence: list[str]
    ) -> dict:
        """Propose an unverified fact using a host-registered schemaand observed evidence
        ids.
        """
        proposal = manager.store.propose_fact(
            manager.principal, scope, schema_id, key, value, evidence
        )
        return {"proposal_id": proposal, "verification": "unverified"}

    @guarded
    async def context_revise_fact(
        part_id: str, expected_revision: int, value: dict[str, Any], evidence: list[str]
    ) -> dict:
        """Propose a checked revision; concurrent changes cause version_conflict on commit."""
        meta = manager.store.metadata(manager.principal, part_id)
        if meta["scope_id"] != scope:
            from ..errors import AccessRevoked

            raise AccessRevoked()
        proposal = manager.store.propose_fact(
            manager.principal,
            scope,
            meta["schema_id"],
            meta["key"],
            value,
            evidence,
            schema_version=meta["schema_version"],
            expected_revision=expected_revision,
        )
        return {"proposal_id": proposal, "verification": "unverified"}

    @guarded
    async def context_commit_fact(proposal_id: str) -> dict:
        """Commit an unverified user-authority fact. Activate relevant facts separately
        with context_activate so future short-history requests retain them.
        """
        proposal = manager.store.db.execute(
            "SELECT scope_id FROM proposals WHERE id=?", (proposal_id,)
        ).fetchone()
        if proposal is None or proposal[0] != scope:
            from ..errors import AccessRevoked

            raise AccessRevoked()
        record = manager.store.commit(manager.principal, proposal_id)
        return {
            "part_id": record.id,
            "revision": record.revision,
            "verification": record.verification,
        }

    @guarded
    async def context_retire_fact(
        part_id: str, status: str = "revoked", superseded_by: str | None = None
    ) -> dict:
        """Revoke, supersede or forget an owned fact; cannot retire host resources."""
        meta = manager.store.metadata(manager.principal, part_id)
        if meta["scope_id"] != scope or meta["kind"] != "fact":
            from ..errors import AccessRevoked

            raise AccessRevoked()
        manager.store.retire(manager.principal, part_id, status=status, superseded_by=superseded_by)
        return {"part_id": part_id, "status": status}

    @guarded
    async def context_search_facts(ctx: RunContext, intent: str):
        """Search authorized persistent facts. Results containmetadata, not all fact
        payloads.
        """
        return await retrieve(
            ctx,
            "fact_search",
            (intent,),
            lambda: manager.search_facts(scope, intent),
            located,
            intent=intent,
        )

    @guarded
    async def context_inspect() -> dict:
        """Explain selected parts, versions and budget without exporting sensitive payloads."""
        snapshot = await manager.next_context(scope)
        compiled = manager.compile(snapshot)
        return {
            "compile_id": compiled.compile_id,
            "revisions": dict(snapshot.revision_vector),
            "estimated_tokens": compiled.token_estimate,
            "trace": [asdict(item) for item in compiled.trace],
        }

    @guarded
    async def context_request_compaction(reason: str) -> dict:
        """Request history compaction on the next model boundary, subject to host policy."""
        if history is None:
            return {"error": "history_manager_not_configured"}
        if history.compactor is None:
            return {"error": "history_compactor_not_configured"}
        history.request_compaction(scope, reason)
        return {"requested": True}

    @guarded
    async def context_history_lookup(ctx: RunContext, event_ids: list[str]):
        """Recover exact archived evidence by id under current source permissions."""
        if history is None:
            return [{"error": "history_manager_not_configured"}]
        if len(event_ids) > 50:
            return [{"error": "lookup_limit_exceeded"}]
        events = await retrieve(
            ctx,
            "history_read",
            tuple(sorted(event_ids)),
            lambda: manager.load_history(scope, event_ids),
            lambda items: tuple("read:" + digest(freeze(item)) for item in items),
        )
        if isinstance(events, dict) and events.get("status") == "retrieval_stopped":
            return events
        return [
            {
                "event_id": event_id,
                "event": event,
                "evidence_id": manager.store.observe_evidence(
                    manager.principal, scope, "history:" + event_id, event_id
                ),
            }
            for event_id, event in zip(event_ids, events, strict=True)
        ]

    @guarded
    async def context_history_search(
        ctx: RunContext, intent: str, kind: str = "all", limit: int = 3
    ):
        """Search archived history only when current context cannot resolve the user's request.
        Use kind='summary' or 'event' to restrict search to summaries or original events.
        Results are historical evidence and are not automatically activated.
        """
        if history is None:
            return [{"error": "history_manager_not_configured"}]
        if kind not in {"all", "event", "summary"} or not 1 <= limit <= 10:
            return [{"error": "invalid_history_search_options"}]
        try:
            from .history import history_groups

            messages = ctx.messages
            starts = [
                i
                for i, message in enumerate(messages)
                if isinstance(message, ModelRequest)
                and any(isinstance(p, UserPromptPart) for p in message.parts)
                and not any(isinstance(p, ToolReturnPart) for p in message.parts)
            ]
            current = messages[starts[-1] :] if starts else messages
            excluded = [
                history.event_id(event)
                for group in history_groups(current)
                for event in group.events
            ]

            async def search():
                return await manager.search_history(
                    scope,
                    intent,
                    kind=kind,
                    limit=limit,
                    exclude_event_ids=excluded,
                    preview_chars=400,
                )

            hits = await retrieve(
                ctx, "history_search", (intent, kind, limit), search, located, intent=intent
            )
            if isinstance(hits, dict) and hits.get("status") == "retrieval_stopped":
                return hits
        except ValueError:
            return [{"error": "invalid_history_query"}]
        return [{**hit, "text": hit["text"][:400]} for hit in hits]

    @guarded
    async def context_recent_history(ctx: RunContext, limit: int = 12):
        """Read recent archived events with evidence IDs for extracting persistent Facts."""
        if history is None:
            return [{"error": "history_manager_not_configured"}]
        return await context_history_lookup(ctx, history.tail_ids(scope, limit=limit))

    @guarded
    async def read_evidence(ctx: RunContext, ref: str):
        """Read a selected evidence reference under current source permissions."""
        assert evidence_reader is not None
        reader = evidence_reader
        return await retrieve(
            ctx,
            "evidence_read",
            (ref,),
            lambda: reader.read(ref, scope),
            lambda value: ("read:" + digest(freeze(value)),),
        )

    return FunctionToolset(
        ([read_evidence] if evidence_reader is not None else [])
        + [
            context_query,
            context_load,
            context_activate,
            context_deactivate,
            context_save_fact,
            context_propose_fact,
            context_revise_fact,
            context_commit_fact,
            context_retire_fact,
            context_search_facts,
            context_inspect,
            context_request_compaction,
            context_history_lookup,
            context_history_search,
            context_recent_history,
        ],
        sequential=True,
        id="reactive-context-operations",
    )


def objective_tools(manager, scope):
    """Small, host-bound toolset; automatically installed in agent-managed objective mode."""

    def guarded(function):
        @wraps(function)
        async def call(*args, **kwargs):
            try:
                return await function(*args, **kwargs)
            except FactAdmissionRejected as error:
                return {"error": error.code, "decision": error.decision, "reasons": error.reasons}
            except ContextError as error:
                return {"error": error.code}

        return call

    history = manager.history if manager.history is not False else None

    async def current_result():
        snapshot = await manager.next_context(scope)
        part = snapshot.part(manager.objective.key)
        return managed_result(
            part.value if part.value is not None else {"revision": 0, "objective": None}, part
        )

    @guarded
    async def context_get_objective():
        """Read the current versioned work objective, not a historical summary."""
        if manager.objective is None:
            return {"error": "objective_not_configured"}
        return await current_result()

    @guarded
    async def context_update_objective(value: WorkObjective, expected_revision: int):
        """Update goal, constraints and next steps using a checked current revision.
        Cannot change host system instructions or source permissions.
        """
        if manager.objective is None:
            return {"error": "objective_not_configured"}
        manager.objective.update_from_agent(scope, value, expected_revision=expected_revision)
        return await current_result()

    @guarded
    async def context_refresh_objective():
        """Ask the configured generative updater to revise the objective from recent evidence."""
        if manager.objective is None or history is None:
            return {"error": "objective_or_history_not_configured"}
        ids = history.tail_ids(scope, limit=min(50, manager.objective.policy.max_update_groups))
        events = await manager.load_history(scope, ids)
        groups = tuple(HistoryGroup((e,)) for e in events)
        await manager.objective.refresh(
            scope,
            groups,
            reason="llm",
            estimator=manager.compiler.policy.estimator,
        )
        return await current_result()

    return FunctionToolset(
        [context_get_objective, context_update_objective, context_refresh_objective],
        sequential=True,
        id="reactive-work-objective",
    )
