"""Real archive writes, normal finishing, ACL changes and cross-turn recovery; no paid model."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import ContextManager, ContextStore, HistoryManager, Principal
from reactive_context.adapters.history import history_groups
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol
from reactive_context.adapters.recovery import ResearchRecoveryCapability
from reactive_context.errors import AccessRevoked
from reactive_context.evidence import ResearchSession, ResearchStopped, SessionBudget

HOST = Principal("host", True)
USER = Principal("user")


def setup(store):
    store.create_scope(HOST, "s")
    store.grant_scope(HOST, "s", USER.id)


async def test_real_history_writes_do_not_bypass_negative_cache_and_finish_is_valid(tmp_path):
    with ContextStore(tmp_path) as store:
        setup(store)
        async with ContextManager(store=store, principal=USER) as manager:
            session = ResearchSession("task", "snapshot", "user:s")
            recovery = ResearchRecoveryCapability(session, max_model_requests=6)
            calls = []

            def model(messages, info):
                calls.append(store.db.total_changes)
                if not info.function_tools:
                    return ModelResponse(parts=[TextPart("The prior dinner is unknown.")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "context_history_search",
                            {"intent": "missing dinner"},
                            f"call{len(calls)}",
                        )
                    ]
                )

            agent = Agent(
                FunctionModel(model),
                toolsets=[context_tools(manager, "s", research_session=session)],
                capabilities=[ReactiveContextCapability(manager, scope="s", cache=None), recovery],
            )
            result = await agent.run("Find the missing dinner")
            assert result.output == "The prior dinner is unknown."
            assert len(calls) == 6
            assert len(set(calls)) > 1  # real archive bookkeeping changed between requests
            assert session.steps == 1  # only the first actual search ran
            assert session.finalizing and recovery.finishes == 1
            validate_protocol(result.all_messages())
            assert store.db.execute("SELECT count(*) FROM history_events").fetchone()[0] > 0


async def test_fact_load_loop_is_bounded_without_promoting_observations(tmp_path):
    class Note(BaseModel):
        text: str

    with ContextStore(tmp_path) as store:
        setup(store)
        store.register_schema("note", Note)
        e = store.observe_evidence(USER, "s", "source", "1")
        fact = store.commit(
            USER, store.propose_fact(USER, "s", "note", "one", {"text": "value"}, [e])
        )
        async with ContextManager(store=store, principal=USER) as manager:
            session = ResearchSession("task", "snap", "user:s")
            recovery = ResearchRecoveryCapability(session, max_model_requests=6)
            calls = []

            def model(messages, info):
                calls.append(1)
                if not info.function_tools:
                    return ModelResponse(parts=[TextPart("Observed value; no new decisions.")])
                return ModelResponse(
                    parts=[ToolCallPart("context_load", {"part_id": fact.id}, str(len(calls)))]
                )

            result = await Agent(
                FunctionModel(model),
                toolsets=[context_tools(manager, "s", research_session=session)],
                capabilities=[ReactiveContextCapability(manager, scope="s", cache=None), recovery],
            ).run("Read fact")
            assert result.output.startswith("Observed")
            assert session.steps == 1 and len(calls) == 6
            assert session.observed_refs
            assert not session._covered
            assert store.load(USER, fact.id).verification == "unverified"
            assert all(not t.get("new_evidence_hashes") for t in session.trace)


def test_content_revision_excludes_bookkeeping_but_changes_for_content_and_acl(tmp_path):
    with ContextStore(tmp_path) as store:
        setup(store)
        history = HistoryManager(store, USER)
        messages = [ModelRequest.user_text_prompt("original"), ModelResponse(parts=[TextPart("A")])]
        history.observe("s", history_groups(messages))
        before = store.retrieval_revision(USER, "s")
        history.observe("s", history_groups(messages))
        store.observe_evidence(USER, "s", "new-receipt", "1")
        assert store.retrieval_revision(USER, "s") == before
        current = ModelRequest.user_text_prompt("current question")
        history.observe("s", history_groups([current]))
        excluded = (history.event_id(history_groups([current])[0].events[0]),)
        assert store.retrieval_revision(USER, "s", exclude_event_ids=excluded) == before
        assert store.retrieval_revision(USER, "s") != before
        before = store.retrieval_revision(USER, "s")
        store.grant_scope(HOST, "s", "other")
        assert store.retrieval_revision(USER, "s") != before
        store.revoke_scope(HOST, "s", USER.id)
        with pytest.raises(AccessRevoked):
            store.retrieval_revision(USER, "s")


async def test_finish_reserves_original_model_budget_and_rejects_hidden_tool_calls():
    executed = []
    session = ResearchSession("a", "b", "c")
    recovery = ResearchRecoveryCapability(session, max_model_requests=3, finish_requests=1)

    async def business():
        executed.append(1)
        return "observed"

    def model(messages, info):
        if not info.function_tools:
            return ModelResponse(parts=[TextPart("Done with available evidence.")])
        return ModelResponse(parts=[ToolCallPart("business", {}, str(len(executed)))])

    result = await Agent(FunctionModel(model), tools=[business], capabilities=[recovery]).run(
        "Task"
    )
    assert result.output.startswith("Done")
    assert len(executed) == 2 and recovery.requests == 3
    session = ResearchSession("d", "e", "f")
    recovery = ResearchRecoveryCapability(session)
    session.begin_finish("budget")

    def bad(messages, info):
        return ModelResponse(parts=[ToolCallPart("business", {})])

    with pytest.raises(ResearchStopped, match="tool_call_during_finish"):
        await Agent(FunctionModel(bad), tools=[business], capabilities=[recovery]).run("Task")
    assert len(executed) == 2


async def test_partial_run_is_available_for_recovery_not_a_final_decision():
    recovery = ResearchRecoveryCapability()
    calls = []

    async def read():
        return "observed raw result"

    def model(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("read", {}, "read1")])
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        await Agent(FunctionModel(model), tools=[read], capabilities=[recovery]).run(
            "current",
            message_history=[
                ModelRequest.user_text_prompt("previous"),
                ModelResponse(parts=[TextPart("previous reply")]),
            ],
        )
    current = recovery.current_messages()
    assert any(
        isinstance(p, ToolReturnPart) and p.content == "observed raw result"
        for m in current
        for p in m.parts
    )
    assert current[0].parts[0].content == "current"


def test_domain_paraphrases_share_slot_and_missing_plan_is_explicit_next_turn(tmp_path):
    from reactive_context.benchmarking.intelligence_pilot import CoverageIntegration

    integration = CoverageIntegration("dx-baseline-shadow", tmp_path, stop_budget=SessionBudget())
    records = [
        {"name": "Alice", "daily_plans": []},
        {
            "query": "I am Zoe.\nPlan dinner",
            "generated_plan": "",
            "final_status": "incomplete",
            "actual_interaction": [{"observed": "restaurant rows"}],
        },
    ]
    context = integration.recovery_context(records, "Use Zoe's day 5 dinner")
    assert context["unresolved_prior_plans"][0]["entity"] == "Zoe"
    assert context["unresolved_prior_plans"][0]["partial_observations_available"]
    assert integration.retrieval_slots(
        "Zoe day-5 dinner restaurant"
    ) == integration.retrieval_slots("dinner of Zoe on the 5th day")
    session = ResearchSession("a", "b", "c")
    integration.actor_session = session
    integration.on_actor_failure(ResearchStopped("finish_attempt_limit"))
    manager = SimpleNamespace(principal=USER)
    assert integration.context_tool_options(manager, 2)["research_session"] is not session


async def test_new_content_invalidates_negative_cache_and_new_turn_has_fresh_budget(tmp_path):
    with ContextStore(tmp_path) as store:
        setup(store)
        history = HistoryManager(store, USER)
        session = ResearchSession("task", "snapshot", "user:s")
        calls = []

        async def search():
            calls.append(1)
            return []

        async def observe(result):
            return result

        async def execute(target):
            return await target.execute(
                "missing",
                slots=("dinner",),
                scope="s",
                index_revision=store.retrieval_revision(USER, "s"),
                policy_revision="v2",
                operation=search,
                validate=observe,
                progress_kind="observation",
            )

        await execute(session)
        with pytest.raises(ResearchStopped, match="negative_cache"):
            await execute(session)
        history.observe("s", history_groups([ModelRequest.user_text_prompt("new dinner source")]))
        await execute(session)
        assert len(calls) == 2
        session.begin_finish("old_turn_done")
        fresh = ResearchSession("next_task", "new_snapshot", "user:s")
        await execute(fresh)
        assert fresh.steps == 1 and not fresh.finalizing
        assert len(calls) == 3


def test_search_projection_changes_revision_without_raw_event_change(tmp_path):
    with ContextStore(tmp_path) as store:
        setup(store)
        history = HistoryManager(store, USER)
        groups = history_groups([ModelResponse(parts=[TextPart("raw answer")])])
        history.observe("s", groups)
        event_id = history.event_id(groups[0].events[0])
        before = store.retrieval_revision(USER, "s")
        with store.transaction():
            history._index("s", "event", event_id, "projected final answer")
        assert store.retrieval_revision(USER, "s") != before


async def test_observation_progress_does_not_resolve_unknown_requirement():
    session = ResearchSession("task", "snapshot", "user:s")
    session.stop_feedback(ResearchStopped("negative_cache"), ("dinner",))

    async def read():
        return "new candidate restaurant"

    async def observe(result):
        return ("read:new_source",)

    await session.execute(
        "read",
        slots=("dinner",),
        scope="s",
        index_revision="v2",
        policy_revision="v2",
        operation=read,
        validate=observe,
        progress_kind="observation",
    )
    assert session.unresolved["dinner"]["reason"] == "new_observation_requires_validation"
    assert not session._covered
