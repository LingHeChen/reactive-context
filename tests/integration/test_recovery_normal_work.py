"""Positive execution paths: enough real evidence, local stops, truthful maintenance."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import ContextManager, ContextStore, Principal
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol
from reactive_context.adapters.recovery import ResearchRecoveryCapability
from reactive_context.benchmarking.memoryarena_pilot import (
    FactDraft,
    TravelFact,
    classify_fact_drafts,
    completion_state,
    encoded,
    select_facts,
)
from reactive_context.benchmarking.travel_dependencies import dependencies
from reactive_context.core.snapshot import freeze
from reactive_context.evidence import ResearchSession, ResearchStopped, SessionBudget


async def test_nine_distinct_fact_reads_then_business_action(tmp_path):
    class Note(BaseModel):
        text: str

    host, user = Principal("host", True), Principal("user")
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "s")
        store.grant_scope(host, "s", user.id)
        store.register_schema("note", Note)
        evidence = store.observe_evidence(user, "s", "public", "1")
        facts = [
            store.commit(
                user,
                store.propose_fact(
                    user, "s", "note", f"fact-{i}", {"text": f"value {i}"}, [evidence]
                ),
            )
            for i in range(9)
        ]
        async with ContextManager(store=store, principal=user) as manager:
            session = ResearchSession(
                "task", "snapshot", "user:s", budget=SessionBudget(max_steps=1)
            )
            recovery = ResearchRecoveryCapability(session, max_model_requests=6)
            actions = []
            count = 0

            async def submit():
                actions.append("submitted")
                return "submitted"

            def model(messages, info):
                nonlocal count
                count += 1
                if count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart("context_load", {"part_id": f.id}, f.id) for f in facts]
                    )
                if count == 2:
                    # Every distinct source was actually returned before the business action.
                    contents = str(messages[-1])
                    assert all(f"value {i}" in contents for i in range(9))
                    assert "submit" in {t.name for t in info.function_tools}
                    return ModelResponse(parts=[ToolCallPart("submit", {}, "submit")])
                return ModelResponse(
                    parts=[TextPart("Complete: nine sources read and action submitted.")]
                )

            result = await Agent(
                FunctionModel(model),
                tools=[submit],
                toolsets=[context_tools(manager, "s", research_session=session)],
                capabilities=[ReactiveContextCapability(manager, scope="s", cache=None), recovery],
            ).run("Read all nine sources and submit")
            assert actions == ["submitted"] and session.read_steps == 9
            assert session.search_steps == 0 and not session.finalizing
            validate_protocol(result.all_messages())


async def test_exhausted_branch_does_not_stop_other_search_read_or_business():
    session = ResearchSession("a", "b", "c", budget=SessionBudget(max_steps=1))

    async def operation():
        return "new evidence"

    async def validate(result):
        return [result]

    async def run(query, slot, kind):
        return await session.execute(
            query,
            slots=(slot,),
            scope="s",
            index_revision="1",
            policy_revision="1",
            operation=operation,
            validate=validate,
            progress_kind="observation",
            work_kind=kind,
        )

    await run("search A", "A", "search")
    with pytest.raises(ResearchStopped) as exc:
        await run("paraphrase A", "A", "search")
    receipt = session.stop_feedback(exc.value, ("A",))
    assert not receipt["finalize"]
    await run("search B", "B", "search")
    await run("read A source", "source-A", "read")
    assert not session.finalizing
    session.stop_feedback(ResearchStopped("no progress"), ("B",))
    assert not session.finalizing  # independent stops never accumulate into global shutdown


async def test_selection_uses_content_and_logical_keys_not_uuid(monkeypatch):
    from reactive_context.benchmarking import memoryarena_pilot as pilot

    facts = [
        SimpleNamespace(
            id=f"id-{i}",
            key=f"Person/day-{i}",
            value=freeze(
                TravelFact(
                    subject="Person",
                    topic=f"day-{i}",
                    content="unique needle" if i == 9 else "ordinary",
                    quotes=["source"],
                )
            ),
        )
        for i in range(10)
    ]
    assert json.loads(encoded(facts[0].value))["subject"] == "Person"
    assert "FrozenRecord object" not in encoded(facts[0].value)
    store = SimpleNamespace(active_refs=lambda *a: ())

    async def activate(*a):
        pass

    manager = SimpleNamespace(store=store, activate=activate)
    monkeypatch.setattr(pilot, "fact_catalog", lambda store: list(reversed(facts)))
    first = await select_facts(manager, "unique needle")
    expected = [f.key for id in first for f in facts if f.id == id]
    assert expected[0] == "Person/day-9"
    for i, f in enumerate(facts):
        f.id = f"new-{99 - i}"
    second = await select_facts(manager, "unique needle")
    assert expected == [f.key for id in second for f in facts if f.id == id]
    preferred = await select_facts(manager, "ordinary", required_keys=("Person/day-9",))
    assert preferred[0] == facts[9].id


def test_dependency_hints_bind_local_people_days_not_cross_product():
    assert dependencies("Zoe day 5 dinner, Joseph day 2 lunch", ("Zoe", "Joseph")) == (
        ("Joseph", 2, "lunch"),
        ("Zoe", 5, "dinner"),
    )
    assert dependencies(
        "For breakfast on the second day I'd like to join Joseph.", ("Joseph",)
    ) == (("Joseph", 2, "breakfast"),)
    assert dependencies("priced within 10% of Joseph's first-day dinner", ("Joseph",)) == (
        ("Joseph", 1, "dinner"),
    )


def test_incomplete_return_classified_before_writer_and_cannot_promote_observations():
    projection = SimpleNamespace(final_plan=None, audit={"errors": ["empty_plan"]})
    state = completion_state("I cannot finish", projection)
    assert state["execution_status"] == "incomplete"
    fact = FactDraft(
        subject="Gary",
        topic="day-2",
        content="candidate",
        quotes=["candidate"],
        record_state="submitted_plan",
    )
    accepted, _ = classify_fact_drafts([fact], state, [])
    assert accepted[0].record_state == "observation"
    accepted, rejected = classify_fact_drafts(
        [fact], state, [{"key": "Gary/day-2", "value": {"record_state": "submitted_plan"}}]
    )
    assert not accepted and rejected[0]["reason"] == "preserve_submitted_plan"
    valid = SimpleNamespace(final_plan=[{"day": 1}], audit={"errors": []})
    assert (
        completion_state("<｜DSML｜invoke> unfinished", valid)["execution_status"] == "incomplete"
    )
    assert completion_state("normal final", valid)["execution_status"] == "completed"


async def test_real_agent_can_execute_business_after_local_search_stop(tmp_path):
    host, user = Principal("host", True), Principal("user")
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "s")
        store.grant_scope(host, "s", user.id)
        async with ContextManager(store=store, principal=user) as manager:
            session = ResearchSession(
                "task", "snapshot", "user:s", budget=SessionBudget(max_steps=1)
            )
            recovery = ResearchRecoveryCapability(session, max_model_requests=6)
            count, actions = 0, []

            async def business():
                actions.append(True)
                return "done"

            def model(messages, info):
                nonlocal count
                count += 1
                if count <= 3:
                    if count == 3:
                        assert "retrieval_stopped" in str(messages[-1])
                        assert not session.finalizing
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "context_search_facts",
                                {"intent": "A" if count < 3 else "B"},
                                str(count),
                            )
                        ]
                    )
                if count == 4:
                    assert "business" in {t.name for t in info.function_tools}
                    return ModelResponse(parts=[ToolCallPart("business", {}, "business")])
                return ModelResponse(
                    parts=[TextPart("Done; A and B unavailable, independent work completed.")]
                )

            result = await Agent(
                FunctionModel(model),
                tools=[business],
                toolsets=[
                    context_tools(
                        manager, "s", research_session=session, requirement_key=lambda text: (text,)
                    )
                ],
                capabilities=[ReactiveContextCapability(manager, scope="s", cache=None), recovery],
            ).run("Task")
            assert actions == [True] and session.search_steps == 2
            assert not session.finalizing and result.output.startswith("Done")


async def test_incomplete_status_reaches_real_episode_writer(tmp_path, monkeypatch):
    from reactive_context.benchmarking import memoryarena_pilot as pilot
    from reactive_context.benchmarking.config import BenchmarkConfig
    from reactive_context.benchmarking.runtime import BudgetLedger

    monkeypatch.setattr(pilot, "business_tools", lambda executor: [])
    monkeypatch.setattr(pilot.upstream, "system_prompt", lambda: "Produce a plan")
    monkeypatch.setattr(pilot.upstream, "parser", lambda: lambda text: [])
    sources = []

    async def writer(model, ledger, trace, manager, source, stage, **kwargs):
        sources.append((stage, source.copy()))
        return []

    monkeypatch.setattr(pilot, "extract", writer)

    def model(messages, info):
        return ModelResponse(parts=[TextPart("Cannot produce a complete plan.")])

    row = {
        "id": 999,
        "base_person": {"name": "Alice", "query": "Trip", "daily_plans": []},
        "questions": ["I am Bob. Join Alice."],
        "answers": ["NEVER_SHOW_TO_MODEL"],
    }
    result = await pilot.episode(
        row, "fact-small", tmp_path, FunctionModel(model), BudgetLedger(BenchmarkConfig()), None
    )
    current = next(source for stage, source in sources if stage == 1)
    assert current["execution_status"] == current["final_status"] == "incomplete"
    assert current["completion_errors"]
    assert result["persons"][0]["plan"] is None


async def test_finish_rejects_textual_pseudo_tool_calls():
    session = ResearchSession("a", "b", "c")
    recovery = ResearchRecoveryCapability(session)
    session.begin_finish("host_request_budget")

    def model(messages, info):
        return ModelResponse(
            parts=[TextPart("<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name='search'>")]
        )

    with pytest.raises(ResearchStopped, match="unexecuted_tool_markup"):
        await Agent(FunctionModel(model), capabilities=[recovery]).run("Task")
    assert "DSML" in str(recovery.current_messages())
