import json
from dataclasses import replace

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from reactive_context import CompilerPolicy, ContextManager, ContextStore
from reactive_context.benchmarking import memoryarena as upstream
from reactive_context.benchmarking.config import BenchmarkConfig
from reactive_context.benchmarking.memoryarena_pilot import (
    ACTOR,
    HOST,
    SCHEMA,
    SCOPE,
    FactOnlyCapability,
    TravelFact,
    commit_batch,
    extract,
    fact_catalog,
    select_facts,
    stage_input,
)
from reactive_context.benchmarking.runtime import BudgetLedger, CallTrace


def provision(path):
    store = ContextStore(path)
    store.create_scope(HOST, SCOPE)
    store.grant_scope(HOST, SCOPE, ACTOR.id)
    store.register_schema(SCHEMA, TravelFact)
    return store


async def test_real_writer_commit_restart_and_next_request(tmp_path):
    fact = TravelFact(
        subject="Alice", topic="day-1", content="Hotel Cedar, Seattle", quotes=["Hotel Cedar"]
    )
    seen = []

    def writer(messages, info):
        seen.append(str(messages))
        assert "FUTURE_GOLD_SENTINEL" not in str(messages)
        return ModelResponse(
            parts=[ToolCallPart("context_commit_facts", {"facts": [fact.model_dump()]}, "write-1")],
            usage=RequestUsage(input_tokens=50, output_tokens=30),
        )

    with provision(tmp_path / "store") as store:
        async with ContextManager(store=store, principal=ACTOR) as manager:
            receipts = await extract(
                FunctionModel(writer),
                BudgetLedger(replace(BenchmarkConfig(), max_requests=10)),
                CallTrace(),
                manager,
                {"generated_plan": "Hotel Cedar"},
                0,
            )
            assert receipts[0]["activated"]
            assert fact_catalog(store)[0].value["content"] == "Hotel Cedar, Seattle"
    with ContextStore(tmp_path / "store") as store:
        store.register_schema(SCHEMA, TravelFact)
        async with ContextManager(
            store=store, principal=ACTOR, compiler_policy=CompilerPolicy(max_tokens=128000)
        ) as manager:
            assert len(await select_facts(manager, "Alice hotel")) == 1

            def actor(messages, info):
                assert "Hotel Cedar, Seattle" in str(messages)
                assert not {"context_history_search", "context_history_lookup"}.intersection(
                    tool.name for tool in info.function_tools
                )
                return ModelResponse(parts=[TextPart("Cedar")])

            await Agent(
                FunctionModel(actor),
                capabilities=[FactOnlyCapability(manager, scope=SCOPE, cache=None)],
            ).run("Which hotel did Alice choose?")
            revised = fact.model_copy(update={"content": "Hotel Pine", "quotes": ["Pine"]})
            receipts = await commit_batch(manager, [revised], {"plan": "Pine"}, 1)
            assert receipts[0]["revision"] == 2
            assert fact_catalog(store)[0].revision == 2
            assert fact_catalog(store)[0].value["content"] == "Hotel Pine"


async def test_invalid_evidence_is_not_committed(tmp_path):
    with provision(tmp_path) as store:
        async with ContextManager(store=store, principal=ACTOR) as manager:
            with pytest.raises(ModelRetry):
                await commit_batch(
                    manager,
                    [
                        TravelFact(
                            subject="A", topic="day-1", content="invention", quotes=["NOT OBSERVED"]
                        )
                    ],
                    {"query": "hello"},
                    0,
                )
            assert fact_catalog(store) == []


def test_public_boundary_and_official_parser_scorer(tmp_path):
    if not upstream.DATA.exists():
        pytest.skip("pinned public MemoryArena data not downloaded")
    row = upstream.tasks(1)[0]
    assert row["id"] == 218
    current = stage_input(row["base_person"], row["questions"][0])
    assert set(current) == {"current_query", "initial_traveler"}
    assert "answers" not in current
    assert upstream.parser()("Day 1:\nDinner: Cafe, Seattle")[0]["dinner"] == "Cafe, Seattle"
    path = tmp_path / "submission.jsonl"
    complete = {
        "id": row["id"],
        "persons": [{"person_idx": i, "plan": plan} for i, plan in enumerate(row["answers"], 1)],
    }
    path.write_text(json.dumps(complete) + "\n")
    assert upstream.score([row], path) == {"ps": 100, "sps": 100, "sr": 100}
    complete["persons"][-1]["plan"] = None
    path.write_text(json.dumps(complete) + "\n")
    scores = upstream.score([row], path)
    assert scores["ps"] == 87.5
    assert scores["sr"] == 0


async def test_deactivate_persisted_facts_before_first_snapshot(tmp_path):
    with provision(tmp_path) as store:
        async with ContextManager(store=store, principal=ACTOR) as manager:
            facts = [
                TravelFact(
                    subject="Alice", topic=f"day-{i}", content=f"Hotel {i}", quotes=["Hotel"]
                )
                for i in range(10)
            ]
            receipts = await commit_batch(manager, facts, {"observed": "Hotel"}, 0)
        async with ContextManager(store=store, principal=ACTOR) as manager:
            # New registry starts empty while active references persist in SQLite.
            await manager.deactivate(SCOPE, receipts[0]["id"])
            await manager.deactivate(SCOPE, receipts[1]["id"])
            snapshot = await manager.next_context(SCOPE)
            assert len(snapshot.parts) == 8
            assert len(store.active_refs(ACTOR, SCOPE)) == 8


async def test_writer_failure_is_scored_without_aborting_episode(tmp_path, monkeypatch):
    from reactive_context.benchmarking import memoryarena_pilot as pilot

    monkeypatch.setattr(pilot, "business_tools", lambda executor: [])
    monkeypatch.setattr(pilot.upstream, "system_prompt", lambda: "Produce a travel plan.")
    monkeypatch.setattr(pilot.upstream, "parser", lambda: lambda text: [{"day": 1}])

    def reply(messages, info):
        assert "REFERENCE_ONLY" not in str(messages)
        if info.output_tools:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "context_commit_facts",
                        {
                            "facts": [
                                {
                                    "subject": "Alice",
                                    "topic": "day-1",
                                    "content": "Cedar",
                                    "quotes": ["UNOBSERVED QUOTE"],
                                }
                            ]
                        },
                        "invalid-write",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("Day 1:\nAccommodation: Cedar")])

    row = {
        "id": 999,
        "base_person": {"name": "Alice", "query": "Trip", "daily_plans": []},
        "questions": ["I am Bob. Join Alice.", "I am Carol. Join Bob."],
        "answers": ["REFERENCE_ONLY", "REFERENCE_ONLY"],
    }
    ledger = BudgetLedger(replace(BenchmarkConfig(), max_requests=20))
    result = await pilot.episode(row, "fact-small", tmp_path, FunctionModel(reply), ledger, None)
    assert len(result["persons"]) == 2
    errors = [
        json.loads(line) for line in (tmp_path / "writer-errors.jsonl").read_text().splitlines()
    ]
    assert [error["stage"] for error in errors] == [0, 1, 2]
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["completed"] == 2


async def test_source_paths_commit_good_entries_and_reject_only_bad_ones(tmp_path):
    from reactive_context.benchmarking.memoryarena_pilot import FactDraft, SourceAnchor

    source = {"given_fixed_initial_plan": {"daily_plans": [{"accommodation": "Hotel Cedar"}]}}
    valid = FactDraft(
        subject="Alice",
        topic="day-1",
        content="Hotel Cedar",
        anchors=[SourceAnchor(path="/given_fixed_initial_plan/daily_plans/0/accommodation")],
    )
    bad = FactDraft(
        subject="Bob",
        topic="day-1",
        content="invented",
        anchors=[SourceAnchor(path="/future_reference_answer")],
    )
    with provision(tmp_path) as store:
        async with ContextManager(store=store, principal=ACTOR) as manager:
            receipts = await commit_batch(manager, [valid, bad], source, 0)
            assert receipts[0]["activated"]
            assert receipts[1]["status"] == "rejected"
            assert len(fact_catalog(store)) == 1
            assert list(fact_catalog(store)[0].value["quotes"]) == ["Hotel Cedar"]
            assert receipts[0]["source_paths"] == [valid.anchors[0].path]
            # Replaying the accepted entry does not create another revision.
            again = await commit_batch(manager, [valid], source, 0)
            assert again[0]["unchanged"]
            assert fact_catalog(store)[0].revision == 1


def test_source_anchor_paths_cannot_match_catalog_or_invent_field_prefixes():
    from reactive_context.benchmarking.memoryarena_pilot import SourceAnchor, anchored_quote

    source = {"a/b": {"~value": "Hotel Cedar"}, "rows": ["literal"]}
    assert anchored_quote(source, SourceAnchor(path="/a~1b/~0value")) == "Hotel Cedar"
    assert anchored_quote(source, SourceAnchor(path="/a~1b/~0value", quote="Cedar")) == "Cedar"
    for anchor in (
        SourceAnchor(path="/catalog/value"),
        SourceAnchor(path="/a~1b/~0value", quote="hotel: Hotel Cedar"),
        SourceAnchor(path="/rows", quote="literal"),
        SourceAnchor(path="/rows/4"),
    ):
        with pytest.raises((ValueError, KeyError, IndexError)):
            anchored_quote(source, anchor)


async def test_native_writer_accepts_path_only_source_anchor(tmp_path):
    def writer(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "context_commit_facts",
                    {
                        "facts": [
                            {
                                "subject": "Maria",
                                "topic": "day-1",
                                "content": "Hotel Cedar",
                                "anchors": [
                                    {
                                        "path": (
                                            "/given_fixed_initial_plan/daily_plans/0/accommodation"
                                        )
                                    }
                                ],
                            }
                        ]
                    },
                    "anchored-write",
                )
            ]
        )

    source = {"given_fixed_initial_plan": {"daily_plans": [{"accommodation": "Hotel Cedar"}]}}
    with provision(tmp_path) as store:
        async with ContextManager(store=store, principal=ACTOR) as manager:
            receipts = await extract(
                FunctionModel(writer),
                BudgetLedger(BenchmarkConfig()),
                CallTrace(),
                manager,
                source,
                0,
            )
            assert receipts[0]["activated"]
            assert list(fact_catalog(store)[0].value["quotes"]) == ["Hotel Cedar"]


async def test_partial_batch_repairs_only_rejected_entries_without_duplicate_revisions(tmp_path):
    calls = []

    def writer(messages, info):
        calls.append(str(messages))
        good = {
            "subject": "Alice",
            "topic": "day-1",
            "content": "Cedar",
            "anchors": [{"path": "/day"}],
        }
        bad = {
            "subject": "Bob",
            "topic": "day-1",
            "content": "Pine",
            "anchors": [{"path": "/missing" if len(calls) == 1 else "/hotel"}],
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "context_commit_facts",
                    {"facts": [good, bad] if len(calls) == 1 else [bad]},
                    f"batch-{len(calls)}",
                )
            ]
        )

    source = {"day": {"days": 1, "accommodation": "Cedar"}, "hotel": "Pine"}
    with provision(tmp_path) as store:
        async with ContextManager(store=store, principal=ACTOR) as manager:
            receipts = await extract(
                FunctionModel(writer),
                BudgetLedger(BenchmarkConfig()),
                CallTrace(),
                manager,
                source,
                0,
            )
            assert len(calls) == 2
            assert len(receipts) == 2
            assert all(receipt.get("status") != "rejected" for receipt in receipts)
            assert {fact.key for fact in fact_catalog(store)} == {"Alice/day-1", "Bob/day-1"}
            assert all(fact.revision == 1 for fact in fact_catalog(store))
            alice = next(f for f in fact_catalog(store) if f.key.startswith("Alice"))
            assert json.loads(alice.value["quotes"][0]) == source["day"]
