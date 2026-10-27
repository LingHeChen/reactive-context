"""Only declared format projection and generic provenance checks; no target answers."""

import json
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import ContextManager, ContextStore
from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.config import BenchmarkConfig
from reactive_context.benchmarking.memoryarena_generic_pilot import GenericIntegration
from reactive_context.benchmarking.runtime import BudgetLedger
from reactive_context.fact_checks import FactCheckRequest


def test_declared_boundary_projection_never_changes_plan_fields():
    integration = GenericIntegration()
    plan = "=== Alice's Plan ===\nDay 1:\nDinner: ANY_VALUE\n"
    assert integration.final_view("PROCESS_ONLY\n" + plan) == plan.strip()
    assert integration.final_view("no declared result boundary") is None


async def test_checked_persistence_checks_source_then_avoids_duplicate_versions(tmp_path):
    integration = GenericIntegration()
    source = {"observed": {"meal": "A"}}
    fact = base.FactDraft(
        subject="Alice",
        topic="day-1",
        content="decision A",
        anchors=[base.SourceAnchor(path="/observed")],
    )
    with ContextStore(tmp_path) as store:
        store.create_scope(base.HOST, base.SCOPE)
        store.grant_scope(base.HOST, base.SCOPE, base.ACTOR.id)
        store.register_schema(base.SCHEMA, base.TravelFact, check=integration.check)
        async with ContextManager(store=store, principal=base.ACTOR) as manager:
            first = await integration.commit(manager, [fact], source, 0)
            again = await integration.commit(manager, [fact], source, 1)
            assert (
                first[0]["committed"]
                and first[0]["admission"]["validation"] == "host_checks_passed"
            )
            assert again[0]["unchanged"]
            assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
    bad = FactCheckRequest(
        "trip",
        base.SCHEMA,
        1,
        "Alice/day-1",
        {"subject": "Alice", "topic": "day-1", "quotes": ["never observed"]},
        ("unknown-evidence",),
    )
    assert integration.check(bad).action == "reject"


@pytest.mark.parametrize("variant", ["legacy", "dx-fact-small", "dx-fact-archive"])
async def test_full_episode_uses_projection_and_checked_save_without_future_labels(
    tmp_path, monkeypatch, variant
):
    monkeypatch.setattr(base, "business_tools", lambda executor: [])
    monkeypatch.setattr(base.upstream, "system_prompt", lambda: "Output the declared plan format.")
    monkeypatch.setattr(base.upstream, "parser", lambda: lambda text: [{"day": 1, "dinner": "A"}])
    actor_calls = 0

    def reply(messages, info):
        nonlocal actor_calls
        assert "REFERENCE_ONLY" not in str(messages)
        if info.output_tools:
            initial = "given_fixed_initial_plan" in str(messages) and actor_calls == 0
            path = "/given_fixed_initial_plan/query" if initial else "/generated_plan"
            fact = {
                "subject": "Base" if initial else "Alice",
                "topic": "day-1",
                "content": "decision A",
                "anchors": [{"path": path}],
            }
            return ModelResponse(
                parts=[ToolCallPart("context_commit_facts", {"facts": [fact]}, "write")]
            )
        actor_calls += 1
        if actor_calls == 2:
            previous = [
                p.content
                for m in messages
                if isinstance(m, ModelResponse)
                for p in m.parts
                if isinstance(p, TextPart)
            ]
            assert previous and all("PROCESS_ONLY" not in text for text in previous)
        return ModelResponse(
            parts=[TextPart("PROCESS_ONLY\n=== Alice's Plan ===\nDay 1:\nDinner: A")]
        )

    row = {
        "id": 999,
        "base_person": {"name": "Base", "query": "known input", "daily_plans": []},
        "questions": ["current one", "current two"],
        "answers": ["REFERENCE_ONLY", "REFERENCE_ONLY"],
    }
    from reactive_context.benchmarking.memoryarena_dx_pilot import Integration

    integration = GenericIntegration() if variant == "legacy" else Integration(variant)
    result = await base.episode(
        row,
        "fact-small",
        tmp_path,
        FunctionModel(reply),
        BudgetLedger(replace(BenchmarkConfig(), max_requests=20)),
        None,
        integration=integration,
    )
    assert len(result["persons"]) == 2 and actor_calls == 2
    assert integration.decisions["created"] == 2
    receipts = [
        json.loads(line) for line in (tmp_path / "fact-writes.jsonl").read_text().splitlines()
    ]
    assert all(
        item["receipts"][0]["admission"]["validation"] == "host_checks_passed" for item in receipts
    )
