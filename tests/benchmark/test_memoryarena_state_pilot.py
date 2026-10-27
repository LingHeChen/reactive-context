"""Real episode boundary, including model intent extraction; all calls use FunctionModel."""

import json
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import ContextStore
from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.config import BenchmarkConfig
from reactive_context.benchmarking.memoryarena_state_pilot import (
    IntentCache,
    StateIntegration,
    validate_intent,
)
from reactive_context.benchmarking.runtime import BudgetLedger


def test_permission_boundary_rejects_future_people_unobserved_quotes_and_wrong_days():
    value = {
        "subject": "Alice",
        "subject_quote": "I am Alice.",
        "template_person": "Seed",
        "template_reason": "visible prior plan",
        "edits": [{"day": 1, "field": "dinner", "quote": "change dinner"}],
    }
    query = "I am Alice. Please change dinner."
    assert validate_intent(value, query, ["Seed"], [1]).subject == "Alice"
    with pytest.raises(ValueError):
        validate_intent({**value, "template_person": "Future"}, query, ["Seed"], [1])
    with pytest.raises(ValueError):
        validate_intent(value, query, ["Seed"], [2])
    with pytest.raises(ValueError):
        validate_intent(value, "I am Alice.", ["Seed"], [1])


async def test_both_arms_share_current_query_intents_and_retry_unauthorized_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(base, "business_tools", lambda executor: [])
    monkeypatch.setattr(base.upstream, "system_prompt", lambda: "Output exact final plan format.")
    plan = [
        {
            "day": 1,
            "current_city": "City",
            "transportation": "KEEP_TRANSPORT",
            "breakfast": "B",
            "attraction": "KEEP_ATTRACTION",
            "lunch": "L",
            "dinner": "D",
            "accommodation": "H",
        }
    ]
    queries = [
        "I am Alice. Please change dinner.",
        "I am Bob. Use Alice's plan and change breakfast.",
    ]
    row = {
        "id": 999,
        "base_person": {"name": "Seed", "daily_plans": plan, "query": "initial"},
        "questions": queries,
        "answers": ["GOLD_MUST_NOT_APPEAR", "GOLD_MUST_NOT_APPEAR"],
    }
    intent_calls, actor_calls = 0, 0
    integration = None
    seen_per_stage = {}

    def reply(messages, info):
        nonlocal intent_calls, actor_calls
        assert "GOLD_MUST_NOT_APPEAR" not in str(messages)
        if info.output_tools and info.output_tools[0].name != "submit_plan_changes":
            if info.output_tools[0].name == "context_commit_facts":
                source = json.loads(messages[-1].parts[0].content.split("\nSOURCE:\n", 1)[1])
                initial = "given_fixed_initial_plan" in source
                subject = "Seed" if initial else integration.intent.subject
                path = "/given_fixed_initial_plan/daily_plans/0" if initial else "/generated_plan"
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "context_commit_facts",
                            {
                                "facts": [
                                    {
                                        "subject": subject,
                                        "topic": "day-1",
                                        "content": "observed plan",
                                        "anchors": [{"path": path}],
                                    }
                                ]
                            },
                            "write",
                        )
                    ]
                )
            intent_calls += 1
            current = json.loads(messages[-1].parts[0].content)
            query = current["current_query"]
            if query == queries[0]:
                assert queries[1] not in str(messages)
                subject, template, clause = "Alice", "Seed", "change dinner"
            else:
                subject, template, clause = "Bob", "Alice", "change breakfast"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {
                            "subject": subject,
                            "subject_quote": f"I am {subject}.",
                            "template_person": template,
                            "template_reason": "model chooses visible source",
                            "edits": [
                                {
                                    "day": 1,
                                    "field": "dinner" if subject == "Alice" else "breakfast",
                                    "action": "replace",
                                    "quote": clause,
                                }
                            ],
                        },
                        "intent",
                    )
                ]
            )
        actor_calls += 1
        intent = integration.intent
        key = (integration.folder.name, intent.subject)
        seen_per_stage[key] = seen_per_stage.get(key, 0) + 1
        operations = []
        if intent.subject == "Alice":
            operations.append({"op": "set", "path": "/days/1/dinner", "value": "NEW_DINNER"})
            if seen_per_stage[key] == 1:
                operations.append(
                    {"op": "set", "path": "/days/1/transportation", "value": "WRONG_TRANSPORT"}
                )
        else:
            assert integration.adapter.plan(base.SCOPE)[0]["dinner"] == "NEW_DINNER"
            operations.append({"op": "set", "path": "/days/1/breakfast", "value": "NEW_BREAKFAST"})
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "submit_plan_changes", {"operations": operations}, f"finish-{actor_calls}"
                )
            ]
        )

    model = FunctionModel(reply)
    ledger = BudgetLedger(replace(BenchmarkConfig(), max_requests=50))
    cache = IntentCache(tmp_path / "intents", model, ledger)
    for arm in ("fact-small", "fact-archive"):
        folder = tmp_path / arm
        integration = StateIntegration(row["id"], folder, cache)
        result = await base.episode(row, arm, folder, model, ledger, None, integration=integration)
        assert len(result["persons"]) == 2
        for person in result["persons"]:
            day = person["plan"][0]
            assert day["transportation"] == "KEEP_TRANSPORT"
            assert day["attraction"] == "KEEP_ATTRACTION"
            assert day["dinner"] == "NEW_DINNER"
        assert integration.state_events["output_rejected"] == 1
        assert integration.state_events["output_accepted"] == 2
        assert all(
            not json.loads(line)["failure"]
            for line in (folder / "stages.jsonl").read_text().splitlines()
        )
    assert intent_calls == 2 and actor_calls == 6
    # Already committed structured state survives a failed textual acknowledgement.
    with ContextStore(tmp_path / "fact-small" / "store") as store:
        recovered = StateIntegration(row["id"], tmp_path / "fact-small", cache)
        await recovered.prepare_stage(
            store, row["base_person"], queries[1], 2, [{"person_idx": 1, "plan": None}]
        )
        assert recovered.adapter.plan(base.SCOPE)[0]["dinner"] == "NEW_DINNER"


async def test_preparation_failure_is_recorded_and_does_not_abort_later_stages(
    tmp_path, monkeypatch
):
    from pydantic_ai.messages import TextPart

    from reactive_context.benchmarking.memoryarena_generic_pilot import GenericIntegration

    class Integration(GenericIntegration):
        async def prepare_stage(self, store, initial, query, index, persons):
            if index == 1:
                raise ValueError("missing template")

    monkeypatch.setattr(base, "business_tools", lambda executor: [])
    monkeypatch.setattr(base.upstream, "system_prompt", lambda: "Return a plan.")
    monkeypatch.setattr(base.upstream, "parser", lambda: lambda text: [{"day": 1}])

    def reply(messages, info):
        if info.output_tools:
            source = json.loads(messages[-1].parts[0].content.split("\nSOURCE:\n", 1)[1])
            initial = "given_fixed_initial_plan" in source
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "context_commit_facts",
                        {
                            "facts": [
                                {
                                    "subject": "Seed" if initial else "Later",
                                    "topic": "day-1",
                                    "content": "observed",
                                    "anchors": [
                                        {
                                            "path": "/given_fixed_initial_plan/query"
                                            if initial
                                            else "/generated_plan"
                                        }
                                    ],
                                }
                            ]
                        },
                        "write",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("=== Later's Plan ===\nDay 1:")])

    row = {
        "id": 999,
        "base_person": {"name": "Seed", "query": "initial", "daily_plans": []},
        "questions": ["fails preparation", "later request"],
        "answers": ["PRIVATE", "PRIVATE"],
    }
    result = await base.episode(
        row,
        "fact-small",
        tmp_path,
        FunctionModel(reply),
        BudgetLedger(replace(BenchmarkConfig(), max_requests=20)),
        None,
        integration=Integration(),
    )
    assert result["persons"][0]["plan"] is None
    # A completed model call with missing fields is still a scoring-contract failure.
    assert result["persons"][1]["plan"] is None
    projections = [
        json.loads(line)
        for line in (tmp_path / "scoring-projections.jsonl").read_text().splitlines()
    ]
    assert projections[0]["raw_plan"] == [{"day": 1}]
    assert "missing_required_field" in projections[0]["audit"]["errors"]
    stages = [json.loads(line) for line in (tmp_path / "stages.jsonl").read_text().splitlines()]
    assert stages[0]["failure"]["phase"] == "state-preparation"
    assert stages[1]["failure"] is None
