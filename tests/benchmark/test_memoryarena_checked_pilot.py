"""No paid calls: actual candidate gates, recorded tool rows and delta-only Facts."""

import json
from collections import Counter
from dataclasses import replace

import pandas as pd
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.config import BenchmarkConfig
from reactive_context.benchmarking.memoryarena_checked_pilot import CheckedIntegration
from reactive_context.benchmarking.memoryarena_state_pilot import IntentCache
from reactive_context.benchmarking.runtime import BudgetLedger


class TableAPI:
    def __init__(self, rows):
        self.rows = pd.DataFrame(rows)

    def run(self, city):
        return self.rows


class Engine:
    restaurants = TableAPI(
        [
            {
                "Name": "Soni Bakers",
                "Average Cost": 42,
                "Aggregate Rating": 3.4,
                "Cuisines": "Pizza, Seafood",
                "City": "City",
            },
            {
                "Name": "Valid Choice",
                "Average Cost": 50,
                "Aggregate Rating": 3.8,
                "Cuisines": "Pizza, Seafood",
                "City": "City",
            },
        ]
    )
    accommodations = TableAPI(
        [
            {
                "NAME": "Seed Hotel",
                "price": 100,
                "room type": "Private room",
                "house_rules": "No pets",
                "maximum occupancy": 2,
                "review rate number": 3,
                "city": "City",
            },
            {
                "NAME": "New Hotel",
                "price": 110,
                "room type": "Private room",
                "house_rules": "No pets",
                "maximum occupancy": 2,
                "review rate number": 4,
                "city": "City",
            },
        ]
    )

    def _format_result(self, result, name):
        return result.to_string(index=False)


async def test_full_checked_episode_repairs_real_bad_choice_and_writes_deltas_without_model_writer(
    tmp_path,
):
    queries = [
        "I am Alice. Breakfast on day 1 costs 42.6 to 61.4 and is rated 3.5 to 4.1.",
        "I am Bob. My day 1 accommodation costs at least 10% more than Alice's day 1 stay "
        "and allows no pets.",
    ]
    initial = {
        "name": "Seed",
        "query": "given public plan",
        "daily_plans": [
            {
                "day": 1,
                "current_city": "City",
                "transportation": "KEEP_TRANSPORT",
                "breakfast": "Original, City",
                "attraction": "KEEP_MUSEUM",
                "lunch": "L",
                "dinner": "D",
                "accommodation": "Seed Hotel, City",
            }
        ],
    }
    row = {
        "id": 999,
        "base_person": initial,
        "questions": queries,
        "answers": ["GOLD_PRIVATE", "FUTURE_PRIVATE"],
    }
    counts, actors = Counter(), Counter()
    integration = None

    def reply(messages, info):
        assert "GOLD_PRIVATE" not in str(messages) and "FUTURE_PRIVATE" not in str(messages)
        assert "context_propose_state_candidate" not in {tool.name for tool in info.function_tools}
        tool = info.output_tools[0]
        assert tool.name != "context_commit_facts"  # no full-source extraction model
        if tool.name != "submit_plan_changes":
            payload = json.loads(messages[-1].parts[0].content)
            query = payload["current_query"]
            i = queries.index(query)
            if i == 0:
                assert queries[1] not in str(messages)
            if "slots" not in tool.parameters_json_schema["properties"]:
                counts["intent"] += 1
                value = {
                    "subject": "Alice" if i == 0 else "Bob",
                    "subject_quote": "I am Alice." if i == 0 else "I am Bob.",
                    "template_person": "Seed" if i == 0 else "Alice",
                    "template_reason": "visible source",
                    "edits": [
                        {
                            "day": 1,
                            "field": "breakfast" if i == 0 else "accommodation",
                            "quote": query,
                        }
                    ],
                }
            else:
                counts["constraints"] += 1
                clauses = (
                    [
                        {"attribute": a, "op": op, "value": value, "quote": query}
                        for a, op, value in [
                            ("price", "ge", 42.6),
                            ("price", "le", 61.4),
                            ("rating", "ge", 3.5),
                            ("rating", "le", 4.1),
                        ]
                    ]
                    if i == 0
                    else [
                        {
                            "attribute": "price",
                            "op": "ge",
                            "reference": {
                                "person": "Alice",
                                "day": 1,
                                "field": "accommodation",
                                "attribute": "price",
                            },
                            "scale": 1.1,
                            "quote": query,
                        },
                        {
                            "attribute": "rules",
                            "op": "contains_all",
                            "value": ["No pets"],
                            "quote": query,
                        },
                    ]
                )
                value = {
                    "slots": [
                        {
                            "day": 1,
                            "field": "breakfast" if i == 0 else "accommodation",
                            "clauses": clauses,
                        }
                    ]
                }
            return ModelResponse(
                parts=[ToolCallPart(tool.name, value, f"shared-{sum(counts.values())}")]
            )
        subject = integration.intent.subject
        key = (integration.folder.name, subject)
        actors[key] += 1
        step = actors[key]
        if step == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "RestaurantSearch" if subject == "Alice" else "AccommodationSearch",
                        {"city": "City"},
                        f"search-{key}",
                    )
                ]
            )
        if subject == "Alice":
            value = "Soni Bakers, City" if step == 2 else "Valid Choice, City"
            if step == 3:
                assert "constraint_price_ge" in str(messages) and "constraint_rating_ge" in str(
                    messages
                )
                assert integration.adapter.state.read(base.SCOPE)["revision"] == 1
            path = "/days/1/breakfast"
        else:
            path, value = "/days/1/accommodation", "New Hotel, City"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "submit_plan_changes",
                    {"operations": [{"op": "set", "path": path, "value": value}]},
                    f"finish-{key}-{step}",
                )
            ]
        )

    model = FunctionModel(reply)
    ledger = BudgetLedger(replace(BenchmarkConfig(), max_requests=50))
    cache = IntentCache(tmp_path / "intents", model, ledger)
    for arm in ["fact-small", "fact-archive"]:
        folder = tmp_path / arm
        integration = CheckedIntegration(999, folder, cache)
        result = await base.episode(
            row, arm, folder, model, ledger, Engine(), integration=integration
        )
        assert result["persons"][0]["plan"][0]["breakfast"] == "Valid Choice, City"
        assert result["persons"][1]["plan"][0]["accommodation"] == "New Hotel, City"
        assert all(p["plan"][0]["transportation"] == "KEEP_TRANSPORT" for p in result["persons"])
        calls = [json.loads(line) for line in (folder / "calls.jsonl").read_text().splitlines()]
        assert all(call["phase"] == "planning" for call in calls)
        assert integration.state_events["deterministic_memory_batches"] == 3
        assert integration.state_events["output_rejected"] == 1
        assert integration.state_events["output_accepted"] == 2
        writes = [
            json.loads(line) for line in (folder / "fact-writes.jsonl").read_text().splitlines()
        ]
        assert [len(w["receipts"]) for w in writes] == [1, 2, 2]
    assert counts == {"intent": 2, "constraints": 2}
