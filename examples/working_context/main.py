"""Offline demo: short history + work objective + extracted Fact + searchable archive.

Run: uv run python examples/working_context/main.py
FunctionModel verifies integration, not real-model quality. No network or paid requests.
"""

import asyncio
import json
from tempfile import TemporaryDirectory

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    ContextManager,
    ContextStore,
    HistoryManager,
    HistoryPolicy,
    Objective,
    ObjectivePolicy,
    Principal,
    SummaryDraft,
)
from reactive_context.adapters.history import history_groups
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability


async def main(directory):
    host, actor = Principal("host", True), Principal("demo-agent")
    with ContextStore(directory) as store:
        store.create_scope(host, "orders")
        store.grant_scope(host, "orders", actor.id)
        store.register_schema("finding", dict[str, str])

        def compact(events):
            return SummaryDraft(
                text="Earlier schema work renamed amount to total; original evidence is archived.",
                evidence_ids=[HistoryManager.event_id(events[0])],
            )

        history = HistoryManager(
            store,
            actor,
            compactor=compact,
            validator=lambda *_: True,
            policy=HistoryPolicy(recent_groups=4),
        )
        objective = Objective(
            store,
            actor,
            policy=ObjectivePolicy(
                update_mode="agent",
                update_conditions=(
                    "Create or update the goal when the user changes the deliverable.",
                    "Save durable findings as Facts with observed evidence IDs.",
                ),
            ),
        )
        raw = []
        for i in range(10):
            raw += [
                ModelRequest.user_text_prompt(f"Earlier schema discussion {i}"),
                ModelResponse(parts=[TextPart("The current monetary column is total.")]),
            ]
        step, saved = 0, {}
        async with ContextManager(
            store=store, principal=actor, history=history, objective=objective
        ) as manager:
            # Explicit archival compaction. Its output never enters the default request.
            await manager.compact_history("orders", history_groups(raw))

            def respond(messages, info):
                nonlocal step
                step += 1
                value = None
                if step > 1:
                    value = next(
                        p.content
                        for m in reversed(messages)
                        if isinstance(m, ModelRequest)
                        for p in reversed(m.parts)
                        if isinstance(p, ToolReturnPart)
                    )
                if step == 1:
                    tool, args = (
                        "context_update_objective",
                        {
                            "value": {
                                "goal": "Query orders using the current schema",
                                "next_steps": ["Recover earlier schema evidence", "Run a query"],
                            },
                            "expected_revision": 0,
                        },
                    )
                elif step == 2:
                    tool, args = "context_history_search", {"intent": "schema", "kind": "summary"}
                elif step == 3:
                    tool, args = "context_history_lookup", {"event_ids": value[0]["event_ids"][:1]}
                elif step == 4:
                    tool, args = (
                        "context_propose_fact",
                        {
                            "schema_id": "finding",
                            "key": "monetary-column",
                            "value": {"text": "orders monetary column is total"},
                            "evidence": [value[0]["evidence_id"]],
                        },
                    )
                elif step == 5:
                    tool, args = "context_commit_fact", {"proposal_id": value["proposal_id"]}
                elif step == 6:
                    saved["fact_id"] = value["part_id"]
                    tool, args = "context_activate", {"part_id": value["part_id"]}
                else:
                    assert "orders monetary column is total" in str(messages)
                    assert "historical_summaries" not in str(messages)
                    return ModelResponse(
                        parts=[TextPart("Current goal and the extracted Fact are available.")]
                    )
                return ModelResponse(parts=[ToolCallPart(tool, args, tool_call_id=f"demo-{step}")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="orders")],
                toolsets=[context_tools(manager, "orders")],
            )
            result = await agent.run("Continue the order query task.", message_history=raw)
            print(result.output)
            print(json.dumps(objective.read("orders"), ensure_ascii=False, indent=2))
            print("Fact verification:", store.load(actor, saved["fact_id"]).verification)
            print("Original message count:", len(result.all_messages()))
            print(
                "Archive summary hits:",
                len(await manager.search_history("orders", "schema", kind="summary")),
            )
            print("Generative model calls:", step, "(all deterministic, local)")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="working-context-demo-") as directory:
        asyncio.run(main(directory))
