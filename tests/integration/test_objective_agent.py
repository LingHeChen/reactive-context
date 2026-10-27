from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    Objective,
    ObjectivePolicy,
    Principal,
)
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol
from reactive_context.sources.catalog import ToolCatalogSource


async def test_agent_mode_supplies_conditions_and_update_tools_automatically(tmp_path):
    host, actor = Principal("host", True), Principal("agent")
    seen = []
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        store.grant_scope(host, "chat", actor.id)
        objective = Objective(store, actor, policy=ObjectivePolicy(update_mode="agent"))
        async with ContextManager(store=store, principal=actor, objective=objective) as manager:

            def respond(messages, info):
                validate_protocol(messages)
                seen.append(str(messages))
                assert "context_update_objective" in {t.name for t in info.function_tools}
                assert "update_conditions" in str(messages)
                if len(seen) == 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "context_update_objective",
                                {
                                    "value": {
                                        "goal": "Implement the requested task",
                                        "next_steps": ["Run tests"],
                                    },
                                    "expected_revision": 0,
                                },
                                tool_call_id="goal",
                            )
                        ]
                    )
                assert "Implement the requested task" in str(messages)
                return ModelResponse(parts=[TextPart("done")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
                toolsets=[context_tools(manager, "chat")],
            )
            result = await agent.run("Implement the requested task")
            assert result.output == "done" and objective.read("chat")["revision"] == 1
            assert len(seen) == 2


async def test_manager_default_projects_short_history_and_fact_without_old_summary(tmp_path):
    host, actor = Principal("host", True), Principal("agent")
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        store.grant_scope(host, "chat", actor.id)
        store.register_schema("note", dict)
        evidence = store.observe_evidence(actor, "chat", "current", "1")
        fact = store.commit(
            actor,
            store.propose_fact(
                actor, "chat", "note", "fact", {"text": "KEEP_THIS_FACT"}, [evidence]
            ),
        )
        objective = Objective(store, actor)
        objective.update("chat", {"goal": "CURRENT_WORK_GOAL"})
        async with ContextManager(
            store=store,
            principal=actor,
            objective=objective,
            compiler_policy=CompilerPolicy(max_tokens=8000),
        ) as manager:
            await manager.activate("chat", fact.id)
            raw = []
            for i in range(20):
                raw += [
                    ModelRequest.user_text_prompt(f"OLD_{i}_QUESTION " + "x" * 500),
                    ModelResponse(parts=[TextPart(f"OLD_{i}_ANSWER " + "y" * 500)]),
                ]

            def respond(messages, info):
                body = str(messages)
                assert "OLD_0_QUESTION" not in body and "CURRENT_QUESTION" in body
                assert "KEEP_THIS_FACT" in body and "CURRENT_WORK_GOAL" in body
                assert "historical_summaries" not in body
                validate_protocol(messages)
                return ModelResponse(parts=[TextPart("done")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
            )
            result = await agent.run("CURRENT_QUESTION", message_history=raw)
            assert len(result.all_messages()) > 40
            assert len(manager.history.tail_ids("chat", limit=50)) > 40


async def test_objective_update_tool_survives_business_catalog_filter(tmp_path):
    from pydantic_ai import Tool

    def query() -> str:
        return "query result"

    tool = Tool(query)
    catalog = ToolCatalogSource(
        "tools",
        [{"name": tool.name, "parameters_json_schema": tool.tool_def.parameters_json_schema}],
    )
    host, actor = Principal("host", True), Principal("agent")
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        store.grant_scope(host, "chat", actor.id)
        objective = Objective(store, actor, policy=ObjectivePolicy(update_mode="agent"))
        async with ContextManager(
            [catalog.part], store=store, principal=actor, objective=objective
        ) as manager:
            manager.publish_catalog(host, "chat")
            await manager.activate("chat", (await manager.query("chat"))[0]["id"])

            def respond(messages, info):
                names = {t.name for t in info.function_tools}
                assert {"query", "context_update_objective"} <= names
                return ModelResponse(parts=[TextPart("done")])

            result = await Agent(
                FunctionModel(respond),
                tools=[tool],
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
            ).run("work")
            assert result.output == "done"


async def test_threshold_goal_update_precedes_current_request_projection(tmp_path):
    from reactive_context import HistoryPolicy

    host, actor = Principal("host", True), Principal("agent")
    updates = []
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        store.grant_scope(host, "chat", actor.id)

        def updater(request):
            updates.append(request)
            return {"goal": "THRESHOLD_UPDATED_GOAL"}

        objective = Objective(
            store,
            actor,
            updater=updater,
            policy=ObjectivePolicy(
                update_mode="threshold", context_ratio=0.25, min_update_interval=0
            ),
        )
        async with ContextManager(
            store=store,
            principal=actor,
            objective=objective,
            compiler_policy=CompilerPolicy(max_tokens=15000),
        ) as manager:
            manager.history.policy = HistoryPolicy(recent_groups=2)
            raw = []
            for _i in range(20):
                raw += [
                    ModelRequest.user_text_prompt("past " + "x" * 1000),
                    ModelResponse(parts=[TextPart("answer " + "y" * 1000)]),
                ]

            def respond(messages, info):
                assert "THRESHOLD_UPDATED_GOAL" in str(messages)
                assert "context_update_objective" not in {t.name for t in info.function_tools}
                return ModelResponse(parts=[TextPart("done")])

            await Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
            ).run("current " + "c" * 5000, message_history=raw)
            assert len(updates) == 1


async def test_agent_extracts_history_fact_and_keeps_it_after_short_history_moves(tmp_path):
    from pydantic_ai.messages import ToolReturnPart

    host, actor = Principal("host", True), Principal("agent")
    state, calls = {}, []
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        store.grant_scope(host, "chat", actor.id)
        store.register_schema("finding", dict[str, str])
        async with ContextManager(store=store, principal=actor) as manager:

            def respond(messages, info):
                calls.append(messages)
                if len(calls) > 1:
                    value = next(
                        p.content
                        for m in reversed(messages)
                        if isinstance(m, ModelRequest)
                        for p in reversed(m.parts)
                        if isinstance(p, ToolReturnPart)
                    )
                if len(calls) == 1:
                    name, args = "context_recent_history", {"limit": 1}
                elif len(calls) == 2:
                    name, args = (
                        "context_propose_fact",
                        {
                            "schema_id": "finding",
                            "key": "budget",
                            "value": {"text": "CUSTOMER_BUDGET_FIVE"},
                            "evidence": [value[0]["evidence_id"]],
                        },
                    )
                elif len(calls) == 3:
                    name, args = "context_commit_fact", {"proposal_id": value["proposal_id"]}
                elif len(calls) == 4:
                    state["fact_id"] = value["part_id"]
                    name, args = "context_activate", {"part_id": state["fact_id"]}
                else:
                    assert "CUSTOMER_BUDGET_FIVE" in str(messages)
                    return ModelResponse(parts=[TextPart("remembered")])
                return ModelResponse(
                    parts=[ToolCallPart(name, args, tool_call_id=f"call-{len(calls)}")]
                )

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
                toolsets=[context_tools(manager, "chat")],
            )
            result = await agent.run("The customer budget is five dollars. Remember it as a fact.")
            assert result.output == "remembered"
            fact = store.load(actor, state["fact_id"])
            assert fact.verification == "unverified" and fact.evidence
            snapshot = await manager.next_context("chat")
            assert all(m.role == "user" for m in manager.compile(snapshot).provider_messages)


async def test_generative_objective_updater_uses_structured_output():
    from reactive_context import ObjectiveUpdateRequest
    from reactive_context.adapters.objective import PydanticObjectiveUpdater

    def respond(messages, info):
        assert "CURRENT_GOAL" in str(messages) and "NEW_CONSTRAINT" in str(messages)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "goal": "CURRENT_GOAL",
                        "constraints": ["NEW_CONSTRAINT"],
                        "next_steps": ["Run tests"],
                    },
                    tool_call_id="result",
                )
            ]
        )

    updater = PydanticObjectiveUpdater(FunctionModel(respond))
    value = await updater(
        ObjectiveUpdateRequest(
            "chat", {"goal": "CURRENT_GOAL"}, 1, ({"content": "NEW_CONSTRAINT"},), "threshold"
        )
    )
    assert value.goal == "CURRENT_GOAL" and value.constraints == ["NEW_CONSTRAINT"]
