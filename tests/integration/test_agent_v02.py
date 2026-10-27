from copy import deepcopy

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from reactive_context import CompilerPolicy
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol
from reactive_context.history import HistoryManager, HistoryPolicy, SummaryDraft
from reactive_context.manager import ContextManager
from reactive_context.sources.config import ConfigSource
from reactive_context.store import ContextStore, Principal

HOST = Principal("host", admin=True)
USER = Principal("agent")


class Note(BaseModel):
    text: str


def provision(tmp_path):
    store = ContextStore(tmp_path)
    store.create_scope(HOST, "chat")
    store.grant_scope(HOST, "chat", USER.id)
    store.register_schema("note", Note)
    return store


def last_return(messages):
    return next(
        p.content
        for m in reversed(messages)
        if isinstance(m, ModelRequest)
        for p in reversed(m.parts)
        if isinstance(p, ToolReturnPart)
    )


async def test_agent_discovers_activates_refreshes_and_creates_fact_without_new_agent_loop(
    tmp_path,
):
    store = provision(tmp_path)
    source = ConfigSource("skill", "RULE_A", revision_mode="release")
    seen = []
    memory = {}
    async with ContextManager(
        [source.part],
        store=store,
        principal=USER,
        compiler_policy=CompilerPolicy(max_tokens=150000, output_reserve=1000),
    ) as manager:
        manager.publish_catalog(HOST, "chat")
        history = HistoryManager(store, USER)

        def respond(messages, info):
            seen.append(deepcopy(messages))
            step = len(seen)
            if step == 1:
                tool, args = "context_query", {"intent": "skill"}
            elif step == 2:
                memory["part"] = last_return(messages)[0]["id"]
                tool, args = "context_load", {"part_id": memory["part"]}
            elif step == 3:
                tool, args = "context_activate", {"part_id": memory["part"]}
            elif step == 4:
                assert "RULE_A" in str(messages)
                tool, args = "publish_update", {}
            elif step == 5:
                assert "RULE_B" in str(messages) and "RULE_A" not in str(messages)
                tool, args = "context_load", {"part_id": memory["part"]}
            elif step == 6:
                evidence = last_return(messages)["evidence_id"]
                tool, args = (
                    "context_propose_fact",
                    {
                        "schema_id": "note",
                        "key": "current",
                        "value": {"text": "uses B"},
                        "evidence": [evidence],
                    },
                )
            elif step == 7:
                tool, args = (
                    "context_commit_fact",
                    {"proposal_id": last_return(messages)["proposal_id"]},
                )
            elif step == 8:
                memory["fact"] = last_return(messages)["part_id"]
                tool, args = "context_activate", {"part_id": memory["fact"]}
            else:
                assert "uses B" in str(messages)
                return ModelResponse(parts=[TextPart("done")])
            return ModelResponse(parts=[ToolCallPart(tool, args, tool_call_id=f"call-{step}")])

        agent = Agent(
            FunctionModel(respond),
            toolsets=[context_tools(manager, "chat", history=history)],
            capabilities=[ReactiveContextCapability(manager, scope="chat", history=history)],
            name="v02_e2e",
        )

        @agent.tool_plain
        def publish_update() -> str:
            source.set("RULE_B")
            return "published new release"

        result = await agent.run("Use the current rule and save a supporting fact")
        assert result.output == "done"
        assert "RULE_A" in str(result.all_messages())
        validate_protocol(result.all_messages())
        original = deepcopy(result.all_messages())
        await agent.run("continue", message_history=result.all_messages())
        assert result.all_messages() == original
        assert store.load(USER, memory["fact"]).verification == "unverified"
        rows = store.db.execute("SELECT event_id FROM history_events ORDER BY seq").fetchall()
        assert "RULE_A" in str(history.retrieve("chat", [r[0] for r in rows]))
        assert "uses B" in str(history.retrieve("chat", [r[0] for r in rows]))
    assert not source._subscribers
    store.close()


async def test_streaming_and_resume_keep_archive_and_refresh_between_tool_calls(tmp_path):
    store = provision(tmp_path)
    source = ConfigSource("schema", "SCHEMA_A")
    counter = 0
    async with ContextManager([source.part], store=store, principal=USER) as manager:
        manager.publish_catalog(HOST, "chat")
        row = (await manager.query("chat"))[0]
        await manager.activate("chat", row["id"])
        history = HistoryManager(store, USER)

        async def stream(messages, info):
            nonlocal counter
            counter += 1
            if counter == 1:
                yield {0: DeltaToolCall(name="change", json_args="{}", tool_call_id="s1")}
            else:
                assert "SCHEMA_B" in str(messages)
                yield "done"

        agent = Agent(
            FunctionModel(stream_function=stream),
            capabilities=[ReactiveContextCapability(manager, scope="chat", history=history)],
            name="v02_stream",
        )

        @agent.tool_plain
        def change() -> str:
            source.set("SCHEMA_B")
            return "updated"

        async with agent.run_stream("start") as result:
            assert await result.get_output() == "done"
        validate_protocol(result.all_messages())
        assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] >= 4
    store.close()


async def test_automatic_compaction_uses_closed_tool_groups_and_preserves_raw_history(tmp_path):
    from reactive_context.adapters.history import history_groups

    store = provision(tmp_path)
    calls = []

    def summarize(events):
        calls.append(events)
        return SummaryDraft(
            text="Earlier lookup completed. Goal: inspect current schema.",
            evidence_ids=[HistoryManager.event_id(events[0])],
        )

    history = HistoryManager(
        store,
        USER,
        compactor=summarize,
        validator=lambda source, draft: True,
        policy=HistoryPolicy(
            projection="compact", recent_groups=2, segment_groups=4, max_compactions_per_boundary=20
        ),
    )
    raw = []
    for _i in range(8):
        raw.extend(
            [
                ModelRequest.user_text_prompt("old question " + "x" * 2500),
                ModelResponse(parts=[TextPart("old answer " + "y" * 2500)]),
            ]
        )
    saved = deepcopy(raw)
    async with ContextManager(
        store=store,
        principal=USER,
        compiler_policy=CompilerPolicy(max_tokens=15000, output_reserve=1000),
    ) as manager:

        def respond(messages, info):
            validate_protocol(messages)
            assert "historical_summaries" in str(messages)
            assert "CURRENT_REQUEST" in str(messages)
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[ReactiveContextCapability(manager, scope="chat", history=history)],
            name="v02_compaction",
        )
        result = await agent.run("CURRENT_REQUEST", message_history=raw)
        assert raw == saved and calls
        assert "historical_summaries" not in str(result.all_messages())
        assert len(history_groups(result.all_messages())) >= len(raw)
        assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] >= len(raw)
    store.close()
