"""Minimal request views keep raw archives complete and retrieval executable."""

import json
from copy import deepcopy

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from reactive_context import ContextManager, ContextStore, HistoryManager, HistoryPolicy, Principal
from reactive_context.adapters.history import history_groups, minimal_history, render_history
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol

HOST = Principal("host", True)


def dialogue():
    return [
        ModelRequest(
            parts=[
                SystemPromptPart("STANDING_RULE"),
                *ModelRequest.user_text_prompt("ancient").parts,
            ]
        ),
        ModelResponse(parts=[TextPart("ancient answer")]),
        ModelRequest.user_text_prompt("PREVIOUS_QUESTION"),
        ModelResponse(parts=[TextPart("INTERMEDIATE_NOISE"), ToolCallPart("fetch", {}, "past")]),
        ModelRequest(parts=[ToolReturnPart("fetch", "UNRELATED_TOOL_DUMP", "past")]),
        ModelResponse(parts=[TextPart("PREVIOUS_FINAL")]),
        ModelRequest.user_text_prompt("CURRENT_QUESTION"),
        ModelResponse(parts=[ToolCallPart("fetch", {}, "current")]),
        ModelRequest(parts=[ToolReturnPart("fetch", "CURRENT_TOOL_RESULT", "current")]),
    ]


def project(messages, policy=None, **kwargs):
    groups = minimal_history(
        messages,
        policy=policy or HistoryPolicy(),
        total_tokens=100000,
        estimator=len,
        **kwargs,
    )
    view = render_history((), groups)
    validate_protocol(view)
    return view


def test_projection_keeps_one_pair_and_complete_active_execution_without_mutation():
    raw = dialogue()
    original = deepcopy(raw)
    view = str(project(raw))
    for retained in (
        "STANDING_RULE",
        "PREVIOUS_QUESTION",
        "PREVIOUS_FINAL",
        "CURRENT_QUESTION",
        "CURRENT_TOOL_RESULT",
    ):
        assert retained in view
    for removed in ("ancient", "INTERMEDIATE_NOISE", "UNRELATED_TOOL_DUMP"):
        assert removed not in view
    assert raw == original


@pytest.mark.parametrize(
    "policy", [HistoryPolicy(recent_turns=0), HistoryPolicy(recent_context_ratio=0)]
)
def test_zero_optional_history_keeps_current_tools(policy):
    view = str(project(dialogue(), policy))
    assert "PREVIOUS_FINAL" not in view and "CURRENT_TOOL_RESULT" in view


def test_large_previous_turn_is_omitted_whole_and_budget_can_evict_optional_pair():
    raw = dialogue()
    raw[5] = ModelResponse(parts=[TextPart("OVERSIZED_REPLY" + "x" * 2000)])
    view = str(project(raw))
    assert "PREVIOUS_QUESTION" not in view and "OVERSIZED_REPLY" not in view
    assert "CURRENT_TOOL_RESULT" in view
    view = str(project(dialogue(), reserved_tokens=99999))
    assert "PREVIOUS_FINAL" not in view and "CURRENT_TOOL_RESULT" in view


def test_structured_final_result_can_be_retained_as_data_without_replaying_tool():
    raw = [
        ModelRequest.user_text_prompt("PREVIOUS_QUESTION"),
        ModelResponse(parts=[ToolCallPart("final_result", {"decision": "APPROVED"}, "output")]),
        ModelRequest(parts=[ToolReturnPart("final_result", "Final result processed.", "output")]),
        ModelRequest.user_text_prompt("CURRENT_QUESTION"),
    ]
    view = project(raw, output_tools=("final_result",))
    assert "APPROVED" in str(view)
    assert not any(isinstance(p, ToolCallPart) for m in view for p in m.parts)


async def test_all_raw_events_are_files_and_survive_reopen_and_legacy_migration(tmp_path):
    raw = history_groups(dialogue())
    events = [e for g in raw for e in g.events]
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        history = HistoryManager(store, HOST)
        history.observe("chat", raw)
        ids = [history.event_id(e) for e in events]
        rows = store.db.execute(
            "SELECT payload, object_path FROM history_events ORDER BY seq"
        ).fetchall()
        assert len(rows) == len(events)
        for row, event in zip(rows, events, strict=True):
            assert row["payload"] is None
            assert json.loads((tmp_path / row["object_path"]).read_bytes()) == event
        # Simulate a pre-upgrade inline event, then migrate via the existing bounded API.
        body = json.dumps(events[0], sort_keys=True, ensure_ascii=False).encode()
        store.db.execute(
            "UPDATE history_events SET payload=?,object_path=NULL WHERE event_id=?", (body, ids[0])
        )
        history.reindex("chat", limit=1)
        assert (
            store.db.execute(
                "SELECT payload FROM history_events WHERE event_id=?", (ids[0],)
            ).fetchone()[0]
            is None
        )
        store.gc(HOST)
    with ContextStore(tmp_path) as store:
        history = HistoryManager(store, HOST)
        assert history.retrieve("chat", ids) == events
        assert history.search("chat", "UNRELATED_TOOL_DUMP", kind="event")


@pytest.mark.parametrize("explicit_tools", [False, True])
async def test_ambiguous_reference_searches_files_only_when_needed_without_duplicate_tools(
    tmp_path, explicit_tools
):
    seen = []
    raw = [
        ModelRequest.user_text_prompt("Store the orchid approval code: SECRET_7921"),
        ModelResponse(parts=[TextPart("Saved")]),
        ModelRequest.user_text_prompt("Unrelated intervening work"),
        ModelResponse(parts=[TextPart("TOOL_NOISE" + "x" * 2000)]),
    ]
    original = deepcopy(raw)
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        async with ContextManager(store=store, principal=HOST) as manager:

            def respond(messages, info):
                validate_protocol(messages)
                seen.append(deepcopy(messages))
                names = [t.name for t in info.function_tools]
                assert (
                    names.count("context_history_search")
                    == names.count("context_history_lookup")
                    == 1
                )
                if len(seen) == 1:
                    assert "SECRET_7921" not in str(messages)
                    assert "TOOL_NOISE" not in str(messages)
                    return ModelResponse(
                        parts=[
                            ToolCallPart("context_history_search", {"intent": "orchid"}, "search")
                        ]
                    )
                result = next(
                    p.content
                    for m in reversed(messages)
                    for p in reversed(m.parts)
                    if isinstance(p, ToolReturnPart)
                )
                if len(seen) == 2:
                    hit = next(h for h in result if "SECRET_7921" in h["text"])
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "context_history_lookup", {"event_ids": hit["event_ids"]}, "lookup"
                            )
                        ]
                    )
                assert "SECRET_7921" in str(result)
                return ModelResponse(parts=[TextPart("The code is SECRET_7921")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
                toolsets=[context_tools(manager, "chat")] if explicit_tools else [],
            )
            result = await agent.run(
                "Use the approval code we decided earlier", message_history=raw
            )
            assert result.output == "The code is SECRET_7921" and len(seen) == 3
            assert raw == original
            assert "SECRET_7921" in str(result.all_messages())
            expected_ids = {
                manager.history.event_id(event)
                for group in history_groups(result.all_messages())
                for event in group.events
            }
            archived_ids = {
                row[0] for row in store.db.execute("SELECT event_id FROM history_events")
            }
            assert archived_ids == expected_ids
            assert (
                store.db.execute(
                    "SELECT COUNT(*) FROM history_events WHERE object_path IS NULL"
                ).fetchone()[0]
                == 0
            )


async def test_self_contained_request_does_not_search_or_summarize(tmp_path):
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        async with ContextManager(store=store, principal=HOST) as manager:

            def respond(messages, info):
                assert manager.history.tail_ids("chat")
                return ModelResponse(parts=[TextPart("4")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
            )
            result = await agent.run("What is 2+2?")
            assert result.output == "4" and len(result.all_messages()) == 2


async def test_archived_volume_does_not_trigger_objective_context_threshold(tmp_path):
    from reactive_context import Objective, ObjectivePolicy

    calls = []

    def updater(request):
        calls.append(request)
        return {"goal": "SHOULD_NOT_RUN_FOR_ARCHIVE_VOLUME"}

    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        objective = Objective(
            store,
            HOST,
            updater=updater,
            policy=ObjectivePolicy(update_mode="threshold", min_update_interval=0),
        )
        async with ContextManager(store=store, principal=HOST, objective=objective) as manager:
            raw = []
            for index in range(10):
                raw.extend(
                    [
                        ModelRequest.user_text_prompt(f"Archived task {index}"),
                        ModelResponse(parts=[TextPart("UNNEEDED_ARCHIVED_BODY" + "x" * 10000)]),
                    ]
                )

            def respond(messages, info):
                assert "UNNEEDED_ARCHIVED_BODY" not in str(messages)
                return ModelResponse(parts=[TextPart("4")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="chat")],
            )
            assert (await agent.run("What is 2+2?", message_history=raw)).output == "4"
            assert not calls and objective.read("chat") is None


def test_tool_attached_user_content_does_not_start_a_new_turn_or_orphan_results():
    from pydantic_ai import BinaryContent
    from pydantic_ai.messages import UserPromptPart

    raw = [
        ModelRequest.user_text_prompt("Inspect the previously confirmed screen"),
        ModelResponse(parts=[ToolCallPart("read_image", {}, "screen")]),
        ModelRequest(
            parts=[
                ToolReturnPart("read_image", "Screenshot evidence", "screen"),
                UserPromptPart([BinaryContent(data=b"image fixture", media_type="image/png")]),
            ]
        ),
    ]
    view = project(raw)
    assert "Inspect the previously confirmed screen" in str(view)
    assert "Screenshot evidence" in str(view)
    assert len(view) == 3
