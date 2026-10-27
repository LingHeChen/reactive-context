"""Clean projections preserve raw history and do not infer any application business rules."""

import json
from copy import deepcopy

from pydantic_ai import Agent, ToolOutput
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    HistoryGroup,
    HistoryManager,
    Principal,
)
from reactive_context.adapters.history import history_groups
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
from reactive_context.history_content import matching_preview

HOST = Principal("host", True)


def provision(path, **kwargs):
    store = ContextStore(path)
    store.create_scope(HOST, "session")
    return store, HistoryManager(store, HOST, **kwargs)


def test_protocol_and_control_chatter_not_indexed_but_raw_data_retrievable(tmp_path):
    with provision(tmp_path)[0] as store:
        history = HistoryManager(store, HOST)
        events = [
            {
                "kind": "request",
                "instructions": "INSTRUCTION_NOISE",
                "conversation_id": "TRANSPORT_NOISE",
                "parts": [{"part_kind": "user-prompt", "content": "confirmed language Mandarin"}],
            },
            {
                "kind": "response",
                "parts": [
                    {"part_kind": "text", "content": "PROCESS_NOISE"},
                    {
                        "part_kind": "tool-call",
                        "tool_name": "context_history_search",
                        "args": {"intent": "Mandarin"},
                    },
                ],
            },
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_name": "context_history_search",
                        "content": "CONTROL_NOISE",
                    }
                ],
            },
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_name": "lookup_account",
                        "content": "region: EU_WEST",
                    }
                ],
            },
            {"application_specific": {"ticket": "OPAQUE_HOST_DATA"}},
        ]
        history.observe("session", [HistoryGroup(tuple(events))])
        for term in ("INSTRUCTION_NOISE", "TRANSPORT_NOISE", "PROCESS_NOISE", "CONTROL_NOISE"):
            assert history.search("session", term) == []
        for term in ("Mandarin", "EU_WEST", "OPAQUE_HOST_DATA"):
            assert history.search("session", term)
        ids = [history.event_id(e) for e in events]
        assert history.retrieve("session", ids) == events
        # Simulate a legacy envelope index, then migrate using the existing paginated API.
        row = store.db.execute(
            "SELECT rowid FROM history_search_entries WHERE record_id=?", (ids[0],)
        ).fetchone()
        store.db.execute(
            "UPDATE history_search SET body=? WHERE rowid=?", (json.dumps(events[0]), row[0])
        )
        history.reindex("session", limit=100)
        assert history.search("session", "INSTRUCTION_NOISE") == []
        assert history.retrieve("session", ids) == events


def test_previews_show_matched_content_and_exclusion_happens_before_candidate_limit(tmp_path):
    store, history = provision(tmp_path)
    with store:
        old = {"content": "long " * 800 + "needle CONFIRMED_ANSWER"}
        current = [{"content": "needle " * 20 + str(i)} for i in range(30)]
        history.observe("session", [HistoryGroup((old, *current))])
        hits = history.search(
            "session",
            "needle",
            limit=1,
            preview_chars=400,
            exclude_event_ids=[history.event_id(e) for e in current],
        )
        assert len(hits) == 1 and hits[0]["id"] == history.event_id(old)
        assert "CONFIRMED_ANSWER" in hits[0]["text"]
        assert len(hits[0]["text"]) <= 400
        assert (
            history.search(
                "session",
                "needle",
                exclude_event_ids=[history.event_id(e) for e in (old, *current)],
            )
            == []
        )
    assert "specific_target" in matching_preview(
        "common " * 8000 + "specific_target", ["common", "specific_target"]
    )


async def test_native_search_excludes_its_current_user_turn(tmp_path):
    store, history = provision(tmp_path)
    with store:
        old = {"content": "PROJECT_ID is evergreen"}
        history.observe("session", [HistoryGroup((old,))])
        steps = []
        async with ContextManager(
            store=store,
            principal=HOST,
            history=history,
            compiler_policy=CompilerPolicy(max_tokens=128000),
        ) as manager:

            def reply(messages, info):
                steps.append(messages)
                if len(steps) == 1:
                    return ModelResponse(
                        parts=[
                            TextPart("PROCESS_SEARCH_ANNOUNCEMENT"),
                            ToolCallPart(
                                "context_history_search", {"intent": "PROJECT_ID"}, "search"
                            ),
                        ]
                    )
                content = messages[-1].parts[0].content
                if isinstance(content, str):
                    content = json.loads(content)
                assert [hit["id"] for hit in content] == [history.event_id(old)]
                return ModelResponse(parts=[TextPart("evergreen")])

            await Agent(
                FunctionModel(reply),
                capabilities=[ReactiveContextCapability(manager, scope="session", cache=None)],
            ).run("Find my PROJECT_ID, CURRENT_QUESTION")
        assert history.search("session", "PROCESS_SEARCH_ANNOUNCEMENT") == []


async def test_result_projection_is_opt_in_and_raw_transcript_is_preserved(tmp_path):
    def project(text):
        return text.split("<final>", 1)[1].split("</final>", 1)[0] if "<final>" in text else None

    store, history = provision(tmp_path, final_projector=project)
    with store:
        seen = []
        async with ContextManager(
            store=store,
            principal=HOST,
            history=history,
            compiler_policy=CompilerPolicy(max_tokens=128000),
        ) as manager:

            def reply(messages, info):
                seen.append(deepcopy(messages))
                if len(seen) == 1:
                    return ModelResponse(
                        parts=[TextPart("EXPLANATION_ONLY <final>VISIBLE_ANSWER</final>")]
                    )
                assert "VISIBLE_ANSWER" in str(messages) and "EXPLANATION_ONLY" not in str(messages)
                return ModelResponse(parts=[TextPart("<final>done</final>")])

            agent = Agent(
                FunctionModel(reply),
                capabilities=[ReactiveContextCapability(manager, scope="session", cache=None)],
            )
            first = await agent.run("first request")
            raw = deepcopy(first.all_messages())
            await agent.run("next request", message_history=raw)
            assert raw == first.all_messages()
        assert history.search("session", "VISIBLE_ANSWER")
        assert history.search("session", "EXPLANATION_ONLY") == []
        events = [e for g in history_groups(first.all_messages()) for e in g.events]
        assert history.retrieve("session", [history.event_id(e) for e in events]) == events


async def test_successful_output_arguments_are_indexed_without_callback_secrets(tmp_path):
    store, history = provision(tmp_path)
    with store:
        async with ContextManager(
            store=store,
            principal=HOST,
            history=history,
            compiler_policy=CompilerPolicy(max_tokens=128000),
        ) as manager:

            def final(answer: str):
                return {"host_internal": "HOST_CALLBACK_SECRET"}

            def reply(messages, info):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            info.output_tools[0].name, {"answer": "MODEL_VISIBLE_RESULT"}, "output"
                        )
                    ]
                )

            result = await Agent(
                FunctionModel(reply),
                output_type=ToolOutput(final),
                capabilities=[ReactiveContextCapability(manager, scope="session", cache=None)],
            ).run("provide output")
            assert result.output["host_internal"] == "HOST_CALLBACK_SECRET"
        assert history.search("session", "MODEL_VISIBLE_RESULT")
        assert history.search("session", "HOST_CALLBACK_SECRET") == []
        history.reindex("session")
        assert history.search("session", "MODEL_VISIBLE_RESULT")
