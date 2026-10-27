"""Loaded evidence survives the current turn, without bypassing live authorization."""

from copy import deepcopy

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import CompilerPolicy, ContextManager, ContextStore, Principal
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, managed_result
from reactive_context.sources.config import ConfigSource

HOST = Principal("host", True)
USER = Principal("reader")
SECRET = "LOADED_BODY_ONLY"


class Note(BaseModel):
    text: str


def provision(path):
    store = ContextStore(path)
    store.create_scope(HOST, "chat")
    store.grant_scope(HOST, "chat", USER.id)
    store.register_schema("note", Note)
    evidence = store.observe_evidence(USER, "chat", "original", "1")
    proposal = store.propose_fact(USER, "chat", "note", "one", {"text": SECRET}, [evidence])
    return store, store.commit(USER, proposal).id


@pytest.mark.parametrize("with_history", [False, True])
async def test_load_visible_through_turn_then_withdrawn_without_activation(tmp_path, with_history):
    store, part_id = provision(tmp_path)
    seen = []
    async with ContextManager(
        store=store,
        principal=USER,
        history=None if with_history else False,
        compiler_policy=CompilerPolicy(max_tokens=128000),
    ) as manager:

        def reply(messages, info):
            seen.append(deepcopy(messages))
            step = len(seen)
            if step == 1:
                return ModelResponse(
                    parts=[ToolCallPart("context_load", {"part_id": part_id}, "load")]
                )
            if step == 2:
                assert SECRET in str(messages)
                assert not store.active_refs(USER, "chat")
                return ModelResponse(parts=[ToolCallPart("noop", {}, "next")])
            if step == 3:
                assert SECRET in str(messages)
            else:
                assert SECRET not in str(messages)
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(reply),
            toolsets=[context_tools(manager, "chat")],
            capabilities=[ReactiveContextCapability(manager, scope="chat", cache=None)],
        )

        @agent.tool_plain
        def noop() -> str:
            return "no change"

        result = await agent.run("read a note")
        assert not store.active_refs(USER, "chat")
        assert SECRET in str(
            result.all_messages()
        )  # request projection does not mutate raw history
        await agent.run("continue", message_history=result.all_messages())
        assert SECRET not in str(seen[-1])
    store.close()


@pytest.mark.parametrize("change", ["revoke", "revise", "retire", "scope"])
async def test_transient_read_rechecks_authority_version_and_scope(tmp_path, change):
    store, part_id = provision(tmp_path)
    step = 0
    async with ContextManager(
        store=store,
        principal=USER,
        history=False,
        compiler_policy=CompilerPolicy(max_tokens=128000),
    ) as manager:

        def reply(messages, info):
            nonlocal step
            step += 1
            if step == 1:
                return ModelResponse(parts=[ToolCallPart("read_then_change", {}, "read")])
            assert SECRET not in str(messages)
            assert "STALE_OR_WITHDRAWN" in str(messages)
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(reply),
            capabilities=[ReactiveContextCapability(manager, scope="chat", cache=None)],
        )

        @agent.tool_plain
        async def read_then_change():
            loaded = await manager.load("chat", part_id)
            if change == "revoke":
                store.revoke_part_access(HOST, part_id, USER.id)
            elif change == "retire":
                store.retire(USER, part_id)
            elif change == "revise":
                evidence = store.observe_evidence(USER, "chat", "update", "2")
                proposal = store.propose_fact(
                    USER, "chat", "note", "one", {"text": "NEW"}, [evidence], expected_revision=1
                )
                store.commit(USER, proposal)
            return managed_result(
                {"text": SECRET},
                loaded.snapshot,
                read_part_id=part_id,
                read_scope="wrong" if change == "scope" else "chat",
            )

        await agent.run("read a note")
    store.close()


async def test_resource_read_rechecks_source_revision_and_activation_persists(tmp_path):
    store, _ = provision(tmp_path)
    source = ConfigSource("skill", SECRET)
    step = 0
    async with ContextManager(
        [source.part],
        store=store,
        principal=USER,
        history=False,
        compiler_policy=CompilerPolicy(max_tokens=128000),
    ) as manager:
        manager.publish_catalog(HOST, "chat")
        part_id = (await manager.query("chat", kind="resource"))[0]["id"]

        def reply(messages, info):
            nonlocal step
            step += 1
            if step == 1:
                return ModelResponse(
                    parts=[ToolCallPart("context_load", {"part_id": part_id}, "load")]
                )
            if step == 2:
                assert SECRET in str(messages)
                return ModelResponse(
                    parts=[ToolCallPart("context_activate", {"part_id": part_id}, "activate")]
                )
            if step == 3:
                assert SECRET in str(messages)
                return ModelResponse(parts=[ToolCallPart("update_source", {}, "update")])
            assert SECRET not in str(messages)
            assert "NEW_SOURCE" in str(messages)
            assert "STALE_OR_WITHDRAWN" in str(messages)
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(reply),
            toolsets=[context_tools(manager, "chat")],
            capabilities=[ReactiveContextCapability(manager, scope="chat", cache=None)],
        )

        @agent.tool_plain
        def update_source() -> str:
            source.set("NEW_SOURCE")
            return "changed"

        await agent.run("read a resource")
    store.close()


async def test_explicit_deactivation_ends_earlier_current_turn_read(tmp_path):
    store, part_id = provision(tmp_path)
    step = 0
    async with ContextManager(
        store=store,
        principal=USER,
        history=False,
        compiler_policy=CompilerPolicy(max_tokens=128000),
    ) as manager:

        def reply(messages, info):
            nonlocal step
            step += 1
            if step == 1:
                return ModelResponse(
                    parts=[ToolCallPart("context_load", {"part_id": part_id}, "load")]
                )
            if step == 2:
                assert SECRET in str(messages)
                return ModelResponse(
                    parts=[ToolCallPart("context_deactivate", {"part_id": part_id}, "drop")]
                )
            assert SECRET not in str(messages)
            assert "STALE_OR_WITHDRAWN" in str(messages)
            return ModelResponse(parts=[TextPart("done")])

        await Agent(
            FunctionModel(reply),
            toolsets=[context_tools(manager, "chat")],
            capabilities=[ReactiveContextCapability(manager, scope="chat", cache=None)],
        ).run("read")
    store.close()


async def test_fact_search_ranks_identity_before_limit_and_preserves_id_paging(tmp_path):
    store, first = provision(tmp_path)
    evidence = store.observe_evidence(USER, "chat", "travel", "1")
    for i in range(30):
        proposal = store.propose_fact(
            USER,
            "chat",
            "note",
            f"Other-{i}/day-3",
            {"text": "Tony day 3 dinner Yakima " * 4},
            [evidence],
        )
        store.commit(USER, proposal)
    proposal = store.propose_fact(
        USER, "chat", "note", "Tony/day-3", {"text": "Tony dinner Yakima"}, [evidence]
    )
    target = store.commit(USER, proposal).id
    async with ContextManager(store=store, principal=USER) as manager:
        hits = await manager.search_facts("chat", "Tony day 3 dinner Yakima", limit=1)
        assert hits[0]["id"] == target
        assert "value" not in hits[0]
        page1 = await manager.query("chat", limit=10)
        page2 = await manager.query("chat", limit=10, after_id=page1[-1]["id"])
        assert max(r["id"] for r in page1) < min(r["id"] for r in page2)
        store.revoke_part_access(HOST, target, USER.id)
        assert all(r["id"] != target for r in await manager.search_facts("chat", "Tony day 3"))
    store.close()
