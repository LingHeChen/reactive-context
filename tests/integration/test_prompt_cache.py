import json
from copy import deepcopy

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, SystemPromptPart, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from reactive_context import CompilerPolicy, Context, SkillRegistry, skill_part, standard
from reactive_context.adapters.cache import PrefixMonitor
from reactive_context.adapters.pydantic_ai import PromptCachePolicy, ReactiveContextCapability
from reactive_context.sources.catalog import ToolCatalogSource
from reactive_context.sources.config import ConfigSource


async def test_skills_are_system_prefix_data_remains_user_and_history_is_untouched():
    skill = ConfigSource("skill", "APPROVED_SKILL_BODY")
    schema = ConfigSource("schema", {"table": "accounts"})
    policy = CompilerPolicy(grants={"skill": "developer"}, prefix_parts={"schema"})
    seen = []

    def respond(messages, info):
        seen.append((deepcopy(messages), deepcopy(info.model_settings)))
        return ModelResponse(parts=[TextPart("answer")])

    async with Context([skill_part(skill.part), schema.part], compiler_policy=policy) as context:
        agent = Agent(
            FunctionModel(respond),
            name="prefix_layout",
            system_prompt="HOST_POLICY",
            capabilities=[
                ReactiveContextCapability(
                    context, scope="session", cache=PromptCachePolicy(messages=False)
                )
            ],
        )
        first = await agent.run("QUESTION_ONE")
        saved = deepcopy(first.all_messages())
        await agent.run("QUESTION_TWO", message_history=first.all_messages())
        assert first.all_messages() == saved
        assert "APPROVED_SKILL_BODY" not in str(saved)
        assert "accounts" not in str(saved)
        prefix = seen[0][0][0]
        assert isinstance(prefix, ModelRequest)
        assert isinstance(prefix.parts[0], SystemPromptPart)
        assert prefix.parts[0].content == "HOST_POLICY"
        assert isinstance(prefix.parts[1], SystemPromptPart)
        assert "APPROVED_SKILL_BODY" in prefix.parts[1].content
        assert '"table"' not in "".join(
            p.content for p in prefix.parts if isinstance(p, SystemPromptPart)
        )
        assert "accounts" in str(prefix.parts[2])
        cache_index = next(
            index
            for index, part in enumerate(prefix.parts)
            if isinstance(getattr(part, "content", None), list)
            and isinstance(part.content[0], CachePoint)
        )
        question_index = next(
            index
            for index, part in enumerate(prefix.parts)
            if getattr(part, "content", None) == "QUESTION_ONE"
        )
        assert 2 < cache_index < question_index
        prefixes = [e for e in context.events.events if e.event == "prompt_prefix"]
        assert prefixes[1].prefix_hash == prefixes[0].prefix_hash
        assert prefixes[1].reused_prefix_bytes == prefixes[1].prefix_bytes
        assert prefixes[1].reason == "prefix_unchanged"


async def test_live_skill_replacement_in_same_run_does_not_append_old_instructions():
    source = ConfigSource("skill", "OLD_SKILL_BODY")
    calls = []

    def respond(messages, info):
        from pydantic_ai.messages import ToolCallPart

        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("change", {}, tool_call_id="change-1")])
        assert "NEW_SKILL_BODY" in str(messages)
        assert "OLD_SKILL_BODY" not in str(messages)
        return ModelResponse(parts=[TextPart("ok")])

    async with Context(
        [skill_part(source.part)], compiler_policy=CompilerPolicy(grants={"skill": "developer"})
    ) as context:
        agent = Agent(
            FunctionModel(respond),
            name="skill_update",
            capabilities=[ReactiveContextCapability(context, scope="live")],
        )

        @agent.tool_plain
        async def change() -> str:
            source.set("NEW_SKILL_BODY")
            return "changed"

        result = await agent.run("start")
        assert result.output == "ok"
        assert "SKILL_BODY" not in str(result.all_messages())
        events = [e for e in context.events.events if e.event == "prompt_prefix"]
        assert events[-1].reason == "changed:instruction:skill"
        assert events[-1].reused_prefix_blocks == 2  # stable settings + tools
        assert "OLD_SKILL_BODY" not in context.export_trace()


async def test_lazy_activation_appends_existing_skill_prefix_and_is_scope_isolated():
    captured = []

    def respond(messages, info):
        captured.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("done")])

    async with Context() as context:
        skills = SkillRegistry(context)
        for key in ["a", "b"]:
            skills.register(
                ConfigSource(key, key + "-BODY").part,
                name=key,
                description="Use " + key,
                trusted=True,
            )
        capability = ReactiveContextCapability(context, scope="one")
        agent = Agent(FunctionModel(respond), capabilities=[capability], name="lazy_skills")
        skills.activate("b")
        await agent.run("first")
        skills.activate("a")
        await agent.run("second")
        first = [p.content for p in captured[0][0].parts if isinstance(p, SystemPromptPart)]
        second = [p.content for p in captured[1][0].parts if isinstance(p, SystemPromptPart)]
        assert first == second[: len(first)]
        events = [e for e in context.events.events if e.event == "prompt_prefix"]
        assert events[-1].reason == "changed:instruction:a"
        assert events[-1].reused_prefix_blocks == len(first) + 2
        other = Agent(
            FunctionModel(respond),
            name="other_scope",
            capabilities=[ReactiveContextCapability(context, scope="two")],
        )
        await other.run("third")
        events = [e for e in context.events.events if e.event == "prompt_prefix"]
        assert events[-1].scope_id == "two" and events[-1].reused_prefix_bytes == 0


async def test_tool_reordering_does_not_bust_prefix_but_schema_change_does():
    async def z(sql: str) -> str:
        return sql

    async def a(sql: str) -> str:
        return sql

    tools = [Tool(z), Tool(a)]
    entries = [
        {"name": tool.name, "parameters_json_schema": tool.tool_def.parameters_json_schema}
        for tool in tools
    ]
    catalog = ToolCatalogSource("tools", entries)
    visible = []

    def respond(messages, info):
        visible.append([tool.name for tool in info.function_tools])
        return ModelResponse(parts=[TextPart("ok")])

    async with Context([catalog.part, standard.task("data")]) as context:
        capability = ReactiveContextCapability(context, scope="catalog")
        agent = Agent(
            FunctionModel(respond), tools=tools, name="stable_tools", capabilities=[capability]
        )
        await agent.run("first")
        catalog.set(list(reversed(entries)))
        await agent.run("second")
        assert visible == [["a", "z"], ["a", "z"]]
        events = [e for e in context.events.events if e.event == "prompt_prefix"]
        assert events[-1].reason == "prefix_unchanged"
        catalog.set([entries[0]])
        await agent.run("third")
        events = [e for e in context.events.events if e.event == "prompt_prefix"]
        assert events[-1].reason == "changed:tools"
        assert events[-1].reused_prefix_blocks == 1  # settings only, tools precede instructions


async def test_model_reported_cache_usage_is_separate_from_local_cache_and_prefix_reuse():
    requests = 0

    def respond(messages, info):
        nonlocal requests
        requests += 1
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(
                input_tokens=100,
                cache_read_tokens=0 if requests == 1 else 80,
                cache_write_tokens=80 if requests == 1 else 0,
            ),
        )

    async with Context([ConfigSource("a", "DO_NOT_LOG_BODY").part]) as context:
        agent = Agent(
            FunctionModel(respond),
            name="usage_fixture",
            capabilities=[ReactiveContextCapability(context, scope="usage")],
        )
        await agent.run("first")
        await agent.run("second")
        summary = context.events.cache_summary("usage")
        assert summary["recorded_requests"] == 2
        assert summary["input_tokens"] == 200
        assert summary["cache_read_tokens"] == 80
        assert summary["cache_write_tokens"] == 80
        assert summary["cache_read_fraction"] == 0.4
        assert "DO_NOT_LOG_BODY" not in context.export_trace()
        assert any(e.cache_hit for e in context.events.events if e.event == "resolve")


@pytest.mark.parametrize(
    "cache, expected",
    [
        (None, None),
        (PromptCachePolicy(retention=False), {"cache": False}),
        (
            PromptCachePolicy(retention="1h", messages=False),
            {"cache": {"retention": "1h", "messages": False}},
        ),
    ],
)
def test_explicit_cache_controls(cache, expected):
    context = Context()
    assert ReactiveContextCapability(context, cache=cache).get_model_settings() == expected


def test_prefix_monitor_counts_whole_blocks_conservatively_and_bounds_memory():
    monitor = PrefixMonitor(max_scopes=1)
    first = monitor.observe("a", [("one", "same"), ("two", "old")])
    second = monitor.observe("a", [("one", "same"), ("two", "new")])
    assert first["reused_prefix_bytes"] == 0
    assert second["reused_prefix_bytes"] == 4
    assert second["reason"] == "changed:two"
    monitor.observe("b", [("one", "same")])
    assert monitor.observe("a", [("one", "same")])["reused_prefix_bytes"] == 0
    assert "same" not in repr(monitor._previous)


async def test_anthropic_serialized_wire_has_skill_system_prefix_and_cache_controls():
    """Actual provider SDK and JSON serialization, with a local mock HTTP transport only."""
    import httpx2
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    wire = []

    def handle(request):
        body = json.loads(request.content)
        wire.append(body)
        return httpx2.Response(
            200,
            json={
                "id": f"msg_{len(wire)}",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 100 if len(wire) == 1 else 0,
                    "cache_read_input_tokens": 0 if len(wire) == 1 else 100,
                },
            },
        )

    skill = ConfigSource("skill", "TRUSTED_SKILL_V1")
    schema = ConfigSource("schema", {"table": "accounts"})

    async def z(sql: str) -> str:
        return sql

    async def a(sql: str) -> str:
        return sql

    executable = [Tool(z), Tool(a)]
    entries = [
        {"name": tool.name, "parameters_json_schema": tool.tool_def.parameters_json_schema}
        for tool in executable
    ]
    catalog = ToolCatalogSource("tools", entries)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
        client = AsyncAnthropic(
            api_key="local-fixture-only", http_client=http, base_url="https://anthropic.test"
        )
        model = AnthropicModel(
            "claude-sonnet-4-6", provider=AnthropicProvider(anthropic_client=client)
        )
        async with Context(
            [skill_part(skill.part), schema.part, catalog.part],
            compiler_policy=CompilerPolicy(grants={"skill": "developer"}, prefix_parts={"schema"}),
        ) as context:
            agent = Agent(
                model,
                name="wire_cache_fixture",
                tools=executable,
                model_settings={"max_tokens": 64},
                capabilities=[
                    ReactiveContextCapability(
                        context, scope="wire", cache=PromptCachePolicy(messages=False)
                    )
                ],
            )
            first = await agent.run("FIRST_QUESTION")
            saved = deepcopy(first.all_messages())
            catalog.set(list(reversed(entries)))
            await agent.run("SECOND_QUESTION", message_history=first.all_messages())
            assert first.all_messages() == saved
            systems = wire[0]["system"]
            assert "TRUSTED_SKILL_V1" in str(systems)
            assert "accounts" not in str(systems)
            assert "cache_control" in str(systems)
            # Shared reference stays user data and gets an explicit end-of-prefix breakpoint.
            first_user = next(m for m in wire[0]["messages"] if m["role"] == "user")
            blocks = first_user["content"]
            schema_index = next(i for i, block in enumerate(blocks) if "accounts" in str(block))
            question_index = next(
                i for i, block in enumerate(blocks) if "FIRST_QUESTION" in str(block)
            )
            assert schema_index < question_index
            assert blocks[schema_index]["cache_control"]["type"] == "ephemeral"
            assert wire[0]["system"] == wire[1]["system"]
            assert [tool["name"] for tool in wire[0]["tools"]] == ["a", "z"]
            assert wire[0]["tools"] == wire[1]["tools"]
            summary = context.events.cache_summary("wire")
            assert summary["cache_read_tokens"] == 100
            assert summary["cache_write_tokens"] == 100
            assert summary["input_tokens"] == 240  # normalized Anthropic reads/writes included
            skill.set("TRUSTED_SKILL_V2")
            await agent.run("THIRD_QUESTION")
            assert "TRUSTED_SKILL_V1" not in str(wire[-1]["system"])
            assert "TRUSTED_SKILL_V2" in str(wire[-1]["system"])
            assert "TRUSTED_SKILL" not in str(first.all_messages())
            assert "TRUSTED_SKILL" not in context.export_trace()


async def test_run_level_disable_cache_does_not_receive_implicit_prefix_breakpoint():
    seen = []

    def respond(messages, info):
        seen.extend(deepcopy(messages))
        return ModelResponse(parts=[TextPart("ok")])

    async with Context(
        [ConfigSource("schema", "reference").part],
        compiler_policy=CompilerPolicy(prefix_parts={"schema"}),
    ) as context:
        agent = Agent(
            FunctionModel(respond),
            name="cache_disabled",
            capabilities=[ReactiveContextCapability(context)],
        )
        await agent.run("q", model_settings={"cache": False})
        assert "CachePoint" not in str(seen)
