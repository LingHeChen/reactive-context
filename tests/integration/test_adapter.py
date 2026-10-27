from copy import deepcopy

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from reactive_context import CompilerPolicy, Context, standard
from reactive_context.adapters.pydantic_ai import (
    ReactiveContextCapability,
    managed_result,
    validate_protocol,
)
from reactive_context.errors import ProtocolInvalid
from reactive_context.sources.catalog import ToolCatalogSource
from reactive_context.sources.config import ConfigSource


async def test_consecutive_tools_refresh_managed_history_and_resume():
    source = ConfigSource("rules", "VERSION_A_ONLY")
    calls, original_results = [], []
    context = Context([source.part])

    def respond(messages, info):
        validate_protocol(messages)
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            assert "VERSION_A_ONLY" in str(messages)
            return ModelResponse(parts=[ToolCallPart("read_rules", {}, tool_call_id="read-1")])
        if len(calls) == 2:
            assert "VERSION_B_ONLY" in str(messages)
            assert "VERSION_A_ONLY" not in str(messages)
            assert "STALE_OR_WITHDRAWN" in str(messages)
            return ModelResponse(parts=[ToolCallPart("act", {}, tool_call_id="act-1")])
        return ModelResponse(parts=[TextPart("completed")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[ReactiveContextCapability(context, scope="session")],
        name="consecutive_tools",
    )

    @agent.tool_plain
    async def read_rules():
        snapshot = (await context.next_context("session")).part("rules")
        result = managed_result(snapshot.value, snapshot)
        original_results.append(result)
        source.set("VERSION_B_ONLY")
        return result

    @agent.tool_plain
    def act() -> str:
        return "action-result"

    async with context:
        result = await agent.run("start")
        assert result.output == "completed"
        assert len(calls) == 3
        history = result.all_messages()
        assert "VERSION_A_ONLY" in str(history)
        assert "VERSION_B_ONLY" not in str(history)
        assert "STALE_OR_WITHDRAWN" not in str(history)
        saved = deepcopy(history)
        await agent.run("resume", message_history=history)
        assert history == saved
        assert "VERSION_A_ONLY" not in str(calls[-1])
        assert "VERSION_B_ONLY" in str(calls[-1])
        validate_protocol(history)


async def test_parallel_tools_and_model_retry():
    source = ConfigSource("a", "data")
    calls = []
    counts = {"left": 0, "right": 0}

    def respond(messages, info):
        validate_protocol(messages)
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("left", {}, tool_call_id="L"),
                    ToolCallPart("right", {}, tool_call_id="R"),
                ]
            )
        if len(calls) == 2:
            return ModelResponse(parts=[ToolCallPart("left", {}, tool_call_id="L2")])
        return ModelResponse(parts=[TextPart("done")])

    async with Context([source.part]) as context:
        agent = Agent(
            FunctionModel(respond),
            capabilities=[ReactiveContextCapability(context)],
            name="parallel_and_retry",
        )

        @agent.tool_plain(retries=2)
        def left() -> str:
            counts["left"] += 1
            if counts["left"] == 1:
                raise ModelRetry("try again")
            return "left"

        @agent.tool_plain
        def right() -> str:
            counts["right"] += 1
            return "right"

        result = await agent.run("start")
        assert result.output == "done"
        assert counts == {"left": 2, "right": 1}
        assert len([e for e in context.events.events if e.event == "invoke"]) == 3
        validate_protocol(result.all_messages())


async def test_streaming_refresh_between_tool_steps():
    source = ConfigSource("a", "A-stream")
    calls = []

    async def stream(messages, info):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            from pydantic_ai.models.function import DeltaToolCall

            yield {0: DeltaToolCall(name="change", json_args="{}", tool_call_id="stream-1")}
        else:
            assert "B-stream" in str(messages)
            assert "A-stream" not in str(messages)
            yield "streamed "
            yield "answer"

    async with Context([source.part]) as context:
        agent = Agent(
            FunctionModel(stream_function=stream),
            name="streaming",
            capabilities=[ReactiveContextCapability(context)],
        )

        @agent.tool_plain
        def change() -> str:
            source.set("B-stream")
            return "updated"

        async with agent.run_stream("start") as result:
            output = await result.get_output()
        assert output == "streamed answer"
        assert len(calls) == 2
        assert "B-stream" not in str(result.all_messages())


async def test_structured_output_testmodel_and_multiple_instances():
    class Output(BaseModel):
        answer: str

    async with Context([ConfigSource("a", "evidence").part]) as context:
        agent = Agent(
            TestModel(),
            output_type=Output,
            name="structured",
            capabilities=[ReactiveContextCapability(context)],
        )
        result = await agent.run("question")
        assert isinstance(result.output, Output)
        validate_protocol(result.all_messages())


async def test_host_instruction_does_not_pollute_history():
    async with Context(
        [standard.system("UNIQUE_HOST_POLICY")],
        compiler_policy=CompilerPolicy(grants={"standard.system": "system"}),
    ) as context:
        observed = []

        def respond(messages, info):
            observed.append(str(messages))
            return ModelResponse(parts=[TextPart("ok")])

        agent = Agent(
            FunctionModel(respond),
            name="trusted_policy",
            capabilities=[ReactiveContextCapability(context)],
        )
        result = await agent.run("hello")
        assert "UNIQUE_HOST_POLICY" in observed[0]
        assert "UNIQUE_HOST_POLICY" not in str(result.all_messages())


@pytest.mark.parametrize(
    "messages",
    [
        [ModelRequest(parts=[ToolReturnPart("unknown", "x", tool_call_id="bad")])],
        [ModelResponse(parts=[ToolCallPart("x", {}, tool_call_id="bad")])],
        [
            ModelResponse(parts=[ToolCallPart("x", {}, tool_call_id="bad")]),
            ModelRequest(parts=[ToolReturnPart("wrong", "x", tool_call_id="bad")]),
        ],
    ],
)
def test_invalid_protocol_fails_closed(messages):
    with pytest.raises(ProtocolInvalid):
        validate_protocol(messages)


async def test_server_side_hidden_state_rejected():
    async with Context([ConfigSource("a", "A").part]) as context:
        agent = Agent(
            TestModel(),
            name="server_state",
            model_settings={"openai_previous_response_id": "auto"},
            capabilities=[ReactiveContextCapability(context)],
        )
        with pytest.raises(ProtocolInvalid, match="hidden state"):
            await agent.run("hello")


async def test_catalog_selects_only_registered_tools():
    catalog = ToolCatalogSource(
        "catalog",
        [
            {
                "name": "allowed",
                "parameters_json_schema": {
                    "additionalProperties": False,
                    "properties": {},
                    "type": "object",
                },
            }
        ],
    )
    async with Context([catalog.part]) as context:
        agent = Agent(
            TestModel(), name="catalog", capabilities=[ReactiveContextCapability(context)]
        )

        @agent.tool_plain
        def allowed() -> str:
            return "allowed"

        @agent.tool_plain
        def hidden() -> str:
            raise AssertionError("catalog must hide this tool")

        await agent.run("test")
        catalog.set([{"name": "invented", "parameters_json_schema": {}}])
        with pytest.raises(ProtocolInvalid):
            await agent.run("test")


async def test_model_exception_leaves_source_history_untouched():
    def respond(messages, info):
        raise RuntimeError("model failed")

    history = [ModelRequest(parts=[])]
    saved = deepcopy(history)
    async with Context([ConfigSource("a", "A").part]) as context:
        agent = Agent(
            FunctionModel(respond),
            name="exception",
            capabilities=[ReactiveContextCapability(context)],
        )
        with pytest.raises(RuntimeError, match="model failed"):
            await agent.run("test", message_history=history)
        assert history == saved


async def test_deferred_tool_manual_resume_uses_new_context():
    from pydantic_ai import DeferredToolRequests, DeferredToolResults
    from pydantic_ai.exceptions import CallDeferred

    source = ConfigSource("rules", "BEFORE_RESUME")
    calls = []

    def respond(messages, info):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("external", {}, tool_call_id="deferred-1")])
        assert "AFTER_RESUME" in str(messages)
        assert "BEFORE_RESUME" not in str(messages)
        return ModelResponse(parts=[TextPart("resumed")])

    async with Context([source.part]) as context:
        agent = Agent(
            FunctionModel(respond),
            name="deferred_resume",
            output_type=[str, DeferredToolRequests],
            capabilities=[ReactiveContextCapability(context, scope="resume-task")],
        )

        @agent.tool_plain
        def external() -> str:
            raise CallDeferred()

        first = await agent.run("start")
        assert isinstance(first.output, DeferredToolRequests)
        saved = deepcopy(first.all_messages())
        source.set("AFTER_RESUME")
        result = await agent.run(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(calls={"deferred-1": "external result"}),
        )
        assert result.output == "resumed"
        assert first.all_messages() == saved
        validate_protocol(result.all_messages())


async def test_output_validation_retry_refreshes_context():
    source = ConfigSource("rules", "BEFORE_VALIDATION")
    calls = []
    async with Context([source.part]) as context:

        def respond(messages, info):
            calls.append(deepcopy(messages))
            return ModelResponse(parts=[TextPart("candidate")])

        agent = Agent(
            FunctionModel(respond),
            name="output_validation_retry",
            capabilities=[ReactiveContextCapability(context)],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if len(calls) == 1:
                source.set("AFTER_VALIDATION")
                raise ModelRetry("validate again")
            return value

        result = await agent.run("start")
        assert result.output == "candidate"
        assert len(calls) == 2
        assert "AFTER_VALIDATION" in str(calls[1])
        assert "BEFORE_VALIDATION" not in str(calls[1])


async def test_hidden_state_in_model_default_settings_also_rejected():
    async with Context([ConfigSource("a", "A").part]) as context:
        agent = Agent(
            TestModel(settings={"openai_previous_response_id": "auto"}),
            name="default_hidden_state",
            capabilities=[ReactiveContextCapability(context)],
        )
        with pytest.raises(ProtocolInvalid, match="hidden state"):
            await agent.run("hello")


@pytest.mark.parametrize("provenance", ["invalid", {}, {"resource_id": [], "revision": "A"}])
async def test_malformed_managed_provenance_fails_closed(provenance):
    history = [
        ModelResponse(parts=[ToolCallPart("read", {}, tool_call_id="p")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "read", "old", tool_call_id="p", metadata={"reactive_context": provenance}
                )
            ]
        ),
    ]
    saved = deepcopy(history)
    async with Context([ConfigSource("a", "A").part]) as context:
        agent = Agent(
            TestModel(),
            name="malformed_provenance",
            capabilities=[ReactiveContextCapability(context)],
        )
        with pytest.raises(ProtocolInvalid, match="managed provenance"):
            await agent.run("hello", message_history=history)
        assert history == saved
