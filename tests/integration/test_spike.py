"""M0: only public Pydantic AI interfaces; no provider credentials."""

from copy import deepcopy
from dataclasses import replace

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel


async def test_request_only_hook_and_tool_history():
    calls = []

    class Spike(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            return replace(
                request_context,
                messages=[
                    *deepcopy(list(request_context.messages)),
                    ModelRequest(parts=[UserPromptPart("request-only-marker")]),
                ],
            )

    def respond(messages, info):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="one")])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), capabilities=[Spike()], name="compatibility_spike")

    @agent.tool_plain
    def lookup() -> str:
        return "original-tool-result"

    result = await agent.run("start")
    assert len(calls) == 2
    assert all("request-only-marker" in str(call) for call in calls)
    assert "request-only-marker" not in str(result.all_messages())
    assert "original-tool-result" in str(result.all_messages())
    original = deepcopy(result.all_messages())
    await agent.run("resume", message_history=result.all_messages())
    assert result.all_messages() == original
