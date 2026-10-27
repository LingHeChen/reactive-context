"""Offline demo: no provider credentials or external service needed."""

import asyncio
from copy import deepcopy

from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import Context
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, managed_result
from reactive_context.sources.config import ConfigSource


async def main():
    source = ConfigSource("rules", "SQL rule A: legacy_table")
    requests = []
    async with Context([source.part]) as context:

        def model(messages, info):
            requests.append(deepcopy(messages))
            if len(requests) == 1:
                return ModelResponse(
                    parts=[ToolCallPart("load_then_update", {}, tool_call_id="load")]
                )
            assert "new_table" in str(messages)
            assert "legacy_table" not in str(messages)
            return ModelResponse(parts=[TextPart("Using rule B: new_table")])

        agent = Agent(
            FunctionModel(model),
            name="reactive_context_demo",
            capabilities=[Caching(), ReactiveContextCapability(context, scope="demo", cache=None)],
        )

        @agent.tool_plain
        async def load_then_update():
            part = (await context.next_context("demo")).part("rules")
            historical_result = managed_result(part.value, part)
            source.set("SQL rule B: new_table")
            return historical_result

        result = await agent.run("Use the current SQL rule.")
        assert "legacy_table" in str(result.all_messages())
        print(result.output)
        print("Model requests:", len(requests))
        print("Original history retained; old body absent from second model input.")
        print(
            "Snapshot:",
            next(e.snapshot_id for e in reversed(context.events.events) if e.event == "invoke"),
        )


if __name__ == "__main__":
    asyncio.run(main())
