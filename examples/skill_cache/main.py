"""Progressive Skill activation, stable order and prompt-cache diagnostics, no API key needed."""

import asyncio
import json
from dataclasses import replace

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import CompilerPolicy, Context, SkillRegistry, standard
from reactive_context.adapters.pydantic_ai import PromptCachePolicy, ReactiveContextCapability
from reactive_context.sources.config import ConfigSource


async def main():
    sources = {
        key: ConfigSource(key, f"{key}: approved workflow version 1")
        for key in ["skill.format", "skill.sql"]
    }
    counts = {key: 0 for key in sources}
    observed = []

    def counted(part):
        async def resolve(ctx):
            counts[part.key] += 1
            return await part.resolve(ctx)

        return replace(part, resolve=resolve)

    def respond(messages, info):
        bodies = []
        order = []
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    bodies.append(part.content)
                    try:
                        value = json.loads(part.content)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and "skill_id" in value:
                        order.append(value["skill_id"])
        observed.append((order, bodies))
        return ModelResponse(parts=[TextPart("active: " + ", ".join(order))])

    policy = CompilerPolicy(grants={"standard.system": "system"}, prefix_parts={"schema"})
    async with Context(
        [
            standard.system("Follow host policy; use only activated Skills."),
            ConfigSource("schema", {"table": "accounts"}).part,
        ],
        compiler_policy=policy,
    ) as context:
        registry = SkillRegistry(context)
        for key, source in sources.items():
            registry.register(
                counted(source.part), name=key, description=f"Use {key}", trusted=True
            )
        agent = Agent(
            FunctionModel(respond),
            name="skill_cache_demo",
            capabilities=[
                ReactiveContextCapability(
                    context, scope="cache-demo", cache=PromptCachePolicy(messages=False)
                )
            ],
        )
        history = []
        labels = [
            "只加载索引，不读取正文",
            "先激活 SQL",
            "再激活 Format，追加到 SQL 后",
            "无变化，再次请求",
            "只更新 SQL 正文，顺序不变",
        ]
        for index, label in enumerate(labels):
            if index == 1:
                registry.activate("skill.sql")
            elif index == 2:
                registry.activate("skill.format")
            elif index == 4:
                sources["skill.sql"].set("skill.sql: approved workflow version 2")
            result = await agent.run("Use the current approved Skills.", message_history=history)
            history = result.all_messages()
            event = next(
                event for event in reversed(context.events.events) if event.event == "prompt_prefix"
            )
            print(f"\n{index}. {label}")
            print("   激活顺序：", list(registry.active_order))
            print("   正文读取次数：", counts)
            print(
                f"   前缀诊断：{event.reason}；整块复用 "
                f"{event.reused_prefix_bytes}/{event.prefix_bytes} bytes"
            )
            if index == 0:
                assert counts == {"skill.format": 0, "skill.sql": 0}
            if index == 2:
                # Registration order was format/sql, activation order is sql/format.
                assert observed[1][1][: -0 or None] == observed[2][1][: len(observed[1][1])]
                assert registry.active_order == ("skill.sql", "skill.format")
            if index == 3:
                assert event.reason == "prefix_unchanged"
            if index == 4:
                sql_instruction = next(
                    json.loads(body)["instructions"]
                    for body in observed[-1][1]
                    if body.startswith("{") and json.loads(body).get("skill_id") == "skill.sql"
                )
                assert sql_instruction.endswith("version 2")
        print("\n模型返回的缓存用量：", context.events.cache_summary("cache-demo"))
        print("FunctionModel 没有真实供应商缓存；前缀字节复用只表示布局稳定性。")


if __name__ == "__main__":
    asyncio.run(main())
