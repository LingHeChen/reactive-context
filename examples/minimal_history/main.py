"""Offline demonstration of tiny history and on-demand file archive retrieval.

Run: uv run python examples/minimal_history/main.py
No API credentials or paid requests are used.
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import ContextManager, ContextStore, Principal
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol


def archive_file_count(directory):
    return sum(p.is_file() for p in Path(directory).glob("objects/sha256/**/*"))


async def main(directory):
    host = Principal("demo-host", True)
    with ContextStore(directory) as store:
        store.create_scope(host, "session")
        async with ContextManager(store=store, principal=host) as manager:
            raw = [
                ModelRequest.user_text_prompt("审批代号 orchid，口令为 CODE_7921。"),
                ModelResponse(parts=[TextPart("已确认。")]),
            ]
            for index in range(12):
                raw.extend(
                    [
                        ModelRequest.user_text_prompt(f"无关旧工作 {index}"),
                        ModelResponse(parts=[TextPart("旧日志内容 " * 200)]),
                    ]
                )
            seen, tool_calls = [], []

            def respond(messages, info):
                validate_protocol(messages)
                seen.append(str(messages))
                returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
                if "算一下 2+2" in str(messages):
                    assert "CODE_7921" not in str(messages)
                    return ModelResponse(parts=[TextPart("4")])
                if not returns:
                    assert "CODE_7921" not in str(messages)
                    name, args = "context_history_search", {"intent": "orchid"}
                elif returns[-1].tool_name == "context_history_search":
                    hit = next(h for h in returns[-1].content if "CODE_7921" in h["text"])
                    name, args = "context_history_lookup", {"event_ids": hit["event_ids"]}
                else:
                    assert "CODE_7921" in str(returns[-1].content)
                    return ModelResponse(parts=[TextPart("此前确认的审批口令为 CODE_7921。")])
                tool_calls.append(name)
                return ModelResponse(parts=[ToolCallPart(name, args, f"read-{len(tool_calls)}")])

            agent = Agent(
                FunctionModel(respond),
                capabilities=[ReactiveContextCapability(manager, scope="session")],
            )
            direct = await agent.run("算一下 2+2", message_history=raw)
            direct_calls = len(seen)
            # A restart can pass only the new question. The old evidence is on disk.
            recovered = await agent.run("之前 orchid 审批口令是什么？")
            file_count = await asyncio.to_thread(archive_file_count, directory)
            print(
                json.dumps(
                    {
                        "direct_answer": direct.output,
                        "direct_model_calls": direct_calls,
                        "direct_history_searches": 0,
                        "first_request_contains_old_secret": "CODE_7921" in seen[0],
                        "recovered_answer": recovered.output,
                        "recovery_tools": tool_calls,
                        "archive_event_count": store.db.execute(
                            "SELECT COUNT(*) FROM history_events"
                        ).fetchone()[0],
                        "archive_file_count": file_count,
                        "original_input_messages_preserved": len(raw) == 26,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    with TemporaryDirectory(prefix="reactive-minimal-history-") as directory:
        asyncio.run(main(directory))
