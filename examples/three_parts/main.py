"""Three independent sources, four stages, real SQLite and actual Pydantic AI tool calls.

The model is deterministic: this demonstrates the input and execution contract, not LLM quality.
Run: uv run python examples/three_parts/main.py --json /tmp/three-parts.json
"""

import argparse
import asyncio
import csv
import io
import json
import re
import sqlite3
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    CachePoint,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from reactive_context import CompilerPolicy, Context, skill_part, source_part
from reactive_context.adapters.pydantic_ai import PromptCachePolicy, ReactiveContextCapability
from reactive_context.core.snapshot import thaw
from reactive_context.sources.catalog import ToolCatalogSource
from reactive_context.sources.file import FileSource

SCOPE = "customer-report"
PROMPT = "按照当前 Skill、可用工具和数据库结构，生成并执行客户名单查询。"
SKILL_V1 = "# 客户名单 Skill\n查询客户名单，按 id 升序排列，最多返回 3 条记录。\n"
SKILL_V2 = "# 客户名单 Skill\n查询客户名单，按 id 升序排列，最多返回 1 条记录。\n"


class TableSchema(BaseModel):
    name: str
    columns: list[str]


class DatabaseSchema(BaseModel):
    tables: list[TableSchema]


def read_schema(db: sqlite3.Connection) -> dict:
    tables = db.execute(
        "SELECT name FROM sqlite_schema "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    schemas = []
    for (name,) in tables:
        quoted_name = name.replace('"', '""')
        columns = [row[1] for row in db.execute(f'PRAGMA table_info("{quoted_name}")')]
        schemas.append({"name": name, "columns": columns})
    return {"tables": schemas}


def active_data(messages) -> dict:
    """Read the last provider-visible, provenance-tagged projection for each source."""
    active = {}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                try:
                    instruction = json.loads(part.content)
                except json.JSONDecodeError:
                    continue
                if isinstance(instruction, dict) and instruction.get("skill_id") == "skill":
                    active["skill://customer-report"] = {
                        "payload": instruction["instructions"],
                        "role": "system",
                    }
                continue
            if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
                continue
            try:
                envelope = json.loads(part.content)
            except json.JSONDecodeError:
                continue
            if isinstance(envelope, dict) and {"resource", "revision", "data"} <= envelope.keys():
                payload = envelope["data"]
                if envelope["resource"] == "dbschema://main":
                    payload = json.loads(payload)
                active[envelope["resource"]] = {
                    "revision": envelope["revision"],
                    "payload": payload,
                }
    return active


def message_layout(messages):
    layout = []
    for message in messages:
        if isinstance(message, ModelResponse):
            layout.append("assistant:history")
            continue
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                layout.append("system:skill")
            elif isinstance(part, ToolReturnPart):
                layout.append("tool-result:" + part.tool_name)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, list) and any(
                    isinstance(p, CachePoint) for p in part.content
                ):
                    layout.append("cache-breakpoint")
                elif (
                    isinstance(part.content, str)
                    and '"resource": "dbschema://main"' in part.content
                ):
                    layout.append("user-prefix:dbschema")
                else:
                    layout.append("user:history/task")
    return layout


class DemoModel:
    """A transparent fixture that reads only actual model inputs and visible tool schemas."""

    def __init__(self):
        self.stage = 0
        self.frames = []

    def respond(self, messages, info):
        data = active_data(messages)
        skill = data["skill://customer-report"]["payload"]
        schema = data["dbschema://main"]["payload"]
        table = schema["tables"][0]
        match = re.search(r"最多返回 (\d+) 条", skill)
        if match is None:
            raise ValueError("demo Skill must declare its row limit")
        limit = int(match[1])
        identifiers = [table["name"], *table["columns"]]
        if any(not re.fullmatch(r"[a-z_]+", name) for name in identifiers):
            raise ValueError("demo supports simple SQL identifiers")
        sql = (
            f"SELECT {', '.join(table['columns'])} FROM {table['name']} ORDER BY id LIMIT {limit};"
        )
        available = [tool.name for tool in info.function_tools]
        if len(available) != 1:
            raise ValueError("the demonstration exposes exactly one executable tool per stage")
        selected = available[0]
        call_id = f"stage-{self.stage}-data"
        result = next(
            (
                part
                for message in reversed(messages)
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_call_id == call_id
            ),
            None,
        )
        self.frames.append(
            {
                "stage": self.stage,
                "phase": "answer" if result else "tool_call",
                "active_data": deepcopy(data),
                "message_layout": message_layout(messages),
                "visible_tools": available,
                "sql": sql,
            }
        )
        if result is None:
            return ModelResponse(parts=[ToolCallPart(selected, {"sql": sql}, tool_call_id=call_id)])
        output = {"tool": selected, "sql": sql, "result": result.content}
        return ModelResponse(parts=[TextPart(json.dumps(output, ensure_ascii=False))])


async def run_demo() -> dict:
    resolves = Counter()
    previous_revisions = {}
    versions = Counter()
    stages = []
    model = DemoModel()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        "CREATE TABLE customers (id INTEGER, name TEXT);"
        "INSERT INTO customers VALUES (1, 'Ada'), (2, 'Lin'), (3, 'Mei');"
    )
    allowed_tools = {"query_database"}
    executed = []

    def instrument(part):
        async def resolve(ctx):
            resolves[part.key] += 1
            return await part.resolve(ctx)

        return replace(part, resolve=resolve)

    def execute_readonly(tool_name, sql):
        # Catalog visibility and execution authorization are separate host checks.
        if tool_name not in allowed_tools:
            raise PermissionError("tool is not authorized")
        db.execute("PRAGMA query_only=ON")
        try:
            cursor = db.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
            executed.append({"tool": tool_name, "sql": sql, "rows": rows})
            return columns, rows
        finally:
            db.execute("PRAGMA query_only=OFF")

    async def query_database(sql: str) -> list[dict]:
        """Execute read-only SQL and return rows as JSON objects."""
        return execute_readonly("query_database", sql)[1]

    async def export_csv(sql: str) -> str:
        """Execute read-only SQL and export its result as CSV."""
        columns, rows = execute_readonly("export_csv", sql)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    query_tool, csv_tool = Tool(query_database), Tool(export_csv)

    def catalog_entry(tool):
        return {"name": tool.name, "parameters_json_schema": tool.tool_def.parameters_json_schema}

    async def resolve_schema(ctx):
        return read_schema(db)

    async def schema_revision(ctx):
        definitions = [
            row[0]
            for row in db.execute(
                "SELECT sql FROM sqlite_schema WHERE type = 'table' ORDER BY name"
            )
        ]
        return sha256(json.dumps(definitions).encode()).hexdigest()

    try:
        with TemporaryDirectory(prefix="reactive-three-parts-") as directory:
            skill_path = Path(directory) / "SKILL.md"
            await asyncio.to_thread(skill_path.write_text, SKILL_V1, encoding="utf-8")
            skill = replace(
                FileSource("skill", skill_path, root=directory).part,
                resource_key="skill://customer-report",
            )
            tools = ToolCatalogSource("tool", [catalog_entry(query_tool)])
            schema = source_part(
                key="dbschema",
                schema=DatabaseSchema,
                resolve=resolve_schema,
                revision=schema_revision,
                resource_key="dbschema://main",
            )

            # Exactly three registered Parts, with no hidden computed fourth Part.
            async with Context(
                [
                    instrument(skill_part(skill, name="客户名单")),
                    instrument(tools.part),
                    instrument(schema),
                ],
                compiler_policy=CompilerPolicy(
                    grants={"skill": "developer"},
                    part_order=("skill", "dbschema", "tool"),
                    prefix_parts={"dbschema"},
                    tool_order=("query_database", "export_csv"),
                ),
            ) as context:
                agent = Agent(
                    FunctionModel(model.respond),
                    tools=[query_tool, csv_tool],
                    name="three_parts_demo",
                    capabilities=[
                        ReactiveContextCapability(
                            context, scope=SCOPE, cache=PromptCachePolicy(messages=False)
                        )
                    ],
                )
                history = []
                labels = ["初始状态", "仅修改 Skill", "仅修改 Tool", "仅修改 DB Schema"]
                for stage, label in enumerate(labels):
                    if stage == 1:
                        await asyncio.to_thread(skill_path.write_text, SKILL_V2, encoding="utf-8")
                    elif stage == 2:
                        allowed_tools.clear()
                        allowed_tools.add("export_csv")
                        tools.set([catalog_entry(csv_tool)])
                    elif stage == 3:
                        db.executescript(
                            "ALTER TABLE customers RENAME TO customer_accounts;"
                            "ALTER TABLE customer_accounts RENAME COLUMN name TO display_name;"
                        )
                    before = resolves.copy()
                    event_start = len(context.events.events)
                    frame_start = len(model.frames)
                    snapshot = await context.next_context(SCOPE)
                    model.stage = stage
                    original = deepcopy(history)
                    result = await agent.run(PROMPT, message_history=history)
                    history_unchanged = history == original
                    if not history_unchanged:
                        raise AssertionError("the input history was mutated")
                    history = result.all_messages()
                    part_states = []
                    for part in snapshot.parts:
                        old = previous_revisions.get(part.part_id)
                        changed = old != part.revision
                        versions[part.part_id] += int(changed)
                        part_states.append(
                            {
                                "part": part.part_id,
                                "version": versions[part.part_id],
                                "revision": part.revision,
                                "status": part.status,
                                "change": "NEW"
                                if old is None
                                else "CHANGED"
                                if changed
                                else "REUSED",
                                "resolves": resolves[part.part_id] - before[part.part_id],
                                "value": thaw(part.value),
                            }
                        )
                        previous_revisions[part.part_id] = part.revision
                    stages.append(
                        {
                            "stage": stage,
                            "label": label,
                            "snapshot_id": snapshot.id,
                            "parts": part_states,
                            "model_requests": len(model.frames) - frame_start,
                            "request_frames": deepcopy(model.frames[frame_start:]),
                            "output": json.loads(result.output),
                            "execution": executed[-1],
                            "input_history_unchanged": history_unchanged,
                            "prefix_diagnostic": vars(
                                next(
                                    event
                                    for event in context.events.events[event_start:]
                                    if event.event == "prompt_prefix"
                                )
                            ),
                            "model_cache_usage": context.events.cache_summary(SCOPE),
                        }
                    )
                return {
                    "model": "deterministic FunctionModel; not a real LLM quality evaluation",
                    "database": "real in-memory SQLite; both tools execute SQL",
                    "scope": SCOPE,
                    "registered_parts": ["skill", "tool", "dbschema"],
                    "stages": stages,
                    "history_note": "原始历史保留先前的 SQL 和结果；当前来源投影使用新版本。",
                }
    finally:
        db.close()


def print_report(report):
    print("三个 Part：文件 Skill / 动态 Tool Catalog / SQLite DB Schema")
    print("同一 Agent、同一 scope、同一任务；每轮只改一个来源。")
    print("模型为确定性 FunctionModel；工具会实际查询 SQLite。")
    for stage in report["stages"]:
        print(f"\n=== {stage['stage']}. {stage['label']} ===")
        for part in stage["parts"]:
            print(
                f"  {part['part']:8} v{part['version']} {part['revision'][:12]} "
                f"{part['change']:7} resolve={part['resolves']} {part['status']}"
            )
        first = stage["request_frames"][0]
        print("  消息布局：" + " → ".join(first["message_layout"][:4]))
        print("  模型可见工具：" + ", ".join(first["visible_tools"]))
        print(
            "  模型读取 Skill："
            + first["active_data"]["skill://customer-report"]["payload"].splitlines()[-1]
        )
        schema = first["active_data"]["dbschema://main"]["payload"]["tables"][0]
        print(f"  模型读取 Schema：{schema['name']}({', '.join(schema['columns'])})")
        print("  生成 SQL：" + stage["output"]["sql"])
        print("  实际执行结果：" + json.dumps(stage["output"]["result"], ensure_ascii=False))
        cache = stage["prefix_diagnostic"]
        print(
            f"  前缀布局诊断：{cache['reason']}；整块复用 "
            f"{cache['reused_prefix_bytes']}/{cache['prefix_bytes']} bytes"
        )
        print("  模型缓存 token：FunctionModel 未实测供应商缓存，不能以此前缀诊断当作命中率")
        print(
            f"  模型请求：{stage['model_requests']} 次；传入的原始历史未修改："
            f"{stage['input_history_unchanged']}"
        )
    print("\n" + report["history_note"])


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Save all versions, active inputs and results")
    args = parser.parse_args()
    report = await run_demo()
    print_report(report)
    if args.json:
        await asyncio.to_thread(
            args.json.write_text,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"详细报告：{args.json}")


if __name__ == "__main__":
    asyncio.run(main())
