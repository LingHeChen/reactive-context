"""Offline Skill / Tool / DB schema updates using the v2 declarative API.

Run: uv run python examples/v2_dx.py
No API key or model request is used. SQLite and the Skill file are real external sources.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from reactive_context import (
    CompilerPolicy,
    Context,
    ContextIRSegment,
    FilePartProvider,
    computed,
    part,
    skill,
    state,
)
from reactive_context.core.snapshot import digest, thaw


def tool_fragment(value, reader):
    return [
        ContextIRSegment(
            reader.part_id,
            reader.part_id,
            reader.revision,
            value,
            kind="tool_definition",
        )
    ]


class SchemaReader:
    def __init__(self, path):
        self.path = path

    def read(self):
        with sqlite3.connect(self.path) as connection:
            return dict(
                connection.execute(
                    "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            )


async def schema_revision(reader):
    return digest(reader.database.read())


class DeveloperContext(Context):
    task: str = state("查询用户")
    coding = skill("coding.md", scope="project")
    available_tools: list[dict] = state(default_factory=list, project=tool_fragment)

    @computed
    def goal(self) -> str:
        return f"当前目标：{self.task}"

    @part(revision=schema_revision, scope="project")
    async def dbschema(self, reader):
        return reader.database.read()


async def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        skill_file = root / "coding.md"
        skill_file.write_text("SQL 只查询必需列。", encoding="utf-8")
        database = root / "app.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE users(id INTEGER PRIMARY KEY,name TEXT)")
        tools = [{"name": "query_users", "parameters": {"type": "object", "properties": {}}}]
        async with DeveloperContext(
            scope_bindings={"project": "demo"},
            providers={"skill": FilePartProvider(root)},
            services={"database": SchemaReader(database)},
            available_tools=tools,
            compiler_policy=CompilerPolicy(
                grants={"coding": "system"},
                part_order=("coding", "available_tools", "dbschema", "goal", "task"),
                prefix_parts=frozenset({"coding"}),
            ),
        ) as context:
            previous = {}

            async def show(label):
                nonlocal previous
                compiled = await context.compile()
                changed = [
                    key
                    for key, revision in compiled.revision_vector.items()
                    if previous.get(key) != revision
                ]
                print(
                    json.dumps(
                        {
                            "step": label,
                            "changed_parts": changed,
                            "snapshot_id": compiled.snapshot_id,
                            "messages": [
                                {"role": m.role, "part": m.part_id, "content": m.content}
                                for m in compiled.messages
                            ],
                            "tools": thaw(compiled.provider_tools),
                            "goal": thaw(context.explain("goal")),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                previous = dict(compiled.revision_vector)

            await show("initial")
            skill_file.write_text("SQL 必须参数化，且只查询必需列。", encoding="utf-8")
            await show("skill changed")
            context.available_tools = [
                *tools,
                {
                    "name": "count_users",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
            await show("tools changed")
            with sqlite3.connect(database) as connection:
                connection.execute("ALTER TABLE users ADD COLUMN email TEXT")
            await show("dbschema changed")


if __name__ == "__main__":
    asyncio.run(main())
