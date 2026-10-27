"""Typed schema + reactive computed documentation, entirely local."""

import asyncio

from pydantic import BaseModel

from reactive_context import Context, ResolveContext, computed_part
from reactive_context.sources.config import ConfigSource


class Schema(BaseModel):
    table: str
    columns: list[str]


async def describe(ctx: ResolveContext) -> str:
    # Validated models are exposed as immutable structural records.
    schema = ctx.dependencies["db"].value
    return f"{schema.table}: {', '.join(schema.columns)}"


async def main():
    source = ConfigSource("db", {"table": "accounts", "columns": ["id"]}, schema=Schema)
    derived = computed_part(key="db-description", schema=str, dependencies=["db"], compute=describe)
    async with Context([source.part, derived]) as context:
        first = await context.next_context("migration")
        source.set({"table": "accounts", "columns": ["id", "email"]})
        second = await context.next_context("migration")
        assert first.part("db-description").value == "accounts: id"
        assert second.part("db-description").value == "accounts: id, email"
        print(second.part("db-description").value)
        print(dict(second.revision_vector))


if __name__ == "__main__":
    asyncio.run(main())
