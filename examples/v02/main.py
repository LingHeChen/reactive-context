"""Offline v02 lifecycle. Run: uv run python examples/v02/main.py."""

import asyncio
import json
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from reactive_context import CompilerPolicy, ContextManager, ContextStore, Principal, skill_part
from reactive_context.core.snapshot import thaw
from reactive_context.sources.config import ConfigSource


class Finding(BaseModel):
    text: str


async def demo(directory):
    host, agent = Principal("host", admin=True), Principal("agent")
    scope = "orders"
    skill = ConfigSource("skill", "按金额升序取三条", revision_mode="release")
    tools = ConfigSource("tools", {"query_api": "v1"})
    schema = ConfigSource("dbschema", {"orders": ["id", "amount"]})
    parts = [skill_part(skill.part, role="system"), tools.part, schema.part]
    policy = CompilerPolicy(grants={"skill": "system"}, part_order=("skill", "tools", "dbschema"))
    with ContextStore(directory) as store:
        store.create_scope(host, scope)
        store.grant_scope(host, scope, agent.id)
        store.register_schema("finding", Finding)
        async with ContextManager(
            parts, store=store, principal=agent, compiler_policy=policy
        ) as context:
            context.publish_catalog(host, scope)
            catalog = await context.query(scope)
            for row in catalog:
                await context.activate(scope, row["id"])

            async def show(stage):
                snapshot = await context.next_context(scope)
                compiled = context.compile(snapshot)
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "compile_id": compiled.compile_id,
                            "versions": dict(compiled.revision_vector),
                            "values": {p.part_id: thaw(p.value) for p in snapshot.parts},
                            "roles": {m.part_id: m.role for m in compiled.provider_messages},
                        },
                        ensure_ascii=False,
                    )
                )

            await show("initial")
            skill.set("按金额降序取一条")
            await show("skill changed")
            tools.set({"query_api": "v2"})
            await show("tool changed")
            schema.set({"orders": ["id", "total"]})
            await show("schema changed")
            loaded = await context.load(
                scope, next(r["id"] for r in catalog if r["key"] == "dbschema")
            )
            proposal = store.propose_fact(
                agent,
                scope,
                "finding",
                "schema-observation",
                {"text": "orders 中存在 total 字段"},
                [loaded.evidence_id],
            )
            fact = store.commit(agent, proposal)
            await context.activate(scope, fact.id)
            await show("unverified fact activated")
        # Active Set persists references. Resource bodies are resolved again after restart.
        async with ContextManager(
            parts, store=store, principal=agent, compiler_policy=policy
        ) as restored:
            snapshot = await restored.next_context(scope)
            print("restored:", sorted(p.part_id for p in snapshot.parts))


if __name__ == "__main__":
    with TemporaryDirectory(prefix="reactive-context-v02-") as directory:
        asyncio.run(demo(directory))
