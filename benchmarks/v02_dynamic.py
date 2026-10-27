"""Three-part dynamic integration matrix, including actual feature ablations.

FunctionModel inspects provider-visible messages and tool definitions; it does not measure LLM
answer quality, tool-choice intelligence, official QA accuracy, or provider cache hit rates.
"""

import argparse
import asyncio
import json
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    CompilerPolicy,
    Context,
    ContextManager,
    ContextStore,
    HistoryManager,
    HistoryPolicy,
    Principal,
    SummaryDraft,
    computed_part,
)
from reactive_context.benchmarking.v02 import VARIANTS, configure_variant
from reactive_context.errors import ContextError
from reactive_context.sources.catalog import ToolCatalogSource
from reactive_context.sources.config import ConfigSource

HOST = Principal("fixture-host", admin=True)
ACTOR = Principal("fixture-agent")


def summary(events):
    first = events[0]
    ids = (
        first["summary"]["evidence_ids"]
        if first.get("kind") == "historical_summary"
        else [HistoryManager.event_id(first)]
    )
    return SummaryDraft(
        text="Historical actions are archived. Keep audit log.",
        constraints=["Keep audit log"],
        evidence_ids=ids[:1],
    )


async def trial(directory, variant, seed, scenario):
    skill = ConfigSource("skill", f"SKILL_A_{seed}", revision_mode="release")
    schema = ConfigSource("dbschema", f"SCHEMA_A_{seed}", revision_mode="release")
    irrelevant = ConfigSource("irrelevant", "unrelated material " * 100)

    def old_api() -> str:
        return "old version"

    def new_api() -> str:
        return "new version"

    def publish_update() -> str:
        skill.set(f"SKILL_B_{seed}")
        schema.set(f"SCHEMA_B_{seed}")
        catalog.set([spec(new_tool), spec(update_tool)])
        return "published three source revisions"

    old_tool, new_tool, update_tool = Tool(old_api), Tool(new_api), Tool(publish_update)

    def spec(tool):
        return {"name": tool.name, "parameters_json_schema": tool.tool_def.parameters_json_schema}

    catalog = ToolCatalogSource("tools", [spec(old_tool), spec(update_tool)])

    async def plan(ctx):
        return "PLAN_FOR_" + ctx.dependencies["dbschema"].value

    derived = computed_part(
        key="plan", schema=str, dependencies=["dbschema"], compute=plan, kind="task_state"
    )
    observed, requests = [], []
    with tempfile.TemporaryDirectory(prefix="rc-v02-dynamic-") as temporary:
        with ContextStore(temporary) as store:
            store.create_scope(HOST, "task")
            store.grant_scope(HOST, "task", ACTOR.id)
            async with ContextManager(
                [skill.part, catalog.part, schema.part, derived, irrelevant.part],
                store=store,
                principal=ACTOR,
                compiler_policy=CompilerPolicy(max_tokens=32000, output_reserve=1000),
            ) as manager:
                manager.publish_catalog(HOST, "task")
                rows = await manager.query("task", limit=100)
                relevant = [r["id"] for r in rows if r["key"] != "irrelevant"]
                initial = await Context.next_context(
                    manager, "task", selected=["skill", "tools", "dbschema", "plan"]
                )
                history = []
                for part in initial.parts:
                    call_id = "read-" + part.part_id
                    history.extend(
                        [
                            ModelResponse(
                                parts=[
                                    ToolCallPart(
                                        "read_source", {"id": part.part_id}, tool_call_id=call_id
                                    )
                                ]
                            ),
                            ModelRequest(
                                parts=[
                                    ToolReturnPart(
                                        "read_source",
                                        str(part.value),
                                        tool_call_id=call_id,
                                        metadata={
                                            "reactive_context": {
                                                "resource_id": part.resource_id,
                                                "revision": part.revision,
                                            }
                                        },
                                    )
                                ]
                            ),
                        ]
                    )
                if scenario == "long":
                    for i in range(12):
                        history.extend(
                            [
                                ModelRequest.user_text_prompt(f"older note {i}: " + "x" * 3000),
                                ModelResponse(parts=[TextPart("acknowledged")]),
                            ]
                        )
                history_manager = HistoryManager(
                    store,
                    ACTOR,
                    compactor=summary,
                    validator=lambda source, draft: "Keep audit log" in draft.constraints,
                    policy=HistoryPolicy(
                        recent_groups=2,
                        segment_groups=4,
                        max_compactions_per_boundary=20,
                        required_constraints=("Keep audit log",),
                    ),
                )
                capability = await configure_variant(
                    manager, "task", variant, selected_part_ids=relevant, history=history_manager
                )

                def respond(messages, info):
                    requests.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
                    body = str(messages)
                    current = {}
                    for message in requests[-1]:
                        for part in message["parts"]:
                            if part["part_kind"] != "user-prompt" or not isinstance(
                                part.get("content"), str
                            ):
                                continue
                            try:
                                envelope = json.loads(part["content"])
                            except ValueError:
                                continue
                            if isinstance(envelope, dict) and envelope.get("status") == "CURRENT":
                                current[envelope.get("resource")] = envelope.get("data")
                    observed.append(
                        {
                            "skill_b_current": current.get("skill") == f"SKILL_B_{seed}",
                            "schema_b_current": current.get("dbschema") == f"SCHEMA_B_{seed}",
                            "old_skill_marker_present": f"SKILL_A_{seed}" in body,
                            "old_schema_marker_present": f"SCHEMA_A_{seed}" in body,
                            "new_tool_exposed": "new_api" in {t.name for t in info.function_tools},
                            "old_tool_exposed": "old_api" in {t.name for t in info.function_tools},
                            "derived_plan_b_present": f"PLAN_FOR_SCHEMA_B_{seed}" in body,
                            "input_digest": sha256(
                                json.dumps(requests[-1], sort_keys=True).encode()
                            ).hexdigest(),
                        }
                    )
                    if len(observed) == 1:
                        return ModelResponse(
                            parts=[ToolCallPart("publish_update", {}, tool_call_id="update")]
                        )
                    return ModelResponse(parts=[TextPart("fixture completed")])

                agent = Agent(
                    FunctionModel(respond),
                    tools=[old_tool, new_tool, update_tool],
                    capabilities=[capability],
                    name="v02_dynamic_fixture",
                )
                failure = None
                raw_result = None
                try:
                    result = await agent.run(
                        "Use current skill, tool, schema and plan. Keep audit log.",
                        message_history=history,
                    )
                    raw_result = ModelMessagesTypeAdapter.dump_python(
                        result.all_messages(), mode="json"
                    )
                except ContextError as error:
                    failure = error.code
                record = {
                    "mode": "protocol",
                    "variant": variant,
                    "features": asdict(VARIANTS[variant]),
                    "seed": seed,
                    "scenario": scenario,
                    "failure": failure,
                    "model_quality_score": None,
                    "provider_cache_hit_rate": None,
                    "model_calls": len(observed),
                    "observations": observed,
                    "compaction_records": store.db.execute(
                        "SELECT COUNT(*) FROM compaction_records"
                    ).fetchone()[0],
                    "source_resolves": sum(
                        e.event == "resolve" and not e.cache_hit for e in manager.events.events
                    ),
                }
                path = directory / "traces" / f"{variant}-{scenario}-{seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "record": record,
                            "provider_requests": requests,
                            "raw_result": raw_result,
                            "events": json.loads(manager.export_trace()),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return record


async def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in ("short", "long"):
        for variant in VARIANTS:
            for seed in (17, 29, 43):
                row = await trial(output, variant, seed, scenario)
                rows.append(row)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "scenario": scenario,
                        "failures": sum(r["failure"] is not None for r in rows[-3:]),
                    }
                ),
                flush=True,
            )
    (output / "dynamic-results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/v02/protocol")
    asyncio.run(run(parser.parse_args().output))
