"""Generic host-checked Fact updates. No model, network or benchmark rules required."""

import asyncio
import json
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel

from reactive_context import ContextManager, ContextStore, FactVerdict, Principal, fact_schema_part

HOST = Principal("example-host", True)
AGENT = Principal("example-agent")
SCOPE = "preferences"
SCHEMA = "user-preference"


class Preference(BaseModel):
    attribute: Literal["language"]
    value: Literal["zh-CN", "en-US"]
    confirmed: bool


def source_checker(observations):
    """The application supplies authoritative observations; the library invents none."""

    def check(request):
        sources = [observations[e] for e in request.evidence if e in observations]
        if not request.value["confirmed"] or not any(s["confirmed"] for s in sources):
            return FactVerdict("skip", ("unconfirmed_inference",))
        if not any(s["value"] == request.value["value"] and s["confirmed"] for s in sources):
            return FactVerdict("reject", ("unsupported_by_source",))
        return FactVerdict("accept", ("confirmed_user_preference",))

    return check


async def main():
    with TemporaryDirectory(prefix="checked-facts-") as path, ContextStore(path) as store:
        observations = {}
        store.create_scope(HOST, SCOPE)
        store.grant_scope(HOST, SCOPE, AGENT.id)
        store.register_schema(SCHEMA, Preference, check=source_checker(observations))
        results = []
        async with ContextManager(store=store, principal=AGENT) as manager:
            for index, (value, confirmed) in enumerate(
                [("zh-CN", True), ("zh-CN", True), ("en-US", False), ("en-US", True)], 1
            ):
                evidence = store.observe_evidence(AGENT, SCOPE, f"user-input:{index}", str(index))
                observations[evidence] = {"value": value, "confirmed": confirmed}
                result = await manager.save_fact(
                    SCOPE,
                    SCHEMA,
                    "user.language",
                    {"attribute": "language", "value": value, "confirmed": confirmed},
                    [evidence],
                    expected_revision=1 if index == 4 else None,
                )
                results.append(result)
        assert [r["decision"] for r in results] == ["created", "unchanged", "skip", "updated"]
        assert store.db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 1
        assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2
        print(json.dumps(results, ensure_ascii=False, indent=2))


async def live(output):
    """Optional native-tool smoke test, charged to the existing authorized shared ledger."""
    import fcntl
    from dataclasses import asdict, replace
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Caching
    from pydantic_ai.usage import UsageLimits

    from reactive_context import CompilerPolicy
    from reactive_context.adapters.operations import context_tools
    from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
    from reactive_context.benchmarking.config import BenchmarkConfig
    from reactive_context.benchmarking.history_regression import write_json
    from reactive_context.benchmarking.official import ROOT
    from reactive_context.benchmarking.runtime import (
        AuditedModel,
        BudgetLedger,
        CallTrace,
        make_model,
    )

    directory = Path(output)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
    config = replace(BenchmarkConfig.load(), max_requests=2000, max_cost_usd=5)
    with (ROOT / "results/v02/pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = BudgetLedger(config, ROOT / "results/history-regression-budget.json")
        start = {"requests": ledger.requests, "cost_usd": ledger.spent}
        trace = CallTrace(capture_directory=directory / "requests")
        observations, records = {}, []
        passed = False
        try:
            with ContextStore(directory / "store") as store:
                store.create_scope(HOST, SCOPE)
                store.grant_scope(HOST, SCOPE, AGENT.id)
                store.register_schema(SCHEMA, Preference, check=source_checker(observations))
                schema_part = fact_schema_part(store, {SCHEMA: 1})
                async with (
                    make_model(config) as model,
                    ContextManager(
                        [schema_part],
                        mandatory=[schema_part.key],
                        store=store,
                        principal=AGENT,
                        compiler_policy=CompilerPolicy(max_tokens=128000),
                    ) as manager,
                ):
                    agent = Agent(
                        AuditedModel(model, ledger, trace, "generic-admission"),
                        retries=0,
                        instructions=(
                            "Manage durable user preferences. The host supplies the current "
                            "observed input with its evidence ID and confirmation state. "
                            "Use schema "
                            "user-preference and stable key user.language. "
                            "Save confirmed preferences "
                            "using context_save_fact when new or changed. "
                            "Do not persist speculation. "
                            "Existing updates require the observed expected_revision; "
                            "unchanged values "
                            "need no revision. Never invent evidence IDs. Reply briefly."
                        ),
                        toolsets=[
                            context_tools(manager, SCOPE).filtered(
                                lambda ctx, tool: (
                                    tool.name
                                    in {"context_save_fact", "context_query", "context_load"}
                                )
                            )
                        ],
                        capabilities=[
                            Caching(messages=False),
                            ReactiveContextCapability(
                                manager, scope=SCOPE, history=False, cache=None
                            ),
                        ],
                        model_settings={
                            "temperature": 0,
                            "max_tokens": 768,
                            "extra_body": {"thinking": {"type": "disabled"}},
                        },
                    )
                    cases = [
                        (
                            "以后默认用中文回复。我确认这是长期偏好，请保存。",
                            "zh-CN",
                            True,
                            "zh-CN",
                            1,
                        ),
                        ("继续保持中文，这是同一个偏好，没有新变化。", "zh-CN", True, "zh-CN", 1),
                        (
                            "我猜以后也许改成英语，但尚未决定，请不要修改长期偏好。",
                            "en-US",
                            False,
                            "zh-CN",
                            1,
                        ),
                        ("现在确认将默认语言改为英语，请更新长期偏好。", "en-US", True, "en-US", 2),
                    ]
                    for index, (message, value, confirmed, expected, revision) in enumerate(
                        cases, 1
                    ):
                        evidence = store.observe_evidence(
                            AGENT, SCOPE, f"user-input:{index}", str(index)
                        )
                        observations[evidence] = {"value": value, "confirmed": confirmed}
                        write_json(directory / "observations.json", observations)
                        result = await agent.run(
                            message
                            + "\nObserved input: "
                            + json.dumps({"evidence_id": evidence, **observations[evidence]}),
                            usage_limits=UsageLimits(request_limit=4),
                        )
                        metadata = store.query(AGENT, SCOPE, kind="fact")
                        assert len(metadata) == 1, "unexpected number of persistent Facts"
                        fact = store.load(AGENT, metadata[0]["id"])
                        assert fact.value["value"] == expected and fact.revision == revision
                        await manager.activate(SCOPE, fact.id)
                        records.append(
                            {
                                "phase": index,
                                "message": message,
                                "answer": result.output,
                                "value": fact.value["value"],
                                "revision": fact.revision,
                                "fact_count": len(metadata),
                            }
                        )
                        write_json(directory / "phases.json", records)
            passed = True
        finally:
            write_json(directory / "calls.json", [asdict(call) for call in trace.calls])
            write_json(
                directory / "result.json",
                {
                    "passed": passed,
                    "phases": records,
                    "requests": ledger.requests - start["requests"],
                    "cost_usd_upper_bound": ledger.spent - start["cost_usd"],
                    "usage": trace.totals(),
                    "scope": "generic native-tool integration; not task accuracy",
                },
            )
        print(json.dumps({"passed": passed, **trace.totals()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="uses existing authorized shared model budget"
    )
    parser.add_argument("--output", default="results/generic-admission-pilot")
    args = parser.parse_args()
    asyncio.run(live(args.output) if args.live else main())
