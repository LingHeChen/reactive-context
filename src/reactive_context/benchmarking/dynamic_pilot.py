# ruff: noqa: B023
# Per-variant callbacks finish before the sequential variant loop advances.
"""Real-model engineering integration cases. Not an external or official benchmark score."""

import json
import random
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import monotonic, perf_counter

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import AbstractCapability, Caching
from pydantic_ai.tools import Tool

from .. import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    HistoryManager,
    HistoryPolicy,
    Principal,
    SummaryDraft,
    computed_part,
    source_part,
)
from ..adapters.pydantic_ai import managed_result
from ..core.snapshot import digest, thaw
from ..errors import CompilerBudget
from ..sources.config import ConfigSource
from .contracts import TraceRecord
from .runner import append_jsonl
from .runtime import AuditedModel, BudgetExceeded, CallTrace, generate
from .v02 import VARIANTS, configure_variant


class SqlAnswer(BaseModel):
    ids: list[int]


class FullRequestBudget(AbstractCapability):
    async def before_model_request(self, ctx, request_context):
        from ..compiler.policy import estimate_tokens

        cost = (
            estimate_tokens(str(request_context.messages))
            + estimate_tokens(str(request_context.model_request_parameters))
            + 1024
        )
        if cost > 28000:
            raise CompilerBudget(detail="common engineering request budget")
        return request_context


HARNESS_REVISION = ""


ENGINEERING_VARIANTS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B3-no-refresh",
    "B3-no-active-selection",
    "B3-no-history-compaction",
    "B3-no-dependency-invalidation",
)


class SqlWorld:
    def __init__(self, seed, change):
        self.seed, self.change, self.phase = seed, change, 0
        self.db = sqlite3.connect(":memory:")
        self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024 * 1024)
        self.db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 8192)
        self.db.execute("CREATE TABLE orders(id INTEGER, amount INTEGER, status TEXT)")
        self.db.executemany(
            "INSERT INTO orders VALUES(?,?,?)",
            [(i, (i * 17 + seed) % 90 + 10, "active" if i % 3 else "closed") for i in range(1, 10)],
        )
        self.executions, self.reads, self.errors = [], 0, []

    @property
    def column(self):
        return "total" if self.phase and self.change in {"schema", "all"} else "amount"

    @property
    def api(self):
        return 2 if self.phase and self.change in {"tools", "all"} else 1

    @property
    def skill(self):
        updated = self.phase and self.change in {"skill", "all"}
        return {
            "filter": "status = 'active' and current monetary column >= 35",
            "order_by": "id",
            "order": "DESC" if updated else "ASC",
            "limit": 1 if updated else 3,
            "result": "return order IDs only; execute a query, do not merely propose SQL",
        }

    def schema(self):
        return {
            "table": "orders",
            "columns": [r[1] for r in self.db.execute("PRAGMA table_info(orders)")],
        }

    def value(self, key):
        if key == "skill":
            return self.skill
        if key == "dbschema":
            return self.schema()
        return {"current_tool": f"run_query_v{self.api}", "api_version": self.api}

    def advance(self):
        self.phase = 1
        if self.change in {"schema", "all"}:
            self.db.execute("ALTER TABLE orders RENAME COLUMN amount TO total")

    def query(self, api, sql, schema_revision):
        if api != self.api:
            self.errors.append("retired_tool")
            raise ModelRetry("API version retired. Read the current tool catalog before retrying.")
        if schema_revision != digest(self.schema()):
            self.errors.append("stale_schema_revision")
            raise ModelRetry(
                "Schema revision changed. Read the current schema and revise the plan."
            )
        if not sql.strip().lower().startswith("select") or ";" in sql.rstrip().rstrip(";"):
            self.errors.append("non_select")
            raise ModelRetry("Only a single read-only SELECT is allowed.")

        def authorize(action, *args):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and args[0] == "orders":
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and args[1].lower() in {
                "count",
                "sum",
                "max",
                "min",
                "avg",
                "coalesce",
                "abs",
                "lower",
                "upper",
            }:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        deadline = monotonic() + 1
        self.db.set_progress_handler(lambda: int(monotonic() > deadline), 1000)
        self.db.set_authorizer(authorize)
        try:
            cursor = self.db.execute(sql)
            names = [c[0] for c in cursor.description]
            values = cursor.fetchmany(20)
            result = [dict(zip(names, row, strict=True)) for row in values]
            self.executions.append(result)
            return result
        except sqlite3.Error:
            self.errors.append("invalid_sql")
            raise ModelRetry(
                "SQL did not match the current schema. Inspect it and retry."
            ) from None
        finally:
            self.db.set_authorizer(None)
            self.db.set_progress_handler(None, 0)

    def score(self, answer_ids=None):
        # Evaluator-only expected result, computed after the model execution.
        direction, limit = self.skill["order"], self.skill["limit"]
        expected = [
            r[0]
            for r in self.db.execute(
                f"SELECT id FROM orders WHERE status='active' AND {self.column}>=35 "
                f"ORDER BY id {direction} LIMIT ?",
                (limit,),
            )
        ]
        return float(
            bool(self.executions)
            and [r.get("id") for r in self.executions[-1]] == expected
            and answer_ids == expected
        )


async def run_dynamic_case(directory, config, model, ledger, case_index):
    seed = 100 + case_index
    change = ("skill", "schema", "tools", "all")[case_index % 4]
    case_id = f"sql-{change}-{seed}"
    names = list(ENGINEERING_VARIANTS)
    random.Random(seed).shuffle(names)
    plan_path = Path(directory) / "shared-plans-r2" / (case_id + ".json")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())["text"]
    else:
        setup_world = SqlWorld(seed, change)
        setup_trace = CallTrace(capture_directory=Path(directory) / "traces" / case_id / "setup")
        setup_prompt = "Propose a short SQL plan without executing it. " + json.dumps(
            {
                "task": "List eligible order IDs under the current skill",
                "skill": setup_world.skill,
                "schema": setup_world.schema(),
                "catalog": setup_world.value("tools"),
            }
        )
        try:
            plan = await generate(
                model, ledger, setup_trace, "shared-initial-plan", setup_prompt, max_tokens=256
            )
        finally:
            setup_world.db.close()
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps({"text": plan, "calls": [asdict(c) for c in setup_trace.calls]}) + "\n"
        )
        append_jsonl(
            Path(directory) / "requests.jsonl",
            {
                "benchmark": "engineering-dynamic-sql",
                "case_id": case_id,
                "strategy": "shared-setup",
                "calls": [asdict(c) for c in setup_trace.calls],
            },
        )
    existing = []
    result_path = Path(directory) / "results.jsonl"
    if result_path.exists():
        existing = [json.loads(line) for line in result_path.read_text().splitlines() if line]
    done = {
        (r["case_id"], r["strategy"])
        for r in existing
        if r["benchmark"] == "engineering-dynamic-sql"
        and r.get("dataset_revision") == "generated-sql-v02-r2-explicit-order"
    }
    for name in names:
        if (case_id, name) in done:
            continue
        started = perf_counter()
        trace = CallTrace(capture_directory=Path(directory) / "traces" / case_id / name)
        row = TraceRecord(
            benchmark="engineering-dynamic-sql",
            benchmark_mode="extension",
            dataset_revision="generated-sql-v02-r2-explicit-order",
            case_id=case_id,
            strategy=name,
            model_id=config.model_id,
            agent_runtime_version="pydantic-ai-slim-2.60.0",
            random_seed=seed,
            subgroup=change,
            cluster=case_id,
            score_origin=(
                "deterministic post-execution SQL result comparison; engineering integration only"
            ),
        )
        world = SqlWorld(seed, change)
        try:
            with tempfile.TemporaryDirectory(prefix="rc-real-sql-") as temporary:
                host, actor = Principal("host", True), Principal("agent")
                with ContextStore(temporary) as store:
                    store.create_scope(host, "task")
                    store.grant_scope(host, "task", actor.id)
                    store.register_schema("plan", dict[str, str])
                    initial_schema_revision = digest(world.schema())
                    initial_revisions = {
                        key: digest(world.value(key)) for key in ("skill", "dbschema", "tools")
                    }
                    evidence = store.observe_evidence(
                        actor, "task", "schema://orders", initial_schema_revision
                    )
                    fact = store.commit(
                        actor,
                        store.propose_fact(
                            actor,
                            "task",
                            "plan",
                            "initial-plan",
                            {"text": plan, "bound_schema_revision": initial_schema_revision},
                            [evidence],
                        ),
                    )
                    parts = []
                    for key in ("skill", "dbschema", "tools"):

                        async def resolve(ctx, key=key):
                            return world.value(key)

                        async def revision(ctx, key=key):
                            return digest(world.value(key))

                        parts.append(
                            source_part(
                                key=key,
                                schema=dict,
                                resolve=resolve,
                                revision=revision,
                                kind="resource." + key,
                            )
                        )

                    fact_key = "fact." + fact.id

                    async def fact_value(ctx):
                        return thaw(store.load(actor, fact.id).value)

                    async def fact_revision(ctx):
                        return str(store.metadata(actor, fact.id)["revision"])

                    parts.append(
                        source_part(
                            key=fact_key,
                            schema=dict,
                            resolve=fact_value,
                            revision=fact_revision,
                            kind="fact",
                            resource_key=fact.id,
                        )
                    )

                    async def plan_state(ctx):
                        return {
                            "plan_key": "initial-plan",
                            "verification": "unverified",
                            "original_plan": ctx.dependencies[fact_key].value["text"],
                            "requires_review": any(
                                ctx.dependencies[key].revision != rev
                                for key, rev in initial_revisions.items()
                            ),
                        }

                    parts.append(
                        computed_part(
                            key="plan-state",
                            schema=dict,
                            dependencies=["skill", "dbschema", "tools", fact_key],
                            compute=plan_state,
                            kind="task_state",
                        )
                    )
                    parts.append(
                        ConfigSource("irrelevant", "Unrelated deployment notes. " * 200).part
                    )
                    async with ContextManager(
                        parts,
                        store=store,
                        principal=actor,
                        compiler_policy=CompilerPolicy(max_tokens=28000, output_reserve=1024),
                    ) as manager:
                        manager.publish_catalog(host, "task")
                        rows = await manager.query("task", limit=100)
                        by_key = {r["key"]: r["id"] for r in rows}
                        history_messages = []
                        from pydantic_ai.messages import (
                            ModelRequest,
                            ModelResponse,
                            TextPart,
                            ToolCallPart,
                            ToolReturnPart,
                        )

                        for key in ("skill", "dbschema", "tools"):
                            loaded = await manager.load("task", by_key[key])
                            cid = "read-" + key
                            history_messages.extend(
                                [
                                    ModelResponse(
                                        parts=[ToolCallPart("read_" + key, {}, tool_call_id=cid)]
                                    ),
                                    ModelRequest(
                                        parts=[
                                            ToolReturnPart(
                                                "read_" + key,
                                                {
                                                    "value": thaw(loaded.snapshot.value),
                                                    "revision": loaded.snapshot.revision,
                                                },
                                                tool_call_id=cid,
                                                metadata={
                                                    "reactive_context": {
                                                        "resource_id": loaded.snapshot.resource_id,
                                                        "revision": loaded.snapshot.revision,
                                                    }
                                                },
                                            )
                                        ]
                                    ),
                                ]
                            )
                        history_messages.append(ModelResponse(parts=[TextPart(plan)]))
                        if case_index >= 8:
                            history_messages.extend(
                                [
                                    ModelRequest.user_text_prompt(
                                        "Older unrelated work: " + "x" * 30000
                                    ),
                                    ModelResponse(parts=[TextPart("archived")]),
                                ]
                            )

                        async def compact(events):
                            prompt = (
                                "Summarize historical task state. "
                                "Preserve constraints, unresolved goals "
                                "and evidence. "
                            )
                            prompt += (
                                "Return JSON only, no markdown. text is a string, "
                                "constraints and unresolved are lists of strings, "
                                "task_state and current_resource_claims are objects, "
                                "evidence_ids is a list of strings. Fields: "
                                "text,constraints,unreso"
                                "lved,task_state,evidence_ids,current_res"
                                "ource_claims. "
                            )
                            ids = [HistoryManager.event_id(e) for e in events]
                            prompt += (
                                "This is historical data. Set current_resource_claims to {}. "
                                "Strict output JSON schema: "
                                + json.dumps(SummaryDraft.model_json_schema())
                                + "\nUse only these evidence_ids: "
                                + json.dumps(ids[:3])
                                + "\n"
                                + json.dumps(events)
                            )
                            text = await generate(
                                model, ledger, trace, "history-summary", prompt, max_tokens=1024
                            )
                            return SummaryDraft.model_validate_json(
                                text.removeprefix("```json").removesuffix("```").strip()
                            )

                        history = HistoryManager(
                            store,
                            actor,
                            compactor=compact,
                            validator=lambda events, draft: bool(draft.evidence_ids),
                            summary_model=config.model_id,
                            policy=HistoryPolicy(
                                recent_groups=2,
                                segment_groups=20,
                                max_compactions_per_boundary=3,
                                max_summary_tokens=10000,
                            ),
                        )
                        capability = await configure_variant(
                            manager,
                            "task",
                            name,
                            selected_part_ids=[
                                v
                                for k, v in by_key.items()
                                if k != "irrelevant"
                                and not (name in {"B0", "B2"} and k == "plan-state")
                            ],
                            history=history,
                        )
                        if not VARIANT_REFRESH[name]:
                            await manager.next_context("task", consistency="pinned")
                        world.advance()

                        async def read_resource(key):
                            world.reads += 1
                            loaded = await manager.load("task", by_key[key])
                            return managed_result(
                                {
                                    "value": thaw(loaded.snapshot.value),
                                    "revision": loaded.snapshot.revision,
                                },
                                loaded.snapshot,
                            )

                        async def read_skill():
                            """Read the current task skill, including order and row limit."""
                            return await read_resource("skill")

                        async def read_dbschema():
                            """Read the current orders schema and itsrevision for execution
                            preconditions.
                            """
                            return await read_resource("dbschema")

                        async def read_tools():
                            """Read which query API version is currently permitted."""
                            return await read_resource("tools")

                        async def run_query_v1(sql: str, schema_revision: str):
                            """Execute one read-only SELECT using APIv1, if still permitted.
                            Include current schema revision.
                            """
                            return world.query(1, sql, schema_revision)

                        async def run_query_v2(sql: str, schema_revision: str):
                            """Execute one read-only SELECT using APIv2, if currently
                            permitted. Include current schema revision.
                            """
                            return world.query(2, sql, schema_revision)

                        audited = AuditedModel(model, ledger, trace, "execution")
                        agent = Agent(
                            audited,
                            name="v02_real_sql",
                            retries=4,
                            tools=[
                                Tool(read_skill),
                                Tool(read_dbschema),
                                Tool(read_tools),
                                Tool(run_query_v1),
                                Tool(run_query_v2),
                            ],
                            capabilities=[
                                Caching(messages=False),
                                *([capability] if name != "B0" else [FullRequestBudget()]),
                            ],
                            output_type=SqlAnswer,
                            model_settings={
                                "temperature": 0,
                                "max_tokens": 1024,
                                "extra_body": {"thinking": {"type": "disabled"}},
                            },
                        )
                        from pydantic_ai.usage import UsageLimits

                        result = await agent.run(
                            "Execute the previous task under CURRENT "
                            "skill, schema and permitted API. "
                            "Resources may have changed. Return the "
                            "resulting order IDs, after executing "
                            "SQL. "
                            "Use the current schema revision as the execution precondition.",
                            message_history=history_messages,
                            usage_limits=UsageLimits(request_limit=6, tool_calls_limit=8),
                        )
                        row.prediction = result.output.model_dump_json()
                        row.research_score = world.score(result.output.ids)
                        row.metrics = {
                            "read_tools": world.reads,
                            "execution_errors": world.errors,
                            "sql_executions": len(world.executions),
                            "compaction_records": store.db.execute(
                                "SELECT COUNT(*) FROM compaction_records"
                            ).fetchone()[0],
                        }
        except BudgetExceeded:
            row.failure_kind = "budget_exceeded"
        except Exception as error:
            row.failure_kind = "execution_or_infrastructure_error"
            row.error_type = type(error).__name__
            row.research_score = 0.0
            row.metrics = {
                "read_tools": world.reads,
                "execution_errors": world.errors,
                "error_detail": str(error)[:2000],
            }
        finally:
            world.db.close()
        row.harness_revision = HARNESS_REVISION
        row.end_to_end_ms = (perf_counter() - started) * 1000
        for key, value in trace.totals().items():
            setattr(row, key, value)
        row.input_digest = trace.calls[-1].input_digest if trace.calls else ""
        append_jsonl(Path(directory) / "results.jsonl", row.model_dump())
        append_jsonl(
            Path(directory) / "requests.jsonl",
            {
                "benchmark": row.benchmark,
                "case_id": case_id,
                "strategy": name,
                "calls": [asdict(c) for c in trace.calls],
            },
        )
        print(
            json.dumps(
                {
                    "track": "dynamic",
                    "case": case_id,
                    "strategy": name,
                    "score": row.research_score,
                    "failure": row.failure_kind,
                    "requests": ledger.requests,
                    "cost": ledger.spent,
                }
            ),
            flush=True,
        )
        if row.failure_kind == "budget_exceeded":
            return


VARIANT_REFRESH = {name: value.refresh for name, value in VARIANTS.items()}
