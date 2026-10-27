"""Reproducible fixture benchmarks; never presented as LLM answer-quality measurements."""

import argparse
import asyncio
import json
import platform
import random
import statistics
import tracemalloc
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from reactive_context import CompilerPolicy, Context, computed_part
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol
from reactive_context.observability.events import EventLog
from reactive_context.sources.config import ConfigSource


class Meter(EventLog):
    def __init__(self):
        super().__init__(max_events=100)
        self.resolves = self.hits = 0

    def emit(self, event, **kwargs):
        if event == "resolve":
            self.resolves += 1
            self.hits += int(kwargs.get("cache_hit", False))
        return super().emit(event, **kwargs)


def percentile(values, fraction):
    return sorted(values)[max(0, int(len(values) * fraction + 0.999) - 1)]


def wilson(success, count):
    z = 1.96
    p = success / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    margin = z * ((p * (1 - p) / count + z * z / (4 * count * count)) ** 0.5) / denominator
    return [round(centre - margin, 4), round(centre + margin, 4)]


async def dag_bench(samples):
    rows = []
    for size in (10, 100, 1000):
        for workload in ("unchanged", "one_change", "continuous_between_boundaries"):
            source = ConfigSource("root", 0)
            parts = [source.part]
            for i in range(1, size):
                dependency = "root" if i == 1 else f"node-{i - 1}"

                async def compute(ctx):
                    return next(iter(ctx.dependencies.values())).value + 1

                parts.append(
                    computed_part(
                        key=f"node-{i}", schema=int, dependencies=[dependency], compute=compute
                    )
                )
            meter = Meter()
            async with Context(
                parts, compiler_policy=CompilerPolicy(max_tokens=2_000_000), event_log=meter
            ) as context:
                context.compile(await context.next_context())
                before_resolves, before_hits = meter.resolves, meter.hits
                timings, compile_times = [], []
                tracemalloc.start()
                for index in range(samples):
                    if workload == "continuous_between_boundaries" or (
                        workload == "one_change" and index == samples // 2
                    ):
                        source.set(index + 1)
                    start = perf_counter()
                    snapshot = await context.next_context()
                    compile_start = perf_counter()
                    context.compile(snapshot)
                    end = perf_counter()
                    timings.append((end - start) * 1000)
                    compile_times.append((end - compile_start) * 1000)
                retained, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                resolves = meter.resolves - before_resolves
                hits = meter.hits - before_hits
                rows.append(
                    {
                        "parts": size,
                        "workload": workload,
                        "samples": samples,
                        "boundary_p50_ms": round(statistics.median(timings), 3),
                        "boundary_p95_ms": round(percentile(timings, 0.95), 3),
                        "compile_p50_ms": round(statistics.median(compile_times), 3),
                        "compile_p95_ms": round(percentile(compile_times, 0.95), 3),
                        "retained_bytes": retained,
                        "peak_bytes": peak,
                        "cache_hit_rate": round(hits / resolves, 4),
                    }
                )
    return rows


async def compare_bench(trials):
    rows = []
    for changed in (False, True):
        for group in ("B0", "B1", "B2", "B3"):
            latest = old = protocol = history_ok = total = resolves = 0
            latencies, sizes = [], []
            fresh_trials = obsolete_trials = 0
            for seed in range(trials):
                latest_before, old_before = latest, old
                nonce = random.Random(seed).randrange(1_000_000)
                a, b = f"RULE_A_ONLY_{nonce}", f"RULE_B_ONLY_{nonce}"
                source = ConfigSource("rules", a)
                original_resolve = source.part.resolve
                actual_resolves = 0

                async def counted(ctx, resolve_hook=original_resolve):
                    nonlocal actual_resolves
                    actual_resolves += 1
                    return await resolve_hook(ctx)

                part = replace(source.part, resolve=counted)
                async with Context([part]) as context:
                    initial = await context.next_context("benchmark")
                    provenance = {
                        "resource_id": "rules",
                        "revision": initial.part("rules").revision,
                    }
                    history = [
                        ModelRequest(parts=[UserPromptPart("Load rules")]),
                        ModelResponse(
                            parts=[ToolCallPart("read_rules", {}, tool_call_id="load-1")]
                        ),
                        ModelRequest(
                            parts=[
                                ToolReturnPart(
                                    "read_rules",
                                    a,
                                    tool_call_id="load-1",
                                    metadata={"reactive_context": provenance},
                                )
                            ]
                        ),
                    ]
                    saved = deepcopy(history)
                    if changed:
                        source.set(b)
                    active = b if changed else a

                    class Baseline(AbstractCapability):
                        def __init__(self, source, group, changed):
                            self.source, self.group, self.changed = source, group, changed

                        async def before_model_request(self, ctx, request_context):
                            messages = deepcopy(list(request_context.messages))
                            if self.group in {"B1", "B2"}:
                                nonlocal actual_resolves
                                actual_resolves += 1
                                value = await self.source.resolve(None)
                                if self.group == "B2" and self.changed:
                                    for message in messages:
                                        if isinstance(message, ModelRequest):
                                            message.parts = [
                                                replace(p, content="historical body withdrawn")
                                                if isinstance(p, ToolReturnPart)
                                                else p
                                                for p in message.parts
                                            ]
                                messages.append(
                                    ModelRequest(
                                        parts=[
                                            UserPromptPart(
                                                json.dumps(
                                                    {
                                                        "resource": "rules",
                                                        "revision": await self.source.revision(
                                                            None
                                                        ),
                                                        "status": "CURRENT",
                                                        "data": value,
                                                    },
                                                    ensure_ascii=False,
                                                )
                                            )
                                        ]
                                    )
                                )
                            return replace(request_context, messages=messages)

                    def respond(
                        messages,
                        info,
                        active=active,
                        changed=changed,
                        a=a,
                        sizes=sizes,
                        context=context,
                    ):
                        nonlocal latest, old, protocol, total
                        text = str(messages)
                        latest += int(active in text)
                        old += int(changed and a in text)
                        validate_protocol(messages)
                        protocol += 1
                        total += 1
                        sizes.append(context.compiler.policy.estimator(text))
                        return ModelResponse(parts=[TextPart("fixture response")])

                    capability = (
                        ReactiveContextCapability(context, scope="benchmark")
                        if group == "B3"
                        else Baseline(source, group, changed)
                    )
                    agent = Agent(
                        FunctionModel(respond), capabilities=[capability], name=f"benchmark_{group}"
                    )
                    for _ in range(3):
                        start = perf_counter()
                        await agent.run("Continue using current rules", message_history=history)
                        latencies.append((perf_counter() - start) * 1000)
                        history_ok += int(history == saved)
                    resolves += actual_resolves
                    fresh_trials += int(latest - latest_before == 3)
                    obsolete_trials += int(old > old_before)
            rows.append(
                {
                    "group": group,
                    "world": "updated" if changed else "static",
                    "requests": total,
                    "independent_fixture_trials": trials,
                    "refresh_without_read_tool_rate": latest / total,
                    "llm_task_success_rate": None,
                    "correct_new_tool_call_rate": None,
                    "latest_body_present_rate": latest / total,
                    "latest_body_wilson95": wilson(fresh_trials, trials),
                    "obsolete_body_exposure_rate": old / total,
                    "obsolete_body_wilson95": wilson(obsolete_trials, trials),
                    "protocol_integrity_rate": protocol / total,
                    "history_fidelity_rate": history_ok / total,
                    "total_source_resolves": resolves,
                    "request_p50_ms": round(statistics.median(latencies), 3),
                    "request_p95_ms": round(percentile(latencies, 0.95), 3),
                    "input_byte_upper_bound_mean": round(statistics.mean(sizes), 1),
                }
            )
    return rows


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pydantic_ai": "2.60.0",
        },
        "limitations": [
            "Deterministic FunctionModel fixtures; no LLM answer-quality or task-success claims.",
            "Wilson intervals use trial-level clusters (3 requests per trial).",
            "Fixture exposure intervals do not estimate real-world model answer quality.",
            "All B0–B3 groups share seeds, inputs, model, tool history, default budget and events.",
            "DAG timing includes tracemalloc overhead and metadata logging; no SLA asserted.",
            "Continuous workload changes between boundaries; in-flight churn is tested separately.",
            "Token figures use a conservative UTF-8 byte upper bound, not billed provider tokens.",
        ],
        "comparison": await compare_bench(args.trials),
        "dag": await dag_bench(args.samples),
    }
    output = Path("docs/benchmark-results.json")
    await asyncio.to_thread(output.write_text, json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}: {len(report['comparison'])} comparison / {len(report['dag'])} DAG rows")


if __name__ == "__main__":
    asyncio.run(main())
