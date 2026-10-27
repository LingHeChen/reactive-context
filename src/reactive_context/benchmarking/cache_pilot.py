"""Real-provider fixed/permuted prefix crossover, independent from quality benchmarks."""

import itertools
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, SystemPromptPart

from .contracts import TraceRecord
from .runner import append_jsonl
from .runtime import BudgetExceeded, CallTrace, generate


class SkillPrefix(AbstractCapability):
    def __init__(self, nonce, order):
        self.nonce, self.order = nonce, order

    async def before_model_request(self, ctx, request_context):
        from .. import CompilerPolicy, Context, SkillRegistry, standard
        from ..sources.config import ConfigSource

        async with Context(
            [standard.system("Experiment " + self.nonce + ".", key="host")],
            compiler_policy=CompilerPolicy(
                max_tokens=200000, grants={"host": "system"}, part_order=("host",)
            ),
        ) as context:
            registry = SkillRegistry(
                context, role="system", order=tuple("skill." + x for x in self.order)
            )
            for letter in "ABC":
                source = ConfigSource(
                    "skill." + letter,
                    f"Reference vocabulary {letter}: alpha beta gamma delta epsilon. " * 250,
                )
                registry.register(
                    source.part,
                    name="Skill " + letter,
                    description="Neutral reference vocabulary",
                    trusted=True,
                )
                registry.activate(source.key)
            snapshot = await context.next_context()
            compiled = context.compile(snapshot)
            prefix: list = [
                SystemPromptPart(item.content)
                for item in compiled.provider_messages
                if item.role in {"system", "developer"}
            ]
            suffix = [item.content for item in compiled.provider_messages if item.role == "user"]
            if suffix:
                from pydantic_ai.messages import UserPromptPart

                prefix += [UserPromptPart(x) for x in suffix]
            return replace(
                request_context, messages=[ModelRequest(parts=prefix), *request_context.messages]
            )


async def run_cache_probe(directory, config, model, ledger, harness_revision):
    for seed in (17, 29, 43):
        orders = list(itertools.permutations("ABC"))
        random.Random(seed).shuffle(orders)
        for step in range(6):
            arms = ["fixed", "permuted"]
            random.Random(seed + step).shuffle(arms)
            for arm in arms:
                trace = CallTrace(
                    capture_directory=Path(directory) / "traces/cache" / f"{seed}-{step}-{arm}"
                )
                row = TraceRecord(
                    benchmark="engineering-prefix-cache",
                    benchmark_mode="extension",
                    dataset_revision="neutral-skills-prefix-v02",
                    case_id=f"seed{seed}-step{step}",
                    strategy=arm,
                    model_id=config.model_id,
                    agent_runtime_version="pydantic-ai-slim-2.60.0",
                    random_seed=seed,
                    cluster=f"seed{seed}",
                    harness_revision=harness_revision,
                )
                order = tuple("ABC") if arm == "fixed" else orders[step]
                # Equal-length namespaces block cross-arm prefix sharing. Same namespace within
                # arm.
                nonce = f"{harness_revision[:12]}-{seed}-arm{'A' if arm == 'fixed' else 'B'}"
                start = perf_counter()
                try:
                    row.prediction = await generate(
                        model,
                        ledger,
                        trace,
                        "cache-probe",
                        "Reply OK only.",
                        capabilities=[SkillPrefix(nonce, order)],
                        max_tokens=4,
                    )
                    row.input_digest = trace.calls[-1].input_digest
                except BudgetExceeded:
                    row.failure_kind = "budget_exceeded"
                except Exception as error:
                    row.failure_kind, row.error_type = "provider_error", type(error).__name__
                for key, value in trace.totals().items():
                    setattr(row, key, value)
                row.end_to_end_ms = (perf_counter() - start) * 1000
                row.metrics = {
                    "order": list(order),
                    "warmup": step == 0,
                    "protocol": (
                        "3 namespaces per arm; 6 paired steps; all 6 "
                        "permutations without replacement"
                    ),
                    "cache_class": (
                        "observed response usage; no assumption first call was actually cold"
                    ),
                }
                append_jsonl(Path(directory) / "results.jsonl", row.model_dump())
                append_jsonl(
                    Path(directory) / "requests.jsonl",
                    {
                        "benchmark": row.benchmark,
                        "case_id": row.case_id,
                        "strategy": arm,
                        "calls": [asdict(c) for c in trace.calls],
                    },
                )
                print(
                    json.dumps(
                        {
                            "track": "cache",
                            "seed": seed,
                            "step": step,
                            "arm": arm,
                            "read_tokens": row.cache_read_tokens,
                            "input_tokens": row.input_tokens,
                            "cost": ledger.spent,
                        }
                    ),
                    flush=True,
                )
                if row.failure_kind == "budget_exceeded":
                    return
