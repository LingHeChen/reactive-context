"""Actual public-data pilot runs, resumable case records and evaluator isolation."""

import asyncio
import importlib.metadata
import json
import platform
import random
import shutil
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from .contracts import TraceRecord
from .datasets import LongMemEvalDataset, pilot_split
from .official import ROOT, LongMemEvalScorer, file_hash
from .report import write_report
from .runtime import BudgetExceeded, BudgetLedger, CallTrace, generate, make_model
from .strategies import MemoryStrategy

STRATEGIES = ("B0", "B1", "B2", "B3")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    path.chmod(0o600)


def append_jsonl(path, value):
    path = Path(path)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    path.chmod(0o600)


def freeze_manifest(directory, config, dataset, selected, heldout, scorer):
    track = directory / "longmemeval"
    track.mkdir(parents=True, exist_ok=True)
    path = track / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["configuration"] != config.public():
            raise RuntimeError("frozen configuration differs; do not tune the same pilot in place")
        return existing
    source = track / "source"
    shutil.copytree(
        ROOT / "src/reactive_context",
        source / "reactive_context",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", source / "uv.lock")
    data_hash = file_hash(dataset.path)
    manifest = {
        "benchmark": "longmemeval",
        "benchmark_mode": "extension",
        "stage": "development_pilot",
        "configuration": config.public(),
        "dataset_revision": dataset.revision,
        "dataset_sha256": data_hash,
        "scorer_revision": scorer.revision,
        "scorer_source_hash": scorer.source_hash,
        "reader_prompt_origin": ("unaltered official prepare_prompt AST, orig-session/json/direct"),
        "judge_profile": ("unaltered official get_anscheck_prompt and yes parser; DeepSeek judge"),
        "official_leaderboard_comparable": False,
        "noncomparability_reasons": [
            "DeepSeek judge replaces official model_zoo choices",
            "development subset, not complete 500",
        ],
        "pilot_ids": [case.inference.case_id for case in selected],
        "heldout_ids": [case.inference.case_id for case in heldout],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pydantic_ai": importlib.metadata.version("pydantic-ai-slim"),
        },
        "source_hashes": {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*.py")},
        "lock_sha256": file_hash(ROOT / "uv.lock"),
        "limitations": [
            "cl100k_base is a token-budget proxy, not the DeepSeek tokenizer",
            "B0/B1/B3 intentionally preserve immutable history; no "
            "supersession labels are inferred",
            "stale misuse, clinical claims and Task Success are not inferred from answer accuracy",
            "cost is a conservative peak-price estimate, not an invoice",
        ],
    }
    write_json(path, manifest)
    return manifest


async def run_longmemeval(directory, config, *, retry_failures=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / "results.jsonl"
    if not records_path.exists():
        records_path.touch(mode=0o600)
    dataset = LongMemEvalDataset()
    selected, heldout = pilot_split(list(dataset.cases()), config.pilot_cases, config.seed)
    scorer = LongMemEvalScorer()
    manifest = freeze_manifest(directory, config, dataset, selected, heldout, scorer)
    prior = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    done = {
        (row["case_id"], row["strategy"])
        for row in prior
        if row["benchmark"] == "longmemeval" and (not retry_failures or not row.get("failure_kind"))
    }
    source_files = {
        str(p.relative_to(ROOT / "src")): file_hash(p)
        for p in (ROOT / "src/reactive_context").rglob("*.py")
    }
    harness_revision = sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    if retry_failures:
        repair = directory / "longmemeval/repairs" / harness_revision[:12]
        if not repair.exists():
            shutil.copytree(
                ROOT / "src/reactive_context",
                repair / "reactive_context",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            write_json(
                repair / "repair.json",
                {
                    "reason": "empty source bootstrap compatibility fix",
                    "harness_revision": harness_revision,
                    "source_hashes": source_files,
                    "disclosure": (
                        "failed attempts retained; rerun cache order confounds cost/latency"
                    ),
                },
            )
    ledger = BudgetLedger(config, directory / "budget-ledger.json")
    semaphore = asyncio.Semaphore(config.concurrency)
    model, judge = make_model(config), make_model(config, judge=True)
    async with model, judge:

        async def one(case):
            async with semaphore:
                strategy_names = list(STRATEGIES)
                order_seed = config.seed + int(
                    sha256(case.inference.case_id.encode()).hexdigest()[:8], 16
                )
                random.Random(order_seed).shuffle(strategy_names)
                for name in strategy_names:
                    if (case.inference.case_id, name) in done:
                        continue
                    started = perf_counter()
                    trace = CallTrace()
                    prepared = None
                    row = TraceRecord(
                        benchmark="longmemeval",
                        attempt=1
                        + sum(
                            r["benchmark"] == "longmemeval"
                            and r["case_id"] == case.inference.case_id
                            and r["strategy"] == name
                            for r in prior
                        ),
                        harness_revision=harness_revision,
                        benchmark_mode="extension",
                        dataset_revision=dataset.revision,
                        case_id=case.inference.case_id,
                        strategy=name,
                        model_id=config.model_id,
                        agent_runtime_version=importlib.metadata.version("pydantic-ai-slim"),
                        random_seed=config.seed,
                        subgroup=case.subgroup,
                        cluster=case.cluster,
                        scorer_revision=scorer.revision,
                        scorer_source_hash=scorer.source_hash,
                        noncomparability_reasons=manifest["noncomparability_reasons"],
                    )
                    try:
                        strategy = MemoryStrategy(name, config, model, ledger, trace)
                        prepared = await strategy.prepare_request(case.inference)
                        row.retrieved, row.included = prepared.retrieved, prepared.included
                        row.source_revisions = {
                            ref["source_uri"]: ref["source_revision"] for ref in prepared.included
                        }
                        prediction = await generate(
                            model,
                            ledger,
                            trace,
                            "answer",
                            prepared.prompt,
                            capabilities=prepared.capabilities,
                        )
                        row.prediction = prediction
                        answer_calls = [call for call in trace.calls if call.phase == "answer"]
                        row.input_digest = answer_calls[0].input_digest
                        if prepared.context:
                            events = prepared.context.events.events
                            row.refresh_ms = sum(
                                e.duration_ms for e in events if e.event == "resolve"
                            )
                            row.compile_ms = sum(
                                e.duration_ms for e in events if e.event == "compile"
                            )
                            snapshot = await prepared.context.next_context(case.inference.case_id)
                            ir = [
                                {
                                    "part_id": s.part_id,
                                    "kind": s.kind,
                                    "role": s.requested_role,
                                    "revision": s.source_revision,
                                    "payload": s.payload,
                                }
                                for s in snapshot.segments
                            ]
                            row.context_ir_digest = sha256(
                                json.dumps(ir, sort_keys=True).encode()
                            ).hexdigest()
                        else:
                            row.refresh_ms = prepared.preparation_ms if name == "B1" else 0
                            row.compile_ms = 0
                        # Labels are first accessed AFTER reader prediction. They are never
                        # given to
                        # the context strategy, source parts, BM25 or compression agent.
                        prompt = scorer.judge_prompt(
                            case.inference.question, case.evaluator_labels, prediction
                        )
                        row.judge_prediction = await generate(
                            judge, ledger, trace, "judge", prompt, max_tokens=10
                        )
                        row.research_score = scorer.parse(row.judge_prediction)
                        row.score_origin = "upstream LongMemEval prompt/parser with DeepSeek judge"
                    except BudgetExceeded:
                        row.failure_kind, row.error_type = "budget_exceeded", "BudgetExceeded"
                    except Exception as error:
                        row.error_type = type(error).__name__
                        row.failure_kind = (
                            "timeout"
                            if isinstance(error, TimeoutError)
                            else "infrastructure_or_model_error"
                        )
                    finally:
                        if prepared:
                            await prepared.close()
                        for key, value in trace.totals().items():
                            setattr(row, key, value)
                        row.end_to_end_ms = (perf_counter() - started) * 1000
                        append_jsonl(records_path, row.model_dump())
                        prior.append(row.model_dump())
                        append_jsonl(
                            directory / "requests.jsonl",
                            {
                                "benchmark": "longmemeval",
                                "case_id": case.inference.case_id,
                                "strategy": name,
                                "calls": [asdict(call) for call in trace.calls],
                            },
                        )
                        done.add((case.inference.case_id, name))
                        print(
                            json.dumps(
                                {
                                    "benchmark": "longmemeval",
                                    "case_id": case.inference.case_id,
                                    "strategy": name,
                                    "score": row.research_score,
                                    "failure": row.failure_kind,
                                    "requests": ledger.requests,
                                    "cost_upper_bound": round(ledger.spent, 4),
                                }
                            ),
                            flush=True,
                        )
                        write_report(directory)

        await asyncio.gather(*(one(case) for case in selected))
    write_report(directory)
    return directory
