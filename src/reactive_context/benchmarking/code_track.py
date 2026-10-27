"""Context quality research pilot using real public repositories and unmodified metric functions.

This is a bounded single-call patch proposal study, not an official agent-outcome leaderboard run.
"""

import asyncio
import importlib.metadata
import json
import random
import shutil
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import tiktoken

from reactive_context import CompilerPolicy, Context, ContextIRSegment
from reactive_context.sources.config import ConfigSource

from .code_data import ContextBenchDataset, checkout_repository, retrieve_repository
from .contracts import TraceRecord
from .datasets import pilot_split
from .official import CACHE, ROOT, ContextBenchScorer, file_hash
from .report import write_report
from .runner import append_jsonl, write_json
from .runtime import BudgetExceeded, BudgetLedger, CallTrace, generate, make_model
from .strategies import reference


def token_proxy(text):
    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=())) + 8


async def select_context(strategy, case, documents, config):
    header = (
        "Resolve this repository issue using the source context below. "
        "Return a unified diff patch proposal; do not claim tests were run.\n\nIssue:\n"
        + case.question
        + "\n\nSource context:\n"
    )
    remaining = config.code_context_tokens_proxy - token_proxy(header)
    if remaining <= 0:
        raise RuntimeError("task alone exceeds the common input budget")
    if strategy == "B3":
        parts = []
        for index, document in enumerate(documents):
            key = f"span-{index:04d}"

            def project(value, ctx, key=key, index=index):
                return [
                    ContextIRSegment(
                        key,
                        key,
                        ctx.revision,
                        value,
                        kind="history",
                        required=False,
                        priority=len(documents) - index,
                    )
                ]

            parts.append(
                ConfigSource(
                    key, document.text, project=project, required=False, resource_key=document.id
                ).part
            )
        context = Context(
            parts,
            compiler_policy=CompilerPolicy(
                max_tokens=remaining + 1, output_reserve=0, estimator=token_proxy
            ),
        )
        try:
            snapshot = await context.next_context(case.case_id)
            compiled = context.compile(snapshot)
            included_ids = {message.resource_id for message in compiled.provider_messages}
            included = [document for document in documents if document.id in included_ids]
            refresh_ms = sum(e.duration_ms for e in context.events.events if e.event == "resolve")
            compile_ms = sum(e.duration_ms for e in context.events.events if e.event == "compile")
            ir_digest = sha256(
                json.dumps(
                    [
                        {
                            "id": segment.id,
                            "revision": segment.source_revision,
                            "payload": segment.payload,
                        }
                        for segment in snapshot.segments
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        finally:
            await context.close()
    else:
        included = []
        used = 0
        seen = set()
        for document in documents:
            if strategy == "B2":
                # A deterministic compact+BM25 baseline; remove exact duplicate source bodies.
                body = "\n".join(document.text.splitlines()[2:])
                fingerprint = sha256(body.encode()).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
            cost = token_proxy(document.text)
            if used + cost <= remaining + 1:
                included.append(document)
                used += cost
        refresh_ms = compile_ms = 0.0
        ir_digest = ""
    prompt = header + "\n\n".join(document.text for document in included)
    return prompt, included, refresh_ms, compile_ms, ir_digest


async def prepare_contextbench(directory, config):
    import os

    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(CACHE / "tokenizers"))
    dataset = ContextBenchDataset()
    selected, heldout = pilot_split(list(dataset.cases()), config.pilot_cases, config.seed)
    track = Path(directory) / "contextbench"
    track.mkdir(parents=True, exist_ok=True)
    scorer = ContextBenchScorer()
    source = track / "source"
    if not source.exists():
        shutil.copytree(
            ROOT / "src/reactive_context",
            source / "reactive_context",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(ROOT / "uv.lock", source / "uv.lock")
    active_hashes = {
        str(p.relative_to(ROOT / "src")): file_hash(p)
        for p in (ROOT / "src/reactive_context").rglob("*.py")
    }
    harness_revision = sha256(json.dumps(active_hashes, sort_keys=True).encode()).hexdigest()
    active_source = track / "repairs" / harness_revision / "source"
    if not active_source.exists():
        shutil.copytree(
            ROOT / "src/reactive_context",
            active_source / "reactive_context",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(ROOT / "uv.lock", active_source / "uv.lock")
    manifest = {
        "harness_revision": harness_revision,
        "active_source": str(active_source.relative_to(track)),
        "active_source_hashes": active_hashes,
        "benchmark": "contextbench",
        "benchmark_mode": "extension",
        "stage": "development_pilot",
        "configuration": config.public(),
        "dataset_revision": dataset.revision,
        "dataset_sha256": file_hash(dataset.path),
        "training_membership_sha256": file_hash(dataset.train_path),
        "eligible_original_training_tasks": len(selected) + len(heldout),
        "pilot_ids": [case.inference.case_id for case in selected],
        "excluded_original_train_tasks": 140,
        "scorer_revision": scorer.revision,
        "scorer_source_hash": scorer.source_hash,
        "official_leaderboard_comparable": False,
        "limits": [
            "common BM25 over pinned repository code, 80-line spans, 80 candidate spans",
            "single-call patch proposal; no official pass@1 or resolved-rate measurement",
            "original training gold format differs; join to JSON full "
            "release by exact issue/repo/base commit",
            "remaining full-release tasks are not used for tuning",
            "same-budget B0/B1/B3 may select identical context; not a causal reactivity experiment",
        ],
        "source_hashes": {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*.py")},
    }
    path = track / "manifest.json"
    if path.exists() and json.loads(path.read_text())["configuration"] != config.public():
        raise RuntimeError("frozen ContextBench configuration differs")
    if not path.exists():
        write_json(path, manifest)
    else:
        write_json(active_source.parent / "manifest.json", manifest)
    ready = {}
    semaphore = asyncio.Semaphore(3)

    async def prepare_one(case):
        async with semaphore:
            try:
                start = perf_counter()
                repo = await asyncio.to_thread(checkout_repository, case.inference)
                docs = await asyncio.to_thread(retrieve_repository, case.inference, repo, config)
                ready[case.inference.case_id] = (repo, docs, (perf_counter() - start) * 1000)
                print(
                    json.dumps(
                        {
                            "track": "contextbench",
                            "phase": "prepared",
                            "case_id": case.inference.case_id,
                            "candidates": len(docs),
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                ready[case.inference.case_id] = type(error).__name__
                print(
                    json.dumps(
                        {
                            "track": "contextbench",
                            "phase": "prepare_failed",
                            "case_id": case.inference.case_id,
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )

    await asyncio.gather(*(prepare_one(case) for case in selected))
    return dataset, selected, scorer, ready, manifest


async def run_contextbench(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset, selected, scorer, ready, manifest = await prepare_contextbench(directory, config)
    records = directory / "results.jsonl"
    if not records.exists():
        records.touch(mode=0o600)
    existing = [json.loads(line) for line in records.read_text().splitlines() if line]
    done = {
        (r["case_id"], r["strategy"])
        for r in existing
        if r["benchmark"] == "contextbench" and not r.get("failure_kind")
    }
    ledger = BudgetLedger(config, directory / "budget-ledger.json")
    semaphore = asyncio.Semaphore(config.concurrency)
    model = make_model(config)
    async with model:

        async def one(case):
            async with semaphore:
                names = ["B0", "B1", "B2", "B3"]
                random.Random(
                    config.seed + int(sha256(case.inference.case_id.encode()).hexdigest()[:8], 16)
                ).shuffle(names)
                for strategy in names:
                    if (case.inference.case_id, strategy) in done:
                        continue
                    start = perf_counter()
                    trace = CallTrace()
                    row = TraceRecord(
                        benchmark="contextbench",
                        harness_revision=manifest["harness_revision"],
                        attempt=1
                        + sum(
                            r["benchmark"] == "contextbench"
                            and r["case_id"] == case.inference.case_id
                            and r["strategy"] == strategy
                            for r in existing
                        ),
                        benchmark_mode="extension",
                        dataset_revision=dataset.revision,
                        case_id=case.inference.case_id,
                        strategy=strategy,
                        model_id=config.model_id,
                        random_seed=config.seed,
                        agent_runtime_version=importlib.metadata.version("pydantic-ai-slim"),
                        subgroup=case.subgroup,
                        cluster=case.cluster,
                        scorer_revision=scorer.revision,
                        scorer_source_hash=scorer.source_hash,
                        noncomparability_reasons=manifest["limits"],
                    )
                    try:
                        prepared = ready[case.inference.case_id]
                        if isinstance(prepared, str):
                            raise RuntimeError(prepared)
                        repo, docs, preparation_ms = prepared
                        prompt, included, refresh, compile_ms, ir_digest = await select_context(
                            strategy, case.inference, docs, config
                        )
                        row.retrieved = [reference(doc) for doc in docs]
                        row.included = [reference(doc) for doc in included]
                        row.source_revisions = {
                            doc.source_uri: doc.source_revision for doc in included
                        }
                        row.context_ir_digest = ir_digest
                        row.refresh_ms = refresh
                        row.compile_ms = compile_ms
                        row.prediction = await generate(model, ledger, trace, "answer", prompt)
                        row.input_digest = trace.calls[-1].input_digest
                        # Only now is the evaluator given gold labels
                        # and actual visible source spans.
                        row.metrics = await asyncio.to_thread(
                            scorer.score, case.evaluator_labels, row.included, repo
                        )
                        row.score_origin = (
                            "unaltered ContextBench compute_granularity_metrics; F1 is derived"
                        )
                        row.metrics["repository_preparation_ms"] = preparation_ms
                        row.metrics["pass_at_1"] = None
                        row.metrics["resolved_rate"] = None
                        write_json(
                            directory
                            / "contextbench/predictions"
                            / strategy
                            / (case.inference.case_id + ".json"),
                            {
                                "instance_id": case.inference.case_id,
                                "commit": case.inference.base_commit,
                                "repo_url": str(repo),
                                "model_patch": row.prediction,
                                "traj_data": {
                                    "pred_files": sorted(
                                        {ref["source_uri"] for ref in row.included}
                                    ),
                                    "pred_spans": {
                                        file: [
                                            {
                                                "start": ref["source_span"][0],
                                                "end": ref["source_span"][1],
                                            }
                                            for ref in row.included
                                            if ref["source_uri"] == file
                                        ]
                                        for file in {ref["source_uri"] for ref in row.included}
                                    },
                                },
                            },
                        )
                    except BudgetExceeded:
                        row.failure_kind = "budget_exceeded"
                        row.error_type = "BudgetExceeded"
                    except Exception as error:
                        row.failure_kind = "infrastructure_or_scoring_error"
                        row.error_type = type(error).__name__
                    for key, value in trace.totals().items():
                        setattr(row, key, value)
                    row.end_to_end_ms = (perf_counter() - start) * 1000
                    append_jsonl(records, row.model_dump())
                    append_jsonl(
                        directory / "requests.jsonl",
                        {
                            "benchmark": "contextbench",
                            "case_id": case.inference.case_id,
                            "strategy": strategy,
                            "calls": [asdict(call) for call in trace.calls],
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "benchmark": "contextbench",
                                "case_id": case.inference.case_id,
                                "strategy": strategy,
                                "failure": row.failure_kind,
                                "error_type": row.error_type,
                                "cost_upper_bound": round(ledger.spent, 4),
                                "requests": ledger.requests,
                            }
                        ),
                        flush=True,
                    )

        await asyncio.gather(*(one(case) for case in selected))
    write_report(directory)
    return directory
