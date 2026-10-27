"""v02 public-data pilots. Original labels reach only the post-prediction evaluators."""

# Per-variant callbacks are fully awaited and disposed before the sequential loop advances.
# ruff: noqa: B023
import asyncio
import json
import random
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from .. import (
    CompilerPolicy,
    ContextIRSegment,
    ContextManager,
    ContextStore,
    HistoryManager,
    HistoryPolicy,
    Principal,
    SummaryDraft,
    source_part,
)
from ..core.snapshot import thaw
from ..errors import ContextBudgetExceeded
from ..history import CompactionRejected, HistoryBudgetExceeded, HistoryGroup
from ..history_part import history_part
from .code_data import ContextBenchDataset, checkout_repository, retrieve_repository
from .code_track import token_proxy
from .contracts import TraceRecord
from .datasets import LongMemEvalDataset, official_memory_prompt, pilot_split
from .official import ROOT, ContextBenchScorer, LongMemEvalScorer
from .runner import append_jsonl
from .runtime import BudgetExceeded, CallTrace, generate
from .strategies import rank_documents, reference


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1 : text.rfind("```")]
    return json.loads(text)


def new_cases(dataset, track, seed=41):
    prior = ROOT / "results/public-pilot" / track / "manifest.json"
    excluded = set(json.loads(prior.read_text())["pilot_ids"]) if prior.exists() else set()
    return pilot_split(
        [c for c in dataset.cases() if c.inference.case_id not in excluded], 30, seed
    )[0]


def result_row(config, case, strategy, revision, *, benchmark=None):
    return TraceRecord(
        benchmark=benchmark or case.inference.benchmark,
        benchmark_mode="extension",
        dataset_revision=revision,
        case_id=case.inference.case_id,
        strategy=strategy,
        model_id=config.model_id,
        agent_runtime_version="pydantic-ai-slim-2.60.0",
        random_seed=41,
        subgroup=case.subgroup,
        cluster=case.cluster,
        noncomparability_reasons=[
            "development pilot; not an official leaderboard score",
            "provider seed unavailable; seed controls paired strategy order only",
        ],
    )


HARNESS_REVISION = ""


def finish(directory, row, trace, started):
    row.harness_revision = HARNESS_REVISION
    for key, value in trace.totals().items():
        setattr(row, key, value)
    row.end_to_end_ms = (perf_counter() - started) * 1000
    row.input_digest = next(
        (c.input_digest for c in reversed(trace.calls) if c.phase == "answer"), ""
    )
    append_jsonl(Path(directory) / "results.jsonl", row.model_dump())
    append_jsonl(
        Path(directory) / "requests.jsonl",
        {
            "benchmark": row.benchmark,
            "case_id": row.case_id,
            "strategy": row.strategy,
            "calls": [asdict(c) for c in trace.calls],
        },
    )


async def run_memory_case(directory, config, model, judge, ledger, case):
    names = ["B0", "B1", "B2", "B3", "B3-no-active-selection", "B3-no-history-compaction"]
    random.Random(41 + sum(case.inference.case_id.encode())).shuffle(names)
    scorer = LongMemEvalScorer()
    for name in names:
        trace = CallTrace(
            capture_directory=Path(directory) / "traces/memory" / case.inference.case_id / name
        )
        row = result_row(config, case, name, LongMemEvalDataset.revision)
        start = perf_counter()
        try:
            documents = list(case.inference.documents)
            selected = (
                documents
                if name in {"B0", "B1", "B3-no-active-selection"}
                else rank_documents(case.inference.question, documents)[: config.rag_sessions]
            )
            row.retrieved = [reference(d) for d in selected]
            row.included = [reference(d) for d in selected]
            prompt = official_memory_prompt(case.inference, selected, config.context_tokens_proxy)
            if name == "B2":
                summary = await generate(
                    model,
                    ledger,
                    trace,
                    "summary",
                    "Extract compact evidence for this question from "
                    "historical chats. Preserve dates, "
                    "updates and uncertainty. Do not invent facts.\n" + prompt,
                    max_tokens=1536,
                )
                prompt = memory_answer_prompt(case, summary)
            elif name in {"B3", "B3-no-active-selection"}:
                with tempfile.TemporaryDirectory(prefix="rc-memory-") as temporary:
                    host = Principal("pilot", True)
                    with ContextStore(temporary) as store:
                        store.create_scope(host, "memory")
                        groups = tuple(HistoryGroup((json.loads(d.text),)) for d in selected)

                        async def compact(events):
                            anchors = [HistoryManager.event_id(e) for e in events]
                            text = await generate(
                                model,
                                ledger,
                                trace,
                                "history-summary",
                                "Extract evidence relevant to the "
                                "QUESTION from historical chats. "
                                "Return JSON with text,constraints,un"
                                "resolved,task_state,evidence_ids,"
                                "current_resource_claims. All "
                                "resource claims are historical, so "
                                "use {} "
                                "for current_resource_claims. Choose "
                                "at most 3 evidence_ids from this "
                                "list: "
                                + json.dumps(anchors)
                                + "\nStrict output JSON schema: "
                                + json.dumps(SummaryDraft.model_json_schema())
                                + "\nQUESTION: "
                                + case.inference.question
                                + "\nHISTORY: "
                                + json.dumps(events),
                                max_tokens=1536,
                            )
                            return SummaryDraft.model_validate(parse_json(text))

                        async def validate(events, draft):
                            sources = [
                                e
                                for e in events
                                if HistoryManager.event_id(e) in draft.evidence_ids
                            ]
                            verdict = await generate(
                                judge,
                                ledger,
                                trace,
                                "summary-validation",
                                "Is every factual claim in the "
                                "summary supported by the supplied "
                                "cited "
                                "historical records, without "
                                "reversing updates? Reply yes or no "
                                "only.\n"
                                + json.dumps({"summary": draft.model_dump(), "sources": sources}),
                                max_tokens=4,
                            )
                            return verdict.strip().lower() == "yes"

                        history = HistoryManager(
                            store,
                            host,
                            compactor=compact,
                            validator=validate,
                            estimator=token_proxy,
                            summary_model=config.model_id,
                            policy=HistoryPolicy(
                                projection="compact",
                                context_ratio=config.history_context_ratio,
                                recent_groups=2,
                                segment_groups=max(1, len(groups) - 2),
                                max_compactions_per_boundary=2,
                                max_summary_tokens=3000,
                            ),
                        )
                        compiler_policy = CompilerPolicy(
                            max_tokens=config.context_tokens_proxy,
                            output_reserve=config.max_output_tokens,
                            estimator=token_proxy,
                        )
                        definitions = []
                        for index, doc in enumerate(selected):

                            async def load_session(ctx, text=doc.text):
                                return json.loads(text)

                            async def session_revision(ctx, rev=doc.source_revision):
                                return rev

                            definitions.append(
                                source_part(
                                    key=f"session-{index}",
                                    schema=dict,
                                    resolve=load_session,
                                    revision=session_revision,
                                    kind="history",
                                    expose=False,
                                    resource_key=f"history-data:{index}",
                                )
                            )

                        def load_groups(ctx):
                            return tuple(
                                HistoryGroup(
                                    (thaw(p.value),), revisions={p.resource_id: p.revision}
                                )
                                for p in ctx.dependencies.values()
                            )

                        definitions.append(
                            history_part(
                                "history.bundle",
                                history,
                                load_groups=load_groups,
                                dependencies=[d.key for d in definitions],
                                context_policy=compiler_policy,
                                reserved_tokens=token_proxy(memory_answer_prompt(case, "")),
                                configuration={"question": case.inference.question},
                            )
                        )
                        async with ContextManager(
                            definitions,
                            store=store,
                            principal=host,
                            mandatory=["history.bundle"],
                            compiler_policy=compiler_policy,
                        ) as context:
                            context.publish_catalog(host, "memory")
                            after_id = None
                            while True:
                                page = await context.query("memory", limit=100, after_id=after_id)
                                for entry in page:
                                    store.activate(host, "memory", entry["id"])
                                if len(page) < 100:
                                    break
                                after_id = page[-1]["id"]
                            try:
                                snapshot = await context.next_context("memory")
                                compiled = context.compile(snapshot)
                                evidence_text = "\n".join(
                                    m.content for m in compiled.provider_messages
                                )
                                row.context_ir_digest = sha256(evidence_text.encode()).hexdigest()
                                prompt = memory_answer_prompt(case, evidence_text)
                            except (CompactionRejected, HistoryBudgetExceeded) as error:
                                row.metrics["compaction_fallback"] = type(error).__name__
                                prompt = official_memory_prompt(
                                    case.inference, selected, config.context_tokens_proxy
                                )
                        row.metrics["compaction_records"] = store.db.execute(
                            "SELECT COUNT(*) FROM compaction_records"
                        ).fetchone()[0]
                        row.metrics["summary_validation"] = (
                            "DeepSeek cited-source entailment; not lossless proof"
                        )
                        row.metrics["included_reference_semantics"] = (
                            "original provenance, not verbatim retention"
                        )
            row.prediction = await generate(model, ledger, trace, "answer", prompt)
            # Reference labels become accessible only after the answer has been generated.
            row.judge_prediction = await generate(
                judge,
                ledger,
                trace,
                "judge",
                scorer.judge_prompt(case.inference.question, case.evaluator_labels, row.prediction),
                max_tokens=10,
            )
            row.research_score = scorer.parse(row.judge_prediction)
            row.scorer_revision, row.scorer_source_hash = scorer.revision, scorer.source_hash
            row.score_origin = "original LongMemEval prompt/parser; DeepSeek judge research variant"
        except BudgetExceeded:
            row.failure_kind = "budget_exceeded"
        except Exception as error:
            row.failure_kind = "preparation_or_model_error"
            row.error_type = type(error).__name__
        finish(directory, row, trace, start)
        print(
            json.dumps(
                {
                    "track": "memory",
                    "case": row.case_id,
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


def memory_answer_prompt(case, evidence):
    return (
        "I will give you several history chats between you and a user. Please answer the "
        "question based on the relevant chat history.\n\n\nHistory Chats:\n\n"
        + evidence
        + "\n\nCurrent Date: "
        + case.inference.question_date
        + "\nQuestion: "
        + case.inference.question
        + "\nAnswer:"
    )


async def prepare_code_cases(config):
    cases = new_cases(ContextBenchDataset(), "contextbench")
    ready = {}
    semaphore = asyncio.Semaphore(3)

    async def one(case):
        async with semaphore:
            try:
                repo = await asyncio.to_thread(checkout_repository, case.inference)
                documents = await asyncio.to_thread(
                    retrieve_repository, case.inference, repo, config
                )
                ready[case.inference.case_id] = repo, documents
            except Exception as error:
                ready[case.inference.case_id] = type(error).__name__
            print(
                json.dumps(
                    {
                        "phase": "code-prepared",
                        "case": case.inference.case_id,
                        "ready": not isinstance(ready[case.inference.case_id], str),
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(one(case) for case in cases))
    return cases, ready


async def code_context(config, case, documents, indices):
    with tempfile.TemporaryDirectory(prefix="rc-code-") as temporary:
        host = Principal("pilot", True)
        with ContextStore(temporary) as store:
            store.create_scope(host, "code")
            question = (
                "Resolve this repository issue. Return a unified diff"
                " proposal; do not claim tests ran.\n" + case.question
            )
            definitions = []
            for index, document in enumerate(documents):

                async def resolve(ctx, text=document.text):
                    return text

                async def revision(ctx, rev=document.source_revision):
                    return rev

                def project(value, ctx, index=index):
                    return [
                        ContextIRSegment(
                            f"span-{index}", f"span-{index}", ctx.revision, value, kind="code"
                        )
                    ]

                definitions.append(
                    source_part(
                        key=f"span-{index}",
                        schema=str,
                        resolve=resolve,
                        revision=revision,
                        project=project,
                        resource_key=document.id,
                        kind="resource.code",
                    )
                )

            async def task(ctx):
                return question

            definitions.append(
                source_part(
                    key="task",
                    schema=str,
                    resolve=task,
                    revision=lambda ctx: sha256(question.encode()).hexdigest(),
                    kind="task_state",
                )
            )
            async with ContextManager(
                definitions,
                store=store,
                principal=host,
                mandatory=["task"],
                compiler_policy=CompilerPolicy(
                    max_tokens=config.code_context_tokens_proxy + config.max_output_tokens,
                    output_reserve=config.max_output_tokens,
                    estimator=token_proxy,
                    part_order=("task",),
                ),
            ) as manager:
                manager.publish_catalog(host, "code")
                ids = {r["key"]: r["id"] for r in await manager.query("code", limit=100)}
                included, rejected = [], []
                for index in indices:
                    try:
                        await manager.activate("code", ids[f"span-{index}"])
                        included.append(documents[index])
                    except ContextBudgetExceeded:
                        rejected.append(index)
                snapshot = await manager.next_context("code")
                compiled = manager.compile(snapshot)
                return (
                    "\n\n".join(m.content for m in compiled.provider_messages),
                    included,
                    rejected,
                    compiled.compile_id,
                )


async def run_code_case(directory, config, model, ledger, case, prepared):
    names = ["B0", "B1", "B2", "B3", "B3-no-active-selection"]
    random.Random(41 + sum(case.inference.case_id.encode())).shuffle(names)
    scorer = ContextBenchScorer()
    candidate_metrics = None
    for name in names:
        trace = CallTrace(
            capture_directory=Path(directory) / "traces/code" / case.inference.case_id / name
        )
        row = result_row(config, case, name, ContextBenchDataset.revision)
        start = perf_counter()
        try:
            if isinstance(prepared, str):
                raise RuntimeError(prepared)
            repo, documents = prepared
            indices = list(range(len(documents)))
            if name == "B2":
                seen = set()
                indices = []
                for index, doc in enumerate(documents):
                    body = "\n".join(doc.text.splitlines()[2:])
                    if body not in seen:
                        indices.append(index)
                        seen.add(body)
            if name == "B3":
                candidates = [
                    {
                        "index": i,
                        "path": d.source_uri,
                        "lines": d.source_span,
                        "cost_tokens_proxy": token_proxy(d.text),
                        "preview": d.text[:350],
                    }
                    for i, d in enumerate(documents)
                ]
                response = await generate(
                    model,
                    ledger,
                    trace,
                    "active-selection",
                    "Select repository source spans needed to resolve the issue. Return JSON "
                    '{"indices":[...]}, in relevance order. Consider '
                    "dependencies and different files. "
                    "Keep total source cost within 8500 tokens. Do "
                    "not invent indices.\nISSUE:\n"
                    + case.inference.question
                    + "\nCANDIDATES:\n"
                    + json.dumps(candidates),
                    max_tokens=512,
                )
                data = parse_json(response)["indices"]
                if not isinstance(data, list) or not all(
                    type(i) is int and 0 <= i < len(documents) for i in data
                ):
                    raise ValueError("invalid source selection")
                indices = list(dict.fromkeys(data))
            prompt, included, rejected, compile_id = await code_context(
                config, case.inference, documents, indices
            )
            row.context_ir_digest = sha256(prompt.encode()).hexdigest()
            row.retrieved = [reference(d) for d in documents]
            row.included = [reference(d) for d in included]
            row.prediction = await generate(model, ledger, trace, "answer", prompt)
            row.metrics = scorer.score(case.evaluator_labels, row.included, repo)
            if candidate_metrics is None:
                candidate_metrics = scorer.score(case.evaluator_labels, row.retrieved, repo)
            row.metrics.update(
                compile_id=compile_id,
                candidate_metrics=candidate_metrics,
                rejected_activation_indices=rejected,
                resolved_rate=None,
                pass_at_1=None,
                selection_method="LLM metadata planner" if name == "B3" else "BM25",
            )
            row.scorer_revision, row.scorer_source_hash = scorer.revision, scorer.source_hash
            row.score_origin = (
                "unmodified ContextBench granularity metric functions; "
                "bounded patch proposal, no task tests"
            )
        except BudgetExceeded:
            row.failure_kind = "budget_exceeded"
        except Exception as error:
            row.failure_kind = "preparation_or_model_error"
            row.error_type = type(error).__name__
        finish(directory, row, trace, start)
        print(
            json.dumps(
                {
                    "track": "code",
                    "case": row.case_id,
                    "strategy": name,
                    "failure": row.failure_kind,
                    "requests": ledger.requests,
                    "cost": ledger.spent,
                }
            ),
            flush=True,
        )
        if row.failure_kind == "budget_exceeded":
            return
