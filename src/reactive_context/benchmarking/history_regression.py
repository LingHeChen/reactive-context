"""Paired, budgeted regression pilots for minimal history and archived memory.

This is an extension using upstream public data and graders, not a leaderboard run.
"""

import argparse
import asyncio
import fcntl
import json
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from math import comb
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from pydantic_ai import Agent, BinaryContent, ToolReturn
from pydantic_ai.capabilities import Caching
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.usage import UsageLimits

from .. import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    HistoryManager,
    HistoryPolicy,
    Principal,
)
from ..adapters.pydantic_ai import ReactiveContextCapability
from . import longmem_v2 as v2
from .config import BenchmarkConfig
from .datasets import LongMemEvalDataset, pilot_split
from .official import ROOT, LongMemEvalScorer, file_hash
from .runtime import AuditedModel, BudgetExceeded, BudgetLedger, CallTrace, generate, make_model
from .strategies import rank_documents

HOST = Principal("memory-benchmark", True)
SEED = 83
ARMS = ("previous-4-groups", "minimal-history", "retrieval-compression")


class PreviousWindow(HistoryManager):
    """Controlled ablation: only restore the pre-change four-group projection.

    Reader, corpus, archive, retrieval tools and instructions are identical.
    """

    async def prepare(self, scope, groups, **kwargs):
        kwargs.pop("projection_groups", None)
        return await super().prepare(scope, groups, **kwargs)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def choose_v2(count):
    # Independent of answers and outcomes; stratify public ability and domain.
    buckets = defaultdict(list)
    for q in v2.load_questions():
        buckets[(q.category, q.domain)].append(q)
    rng = random.Random(SEED)
    for bucket in buckets.values():
        bucket.sort(key=lambda q: q.id)
        rng.shuffle(bucket)
    chosen = []
    while len(chosen) < min(count, 451):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop())
    return chosen


def choose_s(count):
    excluded = set()
    for path in (
        ROOT / "results/public-pilot/longmemeval/manifest.json",
        ROOT / "results/v02/pilot/manifest.json",
    ):
        if path.exists():
            data = json.loads(path.read_text())
            excluded.update(data.get("pilot_ids", data.get("memory_ids", [])))
    return pilot_split(
        [c for c in LongMemEvalDataset().cases() if c.inference.case_id not in excluded],
        count,
        SEED,
    )[0]


def chat_history(case):
    messages = []
    for doc in case.inference.documents:
        session = json.loads(doc.text)
        header = f"Historical session {doc.id}; date {session['date']}\n"
        for turn in session["turns"]:
            if turn["role"] == "user":
                messages.append(ModelRequest.user_text_prompt(header + turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(ModelResponse(parts=[TextPart(turn["content"])]))
            else:
                raise ValueError("unexpected public chat role")
    return messages


def image_content(root, image):
    return BinaryContent(data=v2.safe_image(root, image).read_bytes(), media_type="image/png")


async def answer_agent(
    model, ledger, trace, arm, question, *, raw=(), base_store=None, domain=None, image=None
):
    with tempfile.TemporaryDirectory(prefix="rc-regression-") as temporary:
        directory = Path(temporary) / "store"
        if base_store:
            # DBs are copied; immutable CAS files may safely share inodes. No GC is run here.
            def copy(source, target):
                if "/objects/" in str(source):
                    Path(target).hardlink_to(source)
                    return target
                return shutil.copy2(source, target)

            shutil.copytree(base_store, directory, copy_function=copy)
        with ContextStore(directory) as store:
            if not base_store:
                store.create_scope(HOST, "memory")
            history_cls = PreviousWindow if arm == "previous-4-groups" else HistoryManager
            history = history_cls(store, HOST, policy=HistoryPolicy(recent_groups=4))
            # V2 contains images; UTF-8 transport bytes are a conservative compiler proxy,
            # distinct from provider usage tokens. Both are recorded, never conflated.
            capacity = 8_000_000 if domain else 128000
            async with ContextManager(
                store=store,
                principal=HOST,
                history=history,
                compiler_policy=CompilerPolicy(max_tokens=capacity),
            ) as manager:
                agent = Agent(
                    AuditedModel(model, ledger, trace, "answer"),
                    name="history_regression",
                    retries=0,
                    instructions=v2.domain_prompt(domain)
                    if domain
                    else (
                        "Answer the user's question using historical conversations when needed. "
                        "Preserve dates and updates. If evidence is insufficient, say so."
                    ),
                    capabilities=[
                        Caching(messages=False),
                        ReactiveContextCapability(manager, scope="memory", cache=None),
                    ],
                    model_settings={
                        "temperature": 0,
                        "max_tokens": 768,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                )
                if domain:

                    @agent.tool_plain
                    async def context_history_image(event_id: str):
                        """Read the original screenshot of a previously retrieved history event."""
                        events = await manager.load_history("memory", [event_id])
                        relative = events[0].get("state", {}).get("screenshot")
                        if not relative:
                            return {"error": "no_screenshot"}
                        content = await asyncio.to_thread(image_content, v2.DATA_ROOT, relative)
                        return ToolReturn(return_value={"event_id": event_id}, content=[content])

                prompt = [question, image_content(v2.DATA_ROOT, image)] if image else question
                started = perf_counter()
                result = await agent.run(
                    prompt, message_history=list(raw), usage_limits=UsageLimits(request_limit=6)
                )
                return result.output, {
                    "answer_ms": (perf_counter() - started) * 1000,
                    "raw_message_count": len(result.all_messages()),
                    "archive_events": store.db.execute(
                        "SELECT COUNT(*) FROM history_events"
                    ).fetchone()[0],
                    "compiler_budget_proxy": capacity,
                }


def lexical_v2(base_store, question):
    with ContextStore(base_store) as store:
        history = HistoryManager(store, HOST)
        hits = history.search("memory", question, kind="event", limit=3)
        events = history.retrieve("memory", [hit["id"] for hit in hits])
    texts, images = [], []
    for hit, event in zip(hits, events, strict=True):
        texts.append(json.dumps({"event_id": hit["id"], "event": event}, ensure_ascii=False)[:6000])
        screenshot = event.get("state", {}).get("screenshot")
        if screenshot:
            images.append(image_content(v2.DATA_ROOT, screenshot))
    return texts, images


async def answer_compressed(
    model, ledger, trace, question, *, case=None, base_store=None, domain=None, image=None
):
    if case is not None:
        documents = rank_documents(question, case.inference.documents)[:3]
        texts, images = [d.text[:6000] for d in documents], []
    else:
        texts, images = await asyncio.to_thread(lexical_v2, base_store, question)
    prompt = [
        "Extract compact historical evidence relevant to the QUESTION. Preserve state changes, "
        "dates, workflow steps, exceptions and uncertainty. Do not invent missing information.\n"
        + "QUESTION: "
        + question
        + "\nEVIDENCE:\n"
        + "\n".join(texts),
        *images,
    ]
    if image:
        prompt.append(image_content(v2.DATA_ROOT, image))
    summary = await generate(model, ledger, trace, "summary", prompt, max_tokens=1024)
    instructions = (
        v2.domain_prompt(domain)
        if domain
        else "Answer using the historical evidence. State when evidence is insufficient."
    )
    answer_prompt = [
        ("" if domain else instructions + "\n")
        + "HISTORICAL EVIDENCE:\n"
        + summary
        + "\nQUESTION: "
        + question
    ]
    if image:
        answer_prompt.append(image_content(v2.DATA_ROOT, image))
    prediction = await generate(
        model,
        ledger,
        trace,
        "answer",
        answer_prompt,
        max_tokens=768,
        instructions=instructions if domain else None,
    )
    return prediction, {"retrieved_items": len(texts), "summary_characters": len(summary)}


async def grade_v2(judge, ledger, trace, question, prediction):
    scorer = v2.load_scorer()
    parsed = scorer.extract_boxed_answer(prediction)
    if scorer.is_unknown(parsed):
        return 0.0
    spec = question.labels["eval_function"]
    function, kwargs = scorer.parse_eval_function_spec(spec)
    name = spec.split("|", 1)[0]
    if name in {"llm_abstention_checker", "llm_gotchas_checker"}:
        build = getattr(
            scorer,
            "_build_abstention_judge_messages"
            if name == "llm_abstention_checker"
            else "_build_gotchas_judge_messages",
        )
        messages = build(
            question_text=question.question,
            reference_answer=question.labels["answer"],
            model_full_response=prediction,
            model_final_answer=parsed,
        )
        response = await generate(
            judge,
            ledger,
            trace,
            "judge",
            messages[1]["content"],
            max_tokens=384,
            instructions=messages[0]["content"],
        )
        label, _ = scorer._parse_llm_binary_judgement(response)
        return float(label)
    return float(scorer.score_to_bool(function(parsed, question.labels["answer"], **kwargs)))


async def answer_closed(model, ledger, trace, question, *, domain, image):
    prompt = [question, image_content(v2.DATA_ROOT, image)] if image else question
    prediction = await generate(
        model,
        ledger,
        trace,
        "answer",
        prompt,
        max_tokens=768,
        instructions=v2.domain_prompt(domain),
    )
    return prediction, {"historical_context_items": 0}


async def run_case(output, config, model, judge, ledger, benchmark, case, stores):
    is_v2 = benchmark == "longmemeval-v2"
    case_id = case.id if is_v2 else case.inference.case_id
    question = (
        case.question
        if is_v2
        else f"Question date: {case.inference.question_date}\nQuestion: {case.inference.question}"
    )
    arms = ["minimal-history", "retrieval-compression", "no-memory"] if is_v2 else list(ARMS)
    random.Random(SEED + sum(case_id.encode())).shuffle(arms)
    completed = (
        {
            (r["benchmark"], r["case_id"], r["arm"])
            for line in (output / "results.jsonl").read_text().splitlines()
            if (r := json.loads(line)).get("status") == "completed"
        }
        if (output / "results.jsonl").exists()
        else set()
    )
    for arm in arms:
        if (benchmark, case_id, arm) in completed:
            continue
        trace = CallTrace(capture_directory=output / "traces" / benchmark / case_id / arm)
        row = {
            "benchmark": benchmark,
            "case_id": case_id,
            "arm": arm,
            "category": case.category if is_v2 else case.subgroup,
            "domain": case.domain if is_v2 else "chat",
            "status": "started",
        }
        started = perf_counter()
        try:
            kwargs = (
                {"base_store": stores[case.domain], "domain": case.domain, "image": case.image}
                if is_v2
                else {}
            )
            if arm == "no-memory":
                prediction, metrics = await answer_closed(
                    model, ledger, trace, question, domain=case.domain, image=case.image
                )
            elif arm == "retrieval-compression":
                prediction, metrics = await answer_compressed(
                    model, ledger, trace, question, case=None if is_v2 else case, **kwargs
                )
            else:
                prediction, metrics = await answer_agent(
                    model,
                    ledger,
                    trace,
                    arm,
                    question,
                    raw=() if is_v2 else chat_history(case),
                    **kwargs,
                )
            row.update(prediction=prediction, metrics=metrics)
            if is_v2:
                score = await grade_v2(judge, ledger, trace, case, prediction)
            else:
                scorer = LongMemEvalScorer()
                prompt = scorer.judge_prompt(
                    case.inference.question, case.evaluator_labels, prediction
                )
                judgement = await generate(judge, ledger, trace, "judge", prompt, max_tokens=384)
                score = scorer.parse(judgement)
                row["judge_prediction"] = judgement
            row.update(status="completed", score=score)
        except BudgetExceeded:
            row.update(status="budget_stopped", error_type="BudgetExceeded")
            row.update(trace.totals())
            append(output / "results.jsonl", row)
            append(output / "requests.jsonl", {**row, "calls": [asdict(c) for c in trace.calls]})
            raise
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__)
            if type(error).__name__ in {
                "HistoryBudgetExceeded",
                "ContextBudgetExceeded",
                "UsageLimitExceeded",
                "ProtocolInvalid",
            }:
                row["score"] = 0.0
        row.update(trace.totals())
        row["end_to_end_ms"] = (perf_counter() - started) * 1000
        answer_calls = [c for c in trace.calls if c.phase == "answer"]
        row["initial_input_tokens"] = answer_calls[0].input_tokens if answer_calls else None
        row["initial_input_bytes"] = answer_calls[0].input_bytes if answer_calls else None
        row["answer_request_count"] = len(answer_calls)
        append(output / "results.jsonl", row)
        append(
            output / "requests.jsonl",
            {
                "benchmark": benchmark,
                "case_id": case_id,
                "arm": arm,
                "calls": [asdict(c) for c in trace.calls],
            },
        )
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in ("benchmark", "case_id", "arm", "status", "score", "error_type")
                }
            ),
            flush=True,
        )


def loss_probability_upper(losses, count):
    """Exact one-sided 95% binomial bound; zero observed losses is not proof."""
    if not count or losses == count:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2
        cdf = sum(
            comb(count, k) * middle**k * (1 - middle) ** (count - k) for k in range(losses + 1)
        )
        if cdf > 0.05:
            low = middle
        else:
            high = middle
    return high


def paired_summary(rows, benchmark, baseline):
    cells = {
        (r["case_id"], r["arm"]): r
        for r in rows
        if r["benchmark"] == benchmark and r.get("score") is not None
    }
    ids = sorted(
        {
            key[0]
            for key in cells
            if (key[0], baseline) in cells and (key[0], "minimal-history") in cells
        }
    )
    if not ids:
        return {"paired_cases": 0}
    old, new = [cells[(i, baseline)] for i in ids], [cells[(i, "minimal-history")] for i in ids]
    delta = [n["score"] - o["score"] for o, n in zip(old, new, strict=True)]
    loss_upper = loss_probability_upper(sum(d < 0 for d in delta), len(delta))
    rng = random.Random(SEED)
    boot = sorted(mean(rng.choices(delta, k=len(delta))) for _ in range(2000))
    categories = {}
    for category in sorted({r["category"] for r in new}):
        positions = [j for j, r in enumerate(new) if r["category"] == category]
        categories[category] = {
            "n": len(positions),
            "baseline_accuracy": mean(old[j]["score"] for j in positions),
            "minimal_accuracy": mean(new[j]["score"] for j in positions),
        }
    return {
        "paired_cases": len(ids),
        "baseline": baseline,
        "baseline_accuracy": mean(r["score"] for r in old),
        "minimal_accuracy": mean(r["score"] for r in new),
        "accuracy_delta": mean(delta),
        "paired_bootstrap_95_ci": [boot[49], boot[1949]],
        "noninferiority_margin": 0.05,
        "loss_probability_upper_95": loss_upper,
        "noninferiority_status": "conservative_bound_within_margin"
        if loss_upper <= 0.05
        else "not_established",
        "wins": sum(d > 0 for d in delta),
        "losses": sum(d < 0 for d in delta),
        "ties": sum(d == 0 for d in delta),
        "regression_case_ids": [ids[j] for j, d in enumerate(delta) if d < 0],
        "baseline_initial_tokens_median": median(r["initial_input_tokens"] or 0 for r in old),
        "minimal_initial_tokens_median": median(r["initial_input_tokens"] or 0 for r in new),
        "baseline_total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in old),
        "minimal_total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in new),
        "baseline_cost": sum(r["estimated_cost_usd_upper_bound"] for r in old),
        "minimal_cost": sum(r["estimated_cost_usd_upper_bound"] for r in new),
        "baseline_latency_median_ms": median(r["end_to_end_ms"] for r in old),
        "minimal_latency_median_ms": median(r["end_to_end_ms"] for r in new),
        "categories": categories,
    }


def report(output):
    rows = (
        [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        if (output / "results.jsonl").exists()
        else []
    )
    comparisons = {
        "short_history_vs_previous": paired_summary(rows, "longmemeval", "previous-4-groups"),
        "short_history_vs_compression": paired_summary(
            rows, "longmemeval", "retrieval-compression"
        ),
        "v2_vs_compression": paired_summary(rows, "longmemeval-v2", "retrieval-compression"),
        "v2_vs_no_memory": paired_summary(rows, "longmemeval-v2", "no-memory"),
    }
    write_json(
        output / "summary.json",
        {"comparisons": comparisons, "failures": [r for r in rows if r["status"] != "completed"]},
    )
    lines = [
        "# 最小历史：优化与能力防退化先导",
        "",
        "相同 DeepSeek 模型、同题配对；官方数据与评分规则的研究变体，不是官方排行榜成绩。",
        "",
        "首请求成本与总成本分开统计。LongMemEval-V2 使用完整 small haystack、"
        "问题截图与可检索轨迹截图。",
        "",
        "样本为分层先导，不能保证在未测题目上不退化；bootstrap 在小样本和全平局时会低估不确定性。"
        "防退化判定另用单侧精确二项界，不将零次退化当作证明。",
        "",
    ]
    for name, comparison in comparisons.items():
        lines += [
            f"## {name}",
            "",
            "```json",
            json.dumps(comparison, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "## 失败或预算中止",
        "",
        "```json",
        json.dumps([r for r in rows if r["status"] != "completed"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (output / "conclusions.md").write_text("\n".join(lines))


async def run(output, tracks, cases_s, cases_v2, allocation="existing"):
    output.mkdir(parents=True, exist_ok=True)
    config = replace(
        BenchmarkConfig.load(ROOT / ".env"), max_requests=2000, max_cost_usd=5, seed=SEED
    )
    ledger_path = (
        ROOT / "results/history-regression-budget.json"
        if allocation == "new"
        else ROOT / "results/v02/budget-ledger.json"
    )
    ledger = BudgetLedger(config, ledger_path)
    selected_s = choose_s(cases_s) if "longmemeval" in tracks else []
    selected_v2 = choose_v2(cases_v2) if "longmemeval-v2" in tracks else []
    manifest = {
        "seed": SEED,
        "configuration": config.public(),
        "benchmark_mode": "extension",
        "official_leaderboard_comparable": False,
        "longmemeval_ids": [c.inference.case_id for c in selected_s],
        "v2_ids": [c.id for c in selected_v2],
        "v2_data": v2.describe() if selected_v2 else None,
        "budget": {
            "ledger": str(ledger.state_path),
            "starting_requests": ledger.requests,
            "starting_cost_upper_bound": ledger.spent,
            "allocation_source": (
                "explicit user reply: additional $5/2000 requests, 2026-10-24"
                if allocation == "new"
                else "remaining prior $5/2000 authorization"
            ),
        },
        "controls": (
            "previous-4-groups differs only in recent projection; "
            "same new read-only tools and instructions"
        ),
        "limitations": [
            "DeepSeek reader/judge differ from official fixed models",
            "no provider random seed pinning",
            "V2 screenshots available on demand; lexical retrieval does not embed images",
            "UTF-8 compiler budget proxy differs from provider tokens",
            "pilot alone cannot establish population non-regression",
            "balanced pilot accuracy is not the official aggregate or LAFS metric",
            "V2 questions share two haystacks; question-level uncertainty assumes independence",
            "no pre-generated Facts or work objective; this tests cold archive recall",
        ],
    }
    files = sorted((ROOT / "src/reactive_context").rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    manifest["source_hashes"] = hashes
    revision = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    source = output / "source" / revision
    if not source.exists():
        for p in files:
            target = source / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    manifest_path = output / ("manifest-" + "-".join(tracks) + ".json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("longmemeval_ids", "v2_ids", "source_hashes", "configuration"):
            if previous[key] != manifest[key]:
                raise ValueError("frozen paired run differs; choose a new output path")
    else:
        write_json(manifest_path, manifest)
    stores = {}
    if selected_v2:
        for domain in sorted({q.domain for q in selected_v2}):
            path = output / "haystacks" / domain
            metadata = path / "prepared.json"
            if not metadata.exists():
                with ContextStore(path) as store:
                    store.create_scope(HOST, "memory")
                    value = await asyncio.to_thread(v2.archive_haystack, store, HOST, domain)
                write_json(metadata, value)
            stores[domain] = path
    async with make_model(config) as model, make_model(config, judge=True) as judge:
        semaphore = asyncio.Semaphore(config.concurrency)

        async def bounded(benchmark, case):
            async with semaphore:
                await run_case(output, config, model, judge, ledger, benchmark, case, stores)

        try:
            for benchmark, cases in (("longmemeval", selected_s), ("longmemeval-v2", selected_v2)):
                tasks = [asyncio.create_task(bounded(benchmark, case)) for case in cases]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except BudgetExceeded:
            print("Shared authorized budget reached; no additional calls scheduled.", flush=True)
        finally:
            report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/minimal-history-pilot")
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=["longmemeval", "longmemeval-v2"],
        default=["longmemeval", "longmemeval-v2"],
    )
    parser.add_argument("--cases-s", type=int, default=12)
    parser.add_argument("--cases-v2", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allocation", choices=["existing", "new"], default="existing")
    args = parser.parse_args()
    if args.report_only:
        report(args.output)
        return
    if not 1 <= args.cases_s <= 500 or not 1 <= args.cases_v2 <= 451:
        parser.error("invalid sample counts")
    with (ROOT / "results/v02/pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args.output, args.tracks, args.cases_s, args.cases_v2, args.allocation))


if __name__ == "__main__":
    main()
