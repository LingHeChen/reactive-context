"""Separately authorized v02 pilot, shared $5/2000-request ceiling, sequential tracks."""

import argparse
import asyncio
import fcntl
import json
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from . import dynamic_pilot, public_v02
from .cache_pilot import run_cache_probe
from .config import BenchmarkConfig
from .datasets import LongMemEvalDataset
from .official import ROOT, ContextBenchScorer, LongMemEvalScorer, file_hash
from .public_v02 import new_cases, prepare_code_cases, run_code_case, run_memory_case
from .runtime import BudgetExceeded, BudgetLedger, make_model


def freeze(directory, config):
    files = sorted((ROOT / "src/reactive_context").rglob("*.py")) + [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    revision = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    source = directory / "source" / revision
    if not source.exists():
        for path in files:
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    memory = new_cases(LongMemEvalDataset(), "longmemeval")
    code_dataset = public_v02.ContextBenchDataset()
    code = new_cases(code_dataset, "contextbench")
    lme_scorer, code_scorer = LongMemEvalScorer(), ContextBenchScorer()
    config_path = directory / "manifest.json"
    manifest = {
        "version": "0.2",
        "configuration": config.public(),
        "authorization": {
            "additional_max_requests": 2000,
            "additional_max_cost_usd": 5,
            "source": "explicit user reply 2026-10-24",
        },
        "budget_ledger": str(ROOT / "results/v02/budget-ledger.json"),
        "benchmark_mode": "extension",
        "official_leaderboard_comparable": False,
        "memory_ids": [c.inference.case_id for c in memory],
        "code_ids": [c.inference.case_id for c in code],
        "memory_dataset_revision": LongMemEvalDataset.revision,
        "memory_sha256": file_hash(public_v02.LongMemEvalDataset().path),
        "code_dataset_revision": code_dataset.revision,
        "code_sha256": file_hash(code_dataset.path),
        "scorers": {
            "longmemeval": {"commit": lme_scorer.revision, "sha256": lme_scorer.source_hash},
            "contextbench": {"commit": code_scorer.revision, "sha256": code_scorer.source_hash},
        },
        "engineering_cases": 12,
        "engineering_variants": list(dynamic_pilot.ENGINEERING_VARIANTS),
        "cache_seeds": [17, 29, 43],
        "cache_steps_per_arm": 6,
        "model_seed_semantics": (
            "provider seed unavailable; seed controls paired strategy order/generation only"
        ),
        "static_not_applicable": ["no-refresh", "no-dependency-invalidation"],
        "source_revisions": [revision],
        "source_hashes": hashes,
        "limitations": [
            "DeepSeek QA judge is a research variant",
            "code selector is an extra model planner; include its "
            "cost and do not attribute selection gain to reactivity",
            "code patch proposals are not task pass@1; no Docker test execution",
            "engineering SQL tasks are generated integration tests, not public benchmark scores",
            "aliases cannot pin provider weights; record actual response model IDs",
            "public dataset card licensing must be reviewed before redistribution",
        ],
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ("configuration", "memory_ids", "code_ids", "engineering_variants"):
            if previous[key] != manifest[key]:
                raise ValueError("frozen pilot configuration differs")
        manifest["source_revisions"] = list(
            dict.fromkeys(previous["source_revisions"] + [revision])
        )
    config_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return memory, revision


async def run(directory, config, tracks):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    memory, revision = freeze(directory, config)
    public_v02.HARNESS_REVISION = dynamic_pilot.HARNESS_REVISION = revision
    ledger = BudgetLedger(config, ROOT / "results/v02/budget-ledger.json")
    semaphore = asyncio.Semaphore(config.concurrency)
    async with make_model(config) as model, make_model(config, judge=True) as judge:

        async def limited(function, *args):
            async with semaphore:
                try:
                    await function(*args)
                except BudgetExceeded:
                    return
                except Exception as error:
                    from .runner import append_jsonl

                    append_jsonl(
                        directory / "infrastructure-errors.jsonl",
                        {"operation": function.__name__, "error_type": type(error).__name__},
                    )

        if "cache" in tracks:
            await run_cache_probe(directory, config, model, ledger, revision)
        if "dynamic" in tracks:
            await asyncio.gather(
                *(
                    limited(dynamic_pilot.run_dynamic_case, directory, config, model, ledger, i)
                    for i in range(12)
                )
            )
        if "memory" in tracks:
            await asyncio.gather(
                *(
                    limited(run_memory_case, directory, config, model, judge, ledger, case)
                    for case in memory
                )
            )
        if "code" in tracks:
            cases, ready = await prepare_code_cases(config)
            await asyncio.gather(
                *(
                    limited(
                        run_code_case,
                        directory,
                        config,
                        model,
                        ledger,
                        case,
                        ready[case.inference.case_id],
                    )
                    for case in cases
                )
            )
    from .report_v02 import write_report

    write_report(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results/v02/pilot"))
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=["cache", "dynamic", "memory", "code"],
        default=["cache", "dynamic", "memory", "code"],
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        from .report_v02 import write_report

        write_report(Path(args.output))
        return
    # One allocation shared by every v02 run/output, independent of the old pilot ledger.
    lock_path = ROOT / "results/v02/pilot.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    config = replace(
        BenchmarkConfig.load(ROOT / ".env"),
        max_requests=2000,
        max_cost_usd=5,
        seed=41,
        pilot_cases=30,
    )
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args.output, config, args.tracks))


if __name__ == "__main__":
    main()
