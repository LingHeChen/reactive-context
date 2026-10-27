"""Paired task-level inference; never treat repeated requests as independent cases."""

import csv
import json
import math
import random
import statistics
from pathlib import Path


def bootstrap(values, seed=17, iterations=2000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return [samples[int(iterations * 0.025)], samples[min(iterations - 1, int(iterations * 0.975))]]


def mcnemar_exact(first, second):
    b = sum(a == 1 and c == 0 for a, c in zip(first, second, strict=True))
    c = sum(a == 0 and d == 1 for a, d in zip(first, second, strict=True))
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.0
    return {"discordant_b3_only": b, "discordant_baseline_only": c, "p_value": p}


def write_report(directory):
    directory = Path(directory)
    rows = [
        json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line
    ]
    all_attempts = rows
    latest = {}
    for row in rows:
        latest[(row["benchmark"], row["case_id"], row["strategy"])] = row
    rows = list(latest.values())
    groups = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["strategy"]), []).append(row)
    metrics = []
    for (benchmark, strategy), records in sorted(groups.items()):
        evaluated = [r for r in records if r.get("research_score") is not None]
        success = sum(r["research_score"] for r in evaluated)
        attempts = [
            r for r in all_attempts if r["benchmark"] == benchmark and r["strategy"] == strategy
        ]
        token_total = sum(
            (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) for r in attempts
        )
        metric = {
            "benchmark": benchmark,
            "strategy": strategy,
            "planned_or_recorded": len(records),
            "attempts_total": len(attempts),
            "failed_attempts_total": sum(bool(r.get("failure_kind")) for r in attempts),
            "evaluated": (
                sum(bool(r.get("metrics", {}).get("file")) for r in records)
                if benchmark == "contextbench"
                else len(evaluated)
            ),
            "failures": sum(bool(r.get("failure_kind")) for r in records),
            "research_accuracy": success / len(evaluated) if evaluated else None,
            "accuracy_with_unscored_as_failure": success / len(records) if evaluated else None,
            "request_count": sum(r.get("request_count", 0) for r in attempts),
            "accuracy_ci95": bootstrap([r["research_score"] for r in evaluated]),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in attempts),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in attempts),
            "cache_read_tokens": sum(r.get("cache_read_tokens") or 0 for r in attempts),
            "cost_usd_upper_bound": sum(
                r.get("estimated_cost_usd_upper_bound") or 0 for r in attempts
            ),
            "tokens_per_success": token_total / success if success else None,
            "latency_p50_ms": statistics.median(
                [r["end_to_end_ms"] for r in records if r.get("end_to_end_ms") is not None]
            )
            if any(r.get("end_to_end_ms") is not None for r in records)
            else None,
        }
        latencies = sorted(
            r["end_to_end_ms"] for r in records if r.get("end_to_end_ms") is not None
        )
        metric["latency_p95_ms"] = (
            latencies[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)]
            if latencies
            else None
        )
        metric["failure_breakdown"] = {
            kind: sum(r.get("failure_kind") == kind for r in records)
            for kind in sorted({r["failure_kind"] for r in records if r.get("failure_kind")})
        }
        metric["cache_read_fraction"] = (
            metric["cache_read_tokens"] / metric["input_tokens"] if metric["input_tokens"] else None
        )
        if benchmark == "contextbench":
            for granularity in ("file", "symbol", "line", "span"):
                for measure in ("precision", "coverage", "derived_f1"):
                    values = [
                        r["metrics"][granularity][measure]
                        for r in records
                        if r.get("metrics") and granularity in r["metrics"]
                    ]
                    metric[f"{granularity}_{measure}"] = statistics.mean(values) if values else None
        metrics.append(metric)
    with (directory / "metrics.csv").open("w") as stream:
        columns = sorted({key for row in metrics for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics)
    comparisons = []
    for benchmark in sorted({r["benchmark"] for r in rows}):
        for baseline in ("B1", "B2", "B0"):
            a = {
                r["case_id"]: r
                for r in rows
                if r["benchmark"] == benchmark and r["strategy"] == "B3"
            }
            b = {
                r["case_id"]: r
                for r in rows
                if r["benchmark"] == benchmark and r["strategy"] == baseline
            }
            ids = sorted(a.keys() & b.keys())
            valid = [
                i
                for i in ids
                if a[i].get("research_score") is not None and b[i].get("research_score") is not None
            ]
            scores_a, scores_b = (
                [a[i]["research_score"] for i in valid],
                [b[i]["research_score"] for i in valid],
            )
            delta = [x - y for x, y in zip(scores_a, scores_b, strict=True)]
            comparisons.append(
                {
                    "benchmark": benchmark,
                    "comparison": "B3-" + baseline,
                    "paired_n": len(valid),
                    "difference": statistics.mean(delta) if delta else None,
                    "difference_ci95": bootstrap(delta),
                    "identical_actual_input_pairs": sum(
                        bool(a[i]["input_digest"]) and a[i]["input_digest"] == b[i]["input_digest"]
                        for i in ids
                    ),
                    **mcnemar_exact(scores_a, scores_b),
                }
            )
    # Holm correction covers the three prespecified binary B3 comparisons.
    binary = sorted((c for c in comparisons if c["paired_n"]), key=lambda c: c["p_value"])
    previous = 0.0
    for index, comparison in enumerate(binary):
        previous = max(previous, min(1.0, comparison["p_value"] * (len(binary) - index)))
        comparison["holm_p_value"] = previous
    context_pairs = []
    code_rows = [r for r in rows if r["benchmark"] == "contextbench"]
    for baseline in ("B0", "B1", "B2"):
        a = {r["case_id"]: r for r in code_rows if r["strategy"] == "B3"}
        b = {r["case_id"]: r for r in code_rows if r["strategy"] == baseline}
        for granularity in ("file", "symbol", "line", "span"):
            for measure in ("precision", "coverage", "derived_f1"):
                clusters = {}
                deltas = []
                for case_id in sorted(a.keys() & b.keys()):
                    left = a[case_id].get("metrics", {}).get(granularity, {})
                    right = b[case_id].get("metrics", {}).get(granularity, {})
                    if measure not in left or measure not in right:
                        continue
                    delta = left[measure] - right[measure]
                    deltas.append(delta)
                    clusters.setdefault(a[case_id]["cluster"], []).append(delta)
                rng = random.Random(17)
                names = sorted(clusters)
                draws = (
                    sorted(
                        statistics.mean(
                            [v for name in rng.choices(names, k=len(names)) for v in clusters[name]]
                        )
                        for _ in range(2000)
                    )
                    if names
                    else []
                )
                context_pairs.append(
                    {
                        "comparison": "B3-" + baseline,
                        "metric": f"{granularity}_{measure}",
                        "paired_n": len(deltas),
                        "difference": statistics.mean(deltas) if deltas else None,
                        "task_bootstrap_ci95": bootstrap(deltas),
                        "repository_clusters": len(names),
                        "repository_cluster_bootstrap_ci95": [draws[50], draws[1950]]
                        if draws
                        else [None, None],
                    }
                )
    subgroups = []
    for benchmark, strategy in sorted(groups):
        records = groups[benchmark, strategy]
        for subgroup in sorted({r["subgroup"] for r in records}):
            subset = [r for r in records if r["subgroup"] == subgroup]
            values = [r["research_score"] for r in subset if r["research_score"] is not None]
            subgroups.append(
                {
                    "benchmark": benchmark,
                    "strategy": strategy,
                    "subgroup": subgroup,
                    "recorded": len(subset),
                    "failures": sum(bool(r["failure_kind"]) for r in subset),
                    "research_accuracy": statistics.mean(values) if values else None,
                }
            )
    normalization_sensitivity = []
    audit_file = directory / "contextbench/annotation-source-audit.json"
    if audit_file.exists():
        affected = {
            r["case_id"] for r in json.loads(audit_file.read_text()) if r["missing_source_files"]
        }
        for strategy in ("B0", "B1", "B2", "B3"):
            subset = [
                r
                for r in code_rows
                if r["strategy"] == strategy
                and r["case_id"] not in affected
                and r.get("metrics", {}).get("file")
            ]
            normalization_sensitivity.append(
                {
                    "strategy": strategy,
                    "evaluated": len(subset),
                    "excluded_case_ids": sorted(affected),
                    "file_precision": statistics.mean(
                        r["metrics"]["file"]["precision"] for r in subset
                    )
                    if subset
                    else None,
                    "file_coverage": statistics.mean(
                        r["metrics"]["file"]["coverage"] for r in subset
                    )
                    if subset
                    else None,
                    "label": "exploratory sensitivity; original scores unchanged",
                }
            )
    (directory / "statistics.json").write_text(
        json.dumps(
            {
                "groups": metrics,
                "paired": comparisons,
                "context_paired": context_pairs,
                "subgroups": subgroups,
                "normalization_sensitivity": normalization_sensitivity,
            },
            indent=2,
        )
    )
    lines = [
        "# Public benchmark pilot — actual run",
        "",
        "These are development pilots, not full official leaderboard scores."
        " Official source data and",
        "score functions remain unchanged. DeepSeek QA judging is a research"
        " variant, not the official",
        "LongMemEval judge. No model-quality result is inferred from mechanism fixtures.",
        "",
        "| Benchmark | Strategy | Evaluated | Research accuracy | Failures |"
        " Cost upper bound USD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        acc = f"{row['research_accuracy']:.3f}" if row["research_accuracy"] is not None else "N/A"
        lines.append(
            f"| {row['benchmark']} | {row['strategy']} | {row['evaluated']} | {acc} | "
            f"{row['failures']} | {row['cost_usd_upper_bound']:.4f} |"
        )
    lines += [
        "",
        "ContextBench macro context quality (official metric functions):",
        "",
        "| Strategy | File precision | File coverage | Line precision | Line coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if row["benchmark"] == "contextbench":
            values = [
                row.get(key)
                for key in ("file_precision", "file_coverage", "line_precision", "line_coverage")
            ]
            lines.append(
                "| "
                + row["strategy"]
                + " | "
                + " | ".join(f"{v:.4f}" if v is not None else "N/A" for v in values)
                + " |"
            )
    lines += [
        "",
        "Static equal-input comparisons do not establish a causal reactivity benefit.",
        "Two ContextBench tasks have five gold paths whose leading dots are stripped by the",
        "pinned upstream path normalizer. Original files exist. Main scores are unmodified;",
        "statistics.json includes a separately labelled sensitivity excluding those two tasks.",
        "See contextbench/annotation-source-audit.json for source evidence.",
        "",
        "Paired bootstrap confidence intervals, exact McNemar tests and all negative/null",
        "results and Holm-corrected binary comparisons are in "
        "statistics.json. Context differences are",
        "exploratory. Identical-input pairs cannot establish a causal benefit from reactivity.",
        "",
        "Infrastructure failures and budget stops remain in results.jsonl "
        "and in the failure column;",
        "accuracy excludes unscored failures; no missing result is silently treated as correct.",
        "Raw failed attempts remain visible. Accuracy uses the most recent "
        "attempt per task/strategy;",
        "all attempts contribute to token and cost totals. Repair reruns may have warmer caches,",
        "so their latency/cost differences are descriptive, not evidence of a causal advantage.",
        "",
        "Cost uses response token usage and conservative peak rates, not an "
        "invoice. All compression",
        "and research-judge requests count toward tokens and the common cost cap.",
        "",
        "Original task histories have no version/supersession labels: stale-"
        "misuse is N/A unless an",
        "official annotation explicitly supplies that ground truth. "
        "ContextBench included context is",
        "what was sent to the model; used-context and task pass@1 remain N/A"
        " unless actually measured.",
        "",
    ]
    ledger_path = directory / "budget-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger_accounting = {}
    if ledger:
        unassigned = max(
            0, ledger["requests"] - sum(r.get("request_count", 0) for r in all_attempts)
        )
        ledger_accounting = {
            "requests_without_completed_task_trace": unassigned,
            "preflight_requests": 1 if (directory / "preflight.json").exists() else 0,
            "unassigned_cost_upper_bound": max(
                0.0,
                ledger["cost_upper_bound"]
                - sum(r.get("estimated_cost_usd_upper_bound") or 0 for r in all_attempts),
            ),
        }
        lines += [
            f"Requests without completed task traces: {unassigned}; includes the preflight and",
            "interrupted requests. Reserved costs remain in the ledger; unavailable",
            "token counts remain unknown. See contextbench/repairs/download-timeout-repair.json.",
            "ContextBench preparation transport was repaired during this exploratory run; early",
            "records lack attempt source hashes. Original source and actual input digests remain.",
            "",
            f"Shared ledger: {ledger['requests']} requests; USD "
            f"{ledger['cost_upper_bound']:.6f} conservative usage estimate including preflight.",
            "",
        ]
    manifests = {}
    for benchmark in ("longmemeval", "contextbench"):
        path = directory / benchmark / "manifest.json"
        if path.exists():
            manifests[benchmark] = {
                "path": str(path.relative_to(directory)),
                "configuration": json.loads(path.read_text())["configuration"],
            }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "development_pilot",
                "official_leaderboard_comparable": False,
                "tracks": manifests,
                "shared_ledger": ledger,
                "ledger_accounting": ledger_accounting,
                "raw_attempts": len(all_attempts),
                "latest_task_strategy_results": len(rows),
                "deferred": ["longmemeval-v2", "swe-bench Verified"],
                "blocked": ["StateMemBench: no public official data/scorer located"],
            },
            indent=2,
        )
        + "\n"
    )
    audit_path = directory / "audit.json"
    if audit_path.exists():
        lines += ["Other requested benchmarks (see audit.json):", ""]
        for name, item in json.loads(audit_path.read_text()).items():
            if name not in ("longmemeval", "contextbench"):
                lines.append(f"- {name}: {item['status']}")
    (directory / "report.md").write_text("\n".join(lines))
    return metrics
