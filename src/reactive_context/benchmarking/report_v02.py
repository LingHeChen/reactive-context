"""Separate reporting for real public pilots, real engineering tasks and protocol fixtures."""

import csv
import json
import statistics
from pathlib import Path

from .official import ROOT, file_hash
from .report import bootstrap, mcnemar_exact


def write_report(directory):
    directory = Path(directory)
    path = directory / "results.jsonl"
    attempts = (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )
    correction_path = directory / "evaluation-corrections.json"
    corrections = json.loads(correction_path.read_text()) if correction_path.exists() else {}
    invalid_code_revisions = set(corrections.get("invalid_code_harness_revisions", []))
    eligible = [
        r
        for r in attempts
        if not (
            r["benchmark"] == "contextbench" and r["harness_revision"] in invalid_code_revisions
        )
    ]
    latest = {(r["benchmark"], r["case_id"], r["strategy"]): r for r in eligible}
    groups = {}
    for row in latest.values():
        groups.setdefault((row["benchmark"], row["strategy"]), []).append(row)
    metrics = []
    for (benchmark, strategy), rows in sorted(groups.items()):
        history = [r for r in attempts if r["benchmark"] == benchmark and r["strategy"] == strategy]
        scored = [r for r in rows if r["research_score"] is not None]
        input_tokens = sum(r.get("input_tokens") or 0 for r in rows)
        hits = sum(r.get("cache_read_tokens") or 0 for r in rows)
        metric = {
            "benchmark": benchmark,
            "strategy": strategy,
            "recorded": len(rows),
            "attempts": len(history),
            "failed_attempts": sum(bool(r["failure_kind"]) for r in history),
            "all_attempts_cost_upper_bound": sum(
                r.get("estimated_cost_usd_upper_bound") or 0 for r in history
            ),
            "scored": sum("file" in r["metrics"] for r in rows)
            if benchmark == "contextbench"
            else len(scored),
            "failures": sum(bool(r["failure_kind"]) for r in rows),
            "research_score": statistics.mean(r["research_score"] for r in scored)
            if scored
            else None,
            "cost_upper_bound": sum(r.get("estimated_cost_usd_upper_bound") or 0 for r in rows),
            "input_tokens": input_tokens,
            "cache_read_fraction": hits / input_tokens if input_tokens else None,
        }
        latency = [r["end_to_end_ms"] for r in rows if r.get("end_to_end_ms") is not None]
        metric["p50_ms"] = statistics.median(latency) if latency else None
        if benchmark == "contextbench":
            candidates = [
                r["metrics"]["candidate_metrics"]["file"]["coverage"]
                for r in rows
                if "candidate_metrics" in r["metrics"]
            ]
            metric["candidate_file_coverage"] = statistics.mean(candidates) if candidates else None
            for kind in ("file", "line", "symbol", "span"):
                for measure in ("precision", "coverage", "derived_f1"):
                    values = [r["metrics"][kind][measure] for r in rows if kind in r["metrics"]]
                    metric[f"{kind}_{measure}"] = statistics.mean(values) if values else None
        if benchmark == "engineering-dynamic-sql":
            for key in ("read_tools", "sql_executions"):
                values = [r["metrics"][key] for r in rows if key in r["metrics"]]
                metric[key + "_mean"] = statistics.mean(values) if values else None
        metrics.append(metric)
    paired = []
    for benchmark in sorted({r["benchmark"] for r in latest.values()}):
        b3 = {
            r["case_id"]: r
            for r in latest.values()
            if r["benchmark"] == benchmark and r["strategy"] == "B3"
        }
        for baseline in sorted(
            {r["strategy"] for r in latest.values() if r["benchmark"] == benchmark} - {"B3"}
        ):
            other = {
                r["case_id"]: r
                for r in latest.values()
                if r["benchmark"] == benchmark and r["strategy"] == baseline
            }
            ids = sorted(b3.keys() & other.keys())
            valid = [
                i
                for i in ids
                if b3[i]["research_score"] is not None and other[i]["research_score"] is not None
            ]
            first, second = (
                [b3[i]["research_score"] for i in valid],
                [other[i]["research_score"] for i in valid],
            )
            delta = [a - b for a, b in zip(first, second, strict=True)]
            paired.append(
                {
                    "benchmark": benchmark,
                    "comparison": "B3-" + baseline,
                    "paired_n": len(valid),
                    "difference": statistics.mean(delta) if delta else None,
                    "difference_ci95": bootstrap(delta),
                    **mcnemar_exact(first, second),
                    "identical_input_pairs": sum(
                        bool(b3[i]["input_digest"])
                        and b3[i]["input_digest"] == other[i]["input_digest"]
                        for i in ids
                    ),
                }
            )
            if benchmark == "contextbench":
                for kind in ("file", "line", "symbol", "span"):
                    for measure in ("precision", "coverage", "derived_f1"):
                        measured = [
                            i
                            for i in ids
                            if kind in b3[i]["metrics"] and kind in other[i]["metrics"]
                        ]
                        differences = [
                            b3[i]["metrics"][kind][measure] - other[i]["metrics"][kind][measure]
                            for i in measured
                        ]
                        clusters = {}
                        for i, difference in zip(measured, differences, strict=True):
                            clusters.setdefault(b3[i].get("cluster") or i, []).append(difference)
                        cluster_means = [statistics.mean(v) for v in clusters.values()]
                        paired.append(
                            {
                                "benchmark": benchmark,
                                "comparison": "B3-" + baseline,
                                "metric": kind + "_" + measure,
                                "paired_n": len(measured),
                                "difference": statistics.mean(differences) if differences else None,
                                "difference_ci95": bootstrap(differences),
                                "cluster_n": len(clusters),
                                "cluster_macro_difference": statistics.mean(cluster_means)
                                if cluster_means
                                else None,
                                "cluster_macro_ci95": bootstrap(cluster_means),
                            }
                        )
    # Exploratory family-wise correction across binary comparisons within each track.
    for benchmark in {p["benchmark"] for p in paired}:
        family = sorted(
            [p for p in paired if p["benchmark"] == benchmark and "p_value" in p],
            key=lambda p: p["p_value"],
        )
        adjusted = 0.0
        for index, comparison in enumerate(family):
            adjusted = max(adjusted, min(1.0, comparison["p_value"] * (len(family) - index)))
            comparison["p_value_holm"] = adjusted
    cache = {}
    for arm in ("fixed", "permuted"):
        rows = [
            r
            for r in latest.values()
            if r["benchmark"] == "engineering-prefix-cache"
            and r["strategy"] == arm
            and not r["metrics"].get("warmup")
            and not r["failure_kind"]
        ]
        total = sum(r.get("input_tokens") or 0 for r in rows)
        cache[arm] = {
            "post_warmup_n": len(rows),
            "input_tokens": total,
            "cache_read_fraction": sum(r.get("cache_read_tokens") or 0 for r in rows) / total
            if total
            else None,
        }
    cache_clusters = {}
    for row in latest.values():
        if (
            row["benchmark"] == "engineering-prefix-cache"
            and not row["metrics"].get("warmup")
            and not row["failure_kind"]
        ):
            cluster = cache_clusters.setdefault(row["cluster"], {})
            cluster.setdefault(row["strategy"], []).append(row)
    cache_deltas = []
    for cluster in cache_clusters.values():
        if all(arm in cluster for arm in ("fixed", "permuted")):
            fractions = {
                arm: sum(r["cache_read_tokens"] for r in cluster[arm])
                / sum(r["input_tokens"] for r in cluster[arm])
                for arm in ("fixed", "permuted")
            }
            cache_deltas.append(fractions["fixed"] - fractions["permuted"])
    cache["paired_namespace_difference"] = {
        "n": len(cache_deltas),
        "difference": statistics.mean(cache_deltas) if cache_deltas else None,
        "ci95": bootstrap(cache_deltas),
        "warning": "Only three namespace clusters; exploratory, not broad workload evidence.",
    }
    ledger_path = ROOT / "results/v02/budget-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    planned = {
        "longmemeval": 180,
        "contextbench": 150,
        "engineering-dynamic-sql": 96,
        "engineering-prefix-cache": 36,
    }
    completeness = {
        benchmark: {
            "planned": count,
            "recorded": sum(r["benchmark"] == benchmark for r in latest.values()),
        }
        for benchmark, count in planned.items()
    }
    input_controls = {}
    for benchmark in ("longmemeval", "contextbench"):
        b0 = {
            r["case_id"]: r
            for r in latest.values()
            if r["benchmark"] == benchmark and r["strategy"] == "B0"
        }
        b1 = {
            r["case_id"]: r
            for r in latest.values()
            if r["benchmark"] == benchmark and r["strategy"] == "B1"
        }
        ids = b0.keys() & b1.keys()
        input_controls[benchmark + "_B0_B1"] = {
            "paired_n": len(ids),
            "identical_answer_input_pairs": sum(
                bool(b0[i]["input_digest"]) and b0[i]["input_digest"] == b1[i]["input_digest"]
                for i in ids
            ),
        }
    subgroups = []
    for (benchmark, strategy), rows in sorted(groups.items()):
        names = {}
        for row in rows:
            subgroup = row.get("subgroup") or "unspecified"
            if benchmark == "engineering-dynamic-sql":
                subgroup = "long" if int(row["case_id"].rsplit("-", 1)[1]) >= 108 else "short"
            names.setdefault(subgroup, []).append(row)
        for subgroup, members in sorted(names.items()):
            scored = [r["research_score"] for r in members if r["research_score"] is not None]
            subgroups.append(
                {
                    "benchmark": benchmark,
                    "strategy": strategy,
                    "subgroup": subgroup,
                    "n": len(members),
                    "scored_n": len(scored),
                    "research_score": statistics.mean(scored) if scored else None,
                    "failures": sum(bool(r["failure_kind"]) for r in members),
                    "error_types": {
                        key: sum(r.get("error_type") == key for r in members)
                        for key in {r.get("error_type") for r in members}
                        if key
                    },
                }
            )
    statistics_path = directory / "statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "groups": metrics,
                "paired": paired,
                "subgroups": subgroups,
                "cache_post_warmup": cache,
                "ledger": ledger,
                "input_controls": input_controls,
                "completeness": completeness,
                "analysis_source_sha256": file_hash(Path(__file__)),
                "corrections": corrections,
                "excluded_code_attempts_for_quality": len(attempts) - len(eligible),
            },
            indent=2,
        )
        + "\n"
    )
    with (directory / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for m in metrics for k in m}))
        writer.writeheader()
        writer.writerows(metrics)
    lines = [
        "# v0.2 真实模型先导",
        "",
        "这是开发先导，不是官方全量排行榜成绩。",
        "LongMemEval 使用原评分 prompt/解析规则和 DeepSeek 评审，标为研究变体。",
        "ContextBench 使用原粒度指标；补丁未运行测试，不能报告任务通过率。",
        "SQL 动态任务为独立生成的工程集成集；协议 fixture 不混入此表。",
        "ContextBench 的分数列不设单一任务通过分；已评分指粒度指标有效，具体见下方 P/R/F1。",
        "",
        "| 轨道 | 策略 | 记录数 | 已评分 | 分数 | 失败 | 费用上界估算 USD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        value = f"{m['research_score']:.4f}" if m["research_score"] is not None else "N/A"
        lines.append(
            f"| {m['benchmark']} | {m['strategy']} | {m['recorded']} | {m['scored']} | "
            f"{value} | {m['failures']} | {m['cost_upper_bound']:.6f} |"
        )
    code_metrics = [m for m in metrics if m["benchmark"] == "contextbench"]
    if code_metrics:
        lines += [
            "",
            "| ContextBench 策略 | File P | File R | File F1 | Line P | Line R | Line F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for m in code_metrics:
            values = [
                m.get(kind + "_" + measure)
                for kind in ("file", "line")
                for measure in ("precision", "coverage", "derived_f1")
            ]
            lines.append(
                "| "
                + m["strategy"]
                + " | "
                + " | ".join(f"{v:.4f}" if v is not None else "N/A" for v in values)
                + " |"
            )
    lines += [
        "",
        "计划/已记录实验数量：" + json.dumps(completeness, ensure_ascii=False),
        "配对差异、95% bootstrap 区间、McNemar 检验与相同输入对数见 statistics.json。",
        "先导比较是探索性的；相同静态输入不支持响应式机制的因果收益。",
        "公开静态任务未触发版本变化的 no-refresh/no-dependency 消融标记不适用；",
        "这些开关在独立 SQL 工程集和三部件协议集中实际执行。",
        "",
        "固定/排列 Skill 前缀实验使用实际 SkillRegistry，每个 arm 独立等长 namespace，",
        "3 个重复 × 6 步，两组交错随机顺序；第 0 步作为预热观测，未假定必然冷缓存。",
        "post-warmup 实际缓存读取比例：",
        "",
        "```json",
        json.dumps(cache, indent=2),
        "```",
        "",
        "ContextBench 的 B3 包含额外的模型片段选择调用，费用已计入；"
        "任何改善不能直接归因于版本管理。",
        "摘要验证仅检查被引用证据的支持性，不声称摘要无损或验证器绝对正确。",
        "SQL 工程集摘要验证只验证证据引用非空与结构/版本，不提供独立语义忠实度裁判。",
        "所有失败保留原记录；缺失、超时与预算停止单独列出，不默认为模型答对。",
        "账本还包含共享初始计划，不能仅将策略行费用相加当作总费用。",
        "汇总表使用每个任务的最新记录；metrics.csv 另外保留全部尝试次数、失败数和累计费用。",
        "短/长 SQL 场景分别为 seed 100–107 / 108–111。"
        "无刷新导致的工具重试耗尽计为执行失败；"
        "程序线程/schema 故障修复前记录不作为设计质量结论。",
        "实验修复、版本与完整方法见 docs/v02-evaluation-notes.md。",
        "",
        "共享预算账本：",
        "",
        "```json",
        json.dumps(ledger, indent=2),
        "```",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    return metrics
