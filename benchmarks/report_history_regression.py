"""Combine independently budgeted history pilots without hiding failed cases."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

from reactive_context.benchmarking.history_regression import paired_summary
from reactive_context.benchmarking.official import ROOT

S_RUN = ROOT / "results/minimal-history-validation"
V2_RUN = ROOT / "results/history-regression-expanded"
OUTPUT = ROOT / "results/history-regression"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def collect_requests(directory):
    rows = {}
    for row in read_rows(directory / "requests.jsonl"):
        names = []
        for call in row["calls"]:
            response = call.get("response_file")
            if call["phase"] != "answer" or not response or not resolve(response).exists():
                continue
            for message in json.loads(resolve(response).read_text()):
                names.extend(
                    p["tool_name"] for p in message["parts"] if p.get("part_kind") == "tool-call"
                )
        rows[(row["benchmark"], row["case_id"], row["arm"])] = {
            "tool_calls": names,
            "model_ms_excluding_judge": sum(
                c["duration_ms"] for c in row["calls"] if c["phase"] != "judge"
            ),
            "images_sent": sum(c.get("input_image_count", 0) for c in row["calls"]),
            "actual_models": sorted({c["response_model"] for c in row["calls"]}),
        }
    return rows


def main():
    rows = read_rows(S_RUN / "results.jsonl") + read_rows(V2_RUN / "results.jsonl")
    requests = {**collect_requests(S_RUN), **collect_requests(V2_RUN)}
    comparisons = {
        "S_vs_previous": paired_summary(rows, "longmemeval", "previous-4-groups"),
        "S_vs_compression": paired_summary(rows, "longmemeval", "retrieval-compression"),
        "V2_vs_compression": paired_summary(rows, "longmemeval-v2", "retrieval-compression"),
        "V2_vs_no_memory": paired_summary(rows, "longmemeval-v2", "no-memory"),
    }
    aggregates = []
    for benchmark in ("longmemeval", "longmemeval-v2"):
        for arm in sorted({r["arm"] for r in rows if r["benchmark"] == benchmark}):
            chosen = [r for r in rows if r["benchmark"] == benchmark and r["arm"] == arm]
            scored = [r for r in chosen if r.get("score") is not None]
            reqs = [requests[(benchmark, r["case_id"], arm)] for r in scored]
            no_retrieval_wrong = [
                r["case_id"]
                for r, request in zip(scored, reqs, strict=True)
                if r["score"] == 0 and not request["tool_calls"]
            ]
            aggregates.append(
                {
                    "benchmark": benchmark,
                    "arm": arm,
                    "scored": len(scored),
                    "correct": sum(r["score"] for r in scored),
                    "accuracy": sum(r["score"] for r in scored) / len(scored) if scored else None,
                    "initial_tokens_median": median(
                        r.get("initial_input_tokens") or 0 for r in scored
                    )
                    if scored
                    else None,
                    "total_tokens": sum(
                        r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in chosen
                    ),
                    "cost_upper_bound": sum(
                        r.get("estimated_cost_usd_upper_bound", 0) for r in chosen
                    ),
                    "model_ms_median_excluding_judge": median(
                        r["model_ms_excluding_judge"] for r in reqs
                    )
                    if reqs
                    else None,
                    "retrieval_used_cases": sum(bool(r["tool_calls"]) for r in reqs),
                    "tool_counts": dict(Counter(name for r in reqs for name in r["tool_calls"])),
                    "incorrect_without_retrieval_ids": no_retrieval_wrong,
                    "images_sent_across_calls": sum(r["images_sent"] for r in reqs),
                    "unscored_or_failed": sum(r["status"] != "completed" for r in chosen),
                }
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "comparisons": comparisons,
        "aggregates": aggregates,
        "paid_allocations": {},
        "failures": [r for r in rows if r["status"] != "completed"],
    }
    for name, path in (
        ("old", ROOT / "results/v02/budget-ledger.json"),
        ("new", ROOT / "results/history-regression-budget.json"),
    ):
        if path.exists():
            result["paid_allocations"][name] = json.loads(path.read_text())
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    narrative = []
    previous = comparisons["S_vs_previous"]
    compressed = comparisons["S_vs_compression"]
    if previous["paired_cases"]:
        n = previous["paired_cases"]
        first_drop = (
            1
            - previous["minimal_initial_tokens_median"] / previous["baseline_initial_tokens_median"]
        )
        total_drop = 1 - previous["minimal_total_tokens"] / previous["baseline_total_tokens"]
        narrative += [
            f"LongMemEval S：新方案 {previous['minimal_accuracy'] * n:g}/{n}，"
            f"旧窗口 {previous['baseline_accuracy'] * n:g}/{n}；"
            f"首请求 token 降幅 {first_drop:.1%}，总 token 降幅 {total_drop:.1%}。"
            f"相对传统检索压缩，准确率差 {compressed['accuracy_delta'] * 100:+.1f} 个百分点。",
            "",
        ]
    for key, label in (("V2_vs_compression", "传统检索压缩"), ("V2_vs_no_memory", "无历史")):
        comparison = comparisons[key]
        if comparison["paired_cases"]:
            lower, upper = comparison["paired_bootstrap_95_ci"]
            narrative += [
                f"V2 相对{label}：准确率差 {comparison['accuracy_delta'] * 100:+.1f} 个百分点，"
                f"配对 bootstrap 区间 [{lower * 100:+.1f}, {upper * 100:+.1f}] 个百分点；"
                f"保守防退化判定：{comparison['noninferiority_status']}。",
                "",
            ]
    for row in aggregates:
        if row["arm"] == "minimal-history":
            wrong = row["scored"] - row["correct"]
            narrative += [
                f"{row['benchmark']}：{wrong:g} 道错题中，"
                f"{len(row['incorrect_without_retrieval_ids'])} 道未调用检索；"
                f"共 {row['retrieval_used_cases']}/{row['scored']} 题使用检索，"
                f"轨迹图片工具调用 {row['tool_counts'].get('context_history_image', 0)} 次。",
                "",
            ]
    lines = [
        "# 最小历史的收益与能力防退化",
        "",
        "同题、同 DeepSeek 模型、固定抽样；官方数据与评分规则的研究变体，非官方全量排行榜成绩。",
        "",
        "**结论：缩小历史有明确的输入与成本收益；相对传统检索压缩，记忆质量防退化尚未达标。**",
        "",
        *narrative,
        "提供工具和档案不等于 Agent 会在需要时使用它们。",
        "",
        "修复应优先考虑明确的档案可用性信息，以及在宣称‘没有历史/无法回答’前检查是否需要取证；"
        "不应继续单纯压短首请求。本轮未按参考答案调整生产策略，也未改动评分规则以消除失败。",
        "",
        "| 数据集 | 方案 | 正确/已评分 | 首请求 token 中位数 | 总 token | 费用上界（美元） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    names = {
        "minimal-history": "新小历史＋按需检索",
        "previous-4-groups": "旧 4 组历史＋相同检索工具",
        "retrieval-compression": "传统检索＋LLM 压缩",
        "no-memory": "无历史（保留问题截图）",
    }
    for row in aggregates:
        lines.append(
            f"| {row['benchmark']} | {names[row['arm']]} | "
            f"{row['correct']:g}/{row['scored']} | {row['initial_tokens_median']} | "
            f"{row['total_tokens']} | {row['cost_upper_bound']:.6f} |"
        )
    lines += [
        "",
        "首请求是最终回答 Agent 的第一次请求。检索压缩组的前置摘要请求计入总 token 与费用，"
        "不能只看它已经压缩过的最终回答输入。费用包含压缩、工具推理、回答和评审，按高峰单价估算。",
        "",
        "## 逐类与配对结果",
        "",
    ]
    for name, comparison in comparisons.items():
        lines += [
            f"### {name}",
            "",
            "```json",
            json.dumps(comparison, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "## 诊断",
        "",
        "```json",
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        "```",
        "",
        "错误且未使用检索是可观察的失败模式，不等于证明每道题的唯一根因。"
        "原始预测、每一步工具调用、用量和题目 ID 都保留，可逐题核对。",
        "",
        "## 边界",
        "",
        "V2 为五类能力与两个领域均衡的 100 题子集，完整 small haystack，共享两套轨迹；"
        "准确率不是官方加权聚合或 LAFS。读者和评审均改为 DeepSeek，"
        "没有固定 provider 权重或采样 seed。"
        "问题层面的置信区间受到共享 haystack、分层采样及模型波动影响。",
        "",
        "防退化线为 5 个百分点；没有观察到失败也不能自动判定通过。"
        "精确二项界额外约束逐题退化概率，是平均正确率门槛的保守充分条件，未通过不等于证明总体下降。"
        "Fact/工作目标未预先生成，本轮测试冷归档召回，不能代替预热工作集评测。",
        "",
        "未完成或执行失败记录：",
        "",
        "```json",
        json.dumps(result["failures"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (OUTPUT / "conclusions.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"aggregates": aggregates, "failures": len(result["failures"])},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
