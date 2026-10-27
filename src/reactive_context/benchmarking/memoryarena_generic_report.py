"""Reproducible, offline comparison against the preceding successful Fact pilot."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import memoryarena as upstream
from .history_regression import write_json
from .memoryarena_fix_report import read_lines


def collect(directory, arm, groups, rows):
    calls, stages, receipts, details = [], [], [], []
    decisions, tools = Counter(), Counter()
    fresh_reads, withdrawn = set(), set()
    projections, fact_counts, writer_failures = [], {}, 0
    for group in groups:
        folder = directory / arm / str(group["id"])
        state = json.loads((folder / "checkpoint.json").read_text())
        assert state["completed"] == group["stages"]
        calls.extend(read_lines(folder / "calls.jsonl"))
        episode_stages = read_lines(folder / "stages.jsonl")
        stages.extend(episode_stages)
        fact_counts[str(group["id"])] = episode_stages[-1]["fact_count"]
        receipts.extend(
            receipt
            for write in read_lines(folder / "fact-writes.jsonl")
            for receipt in write["receipts"]
        )
        writer_failures += len(read_lines(folder / "writer-errors.jsonl"))
        audit = folder / "integration.json"
        if audit.exists():
            data = json.loads(audit.read_text())
            decisions.update(data["save_decisions"])
            projections.extend(data["projected_results"].values())
        for capture in (folder / "requests").glob("*-planning-*-response.json"):
            payload = json.loads(capture.read_text())
            tools.update(
                part["tool_name"]
                for message in payload
                for part in message["parts"]
                if part.get("part_kind") == "tool-call"
            )
        for capture in (folder / "requests").glob("*-planning-*.json"):
            if capture.name.endswith("-response.json"):
                continue
            payload = json.loads(capture.read_text())
            for message in reversed(payload["messages"]):
                if message["kind"] == "response":
                    break
                for part in message["parts"]:
                    if (
                        part.get("kind") == "tool-return"
                        and part.get("tool_name") == "context_load"
                    ):
                        identity = (group["id"], part["tool_call_id"])
                        fresh_reads.add(identity)
                        if "STALE_OR_WITHDRAWN" in str(part.get("content")):
                            withdrawn.add(identity)
        for person in state["persons"]:
            details.append(
                {
                    "arm": arm,
                    "group": group["id"],
                    "stage": person["person_idx"],
                    **upstream.per_person_scores(
                        rows[group["id"]], person["person_idx"], person["plan"]
                    ),
                }
            )
    phases = defaultdict(Counter)
    for call in calls:
        phase = phases[call["phase"]]
        phase["requests"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cost_upper_bound"):
            phase[key] += call[key]
    fields = {
        key: sum(detail[key] for detail in details)
        for key in ("constraint_count", "constraints_passed", "inherited_count", "inherited_passed")
    }
    totals = Counter()
    for phase in phases.values():
        totals.update(phase)
    return {
        **upstream.score(list(rows.values()), directory / f"{arm}-submission.jsonl"),
        **dict(totals),
        "stages": len(stages),
        "planning_failures": sum(bool(stage["failure"]) for stage in stages),
        "writer_failures": writer_failures,
        "commits": sum(bool(r.get("committed") or r.get("proposal")) for r in receipts),
        "rejected_entries": sum(r.get("status") == "rejected" for r in receipts),
        "final_fact_counts": fact_counts,
        "save_decisions": dict(decisions),
        "fresh_load_count": len(fresh_reads),
        "fresh_load_withdrawn_count": len(withdrawn),
        "projection": {
            "unique_results": len(projections),
            "boundary_found": sum(p["has_declared_boundary"] for p in projections),
            "original_token_proxy": sum(p["original_token_proxy"] for p in projections),
            "projected_token_proxy": sum(p["projected_token_proxy"] for p in projections),
        },
        "fields": fields,
        "tools": dict(tools),
        "phases": {key: dict(value) for key, value in phases.items()},
    }, details


def report(directory, baseline):
    directory, baseline = Path(directory), Path(baseline)
    manifest = json.loads((directory / "manifest.json").read_text())
    old_manifest = json.loads((baseline / "manifest.json").read_text())
    assert manifest["groups"] == old_manifest["groups"]
    for key in ("dataset_sha256", "parser_sha256", "scorer_sha256", "tool_source_sha256"):
        assert manifest[key] == old_manifest[key]
    rows = {row["id"]: row for row in upstream.tasks(len(manifest["groups"]))}
    comparison, details = {}, []
    for arm in manifest["arms"]:
        comparison[arm] = {}
        for version, root in (("previous", baseline), ("current", directory)):
            metrics, per_stage = collect(root, arm, manifest["groups"], rows)
            comparison[arm][version] = metrics
            details.extend({"version": version, **detail} for detail in per_stage)
    auth = json.loads((directory / "authorization.json").read_text())
    end = json.loads((directory / "validation-budget-end.json").read_text())
    budget = {
        "requests": end["requests"] - auth["ledger_start"]["requests"],
        "cost_upper_bound": end["cost_upper_bound"] - auth["ledger_start"]["cost_upper_bound"],
        "request_limit": auth["additional_requests"],
        "cost_limit": auth["additional_cost_usd"],
        "outstanding_reservation": end["outstanding_reservation"],
    }
    assert budget["requests"] <= budget["request_limit"]
    assert budget["cost_upper_bound"] <= budget["cost_limit"]
    assert abs(budget["outstanding_reservation"]) < 1e-9
    write_json(directory / "comparison.json", comparison)
    write_json(directory / "per-stage-comparison.json", details)
    write_json(directory / "budget-usage.json", budget)
    lines = [
        "# 通用 Fact 检查与历史投影：同题重跑",
        "",
        "同一 3 个旅行组（218、164、36），每方案 21 个连续阶段。"
        "旧版引用上轮成功运行，新版完整重跑；不是同一批次双版本随机实验。"
        "问题、业务工具、actor/writer 指令、逐阶段抽取时机、请求上限与官方评分器保持一致。",
        "",
        "| 方案 | 版本 | SPS | PS | 请求 | 输入 tokens | 费用上界 | Fact 写入 | 最终 Part |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, pair in comparison.items():
        for version, data in pair.items():
            lines.append(
                f"| {arm} | {version} | {data['sps']:.2f}% | {data['ps']:.2f}% | "
                f"{data['requests']} | {data['input_tokens']:,} | "
                f"${data['cost_upper_bound']:.4f} | {data['commits']} | "
                f"{sum(data['final_fact_counts'].values())} |"
            )
    lines += ["", "## 直接结论", ""]
    if any(
        pair["current"]["fields"]["inherited_passed"]
        < pair["previous"]["fields"]["inherited_passed"]
        for pair in comparison.values()
    ):
        lines += [
            "**不能判为整体通过防退化。** 变更字段 SPS 提升的同时，"
            "至少一个方案在其他字段上出现退化；完整计划通过率也没有提高。",
            "",
        ]
    for arm, pair in comparison.items():
        old, new = pair["previous"], pair["current"]
        lines.append(
            f"- {arm}：SPS {new['sps'] - old['sps']:+.2f} 个百分点；"
            f"输入 tokens {(new['input_tokens'] / old['input_tokens'] - 1) * 100:+.1f}%；"
            f"费用 {(new['cost_upper_bound'] / old['cost_upper_bound'] - 1) * 100:+.1f}%。"
            f"规划失败 {new['planning_failures']}，抽取失败 {new['writer_failures']}，"
            f"最终拒绝条目 {new['rejected_entries']}。"
        )
        fields = new["fields"]
        old_fields = old["fields"]
        projection = new["projection"]
        lines.append(
            f"- {arm} 字段命中：用户变更字段 "
            f"{old_fields['constraints_passed']}→{fields['constraints_passed']}"
            f"/{fields['constraint_count']}；其余字段 "
            f"{old_fields['inherited_passed']}→{fields['inherited_passed']}"
            f"/{fields['inherited_count']}。保存决策 {new['save_decisions']}。"
        )
        lines.append(
            f"- {arm} 最终回复投影：{projection['boundary_found']}/"
            f"{projection['unique_results']} 条找到原有格式边界，"
            f"文本 token 代理 {projection['original_token_proxy']}→"
            f"{projection['projected_token_proxy']}。这是独立文本长度，"
            "不是实际账单节省量；Fact 引文仍可能保留推理片段。"
        )
    lines += [
        "",
        "## 成本来源与分组结果",
        "",
        "| 方案 | 阶段 | 旧请求→新请求 | 旧费用→新费用 | 旧输入→新输入 tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for arm, pair in comparison.items():
        for phase in ("planning", "fact-extraction"):
            old = pair["previous"]["phases"][phase]
            new = pair["current"]["phases"][phase]
            lines.append(
                f"| {arm} | {phase} | {old['requests']}→{new['requests']} | "
                f"${old['cost_upper_bound']:.4f}→${new['cost_upper_bound']:.4f} | "
                f"{old['input_tokens']:,}→{new['input_tokens']:,} |"
            )
    lines += ["", "| 方案 | 组 | 旧 SPS→新 SPS |", "|---|---|---:|"]
    for arm in comparison:
        for group in manifest["groups"]:
            values = []
            for version in ("previous", "current"):
                matched = [
                    item
                    for item in details
                    if item["arm"] == arm
                    and item["group"] == group["id"]
                    and item["version"] == version
                    and item["constraint_count"]
                ]
                values.append(
                    sum(item["constraints_passed"] / item["constraint_count"] for item in matched)
                    / len(matched)
                    * 100
                )
            lines.append(f"| {arm} | {group['id']} | {values[0]:.2f}%→{values[1]:.2f}% |")
    reference = json.loads((baseline / "paired-baseline-metrics.json").read_text())
    lines += [
        "",
        f"更早的历史参考（本轮未重跑）：完整历史 SPS {reference['full-history']['sps']:.2f}%，"
        f"费用 ${reference['full-history']['cost_usd']:.4f}；检索压缩 SPS "
        f"{reference['retrieval-compression']['sps']:.2f}%，费用 "
        f"${reference['retrieval-compression']['cost_usd']:.4f}。",
        "",
        "## 原因与边界",
        "",
        "本轮新增的是受控保存、来源检查、干净历史索引，以及按原有输出格式提取最终结果。"
        "检查只确认 schema、作用域、原文锚点与键的一致性，不能判断旅行条件是否满足，"
        "所以不能阻止‘有来源但结论错误’的 Fact。没有添加参考答案推导的继承规则或数值筛选。",
        "",
        "本次仍在初始输入及每阶段结束后抽取，抽取器仍读取既有 Fact 目录和完整当轮交互。"
        "保存接口能够去重，但不会自动减少抽取请求，也不会将新人物/日期键合并为旧 Part。"
        "应将写入检查、抽取频率、输入投影分别衡量；当前结果不能证明已经解决维护成本。",
        "",
        "SPS 是先按人物、再按旅行组宏平均的变更字段参考相似度；"
        "字段计数是微平均，二者不可直接换算。"
        "PS 为整个人物计划通过率，SR 为整组通过率，详见 comparison.json。"
        "只有 3 个独立旅行组，阶段相互依赖；温度为 0 仍存在服务端不确定性，模型别名"
        "及实际缓存未隔离，分差不能全归因于某个改动。",
        "",
        f"新增用量 {budget['requests']} 次请求、费用上界 ${budget['cost_upper_bound']:.6f}，"
        "低于本轮授权 500 次/$2；继续使用原共享账本，没有重置累计消耗。",
        "",
        "复现：.venv/bin/python -m reactive_context.benchmarking.memoryarena_generic_report；"
        "原始调用、检查点和 integration.json 保存在各方案/组目录。",
    ]
    (directory / "conclusions.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results/memoryarena-generic-rerun")
    parser.add_argument("--baseline", default="results/memoryarena-fact-fix-expanded")
    args = parser.parse_args()
    report(args.directory, args.baseline)
