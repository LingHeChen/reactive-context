"""Compare the same sequential episodes before/after the two Fact pipeline fixes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import memoryarena as upstream
from .history_regression import write_json
from .official import ROOT


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(directory, baseline):
    directory, baseline = Path(directory), Path(baseline)
    manifest = json.loads((directory / "manifest.json").read_text())
    previous_manifest = json.loads((baseline / "manifest.json").read_text())
    if any(group not in previous_manifest["groups"] for group in manifest["groups"]):
        raise ValueError("paired groups are not a subset of the original pilot")
    scores = json.loads((directory / "scores.json").read_text())
    rows = {row["id"]: row for row in upstream.tasks(len(manifest["groups"]))}
    expected_stages = sum(group["stages"] for group in manifest["groups"])
    previous = {}
    for arm in previous_manifest["arms"]:
        old_stages, old_calls = [], []
        for group in manifest["groups"]:
            folder = baseline / arm / str(group["id"])
            old_stages.extend(read_lines(folder / "stages.jsonl"))
            old_calls.extend(read_lines(folder / "calls.jsonl"))
        previous[arm] = {
            **upstream.score(list(rows.values()), baseline / f"{arm}-submission.jsonl"),
            "requests": len(old_calls),
            "cost_usd": sum(call["cost_upper_bound"] for call in old_calls),
            "planning_failures": sum(bool(stage["failure"]) for stage in old_stages),
        }
    write_json(directory / "paired-baseline-metrics.json", previous)
    metrics, details = {}, []
    for arm in manifest["arms"]:
        stages, calls, receipts, errors = [], [], [], []
        fresh_reads, fresh_withdrawn, initial_facts = set(), set(), {}
        tools = Counter()
        for group in manifest["groups"]:
            folder = directory / arm / str(group["id"])
            state = json.loads((folder / "checkpoint.json").read_text())
            assert state["completed"] == group["stages"]
            stages.extend(read_lines(folder / "stages.jsonl"))
            calls.extend(read_lines(folder / "calls.jsonl"))
            writes = read_lines(folder / "fact-writes.jsonl")
            receipts.extend(receipt for write in writes for receipt in write["receipts"])
            initial_facts[str(group["id"])] = sum(
                bool(r.get("proposal"))
                for write in writes
                if write["stage"] == 0
                for r in write["receipts"]
            )
            errors.extend(read_lines(folder / "writer-errors.jsonl"))
            for capture in (folder / "requests").glob("*-planning-*.json"):
                payload = json.loads(capture.read_text())
                if capture.name.endswith("-response.json"):
                    tools.update(
                        p["tool_name"]
                        for p in payload[0]["parts"]
                        if p.get("part_kind") == "tool-call"
                    )
                    continue
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
                                fresh_withdrawn.add(identity)
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
        phases = defaultdict(
            lambda: {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0.0,
            }
        )
        for call in calls:
            phase = phases[call["phase"]]
            phase["requests"] += 1
            for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
                phase[key] += call[key]
            phase["cost_usd"] += call["cost_upper_bound"]
        metrics[arm] = {
            **scores[arm],
            "stages": len(stages),
            "requests": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "cost_usd": sum(c["cost_upper_bound"] for c in calls),
            "planning_failures": sum(bool(s["failure"]) for s in stages),
            "writer_failures": len(errors),
            "rejected_entries": sum(r.get("status") == "rejected" for r in receipts),
            "commits": sum(bool(r.get("proposal")) for r in receipts),
            "initial_fact_counts": initial_facts,
            "fresh_load_count": len(fresh_reads),
            "fresh_load_withdrawn_count": len(fresh_withdrawn),
            "tools": dict(tools),
            "phases": dict(phases),
        }
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "per-stage-scores.json", details)
    end_path = directory / "validation-budget-end.json"
    end = json.loads(
        (end_path if end_path.exists() else ROOT / manifest["budget_ledger"]).read_text()
    )
    # Include the interrupted first pass in the total fix-validation cost.
    candidates = [
        directory / "validation-budget-start.json",
        directory.with_name(directory.name + "-first-pass") / "budget-start.json",
        directory / "budget-start.json",
    ]
    start = json.loads(next(path for path in candidates if path.exists()).read_text())
    usage = {
        "requests": end["requests"] - start["requests"],
        "cost_usd": end["cost_upper_bound"] - start["cost_upper_bound"],
        "shared_ledger_end": end,
    }
    write_json(directory / "budget-usage.json", usage)
    names = {"fact-small": "Fact＋小历史", "fact-archive": "Fact＋小历史＋归档"}
    text = [
        "# MemoryArena 修复后对照",
        "",
        f"同一 {len(rows)} 个旅行组、{expected_stages} 个连续阶段/方案。"
        "本次只重跑两个 Fact 方案；完整历史与检索压缩引用上一轮结果，没有重新计费运行。"
        "工具、原始问题、模型配置、30 请求阶段上限与官方评分器保持一致。",
        "",
        "| 方案 | 版本 | 变更字段 SPS | 完整行程 PS | 规划失败 | 请求 | 费用上界 USD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, current in metrics.items():
        for version, data in (("修复前", previous[arm]), ("修复后", current)):
            text.append(
                f"| {names[arm]} | {version} | {data['sps']:.2f}% | {data['ps']:.2f}% | "
                f"{data['planning_failures']}/{expected_stages} | {data['requests']} | "
                f"${data['cost_usd']:.4f} |"
            )
    text += [
        "",
        "## 仍存在的限制",
        "",
        "规划失败归零并不等于完整行程正确：PS 和 SR 仍然很低。"
        "与完整历史相比，Fact 方案还有额外的抽取维护和检索调用；"
        "此次修复消除了循环开销，尚未实现费用优势。",
        "",
        "Fact 抽取阶段费用："
        + "；".join(
            f"{names[arm]} ${data['phases']['fact-extraction']['cost_usd']:.4f}"
            for arm, data in metrics.items()
        )
        + "。当前抽取器仍读取既有 Fact 目录和当轮完整交互，"
        "后续可减少输入、只处理增量。引用路径不存在或来源条目超过 schema 上限时，"
        "仍会拒绝该条目；拒绝不再丢弃其他有效条目，计数见协议验证。",
        "",
        "两个 Fact 方案修复后的差异也包括各自规划路径的分化；不能将其分差全部归因于归档回查开关。",
        "",
        "历史对照（未重跑）：完整历史 SPS "
        f"{previous['full-history']['sps']:.2f}%，检索压缩 "
        f"{previous['retrieval-compression']['sps']:.2f}%。",
        "",
        "## 修复内容",
        "",
        "- context_load 的正文在当前用户回合内可读，每次请求复核作用域、权限、版本；"
        "跨用户回合需 context_activate 保留。显式撤下会结束早先的临时读取，"
        "普通 managed_result 的旧版撤下规则保持有效。",
        "- Fact 抽取支持 JSON Pointer 来源锚点：字符串原文或小型结构化对象的规范化快照。"
        "逐条核验、逐条入库；部分失败只补修被拒条目，最多一次额外模型调用。"
        "没有使用参考行程补齐。有效条目不因其他条目失败而回滚。",
        "- 相同 Fact 重复提交先还原冻结数据再比较，避免无意义修订。",
        "- context_search_facts 先按键中主体/日期和 FTS 相关性排序再截断；"
        "context_query 的 ID 分页协议保持不变。",
        "",
        "## 协议验证",
        "",
    ]
    for arm, data in metrics.items():
        before = previous[arm]
        text.append(
            f"- {names[arm]}：SPS 变化 {data['sps'] - before['sps']:+.2f} 个百分点；"
            f"请求变化 {(data['requests'] / before['requests'] - 1) * 100:+.1f}%；"
            f"费用变化 {(data['cost_usd'] / before['cost_usd'] - 1) * 100:+.1f}%。"
        )
        text.append(
            f"- {names[arm]}：新鲜读取 {data['fresh_load_count']} 次，首次读取即被撤下 "
            f"{data['fresh_load_withdrawn_count']} 次；抽取失败 {data['writer_failures']}，"
            f"未修复的拒绝条目 {data['rejected_entries']}。"
            f"初始入库 Fact：{data['initial_fact_counts']}。"
        )
    text += [
        "",
        f"修复验证共新增 {usage['requests']} 次请求、费用上界 ${usage['cost_usd']:.4f}，"
        "包含两次中断的补测：字符串锚点过窄，以及临时读取生命周期过短。"
        "这两次不参与得分比较，原始记录单独保留。",
        "",
        "少量先导样本不足以证明普遍优势。PS/SPS 使用原版参考字段相似度评分，并非完整的"
        "自然语言约束检查。两个 Fact 方案各自顺序执行，错误可沿后续阶段传播；"
        "费用包含实际服务商缓存命中，未隔离冷缓存。"
        "修复前后均不将金标准或评分反馈喂给 Agent。",
        "",
        "完整数据：metrics.json、scores.json、per-stage-scores.json、budget-usage.json；"
        "每组目录保存检查点、来源校验收据和原始调用。",
    ]
    (directory / "conclusions.md").write_text("\n".join(text) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results/memoryarena-fact-fix-expanded")
    parser.add_argument("--baseline", default="results/memoryarena-fact-pilot")
    args = parser.parse_args()
    report(args.directory, args.baseline)
