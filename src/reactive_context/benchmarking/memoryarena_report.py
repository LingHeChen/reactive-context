"""Summarize the completed sequential pilot without feeding scores back to actors."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import memoryarena as upstream
from .history_regression import write_json
from .memoryarena_pilot import ARMS, SCHEMA, flatten
from .official import ROOT


def lines(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = upstream.tasks(len(manifest["groups"]))
    rows_by_id = {row["id"]: row for row in rows}
    totals = {}
    detail = []
    for arm in ARMS:
        stages, calls, writes, errors = [], [], [], []
        tools = Counter()
        active_read_requests = 0
        withdrawn_reads = set()
        for group in manifest["groups"]:
            folder = directory / arm / str(group["id"])
            state = json.loads((folder / "checkpoint.json").read_text())
            if state["completed"] != group["stages"]:
                raise ValueError(f"incomplete group {arm} {group['id']}")
            stages.extend(lines(folder / "stages.jsonl"))
            calls.extend(lines(folder / "calls.jsonl"))
            writes.extend(lines(folder / "fact-writes.jsonl"))
            errors.extend(lines(folder / "writer-errors.jsonl"))
            for capture in (folder / "requests").glob("*-planning-*.json"):
                if capture.name.endswith("-response.json"):
                    response = json.loads(capture.read_text())
                    for part in response[0].get("parts", []):
                        if part.get("part_kind") == "tool-call":
                            tools[part["tool_name"]] += 1
                else:
                    request = json.loads(capture.read_text())
                    active_read_requests += any(SCHEMA in text for text in flatten(request))
                    for message in request.get("messages", []):
                        for part in message["parts"]:
                            if (
                                part.get("kind") == "tool-return"
                                and part.get("tool_name") == "context_load"
                                and "STALE_OR_WITHDRAWN" in str(part.get("content", ""))
                            ):
                                withdrawn_reads.add((group["id"], part["tool_call_id"]))
            for person in state["persons"]:
                detail.append(
                    {
                        "arm": arm,
                        "group": group["id"],
                        "stage": person["person_idx"],
                        **upstream.per_person_scores(
                            rows_by_id[group["id"]], person["person_idx"], person["plan"]
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
        scores = json.loads((directory / "scores.json").read_text())[arm]
        receipts = [receipt for write in writes for receipt in write["receipts"]]
        totals[arm] = {
            **scores,
            "stages": len(stages),
            "planning_failures": sum(bool(s["failure"]) for s in stages),
            "writer_failures": len(errors),
            "commits": sum(bool(r.get("proposal")) for r in receipts),
            "revisions": sum(r.get("revision", 1) > 1 for r in receipts),
            "active_fact_request_count": active_read_requests,
            "unique_withdrawn_fact_reads": len(withdrawn_reads),
            "tool_calls": dict(tools),
            "phases": dict(phases),
            "input_tokens": sum(p["input_tokens"] for p in phases.values()),
            "output_tokens": sum(p["output_tokens"] for p in phases.values()),
            "cache_read_tokens": sum(p["cache_read_tokens"] for p in phases.values()),
            "cost_usd": sum(p["cost_usd"] for p in phases.values()),
            "requests": len(calls),
            "stage_duration_seconds": sum(s["duration_ms"] for s in stages) / 1000,
        }
        own = [item for item in detail if item["arm"] == arm]
        inherited_count = sum(item["inherited_count"] for item in own)
        totals[arm]["inherited_field_accuracy_percent"] = (
            100 * sum(item["inherited_passed"] for item in own) / inherited_count
            if inherited_count
            else 0
        )
    write_json(directory / "metrics.json", totals)
    write_json(directory / "per-stage-scores.json", detail)
    budget_start = json.loads((directory / "budget-start.json").read_text())
    budget_end = json.loads((ROOT / manifest["budget_ledger"]).read_text())
    budget_usage = {
        "requests": budget_end["requests"] - budget_start["requests"],
        "cost_usd": budget_end["cost_upper_bound"] - budget_start["cost_upper_bound"],
        "shared_ledger_end": budget_end,
        "completed_stage_audit_requests": sum(m["requests"] for m in totals.values()),
        "completed_stage_audit_cost_usd": sum(m["cost_usd"] for m in totals.values()),
    }
    write_json(directory / "budget-usage.json", budget_usage)
    text = [
        "# MemoryArena Fact 连续交互先导",
        "",
        "## 直接结论",
        "",
        "这轮没有证明当前 Fact 实现有综合优势。完整历史的变更字段得分最高，"
        "费用最低；当前 Fact 方案的读取协议存在明显阻塞，优先修复接口后再判断架构收益。",
        "",
        f"归档回查相对 Fact＋小历史，将规划失败从 {totals['fact-small']['planning_failures']} "
        f"降为 {totals['fact-archive']['planning_failures']}，但 SPS 从 "
        f"{totals['fact-small']['sps']:.2f}% 变为 {totals['fact-archive']['sps']:.2f}%。"
        "因此本轮只观察到完成规划的韧性改善，没有观察到答案准确率提升。",
        "",
        "## 测试协议",
        "",
        "选用官方团体旅行规划，固定 3 组（218、164、36），共 21 个连续阶段。"
        "每组从公开的首位旅行者固定行程开始，后续只使用此前实际生成的行程，"
        "不将参考答案或评分反馈写入记忆。四组使用同一 DeepSeek 配置、六个原版环境工具、"
        "原版行程解析器和评分函数。此为接入本项目运行时的扩展先导，非官方榜单复现。",
        "",
        "Fact 由真实模型在阶段结束时提取，经来源原文检查、propose、commit、activate "
        "写入真实存储；每阶段重建管理器验证持久化。默认只保留最多一轮历史，"
        "占上下文预算不超过 1%，初始按当前问题 BM25 加载最多 8 个 Fact，"
        "模型可通过工具继续发现、加载和激活。归档方案只在当前信息不足时按需读取原历史。",
        "",
        "| 方案 | 完整行程 PS | 变更字段 SPS | 全组 SR | "
        "累计输入 token | 输出 token | 请求 | 费用上界 USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "full-history": "完整历史",
        "retrieval-compression": "检索＋压缩",
        "fact-small": "Fact＋小历史",
        "fact-archive": "Fact＋小历史＋归档回查",
    }
    for arm, m in totals.items():
        text.append(
            f"| {names[arm]} | {m['ps']:.2f}% | {m['sps']:.2f}% | {m['sr']:.2f}% | "
            f"{m['input_tokens']:,} | {m['output_tokens']:,} | {m['requests']} | "
            f"${m['cost_usd']:.4f} |"
        )
    text += ["", "诊断指标：未变更的初始行程字段保持率（非官方指标，按字段汇总）："]
    for arm, m in totals.items():
        text.append(f"- {names[arm]}：{m['inherited_field_accuracy_percent']:.2f}%。")
    text += [
        "",
        "## 额外成本与执行情况",
        "",
        f"本轮共享账本新增 {budget_usage['requests']} 次请求，费用上界 "
        f"${budget_usage['cost_usd']:.4f}（包括中断和恢复的额外消耗）。",
        "",
        "以上累计 token/费用包含规划、检索压缩或 Fact 抽取；不是只统计最后一次回答。"
        "token 来自服务端 usage，费用按现有 DeepSeek 峰时价格计算保守上界，并计入缓存命中。",
        "",
    ]
    for arm, m in totals.items():
        text.append(
            f"- {names[arm]}：规划失败 {m['planning_failures']}，写入失败 {m['writer_failures']}，"
            f"Fact 提交 {m['commits']}，修订 {m['revisions']}；阶段耗时合计 "
            f"{m['stage_duration_seconds']:.1f} 秒（并发与初始提取不包含在此耗时比较中）。"
        )
        for phase, p in m["phases"].items():
            text.append(
                f"  - {phase}：{p['requests']} 请求，输入 {p['input_tokens']:,}，"
                f"输出 {p['output_tokens']:,}，费用 ${p['cost_usd']:.4f}。"
            )
        text.append(
            f"  - 历史搜索 {m['tool_calls'].get('context_history_search', 0)} 次；"
            f"历史读取 {m['tool_calls'].get('context_history_lookup', 0)} 次。"
        )
        text.append(f"  - 去重后的 Fact 读取结果被撤下 {m['unique_withdrawn_fact_reads']} 次。")
    text += [
        "",
        "## 具体问题与修复方向",
        "",
        "1. **读取与激活的语义不一致。** context_load 说明可读取而不激活，"
        "但 inactive Fact 的 managed_result 在下一请求被 withdraw_managed 替换为 "
        "STALE_OR_WITHDRAWN。日志中模型随后重复加载，最终触发 30 请求上限。"
        "应统一读取与激活协议：允许有边界的一次性读取，或明确提供原子读取激活工具及可执行反馈；"
        "仍需保留权限、撤销和版本检查。此问题本轮保留为被测实现缺陷，没有中途调策略。",
        "",
        "2. **初始抽取失败会放大为后续失败。** 第 36 组的初始抽取把字段名前缀放入引文，"
        "与源码字符串值无法精确匹配，两次尝试后整批拒绝。应使用结构化来源路径和字段值锚点，"
        "为无效引文返回具体反馈，并支持保留其他有效条目。归档方案能回查公开初始计划，"
        "但其后续得分仍不能说明准确率已恢复。",
        "",
        "3. **Fact 搜索排序有改进空间。** 当前 FTS 将词用 OR 连接并按随机 ID 截取结果，"
        "高频 day/city 词会挤入无关条目。可按相关性及明确的人名、日期、键排序，减少加载试错。",
        "",
        "4. **维护成本需要单独优化。** 本轮 Fact 抽取器每阶段还读取完整既有 Fact 目录，"
        "可改为只加载相关旧值并提交增量；不过当前额外费用的主要问题是重复规划调用。"
        "完整历史输入缓存命中约 81.8%，费用比较包含服务商实际缓存状态，未隔离冷缓存。",
        "",
        "## 解释边界",
        "",
        "PS 检查完整参考行程的所有六类字段，SPS 检查相对初始计划发生变动的字段，"
        "SR 要求整组所有人都通过。评分沿用官方字符串相似度阈值 0.7，"
        "不等于完整的自然语言约束验证：多个合理选项可能被参考答案评分误罚。",
        "",
        "只有 3 个独立旅行组，组内阶段高度相关；不能当作 21 个独立样本证明普遍提升。"
        "两个 Fact 方案各自独立生成和提取，其差别包含路径分化，"
        "不能把全部得分差归因于单个回查工具。",
        "",
        "真实运行共提交 179 个 Fact，但没有产生同一 Fact 的修订；因此本先导覆盖新增、"
        "持久化与读取，不覆盖实际模型驱动的冲突更新。修订 API 只在模拟集成测试中验证通过。",
        "",
        "恢复测试中发现并修复了 ContextManager 在首次快照前撤下持久化 Fact 时缺少依赖定义的问题。"
        "已完成阶段从检查点恢复，没有替换成参考行程；中断的调用仍记在共享预算，"
        "完整预算以 history-regression-budget.json 为准。阶段表列已完成阶段的审计调用，"
        "中断时尚未完成阶段的额外调用保留原始抓取文件。",
        "",
        "此前 LongMemEval 的仅归档检索结果没有提取 Fact，属于冷归档基线，"
        "不能据此评价本轮完整 Fact 管线。",
        "",
        "机器可读详情：metrics.json、per-stage-scores.json、scores.json；"
        "每组目录保存 checkpoint、stages、fact-writes、calls 和原始 requests。",
    ]
    (directory / "conclusions.md").write_text("\n".join(text) + "\n")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results/memoryarena-fact-pilot")
    report(parser.parse_args().directory)
