"""Offline initial-selection replay from captured pre-write catalogs (no model calls)."""

import argparse
import json
from pathlib import Path

from reactive_context.benchmarking.memoryarena_pilot import TravelFact, encoded, ranked, token_count
from reactive_context.benchmarking.travel_dependencies import dependencies
from reactive_context.core.snapshot import freeze


def replay(folder, stage_index):
    calls = [json.loads(x) for x in (folder / "calls.jsonl").read_text().splitlines()]
    stages = [json.loads(x) for x in (folder / "stages.jsonl").read_text().splitlines()]
    seed = json.loads((folder / "checkpoint.json").read_text())["records"][0]
    cursor = 0
    while calls[cursor]["phase"] == "fact-extraction":
        cursor += 1
    for stage in stages:
        window = calls[cursor : cursor + stage["usage"]["request_count"]]
        cursor += stage["usage"]["request_count"]
        if stage["stage"] == stage_index:
            break
    else:
        raise ValueError("stage not found")
    first_writer = next(c for c in window if c["phase"] == "fact-extraction")
    request = json.loads(Path(first_writer["request_file"]).read_text())
    text = next(
        p["content"]
        for m in request["messages"]
        for p in m["parts"]
        if isinstance(p.get("content"), str) and p["content"].startswith("EXISTING FACTS:")
    )
    catalog = json.loads(text.split("EXISTING FACTS:\n")[1].split("\nSOURCE:\n")[0])
    catalog.sort(key=lambda r: r["key"])
    by_id = {r["id"]: r for r in catalog}
    people = {r["value"]["subject"] for r in catalog}
    required = {f"{p}/day-{d}" for p, d, _ in dependencies(stage["query"], people)}
    retained = {f"{seed['name']}/day-{d.get('day', d.get('days'))}" for d in seed["daily_plans"]}
    ordered = ranked(
        stage["query"],
        [(r["id"], encoded(freeze(TravelFact.model_validate(r["value"])))) for r in catalog],
    )
    ordered.sort(key=lambda d: by_id[d.id]["key"] not in required)
    kept = [d for d in ordered if by_id[d.id]["key"] in retained]
    rest = [d for d in ordered if by_id[d.id]["key"] not in retained]
    limit = max(8 - len(kept), sum(by_id[d.id]["key"] in required for d in rest))
    candidate = kept + rest[:limit]
    summary = {
        "stage": stage_index,
        "baseline_person": seed["name"],
        "required_keys": sorted(required),
    }
    for label, selected in (("control", ordered[:8]), ("candidate", candidate)):
        keys = {by_id[d.id]["key"] for d in selected}
        summary[label] = {
            "keys": sorted(keys),
            "baseline_count": len(keys & retained),
            "dependency_count": len(keys & required),
            "fact_tokens_proxy": token_count(encoded([by_id[d.id]["value"] for d in selected])),
        }
    summary["quality_change"] = "NOT_MEASURED"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.episode, args.stage)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
