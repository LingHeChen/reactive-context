"""Offline single-variable smoke replay; never overwrites an existing result directory."""

import argparse
import asyncio
import json
from hashlib import sha256
from pathlib import Path

from reactive_context.benchmarking.intelligence_ablation import (
    EXPERIMENTS,
    preregistration,
    replay_evidence,
)
from reactive_context.core.snapshot import digest, freeze, thaw
from reactive_context.evidence import (
    CompilationBudget,
    EvidenceRequirement,
    IndexedNavigator,
    ResearchBudget,
    SessionBudget,
    SourceBinding,
    SourceSnapshot,
)


def source_hashes():
    return {
        str(p): sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("src/reactive_context").rglob("*.py"))
    }


async def run(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for i in range(4):
        text = f"Entity day{i + 1} dinner: observed value {i}"
        sources.append(
            SourceSnapshot(
                str(i),
                "v1",
                "default",
                text,
                (SourceBinding(f"day{i + 1}.dinner", 0, len(text), str(i)),),
            )
        )
    slots = tuple(
        EvidenceRequirement(s.bindings[0].subject, s.bindings[0].subject) for s in sources
    )
    candidates = tuple(h for s in sources[:2] for h in s.hits())
    fixture_hash = digest(freeze((sources, slots, candidates)))
    manifest = preregistration(
        source_hashes=await asyncio.to_thread(source_hashes),
        fixture_hashes=[fixture_hash],
        model_parameters={"model": None, "fixture": "synthetic"},
        thresholds={
            "quality": "NOT_MEASURED",
            "release": "disabled",
            "frozen_E0_E1_input_equal": True,
        },
    )
    (output / "manifest.json").write_text(json.dumps(thaw(manifest), indent=2))
    rows = []
    for experiment in EXPERIMENTS:
        row = await replay_evidence(
            experiment,
            requirements=slots,
            candidates=candidates,
            navigator=IndexedNavigator(sources, authorize=lambda *_: True),
            research_budget=ResearchBudget(),
            session_budget=SessionBudget(),
            compilation_budget=CompilationBudget(2048, 500, 100, 8192),
            index_revision=fixture_hash,
            estimator=len,
        )
        rows.append(thaw(row))
    assert rows[0]["input_hash"] == rows[1]["input_hash"]
    assert rows[0]["candidate_hash"] == rows[5]["candidate_hash"]
    (output / "replay.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"experiments": len(rows), "model_calls": 0, "release": "NOT_MEASURED"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
