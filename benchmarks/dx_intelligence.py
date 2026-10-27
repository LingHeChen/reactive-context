"""Offline X0/X1 and B1-B4 mechanism artifacts. Never calls a paid model."""

import argparse
import asyncio
import json
import sqlite3
from collections import Counter
from hashlib import sha256
from pathlib import Path

from reactive_context import CompilerPolicy, Context, computed, part, state
from reactive_context.benchmarking import memoryarena as upstream
from reactive_context.benchmarking.frozen_replay import (
    ReplayFixture,
    canonical_replay,
    write_replay,
)
from reactive_context.benchmarking.plan_projection import PROJECTION_REVISION, project_plan
from reactive_context.core.snapshot import digest, thaw
from reactive_context.evidence import (
    EvidenceRequirement,
    EvidenceResearcher,
    IndexedNavigator,
    SourceBinding,
    SourceSnapshot,
)
from reactive_context.sources.config import ConfigSource


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class ReplayContext(Context):
    objective_input = state(default_factory=dict, expose=False)

    @computed(expose=False)
    def target(self):
        return self.objective_input

    @part(revision=lambda reader: digest(reader.read("target")))
    async def work_objective(self, reader):
        return reader.read("target")


async def run(args):
    source, out = Path(args.source), Path(args.output)
    await asyncio.to_thread(out.mkdir, parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    rows = upstream.tasks(3)
    audit, counts = {}, Counter()
    parser = upstream.parser()
    old_hashes = {
        str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest()
        for p in await asyncio.to_thread(lambda: list(source.rglob("*")))
        if p.is_file() and p.suffix in {".json", ".jsonl", ".py"}
    }
    for variant, arm in manifest["variants"].items():
        submissions, traces = [], []
        for row in rows:
            directory = source / variant / arm / str(row["id"])
            records = [
                json.loads(line) for line in (directory / "stages.jsonl").read_text().splitlines()
            ]
            persons = []
            for record in records:
                result = project_plan(
                    record["output"],
                    parser,
                    expected_days=range(1, len(row["base_person"]["daily_plans"]) + 1),
                )
                counts.update(result.audit["errors"])
                counts["outputs"] += 1
                counts["valid_final_structure"] += result.audit["valid"]
                persons.append({"person_idx": record["stage"], "plan": thaw(result.final_plan)})
                traces.append(
                    {
                        "group": row["id"],
                        "stage": record["stage"],
                        "raw_plan": thaw(result.raw_plan),
                        "audit": thaw(result.audit),
                    }
                )
            submissions.append({"id": row["id"], "persons": persons})
        target = out / f"{variant}-validated-submission.jsonl"
        target.write_text("".join(json.dumps(r) + "\n" for r in submissions))
        write(out / f"{variant}-projection-trace.json", traces)
        audit[variant] = upstream.score(rows, target)
    write(out / "validated-scores.json", audit)
    write(
        out / "projection-summary.json",
        {
            "revision": PROJECTION_REVISION,
            "counts": counts,
            "official_raw_scores": str(source / "scores.json"),
            "prior_diagnostic_scores": str(source / "final-boundary-diagnostic.json"),
            "semantic_task_success": "NOT_MEASURED",
        },
    )

    # Freeze real stage-0 model-extracted facts, never evaluator gold or later facts.
    seed_dir = source / "legacy-fact-small/fact-small/36"
    initial = json.loads((seed_dir / "fact-writes.jsonl").read_text().splitlines()[0])
    if initial["stage"] != 0:
        raise ValueError("fixture must use only initial observed facts")
    connection = sqlite3.connect(f"file:{seed_dir / 'store/index.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    facts, provenance = [], []
    try:
        for entry in initial["receipts"]:
            if not entry.get("committed"):
                continue
            row = connection.execute(
                "SELECT * FROM revisions WHERE part_id=? AND rev=?",
                (entry["id"], entry["revision"]),
            ).fetchone()
            body = (
                bytes(row["inline"])
                if row["inline"] is not None
                else (seed_dir / "store" / row["object_path"]).read_bytes()
            )
            if sha256(body).hexdigest() != row["digest"]:
                raise ValueError("source Fact checksum mismatch")
            facts.append(("fact." + entry["id"], json.loads(body)))
            provenance.append(entry)
    finally:
        connection.close()
    write(out / "frozen-initial-facts.json", {"facts": facts, "receipts": provenance})
    seed_row = next(row for row in rows if row["id"] == 36)
    objective = {
        "current_query": seed_row["questions"][0],
        "initial_traveler": seed_row["base_person"]["name"],
    }
    order = ("work_objective", *(key for key, _ in facts))
    policy = CompilerPolicy(max_tokens=256000, part_order=order)
    async with (
        Context(
            [
                ConfigSource("work_objective", objective).part,
                *(ConfigSource(key, value).part for key, value in facts),
            ],
            compiler_policy=policy,
        ) as legacy,
        ReplayContext(
            [ConfigSource(key, value).part for key, value in facts],
            objective_input=objective,
            compiler_policy=policy,
        ) as dx,
    ):
        left, right = await legacy.next_context(), await dx.next_context()
        fixture = ReplayFixture(
            "dx-initial-36",
            order,
            {
                s.part_id: {
                    "payload_hash": digest(s.payload),
                    "resource_id": s.part_id,
                    "revision": "fixed-initial-1",
                }
                for s in left.segments
            },
            manifest.get("upstream_revision", "see-source-manifest"),
            manifest["config"],
            PROJECTION_REVISION,
            "cl100k_base",
        )
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        for release in ("none", "resource", "revision", "order"):
            write_replay(
                out / f"replay-{release}",
                fixture,
                {
                    "legacy": canonical_replay(left, fixture, policy, release=release),
                    "dx": canonical_replay(right, fixture, policy, release=release),
                },
                tokenizer=tokenizer.encode,
            )

    # Explicit synthetic reader-isolation test, not a memory extraction benchmark.
    sources = []
    reqs = tuple(EvidenceRequirement(str(i), f"day{i}.accommodation") for i in range(1, 5))
    for i in range(1, 5):
        text = f"day{i} accommodation=hotel{i}"
        sources.append(
            SourceSnapshot(
                str(i),
                "1",
                "default",
                text,
                (SourceBinding(f"day{i}.accommodation", 0, len(text), f"hotel{i}"),),
            )
        )
    engine = EvidenceResearcher(IndexedNavigator(sources, authorize=lambda *_: True))
    candidates = tuple(h for s in sources[:3] for h in s.hits())
    coverage = {}
    for mode in ("shadow", "dynamic", "research"):
        bundle = await engine.run(reqs, candidates, mode=mode)
        coverage[mode] = thaw(bundle.manifest())
    write(out / "coverage-mechanisms.json", coverage)
    after = {name: sha256((source / name).read_bytes()).hexdigest() for name in old_hashes}
    assert old_hashes == after, "historical artifacts changed"
    write(
        out / "validation.json",
        {
            "original_artifacts_unchanged": True,
            "original_artifact_count": len(old_hashes),
            "model_calls": 0,
            "quality_verdict": "inconclusive",
            "paid_reliability": "NOT_MEASURED",
            "coverage_semantic_precision_recall": "NOT_MEASURED",
            "fact_extraction": "NOT_MEASURED",
            "scope": "X0/X1 + synthetic reader isolation",
        },
    )
    print(json.dumps({"output": str(out), "projection_counts": counts, "model_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/dx-v2-benchmark")
    parser.add_argument("--output", default="results/dx-intelligence")
    asyncio.run(run(parser.parse_args()))
