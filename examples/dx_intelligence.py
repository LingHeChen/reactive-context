"""Run: python examples/dx_intelligence.py. No model, API key or network required."""

import asyncio
import json

from reactive_context import CompilerPolicy, Context, state
from reactive_context.core.snapshot import thaw
from reactive_context.evidence import (
    EvidencePart,
    EvidenceRequirement,
    EvidenceResearcher,
    IndexedNavigator,
    ResearchBudget,
    SourceBinding,
    SourceSnapshot,
)


class Task(Context):
    goal = state("保留四天住宿安排，只调整交通")


def archive():
    # Host schema adapter binds entity/day/field to the immutable original spans.
    # These are reader-isolation fixtures, not an extraction-quality benchmark.
    sources = []
    for day in range(1, 5):
        text = f"旅客 A 第 {day} 天已确认住宿：旅馆 {day}"
        sources.append(
            SourceSnapshot(
                id=f"event-{day}",
                revision="1",
                scope="default",
                text=text,
                bindings=(SourceBinding(f"A.day{day}.accommodation", 0, len(text), f"旅馆{day}"),),
            )
        )
    return sources


async def main():
    sources = archive()
    slots = tuple(
        EvidenceRequirement(str(day), f"A.day{day}.accommodation", predicate="preserve")
        for day in range(1, 5)
    )
    initial_hits = tuple(hit for source in sources[:3] for hit in source.hits())
    for mode in ("shadow", "dynamic", "research"):
        navigator = IndexedNavigator(sources, authorize=lambda scope, _: scope == "default")
        engine = EvidenceResearcher(
            navigator,
            budget=ResearchBudget(
                max_rounds=3,
                max_io=16,
                max_tokens=4096,
                timeout_seconds=2,
            ),
        )
        evidence = EvidencePart(
            "memory.evidence", engine, slots, candidates=initial_hits, mode=mode
        )
        policy = CompilerPolicy(
            research_mode="shadow" if mode == "shadow" else "schema_coverage",
            evidence_fail_closed=mode != "shadow",
            # External navigation lacks an atomic revision API. Strict mode rejects it.
            include_uncertain=True,
        )
        async with Task([evidence.part], compiler_policy=policy) as ctx:
            result = await evidence.compile(ctx)
            report = result.manifest["evidence"]
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "sufficiency": report["sufficiency"],
                        "missing": report["missing_ids"],
                        "search_rounds": report["research_cost"]["search_rounds"],
                        "delivery": thaw(result.manifest["evidence_delivery"]),
                        "messages": [
                            {"role": m.role, "content": m.content} for m in result.messages
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
