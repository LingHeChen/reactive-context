"""Offline DX/core performance sample; no model calls. Run from the repository root."""

import asyncio
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

from reactive_context import Context, computed, computed_part, state
from reactive_context.sources.config import ConfigSource


def percentile(values, fraction):
    return sorted(values)[round((len(values) - 1) * fraction)]


async def measure(context, mutate, count=100):
    await context.compile()
    unchanged, changed = [], []
    for index in range(count):
        start = perf_counter()
        await context.compile()
        unchanged.append((perf_counter() - start) * 1000)
        mutate(index + 2)
        start = perf_counter()
        await context.compile()
        changed.append((perf_counter() - start) * 1000)
    result = {}
    for name, values in (("unchanged", unchanged), ("changed", changed)):
        result[name] = {"p50_ms": median(values), "p95_ms": percentile(values, 0.95)}
    await context.aclose()
    return result


async def main():
    # 100 computed nodes; root affects only ten, so both paths can check unrelated recomputation.
    counts = {"options": [0] * 100, "legacy": [0] * 100}
    declarations = {"root": state(1), "other": state(5)}
    legacy_sources = [ConfigSource("root", 1), ConfigSource("other", 5)]
    legacy = [source.part for source in legacy_sources]
    for index in range(100):
        dependency = "root" if index < 10 else "other"

        def evaluate(self, index=index, dependency=dependency):
            counts["options"][index] += 1
            return getattr(self, dependency) + index

        def compute(reader, index=index, dependency=dependency):
            counts["legacy"][index] += 1
            return reader.dependencies[dependency].value + index

        declarations[f"node_{index}"] = computed(evaluate)
        legacy.append(
            computed_part(
                key=f"node_{index}",
                schema=int,
                dependencies=[dependency],
                compute=compute,
            )
        )
    options = type("Sample", (Context,), declarations)()
    # Rendered budgets have room for all nodes under the existing conservative estimator.
    legacy_context = Context(legacy)
    timings = {
        "legacy": await measure(legacy_context, legacy_sources[0].set),
        "options": await measure(options, lambda value: options.update(root=value)),
    }
    for arm in counts:
        assert counts[arm][:10] == [101] * 10
        assert counts[arm][10:] == [1] * 90
    report = {
        "python": platform.python_version(),
        "samples_per_case": 100,
        "nodes": 102,
        "affected_computed_nodes": 10,
        "unrelated_recomputations": 0,
        "unchanged_recomputations": 0,
        "timings": timings,
        "requests": 0,
        "note": "Local mechanism sample, not LLM quality or cross-machine latency target.",
    }
    target = Path("results/dx-v2/performance.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(target.write_text, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
