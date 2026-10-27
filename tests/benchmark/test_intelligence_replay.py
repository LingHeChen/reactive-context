import re

import pytest

from reactive_context import CompilerPolicy, Context, state
from reactive_context.benchmarking.frozen_replay import (
    ReplayFixture,
    canonical_replay,
    capture,
    paired_bootstrap,
)
from reactive_context.benchmarking.plan_projection import final_plan_view, project_plan
from reactive_context.core.snapshot import digest
from reactive_context.sources.config import ConfigSource


def parse(text):
    rows = []
    for m in re.finditer(r"Day\s*(\d+):(.*?)(?=Day\s*\d+:|$)", text or "", re.S):
        row = {"day": int(m[1])}
        for line in m[2].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                row[k.strip().lower()] = v.strip()
        rows.append(row)
    return rows or None


def test_plan_projection_excludes_outline_and_rejects_ambiguous_or_invalid_structure():
    final = "=== Alice's Plan ===\nDay 1:\nDinner: A\nDay 2:\nDinner: B"
    raw = "Outline\nDay 1: first\nDay 2: second\n" + final
    result = project_plan(raw, parse, required_fields=("dinner",))
    assert len(result.raw_plan) == 4
    assert len(result.final_plan) == 2
    assert result.audit["valid"]
    assert final_plan_view(raw) == final
    for bad in [
        final + "\nDay 1:\nDinner: C",
        final.replace("Dinner: B", "Dinner: "),
        "Day 1:\nDinner: A",
        final + "\n" + final,
        final.replace("Day 2", "Day 3"),
    ]:
        invalid = project_plan(bad, parse, required_fields=("dinner",))
        assert invalid.final_plan is None
        assert invalid.audit["errors"]


async def test_frozen_fact_and_objective_bytes_match_across_interfaces():
    class DX(Context):
        objective = state("current public goal")

    fact = {"subject": "A", "value": "B", "evidence": ["raw:1"]}
    policy = CompilerPolicy(part_order=("objective", "fact.001"))
    async with (
        Context(
            [
                ConfigSource("objective", "current public goal").part,
                ConfigSource("fact.001", fact).part,
            ],
            compiler_policy=policy,
        ) as old,
        DX([ConfigSource("fact.001", fact).part], compiler_policy=policy) as new,
    ):
        a, b = await old.next_context(), await new.next_context()
        fixture = ReplayFixture(
            "frozen-1",
            policy.part_order,
            {
                s.part_id: {
                    "payload_hash": digest(s.payload),
                    "resource_id": s.part_id,
                    "revision": "frozen-1",
                }
                for s in a.segments
            },
            "synthetic-protocol-v1",
            {"calls": 0},
            "declared-final-plan-v2",
            "utf8",
        )
        left, right = [capture(canonical_replay(s, fixture, policy)) for s in (a, b)]
        assert left["bytes_hash"] == right["bytes_hash"]
        assert left["serialized_utf8"] == right["serialized_utf8"]
        assert (
            capture(canonical_replay(a, fixture, policy, release="resource"))["bytes_hash"]
            != (capture(canonical_replay(b, fixture, policy, release="resource"))["bytes_hash"])
        )
        new.objective = "changed"
        with pytest.raises(ValueError, match="payload mismatch"):
            canonical_replay(await new.next_context(), fixture, policy)


def test_small_paired_experiment_remains_inconclusive():
    result = paired_bootstrap([0.8, 0.9, 0.8], [0.9, 0.8, 0.9])
    assert result["episodes"] == 3
    assert result["verdict"] == "inconclusive"


def test_public_trip_days_are_checked_separately_from_boundary_only_diagnostic():
    output = "=== Alice's Plan ===\nDay 1:\nDinner: A"
    assert project_plan(output, parse, required_fields=("dinner",)).final_plan is not None
    result = project_plan(output, parse, required_fields=("dinner",), expected_days=(1, 2))
    assert result.final_plan is None
    assert result.audit["errors"] == ("unexpected_days",)
    assert result.audit["expected_days"] == (1, 2)
